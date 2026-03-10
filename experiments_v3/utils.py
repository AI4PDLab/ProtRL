import torch
import numpy as np
from torch.utils.data import Dataset
try:
    from transformers import AutoTokenizer, EsmForProteinFolding, AutoModelForSequenceClassification
except ImportError:
    pass
try:
    from peft import LoraConfig, inject_adapter_in_model
except ImportError:
    pass
import os
import subprocess
import torch.nn as nn
from Bio.PDB import PDBParser
import argparse
import csv
try:
    import esm
except ImportError:
    pass

class ESMFolder:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EsmForProteinFolding.from_pretrained(model_path).to(self.device)
        self.model.eval()

    def fold_sequence(self, sequence, name, output_dir):
        with torch.no_grad():
            output = self.model.infer_pdb(sequence)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{name}.pdb")
        with open(output_path, "w") as f:
            f.write(output)
        return output_path

def run_foldseek(query_pdb, target_pdb, output_file, foldseek_bin):
    fmt = "query,target,alntmscore,qtmscore,ttmscore,alnlen"
    cmd = [
        foldseek_bin, "easy-search",
        query_pdb, target_pdb, output_file, "tmp_foldseek",
        "--format-output", fmt,
        "--exhaustive-search", "1", "-e", "inf", "--tmscore-threshold", "0.0"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True

def run_mmseqs_clustering(fasta_file, output_prefix, mmseqs_bin, min_seq_id=0.9):
    cmd = [
        mmseqs_bin, "easy-cluster",
        fasta_file, output_prefix, "tmp_mmseqs",
        "--min-seq-id", str(min_seq_id)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Count clusters from output
    cluster_file = output_prefix
    with open(cluster_file, 'r') as f:
        reps = set()
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                reps.add(parts[0])
    return len(reps)

def run_mmseqs_search(query_fasta, target_db, output_m8, mmseqs_bin):
    # Format: query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits
    cmd = [
        mmseqs_bin, "easy-search",
        query_fasta, target_db, output_m8, "tmp_mmseqs_search",
        "--format-output", "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits"
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True

# =============================================================================
# CLEAN / Embedding Utils
# =============================================================================

class LayerNormNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, device, dtype, drop_out=0.1):
        super(LayerNormNet, self).__init__()
        self.fc1 = nn.Linear(1280, hidden_dim, dtype=dtype, device=device)
        self.ln1 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, dtype=dtype, device=device)
        self.ln2 = nn.LayerNorm(hidden_dim, dtype=dtype, device=device)
        self.fc3 = nn.Linear(hidden_dim, out_dim, dtype=dtype, device=device)
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        x = self.dropout(self.ln1(self.fc1(x)))
        x = torch.relu(x)
        x = self.dropout(self.ln2(self.fc2(x)))
        x = torch.relu(x)
        x = self.fc3(x)
        return x

def extract_mean_plddt(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('', pdb_file)
    b_factors = [atom.get_bfactor() for atom in structure.get_atoms()]
    return sum(b_factors) / len(b_factors)

def get_ec_id_dict(csv_name):
    ec_list = []
    seen_ecs = set()
    with open(csv_name, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader) 
        for rows in reader:
            if len(rows) < 2: continue
            ecs = rows[1].split(';')
            for ec in ecs:
                if ec not in seen_ecs:
                    seen_ecs.add(ec)
                    ec_list.append(ec)
    return ec_list

def load_clean_model(model_path, device):
    dtype = torch.float32
    model = LayerNormNet(512, 128, device, dtype)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def load_esm1b_model(model_path, device):
    with torch.serialization.safe_globals([argparse.Namespace]):
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(model_path)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)
    return model, batch_converter

def get_esm_embedding(model, batch_converter, sequence, device):
    data = [("seq", sequence)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    seq_len = len(sequence)
    return token_representations[0, 1 : 1 + seq_len].mean(0)

def map_emb_center(target_ec, ec_list, train_emb):
    positions = [i for i, ec in enumerate(ec_list) if ec == target_ec]
    if not positions:
        return None
    refs = train_emb[positions]
    return refs.mean(0)

# =============================================================================
# Helper Classes and Functions for Activity Prediction
# =============================================================================

class SequenceDataset(Dataset):
    def __init__(self, tokenized_sequences):
        self.input_ids = torch.cat([seq["input_ids"] for seq in tokenized_sequences])
        self.attention_mask = torch.cat([seq["attention_mask"] for seq in tokenized_sequences])

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

def load_esm_classification(checkpoint, num_labels, half_precision=False, full=False, deepspeed=True):
    """
    Loads an ESM model for sequence classification.
    Optionally injects LoRA adapters unless full=True.
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels,
        torch_dtype=torch.float16 if half_precision and deepspeed else None
    )
    if full:
        return model, tokenizer

    peft_config = LoraConfig(
        r=4, lora_alpha=1, bias="all", target_modules=["query", "key", "value", "dense"]
    )
    model = inject_adapter_in_model(peft_config, model)
    for param_name, param in model.classifier.named_parameters():
        param.requires_grad = True
    return model, tokenizer

def load_finetuned_esm(checkpoint, filepath, num_labels=1, mixed=False, full=False, deepspeed=True):
    """
    Loads a base ESM model (with LoRA) and then loads specific finetuned weights.
    """
    model, tokenizer = load_esm_classification(checkpoint, num_labels, mixed, full, deepspeed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if os.path.exists(filepath):
        non_frozen_params = torch.load(filepath, map_location=device)
        for param_name, param in model.named_parameters():
            if param_name in non_frozen_params:
                param.data = non_frozen_params[param_name].data
    else:
        print(f"Warning: LoRA weights not found at {filepath}, using base model.")
        
    return tokenizer, model

def compute_esm1v_pll(model, batch_converter, sequences, device):
    """
    Computes Pseudo-Log-Likelihood (PLL) for a batch of sequences using ESM-1v.
    PLL is the sum of log-probabilities of each residue conditioned on all others (masked).
    Returns list of PLL scores.
    WARNING: Expensive O(L^2) calculation.
    """
    model.eval()
    pll_scores = []
    
    with torch.no_grad():
        for seq in sequences:
            # Prepare batch for this sequence (masking one by one)
            # This is very slow for long sequences.
            # Optimization: Batch the masks.
            
            L = len(seq)
            if L == 0:
                pll_scores.append(0.0)
                continue
                
            # Create L copies, each with one mask
            # If L is large (e.g. 500), we can't do single batch.
            # Batch size limited by memory.
            
            # Simple approach: Loop over positions
            log_probs_sum = 0.0
            
            # Tokenize single sequence
            # batch_converter handles list of (label, seq)
            
            # We need to construct inputs manually to be efficient or use the converter
            # ESM tokenizer adds CLS and EOS.
            # <cls> s1 s2 ... sL <eos>
            # Indices: 0 (cls), 1..L (res), L+1 (eos)
            
            # Use batch converter to get tokens
            labels, strs, tokens = batch_converter([("seq", seq)])
            tokens = tokens.to(device) # shape [1, L+2]
            
            # Target tokens (original)
            target = tokens.clone()
            
            # Iterate 1 to L
            for i in range(1, L + 1):
                # Mask position i
                masked_tokens = tokens.clone()
                masked_tokens[0, i] = model.alphabet.mask_idx
                
                # Forward pass
                results = model(masked_tokens, repr_layers=[], return_contacts=False)
                logits = results["logits"] # [1, L+2, vocab_size]
                
                # Get log prob of the true token at position i
                # LogSoftmax over vocab
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # True token index
                true_idx = target[0, i]
                
                # Accumulate
                log_probs_sum += log_probs[0, i, true_idx].item()
                
            pll_scores.append(log_probs_sum)
            
    return pll_scores
