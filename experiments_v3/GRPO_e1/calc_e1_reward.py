import argparse
import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

# Try importing E1 modules
try:
    from E1.batch_preparer import E1BatchPreparer
    from E1.modeling import E1ForMaskedLM
except ImportError:
    print("Error: Could not import E1. Ensure PYTHONPATH includes the E1 repository.")
    sys.exit(1)

def calculate_pll(model, batch_preparer, sequences, device, batch_size=4):
    """
    Calculates Pseudo-Log-Likelihood (PLL) using E1BatchPreparer and '?' masking.
    PLL = Sum_{i=1 to L} log P(x_i | x_{mask_i})
    """
    model.eval()
    pll_scores = []
    
    for seq in tqdm(sequences, desc="Calculating PLL"):
        try:
            seq_len = len(seq)
            if seq_len == 0:
                pll_scores.append(0.0)
                continue
                
            # Create masked versions of the sequence (one mask per position)
            # Notebook strategy: replace char with '?'
            masked_versions = [seq[:i] + "?" + seq[i+1:] for i in range(seq_len)]
            
            # Helper to get log probs for a batch of masked sequences
            seq_log_probs = []
            
            for i in range(0, len(masked_versions), batch_size):
                batch_masked_seqs = masked_versions[i : i + batch_size]
                
                # Prepare batch
                try:
                    batch = batch_preparer.get_batch_kwargs(batch_masked_seqs, device=device)
                except Exception as e:
                    print(f"Error preparing batch: {e}")
                    raise e

                input_ids = batch["input_ids"]
                
                with torch.no_grad():
                     with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        outputs = model(
                            input_ids=batch["input_ids"],
                            within_seq_position_ids=batch.get("within_seq_position_ids"),
                            global_position_ids=batch.get("global_position_ids"),
                            sequence_ids=batch.get("sequence_ids"),
                            past_key_values=None,
                            use_cache=False,
                            output_attentions=False,
                            output_hidden_states=False,
                        )
                
                logits = outputs.logits # (B, L_tokens, V)
                log_softmax = F.log_softmax(logits, dim=-1)
                
                # We need to extract the log prob of the TRUE token at the MASKED position.
                # Challenge: Tokenization might result in different lengths or offsets? 
                # E1/ProteinBERT tokenization usually maps 1 char to 1 token + specials.
                # batch_preparer handles specials.
                
                # From notebook:
                # "log_probs = [log_probs[i, last_sequence_residue_selector[i]] ...]"
                # This extracts ALL residue probs.
                # We specifically need the prob at the masked position.
                # The masked position in the input string is `i + current_batch_offset`.
                # BUT, tokenizer adds special tokens.
                
                # Identify the mask token index in input_ids
                # E1 tokenizer mask token? We can look for it.
                # batch_preparer.tokenizer.mask_token_id
                mask_token_id = batch_preparer.tokenizer.mask_token_id
                
                for j, masked_seq_str in enumerate(batch_masked_seqs):
                    # Original character at this position
                    global_idx = i + j
                    target_char = seq[global_idx]
                    
                    # Find mask token in input_ids[j]
                    # Note: There should be exactly one mask token per sequence here
                    # (unless '?' appears naturally? No, '?' is the mask instruction)
                    token_ids = input_ids[j]
                    mask_indices = (token_ids == mask_token_id).nonzero(as_tuple=True)[0]
                    
                    if len(mask_indices) != 1:
                        # Fallback or error logic
                        # Maybe tokenizer split '?' or something weird happened
                        # Or structural tokens?
                        # Taking the first one if multiple found (unlikely)
                        mask_idx = mask_indices[0].item()
                    else:
                        mask_idx = mask_indices[0].item()
                        
                    # Get target token ID
                    # We need to tokenize the target character to get its ID
                    # Or rely on vocab.
                    # E1 tokenizer vocab usually maps 'A' -> ID.
                    # batch_preparer.tokenizer.convert_tokens_to_ids(target_char)
                    target_token_id = batch_preparer.tokenizer.convert_tokens_to_ids(target_char)
                    
                    # Extract log prob
                    log_prob = log_softmax[j, mask_idx, target_token_id].item()
                    seq_log_probs.append(log_prob)

            # Sum of log probs (Pseudo-Log-Likelihood)
            pll = sum(seq_log_probs)
            pll_scores.append(pll)
            
        except Exception as e:
            print(f"Error for sequence {seq}: {e}")
            pll_scores.append(-9999.0)
            
    return pll_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--label", type=str, default="enzyme") 
    args = parser.parse_args()

    logs_file = os.path.join(args.output_dir, "logs.csv")
    if not os.path.exists(logs_file):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "Profluent-Bio/E1-150m"
    
    print(f"Loading {model_name}...")
    try:
        # Use E1 classes
        model = E1ForMaskedLM.from_pretrained(model_name, trust_remote_code=True).to(device).eval()
        batch_preparer = E1BatchPreparer()
    except Exception as e:
        print(f"Failed to load E1 model/preparer: {e}")
        return

    df = pd.read_csv(logs_file)
    mask = df['iteration_num'] == args.iteration_num
    
    if not mask.any():
        print(f"No rows for iteration {args.iteration_num}")
        return

    sequences = []
    indices = []
    
    for index, row in df[mask].iterrows():
        raw_seq = row['sequence']
        if "<start>" in raw_seq:
            try: raw_seq = raw_seq.split("<start>")[1].split("<end>")[0]
            except: pass
        raw_seq = raw_seq.replace("<sep>", "").replace("<|endoftext|>", "").strip()
        sequences.append(raw_seq)
        indices.append(index)

    if sequences:
        pll_scores = calculate_pll(model, batch_preparer, sequences, device)
        for idx, score in zip(indices, pll_scores):
            df.at[idx, 'e1_pll'] = score
            df.at[idx, 'clean_reward'] = score
            
        df.to_csv(logs_file, index=False)
        print(f"Updated rewards for {len(sequences)} sequences.")

if __name__ == "__main__":
    main()
