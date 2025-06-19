import os
import math
import random
import argparse
import statistics
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AlbertTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict
from tqdm import tqdm
import matplotlib.pyplot as plt


# ---------------------------
# Hyperparameters and Config
# ---------------------------
CONFIG = {
    "beta": 0.1,              
    "seed": 1998,
    "learning_rate": 1e-7,    
    "batch_size": 8,         
    "num_epochs": 10,        
    "split_percent": 0.2,
    "adam_betas": (0.9, 0.999),  
    "epsilon": 1e-8,
    "adam_decay": 0.01,      
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]

# ---------------------------
# Utility Functions
# ---------------------------
def seed_everything(seed=2003):
    """
    Sets random seed for reproducibility across libraries.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def models_equal(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if not torch.equal(p1, p2):
            return False
    return True

def formatting_aa_sequence(sequence):
    """
    Formats correctly the sequence as in the RexZyme trainset.
    """
    return sequence 

def formatting_smile_sequence(smile):
    """
    Formats correctly the sequence as in the RexZyme trainset.
    """
    return "translation"+smile



def save_model_and_tokenizer(model, output_dir):
    """
    Saves the model and tokenizer to a specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


# ---------------------------
# Dataset Generation
# ---------------------------
def generate_dataset(iteration_num, label, mode):
    reaction = "O=C([O-])O.[H+]>>O.O=C=O"
    data = dict()
    data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        "reaction" : [],
        }
    seq_lenght = []
    with open(f"FDH_reac_generated_iteration{iteration_num-1}.fasta", "r") as f:
        rep_seq = f.readlines()

    sequences_rep = {}
    for line in rep_seq:
        if ">" in line:
            name = line.split("\t")[0].replace(">", "").strip()
        else:
            sequences_rep[name] = {"sequence": line.strip()}

    for entry in sequences_rep:
        
            name = entry
            sequence = sequences_rep[str(name)]['sequence']
            lenght_rew = 60-len(sequence) if len(sequence)>60 else 0   # Here we want to minimize the lenght, so we use a negative value
            seq_lenght.append(len(sequence))  
            
            data["sequence"].append(formatting_aa_sequence(sequence))
            data["seq_name"].append(entry)
            data["weight"].append(float(lenght_rew))
            data["reaction"].append(formatting_smile_sequence(reaction))

    print(f"The average lenght at iteration {iteration_num} is: {statistics.mean(seq_lenght)}")
    # Convert data dictionary to a Hugging Face Dataset
    hf_dataset = Dataset.from_pandas(pd.DataFrame(data))

    # Prepare pairs if mode is 'paired'
    if mode == 'paired':
        hf_dataset = prepare_pairs(hf_dataset)

    # Shuffle and split the dataset
    shuffled_dataset = hf_dataset.shuffle(seed=CONFIG["seed"])
    train_size = int((1 - CONFIG["split_percent"]) * len(shuffled_dataset))
    train_dataset = shuffled_dataset.select(range(train_size))
    eval_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))

    # Save the dataset to disk and return
    final_dataset = DatasetDict({"train": train_dataset, "eval": eval_dataset})
    final_dataset.save_to_disk(f"dataset_iteration{iteration_num}")

    return final_dataset


def prepare_pairs(hf_dataset):
    """
    Prepare paired data from the paired form of DPO.
    """
    # Sort the dataset by weight in descending order
    sorted_dataset = hf_dataset.sort("weight", reverse=False)

    # Split the dataset into two halves
    mid_point = len(sorted_dataset) // 2
    first_half = sorted_dataset.select(range(mid_point))
    second_half = sorted_dataset.select(range(mid_point, len(sorted_dataset)))

    # Create pairs of positive and negative sequences
    pairs = []
    for pos_example, neg_example in zip(first_half, second_half):
        pairs.append({
            "positive_sequence": pos_example["sequence"],
            "negative_sequence": neg_example["sequence"],
        })

    return Dataset.from_list(pairs)


# ---------------------------
# Loss Functions
# ---------------------------
def log_likelihood(sequences, reaction, device, model, tokenizer_aa, tokenizer_smiles):
    all_log_likelihood = []
    for seq, reac in zip(sequences, reaction):
        # Tokenize a single sequence
        inputs = tokenizer_smiles(
            reac, 
            max_length=512, 
            padding=False, 
            truncation=True, 
            return_tensors="pt"
        ).to(device)

        label_ids = tokenizer_aa(
            seq,
            max_length=1024, 
            padding=False, 
            truncation=True, 
            return_tensors="pt"
        ).input_ids.to(device)


        # Forward pass
        outputs = model(**inputs, labels=label_ids)
        neg_log_likelihood = outputs.loss  # averaged negative log-likelihood

        # Convert to likelihood by multiplying by -1
        all_log_likelihood.append(-neg_log_likelihood.unsqueeze(0))
    
    return torch.cat(all_log_likelihood)

def dpo_paired_loss(batch, model, ref_model, tokenizer, device, beta=0.1):
    """
    Calculates the paired DPO loss.
    """
    # Extract positive and negative sequences
    positive_sequence = batch["positive_sequence"]
    negative_sequence = batch["negative_sequence"]

    # Log probabilities for positive sequences
    pos_ref_log_probs = log_likelihood(positive_sequence, device, ref_model, tokenizer)
    pos_policy_log_probs = log_likelihood(positive_sequence, device, model, tokenizer)
    pos_ratios = beta * (pos_policy_log_probs - pos_ref_log_probs)

    # Log probabilities for negative sequences
    neg_ref_log_probs = log_likelihood(negative_sequence, device, ref_model, tokenizer)
    neg_policy_log_probs = log_likelihood(negative_sequence, device, model, tokenizer)
    neg_ratios = beta * (neg_policy_log_probs - neg_ref_log_probs)

    # Compute the DPO paired loss
    loss = -F.logsigmoid(pos_ratios - neg_ratios)

    return  torch.mean(loss)
    
def dpo_weighted_loss(pi_log_likelihood, ref_log_likelihood, weights, beta=0.1):
    """
    Calculates DPO weighted loss. 
    Function kindly provided by Widatalla et.al 2024 "Aligning protein 
    generative models with experimental fitness via Direct Preference Optimization"
    """

    if ref_log_likelihood is not None:
        pi_ratio = beta * (pi_log_likelihood - ref_log_likelihood)
    else:
        pi_ratio = beta * pi_log_likelihood

    weights = torch.softmax(weights, dim=0)
    loss = F.cross_entropy(pi_ratio, weights)
    
    return loss


def dpo_ranked_loss(pi_log_likelihood, pi_ref_loglikelihood, weights, beta=0.1):
    """
    Calculates the Directed Policy Optimization (DPO) ranked loss.
    In this case the ranking is on the batch dimension.
    """
    # Ensure weights have at least one dimension
    weights = torch.softmax(weights, dim=0)
    weights = weights.view(-1)  
    
    sorted_indices = torch.argsort(weights, descending=True)
    pi_log_likelihood = pi_log_likelihood[sorted_indices]
    pi_ref_loglikelihood = pi_ref_loglikelihood[sorted_indices] if not pi_ref_loglikelihood == pi_log_likelihood else None
    weights = weights[sorted_indices]
    print(f"Sorted weights: {weights}")

    if pi_ref_loglikelihood is None:
        pi_ratio = beta * pi_log_likelihood
    else:
        pi_ratio = beta * (pi_log_likelihood - pi_ref_loglikelihood)
    
    uniform_weights = torch.arange(pi_ratio.size(0) -1 , -1, -1, device=advantages.device, dtype=advantages.dtype)        
    uniform_weights = torch.softmax(uniform_weights, dim=0)
    
    loss = F.cross_entropy(pi_ratio, uniform_weights)
    return loss



# ---------------------------
# Training and Evaluation
# ---------------------------
def train(model, iteration_num, ref_model, tokenizer_aa, tokenizer_smiles, train_loader, optimizer, device, mode):
    model.train()
    total_loss = []
    prev_state = copy.deepcopy(model.state_dict())

    for batch in train_loader:
        # Save a snapshot of parameters before this update
        prev_state = copy.deepcopy(model.state_dict())

        if mode != 'paired':
            optimizer.zero_grad()
            sequences = batch["sequence"] 
            labels = batch["reaction"]

            ref_log_probs = log_likelihood(sequences, labels, device, ref_model, tokenizer_aa, tokenizer_smiles)
            policy_log_probs = log_likelihood(sequences, labels, device, model, tokenizer_aa, tokenizer_smiles)

            if iteration_num == 1:
                ref_log_probs = None

            weights = batch["weight"].to(device)

            if mode == "weighted":
                loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
            elif mode == "ranked":
                loss = dpo_ranked_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
            
        elif mode == "paired":
            loss = dpo_paired_loss(batch, model, ref_model, tokenizer, device, CONFIG["beta"])
        
        loss.backward()
        optimizer.step()

       
        total_loss.append(loss.item())
    
    # Compare current state to previous snapshot
    new_state = model.state_dict()
    conserved_par, changed_par = [],[]
    for key in prev_state:
        if not torch.allclose(prev_state[key], new_state[key]):
            conserved_par.append(key)
        else:
            changed_par.append(key)
    print(f"at iteration {iteration_num}: {len(changed_par)} layers has changed over the toatl: {len(prev_state)}")
    torch.cuda.empty_cache()
    return sum(total_loss) / len(total_loss)




def evaluate(model, iteration_num, ref_model, tokenizer_aa, tokenizer_smiles, eval_loader, optimizer, device, mode):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in eval_loader:
            if mode != 'paired':
                sequences = batch["sequence"]
                labels = batch["reaction"]
                ref_log_probs = log_likelihood(sequences, labels, device, ref_model, tokenizer_aa, tokenizer_smiles)
                policy_log_probs = log_likelihood(sequences, labels, device, model, tokenizer_aa, tokenizer_smiles)
                
                if models_equal(ref_model, model) or iteration_num == 1:
                    ref_log_probs = None
                
                weights = batch["weight"].to(device)

                if mode == "weighted":
                    loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])

                elif mode == "ranked":
                    loss = dpo_ranked_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])

            else:
                # Paired mode
                loss = dpo_paired_loss(batch, model, ref_model, tokenizer_aa, tokenizer_smiles, device, CONFIG["beta"])

            total_loss += loss.item()

    # Take the average loss across all batches in the eval_loader
    return total_loss / len(eval_loader)


# ---------------------------
# Main Function
# ---------------------------
def main(train_loader, eval_loader, iteration_num, model_directory, mode):
    """
    Main training loop for a given iteration.
    """

    model_name = model_directory if iteration_num == 1 else f"output_iteration{iteration_num - 1}"
    
    print(f"Model {model_name} has been loaded")

    config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=False,
        revision="main",
        use_auth_token=False,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        from_tf=bool(".ckpt" in model_name),
        config=config,
        cache_dir=False,
        use_auth_token=None,
    ).to(device)

    tokenizer_smiles = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/REXzyme_smiles")
    tokenizer_aa = AutoTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/REXzyme_aa")

    ref_model = AutoModelForSeq2SeqLM.from_pretrained(
        "/home/woody/b114cb/b114cb23/models/checkpoint-106500",
        from_tf=bool(".ckpt" in "/home/woody/b114cb/b114cb23/models/checkpoint-106500"),
        config=config,
        cache_dir=False,
        use_auth_token=None,
    ).to(device)

    n_params = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        betas=CONFIG["adam_betas"],
        eps=CONFIG["epsilon"],
        weight_decay=CONFIG["adam_decay"],
    )

    for epoch in range(CONFIG["num_epochs"]):

        train_loss = train(model, iteration_num, ref_model, tokenizer_aa, tokenizer_smiles, train_loader, optimizer, device, mode)
        eval_loss = evaluate(model, iteration_num, ref_model, tokenizer_aa, tokenizer_smiles, eval_loader, optimizer, device, mode)
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

        save_model_and_tokenizer(model, output_dir=f"output_iteration{iteration_num}")

    del model
    del ref_model
    torch.cuda.empty_cache()

# ---------------------------
#     MAIN
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, required=True)

    args = parser.parse_args()
    seed_everything(CONFIG["seed"])

    if not os.path.exists(f"dataset_iteration{args.iteration_num}"):
        dataset = generate_dataset(args.iteration_num, args.label.strip(), args.mode)
    else:
        dataset = load_from_disk(f"dataset_iteration{args.iteration_num}")

    print("Dataset Loaded!")
    train_loader = DataLoader(dataset["train"], batch_size=CONFIG["batch_size"], shuffle=True)
    eval_loader = DataLoader(dataset["eval"], batch_size=CONFIG["batch_size"], shuffle=False)

    main(train_loader, eval_loader, args.iteration_num, args.model_dir, args.mode)
