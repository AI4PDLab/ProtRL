import os
import math
import random
import argparse
import statistics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict
from tqdm import tqdm
#import matplotlib.pyplot as plt
import subprocess
import json
import tempfile
import glob
from glob import glob
import csv
from collections import defaultdict

#from functions import *
#
#import pdbfixer
#from pdbfixer import PDBFixer
#from openmm.app import PDBFile
import io



# ---------------------------
# Hyperparameters and Config
# ---------------------------
CONFIG = {
    "beta": 0.1,
    "seed": 1998,
    "learning_rate": 1e-5,
    "batch_size": 20,
    "num_epochs": 10,
    "split_percent": 0.2,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "adam_decay": 0.1,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def formatting_sequence(sequence):
    """
    Formats correctly the sequence as in the ProtGPT2 trainset.
    """
    return "<|endoftext|>\n"+"\n".join(sequence[i:i+60] for i in range(0, len(sequence), 60))+"\n"


def save_model_and_tokenizer(model, tokenizer, output_dir):
    """
    Saves the model and tokenizer to a specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


# ---------------------------
# Dataset Generation
# ---------------------------
def generate_dataset(iteration_num, label, mode):
    data = dict()
    data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        }
    seq_lenght = []
    with open(f"seq_gen_{label}_iteration{iteration_num-1}.fasta", "r") as f:
        rep_seq = f.readlines()

    sequences_rep = {}
    for line in rep_seq:
        if ">" in line:
            name = line.split("\t")[0].replace(">", "").strip()
        else:
            sequences_rep[name] = {"sequence": line.strip()}
    
    
   

    for entry in sequences_rep:
        try:
            name = entry
            sequence = sequences_rep[str(name)]['sequence']
            lenght = math.exp(-((((90/len(sequence))-1)**2)/(0.5**2))) # Gaussian center on 1. The closer the ratio between len and aligment is, the higher is the reward
            
            data["sequence"].append(formatting_sequence(sequence))
            data["seq_name"].append(entry)
            data["weight"].append(lenght)
        except:
                print(f"error for sequence {name}")
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
def batch_log_likelihood(sequences, device, model, tokenizer):

    cse_f = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_token_id)

    inputs = tokenizer(
        sequences,
        return_tensors='pt',
        add_special_tokens=False,
        padding=True,
        padding_side='left').to(device)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    outputs = model(input_ids, attention_mask=attention_mask)
    shift_logits = outputs.logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    # attention mask should match labels
    shift_mask = attention_mask[..., 1:].contiguous()

    log_likelihood_per_token = cse_f(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1)
    ).view(shift_labels.shape)


    # zero out log_p for pad tokens
    #log_likelihood_per_token = log_likelihood_per_token * shift_mask

    log_likelihood_per_seq = (
    (log_likelihood_per_token * shift_mask).sum(dim=1)
    / shift_mask.sum(dim=1)
)


    return log_likelihood_per_token

def log_likelihood(sequences, device, model, tokenizer):
    
    all_log_likelihood = []  # List to store loss for each sequence

    for sequence in sequences:
        inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
        #inputs = tokenizer(sequence, return_tensors='pt').input_ids.to(device)
        outputs = model(inputs, labels=inputs)
        neg_log_likelihood, logits = outputs[:2]                        # The HF loss output is the negative log-likelihood averaged over the number of tokens.
        all_log_likelihood.append(-neg_log_likelihood.unsqueeze(0)) # Convert negative log-likelihood to likelihood by multiplying by -1.
        
    all_log_likelihood = torch.cat(all_log_likelihood)
    
    return all_log_likelihood

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
    if models_equal(ref_model, model):
                pos_ref_log_probs = None
                
    if pos_ref_log_probs is None:
        pos_ratios = beta * pos_policy_log_probs
    else:
        pos_ratios = beta * (pos_policy_log_probs - pos_ref_log_probs)

    # Log probabilities for negative sequences
    neg_ref_log_probs = log_likelihood(negative_sequence, device, ref_model, tokenizer)
    neg_policy_log_probs = log_likelihood(negative_sequence, device, model, tokenizer)
    if neg_ref_log_probs is None:
        neg_ratios = beta * (neg_policy_log_probs)
    else:
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
    if ref_log_likelihood is None:
        pi_ratio = beta * pi_log_likelihood
    else:
        pi_ratio = beta * (pi_log_likelihood - ref_log_likelihood)
        
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

    uniform_weights = torch.ones_like(pi_ratio)
    print(f"pi ratios: {pi_ratio}")

    
    loss = F.cross_entropy(pi_ratio, uniform_weights)
    return loss



# ---------------------------
# Training and Evaluation
# ---------------------------
import copy
import torch

def train(model, iteration_num, ref_model, tokenizer, train_loader, optimizer, device, mode):
  
    model.train()
    total_loss = []
    for batch in train_loader:
        # Save a snapshot of parameters before this update
        prev_state = copy.deepcopy(model.state_dict())

        if mode != 'paired':
            optimizer.zero_grad()
            sequences = batch["sequence"] 
            ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
            policy_log_probs = log_likelihood(sequences, device, model, tokenizer)

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

        # Compare current state to previous snapshot
        new_state = model.state_dict()
        for key in prev_state:
            if not torch.allclose(prev_state[key], new_state[key]):
                print(f"Parameter '{key}' has changed after update.")
            else:
                print(f"Parameter '{key}' is unchanged after update.")

        total_loss.append(loss.item())
    
    torch.cuda.empty_cache()
    return sum(total_loss) / len(total_loss)



def evaluate(model, iteration_num, ref_model, tokenizer, eval_loader, optimizer, device, mode):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in eval_loader:
            if mode != 'paired':
                sequences = batch["sequence"]
                ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
                policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
                
                if models_equal(ref_model, model) or iteration_num == 1:
                    ref_log_probs = None
                
                weights = batch["weight"].to(device)

                if mode == "weighted":
                    loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])

                elif mode == "ranked":
                    loss = dpo_ranked_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])

            else:
                # Paired mode
                loss = dpo_paired_loss(batch, model, ref_model, tokenizer, device, CONFIG["beta"])

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

    tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(model_directory).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        betas=CONFIG["adam_betas"],
        eps=CONFIG["epsilon"],
        weight_decay=CONFIG["adam_decay"],
    )

    for epoch in range(CONFIG["num_epochs"]):
        train_loss = train(model, iteration_num, ref_model, tokenizer, train_loader, optimizer, device, mode)
        eval_loss = evaluate(model, iteration_num, ref_model, tokenizer, eval_loader, optimizer, device, mode)
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")

        save_model_and_tokenizer(model, tokenizer, output_dir=f"output_iteration{iteration_num}")

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
