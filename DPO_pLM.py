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
import matplotlib.pyplot as plt


# ---------------------------
# Hyperparameters and Config
# ---------------------------
CONFIG = {
    "beta": 0.01,
    "seed": 1998,
    "learning_rate": 1e-7,
    "batch_size": 4,
    "num_epochs": 5,
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

def formatting_sequence(sequence, ec_label):
    """
    Formats correctly the sequence as in the ZymCTRL trainset.
    """
    return f"{ec_label}<sep><start>{sequence}<end><|endoftext|>"


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
def generate_dataset(iteration_num, ec_label, mode):
    data = dict()
    data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        }
    
    with open(f"seq_gen_{ec_label}_iteration{iteration_num-1}.fasta", "r") as f:
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
                                                                       # if we reach the objective (len < 0), the score is 0
            data["sequence"].append(formatting_sequence(sequence, ec_label))
            data["seq_name"].append(entry)
            data["weight"].append(float(lenght_rew))
     
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
def per_token_log_likelihood(input_texts, device, model, tokenizer):

    inputs = tokenizer(input_texts, padding=True, padding_side='right', return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1)

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    shift_mask = attention_mask[..., 1:]#.contiguous()
    # Use attention_mask to normalise to exclude pad_tokens
    gen_probs = (gen_probs * shift_mask) / shift_mask.sum(dim=1, keepdim=True)

    return gen_probs




def log_likelihood(sequences, device, model, tokenizer):
    
    all_log_likelihood = []  # List to store loss for each sequence

    for sequence in sequences:
        inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
        outputs = model(inputs, labels=inputs)
        neg_log_likelihood, logits = outputs[:2]  # The HF loss output is the negative log-likelihood averaged over the number of tokens.
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
    if ref_log_likelihood is None: # First iteration, where ref_model == model, ratio consider only the model, otherwise loss would be zero
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
    
    uniform_weights = torch.arange(pi_ratio.size(0) -1 , -1, -1, device=advantages.device, dtype=advantages.dtype)        
    uniform_weights = torch.softmax(uniform_weights, dim=0)
    
    loss = F.cross_entropy(pi_ratio, uniform_weights)
    return loss



# ---------------------------
# Training and Evaluation
# ---------------------------
def train(model, ref_model, tokenizer, iteration_num, train_loader, optimizer, device, mode):
    """
    Performs training for one epoch.
    """
    model.train()
    total_loss = []
    for batch in train_loader:

        if mode != 'paired':
            optimizer.zero_grad()
            sequences = batch["sequence"] 
            ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
            policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
            
            if models_equal(ref_model, model) or iteration_num == 1: # If the model == ref_model we set ref_log_probs to be None
                ref_log_probs = None
                
            weights = batch["weight"].to(device)
            
            if mode == "weighted":
                loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
            
            if mode == "ranked":
                loss = dpo_ranked_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
            
        if mode == "paired":
            loss = dpo_paired_loss(batch, model, ref_model, tokenizer, device, CONFIG["beta"])
        
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
    
    torch.cuda.empty_cache()

    return sum(total_loss) / len(total_loss)


def evaluate(model, ref_model, tokenizer, iteration_num, eval_loader, optimizer, device, mode):
    """
    Evaluates the model on the evaluation set.
    """
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch in eval_loader:
            if mode != 'paired':
                optimizer.zero_grad()
                sequences = batch["sequence"] 
                ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
                policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
                
                if models_equal(ref_model, model) or iteration_num == 1: # If the model == ref_model we set ref_log_probs to be None
                    ref_log_probs = None
                    
                weights = batch["weight"].to(device)
                
                if mode == "weighted":
                    loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
                
                if mode == "ranked":
                    loss = dpo_ranked_loss(policy_log_probs, ref_log_probs, weights, CONFIG["beta"])
                
        if mode == "paired":
            loss = dpo_paired_loss(batch, model, ref_model, tokenizer, device, CONFIG["beta"])
        
        total_loss.append(loss.item())
    
    torch.cuda.empty_cache()

    return sum(total_loss) / len(total_loss)


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
        train_loss = train(model, ref_model, tokenizer, iteration_num, train_loader, optimizer, device, mode)
        eval_loss = evaluate(model, ref_model, tokenizer, iteration_num, eval_loader, optimizer, device, mode)
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
