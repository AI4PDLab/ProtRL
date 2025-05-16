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
from math import log, log10, exp
from transformers import EsmTokenizer, EsmModel

# Load the ESM model and tokenizer
tokenizer_esm = EsmTokenizer.from_pretrained("/home/woody/b114cb/b114cb23/models/esm2_t36_3B_UR50D")
model_esm = EsmModel.from_pretrained("/home/woody/b114cb/b114cb23/models/esm2_t36_3B_UR50D")


# ---------------------------
# Hyperparameters and Config
# ---------------------------
CONFIG = {
    "beta": 0.01,
    "seed": 1998,
    "learning_rate": 1e-7,
    "batch_size": 4,
    "num_epochs": 20,
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


def save_model_and_tokenizer(model, tokenizer, output_dir):
    """
    Saves the model and tokenizer to a specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")

def compute_embeddings(seq, model_esm, tokenizer_esm):
    model_esm.to(device)
    with torch.no_grad():
            inputs = tokenizer_esm(seq, return_tensors="pt", max_length = 10000, truncation=True, padding=False).to('cuda')
            # compute single seq embedding, calculate mean across seq len dimension
            output = torch.mean( model_esm(**inputs).last_hidden_state, dim = 1)
    return output

def cosine_similarity(emb, reference_emb):
    cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    similarity = cos(emb, reference_emb)
    return similarity


def compute_distance(data):
    distance = []
    seq = "NSDSECPLSHDGYCLHDGVCMYIEALDKYACNCVVGYIGERCQYRDLKWWE"
    ref_embedding = compute_embeddings(seq, model_esm, tokenizer_esm)
    for seq in data["sequence"]:
        distance.append(cosine_similarity(compute_embeddings(seq, model_esm, tokenizer_esm), ref_embedding).to('cpu').item())
    
    # Create a new list for normalized distances
    max_distance = max(distance)
    normalized_distance = [x / max_distance for x in distance]  # Adjust the normalization as needed

    return normalized_distance
def min_max_normalize(s):
    return (s - s.min()) / (s.max() - s.min())



# ---------------------------
# Dataset Generation
# ---------------------------
def generate_dataset(iteration_num, label, mode):
    data = dict()
    data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        "kd_norm" : [],
        "kon_norm" : [],
        }
    seq_lenght = []

    ec_label = "1.3.3.18"
    min_kd = 1.00e-3
    
    df = pd.read_csv("EGFR_binder_summary.csv")
    
    mapping = {'high': 3, 'medium': 2, 'low': 1}
    df['expression_numeric'] = df['expression'].map(mapping)

    df = df.groupby(['sequence',"name"])[['kd',"expression_numeric", 'kon', 'koff']].mean().reset_index()
    df = df.fillna(min_kd)

    df["kon"] = df["kon"] / df["kon"].max()
    df["kd_norm"] = df["kd"].apply(lambda x:(exp(max(-log10(x) - min_kd, 0)) - 1)*10)
    df["kd_norm"] = df["kd_norm"] / df["kd_norm"].max()
    df["sequence"] = f"{ec_label}<sep><start>" + df["sequence"] + "<end><|endoftext|>"
    min_kd = 3.0
    df["weight"] =  df['expression_numeric']*((df["kd_norm"] + df["kon"])/2)

        
    data["sequence"].extend(df["sequence"].tolist())
    data["seq_name"].extend(df["name"].tolist())
    data["weight"].extend(df["weight"].tolist())  # Ensure float values are stored properly
    data["kd_norm"].extend(df["kd_norm"].tolist())
    data["kon_norm"].extend(df["kon"])

    print(data)
    pd.DataFrame(data).to_csv("training_dataset.csv", index=False)

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
    print(final_dataset["train"][0])
    print("Weights in train dataset:", final_dataset["train"]["weight"])
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
def log_likelihood(sequences, device, model, tokenizer):
    
    all_log_likelihood = []  # List to store loss for each sequence

    for sequence in sequences:
        inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
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
    
    if pi_ref_loglikelihood is not None: 
        pi_ref_loglikelihood = pi_ref_loglikelihood[sorted_indices] 
    
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
def train(model, iteration_num, ref_model, tokenizer, train_loader, optimizer, device, mode):
    """
    Performs training for one epoch.
    """
    model.train()
    total_loss = []
    print(train_loader)
    print("Train Loader")
    for batch in train_loader:
        print("Batch")
        print(batch)
        if mode != 'paired':
            optimizer.zero_grad()
            sequences = batch["sequence"] 
            print(sequences)
            ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
            policy_log_probs = log_likelihood(sequences,  device, model, tokenizer)

            if models_equal(ref_model, model) or iteration_num == 1:
                ref_log_probs = None

            weights = batch["weight"].to(device)
            print(weights)
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
        print(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")
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
    print(dataset["train"][0])
    print("Weights in train dataset:", dataset["train"]["weight"])
    train_loader = DataLoader(dataset["train"], batch_size=CONFIG["batch_size"], shuffle=True)
    eval_loader = DataLoader(dataset["eval"], batch_size=CONFIG["batch_size"], shuffle=False)

    main(train_loader, eval_loader, args.iteration_num, args.model_dir, args.mode)
