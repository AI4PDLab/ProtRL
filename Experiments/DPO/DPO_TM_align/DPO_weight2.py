from torch.nn import CrossEntropyLoss
import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import random
import numpy as np
from functools import partial
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from datasets import Dataset, load_from_disk, DatasetDict
import torch.optim as optim
import os

# HYPERPARAMETERS
beta = 0.01
seed = 1998
learning_rate = 1e-7
batch_size = 5
num_epochs = 5
count = 0
split_percent = 0.2  # of the eval set
beta1 = 0.9
beta2 = 0.98
epsilon = 1e-8
adam_decay = 0.1



device_name = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device_name)

def extract_sequences(rep_seq):
    
    sequences_rep = dict()
     
    for line in rep_seq:
            
            if ">" in line:
                name = line.replace(">", "").replace("\n", "").split('\t')[0]
                
            else:

                aa = line.strip()
                
                try:
                    
                    print(name)
                    sequences_rep[name] = {
                                "sequence" : aa,
                                    }
                except:
                    print(f'WARNING for seq {name}')
    
    return sequences_rep
    
def generate_dataset(iteration_num, ec_label):
     data = dict()
     data = {
        "sequence" : [],
        "seq_name" : [],
        "weight" : [],
        }
    
     with open(f"seq_gen_{ec_label}_iteration{iteration_num-1}.fasta", "r") as f:
        rep_seq = f.readlines()

     with open(f"alpha_{ec_label}_TM_iteration{iteration_num-1}", "r") as f:
        alpha_TM_score = f.readlines()
        
     
     # Get the amminoacid sequences 
     sequences_rep = extract_sequences(rep_seq)
     
     for entry in alpha_TM_score:
            name = entry.split("\t")[0]
            TM = entry.split("\t")[2]
            TM_norm_que = entry.split("\t")[4]
            algn = int(entry.split("\t")[5])
            sequence = sequences_rep[str(name)]['sequence']
            lenght_rew = math.exp(-((((len(sequence)/578)-1)**2)/(0.5**2))) # Gaussian center on 1. The closer the ratio between len and aligment is, the higher is the reward
            
            print(f'logging:name{name}:weight:{float(TM_norm_que)}:algn:{algn}:TM_score:{TM_norm_que}:sequence:{sequence}')
            
            data["sequence"].append(formatting_sequence(sequence, ec_label))
            data["seq_name"].append(entry)
            data["weight"].append(float(TM_norm_que)+(float(algn)/100))
     
     hf_dataset = Dataset.from_pandas(pd.DataFrame(data))
     shuffled_dataset = hf_dataset.shuffle(seed=seed)
     # Split the dataset (80% train, 20% eval)
     train_size = int((1-split_percent) * len(shuffled_dataset))
     train_dataset = shuffled_dataset.select(range(train_size))
     eval_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))
  
     # Create a DatasetDict to hold the train and eval datasets
     final_dataset = DatasetDict({
          'train': train_dataset,
          'eval': eval_dataset
          })
          
     final_dataset.save_to_disk(f"dataset_iteration{iteration_num}")
     

     return final_dataset
     
def seed_everything(seed=2003):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def formatting_sequence(sequence, ec_label):
    sequence = str(f"{ec_label}<sep><start>{sequence}<end><|endoftext|>")
    return sequence
    

def log_likelihood(sequences, device, model, tokenizer):
    
    all_loss = []  # List to store loss for each sequence

    for sequence in sequences:
        inputs = tokenizer.encode(sequence, return_tensors='pt').to(device)
        outputs = model(inputs, labels=inputs)
        loss, logits = outputs[:2]
        all_loss.append(loss.unsqueeze(0))
        
    all_loss = torch.cat(all_loss)
    
    return all_loss

def dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, beta=0.1):

    if ref_log_probs is None:
        log_ratios = beta * policy_log_probs
    else:
        log_ratios = (beta * (policy_log_probs.to(device) - ref_log_probs.to(device)))
    weights = torch.softmax(weights*(-1), dim=0)

    return F.cross_entropy(log_ratios, weights)

# Training function
def train(model, ref_model, tokenizer, train_loader, optimizer, device):
    model.train()
    total_loss = []
    for batch in train_loader:
        optimizer.zero_grad()

        sequences = batch["sequence" ]  
        ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
        policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
        weights = batch["weight"].to(device)
        
        # Calculate DPO loss
        loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, beta)

        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        print(f'loss:{loss}')
        total_loss.append(loss.item())


    return sum(total_loss) / len(total_loss)



def evaluate(model,ref_model,tokenizer, eval_loader, device):
    model.eval()
    total_loss = []
    with torch.no_grad():
        for batch in eval_loader:
            sequences = batch["sequence" ]  
            ref_log_probs = log_likelihood(sequences, device, ref_model, tokenizer)
            policy_log_probs = log_likelihood(sequences, device, model, tokenizer)
            weights = batch["weight"].to(device)
            
            # Calculate DPO loss
            loss = dpo_weighted_loss(policy_log_probs, ref_log_probs, weights, beta)
    
            total_loss.append(loss.item())

    return sum(total_loss) / len(total_loss)


def save_model_and_tokenizer(model, tokenizer, output_dir):
    """
    Saves the model and tokenizer for sequence generation.
    """
    try:
        # Check if the output directory exists; create it if it does not
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Save the model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"Model and tokenizer saved to {output_dir}")

    except Exception as e:
        print(f"An error occurred while saving the model and tokenizer: {e}")





def main(train_loader,eval_loader, iteration_num):
  # Load the model
  
  if int(iteration_num) == 1:

    model_name = "AI4PD/ZymCTRL"
    
  else:
    model_name = f"output_iteration{iteration_num-1}"
  
  print(f'Model {model_name} has been loaded')

  tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
  model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
  ref_model = AutoModelForCausalLM.from_pretrained('AI4PD/ZymCTRL').to(device)
  optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=epsilon, weight_decay=adam_decay)



  for epoch in range(num_epochs):
      train_loss = train(model, ref_model, tokenizer, train_loader, optimizer, device)
      eval_loss = evaluate(model, ref_model, tokenizer, eval_loader, device)
  
      print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}")
      
      save_model_and_tokenizer(model, tokenizer, output_dir=f"output_iteration{iteration_num}")

  del model
  del ref_model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int)
    parser.add_argument("--label", type=str)
    args = parser.parse_args()
    iteration_num = args.iteration_num
    ec_label = args.label
    ec_label = ec_label.strip()
    seed_everything(seed)
    
    if not os.path.exists(f"dataset_iteration{iteration_num}"):
      dataset = generate_dataset(iteration_num, ec_label)
    else:
      dataset = load_from_disk(f"dataset_iteration{iteration_num}")
    
    print('Dataset Loaded!!')
    train_set = dataset["train"]
    eval_set = dataset["eval"]


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=True)

    main(train_loader,eval_loader, iteration_num)

