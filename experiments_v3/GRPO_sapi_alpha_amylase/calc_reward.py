import argparse
import os
import sys
import csv
import torch
import torch.nn as nn
import math
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import SequenceDataset, load_finetuned_esm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()

    # Paths
    # Assuming models are in a local 'model' folder as per instruction
    BASE_MODEL_PATH = "model/esm1v_t33_650M_UR90S_1" 
    LORA_WEIGHTS_PATH = "model/Esm1v_GB1_finetuned.pth"
    
    logs_file = os.path.join(args.output_dir, "logs.csv")
    if not os.path.exists(logs_file):
        print(f"Logs file not found: {logs_file}")
        return

    # Load Activity Model
    print("Loading Activity Prediction Model...")
    try:
        tokenizer, model = load_finetuned_esm(BASE_MODEL_PATH, LORA_WEIGHTS_PATH, num_labels=1)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading activity model: {e}")
        return

    # Process Logs
    df = pd.read_csv(logs_file)
    
    # Check if 'aln_len' column exists, if not add it
    if 'aln_len' not in df.columns:
        df['aln_len'] = 0
    
    # Filter for current iteration
    mask = df['iteration_num'] == args.iteration_num
    
    if not mask.any():
        print(f"No rows found for iteration {args.iteration_num}")
        return

    # Compute Rewards
    
    indices_to_process = []
    raw_sequences = []
    
    for index, row in df[mask].iterrows():
        sequence = row['sequence']
        # Extract raw sequence
        raw_seq = sequence
        if "<start>" in sequence:
            try:
                raw_seq = sequence.split("<start>")[1].split("<end>")[0]
            except:
                pass
        
        indices_to_process.append(index)
        raw_sequences.append(raw_seq)

    if not raw_sequences:
        return

    # Tokenize
    tokenized_sequences = []
    for seq in raw_sequences:
        encoded = tokenizer(
            seq, max_length=1024, padding="max_length", truncation=True, return_tensors="pt"
        )
        tokenized_sequences.append(encoded)
    
    dataset = SequenceDataset(tokenized_sequences)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    all_predictions = []
    
    print(f"Calculating activity for {len(raw_sequences)} sequences...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            batch_preds = logits.squeeze(-1).tolist()
            if isinstance(batch_preds, float): # Single item batch
                batch_preds = [batch_preds]
            all_predictions.extend(batch_preds)


    # Update DataFrame
    for i, index in enumerate(indices_to_process):
        activity_score = all_predictions[i]
        
        # Generic reward column for training
        df.at[index, 'reward'] = activity_score
        
        # Log explicit activity for tracking
        df.at[index, 'activity'] = activity_score
        
        # Cluster info is now logged in fold_sequences.py

    # Save back
    df.to_csv(logs_file, index=False)
    print(f"Updated rewards (activity) for iteration {args.iteration_num}")

if __name__ == "__main__":
    main()
