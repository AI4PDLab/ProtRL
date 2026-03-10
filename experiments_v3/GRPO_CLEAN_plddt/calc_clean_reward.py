import argparse
import os
import sys
import csv
import torch
import torch.nn as nn
import math
import pandas as pd
from utils import load_clean_model, load_esm1b_model, get_esm_embedding, \
                  map_emb_center, get_ec_id_dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    args = parser.parse_args()
    
    CLEAN_DATA_DIR = "../../CLEAN/app/data"
    SPLIT100_CSV = os.path.join(CLEAN_DATA_DIR, "split100.csv")
    PRETRAINED_100_PT = os.path.join(CLEAN_DATA_DIR, "pretrained/100.pt")
    CLEAN_MODEL_PTH = os.path.join(CLEAN_DATA_DIR, "pretrained/split100.pth")
    ESM1B_MODEL_PATH = "../../../models/esm1b_t33_650M_UR50S.pt"

    logs_file = os.path.join(args.output_dir, "logs.csv")
    if not os.path.exists(logs_file):
        print(f"Logs file not found: {logs_file}")
        return

    # Load CLEAN Resources
    clean_model = load_clean_model(CLEAN_MODEL_PTH, device)
    esm1b_model, esm_batch_converter = load_esm1b_model(ESM1B_MODEL_PATH, device)
    
    train_emb = None
    ec_list = []
    if os.path.exists(PRETRAINED_100_PT):
        try: train_emb = torch.load(PRETRAINED_100_PT, map_location="cpu")
        except: pass
    if os.path.exists(SPLIT100_CSV):
        ec_list = get_ec_id_dict(SPLIT100_CSV)
    
    reference_emb = None
    if train_emb is not None and ec_list:
        reference_emb = map_emb_center(args.label, ec_list, train_emb)
        if reference_emb is not None: reference_emb = reference_emb.to(device)

    if reference_emb is None:
        print("Reference embedding not found. Skipping reward calculation.")
        return

    # Process Logs
    df = pd.read_csv(logs_file)
    
    # Check if 'aln_len' column exists, if not add it (backward compatibility or if fold_sequences didn't write it)
    if 'aln_len' not in df.columns:
        df['aln_len'] = 0

    # Filter for current iteration
    # We update rows where iteration_num matches
    mask = df['iteration_num'] == args.iteration_num
    
    # If no rows, return
    if not mask.any():
        print(f"No rows found for iteration {args.iteration_num}")
        return

    # Calculate CLEAN Accuracy and Reward
    # Look for CLEAN inference output
    clean_output_file = os.path.join(args.output_dir, f"seq_gen_{args.label}_iteration{args.iteration_num}_maxsep.csv")
    
    clean_preds = {} # Map seq_id -> (predicted_ec, confidence, is_correct)
    correct_count = 0
    total_count = 0

    if os.path.exists(clean_output_file):
        print(f"Reading CLEAN output from {clean_output_file}")
        try:
            # Read line by line to handle potential format weirdness
            with open(clean_output_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # Format: "4.2.1.1_0\tppl=1.0472\treward=0.0000,EC:4.2.1.1/0.9920"
                # Split by comma first to separate ID/Stats from Result
                # Be careful if comma is used elsewhere. 
                # Assuming the LAST comma separates the result? Or just split by ','
                
                parts = line.split(',')
                if len(parts) < 2:
                    # Maybe it's just the header or malformed
                    continue
                
                # The result part is the last part? 
                # "EC:4.2.1.1/0.9920"
                result_part = parts[-1] 
                name_part = ",".join(parts[:-1]) # Reconstruct name if it had commas?
                
                if "EC:" not in result_part:
                    continue
                    
                # Extract ID from name_part
                # name_part: "4.2.1.1_0\tppl=1.0472\treward=0.0000"
                # ID is before the first tab
                seq_id = name_part.split('\t')[0].strip()
                
                # Extract EC and Score from result_part
                # "EC:4.2.1.1/0.9920"
                cleaned_res = result_part.replace("EC:", "").strip()
                ec_parts = cleaned_res.split('/')
                
                pred_ec = ec_parts[0]
                confidence = 0.0
                if len(ec_parts) > 1:
                    try:
                        confidence = float(ec_parts[1])
                    except:
                        pass
                
                # Check correctness
                # args.label is the target EC
                is_correct = (pred_ec == args.label)
                if is_correct:
                    correct_count += 1
                total_count += 1
                
                clean_preds[seq_id] = (pred_ec, confidence, is_correct)
                
        except Exception as e:
            print(f"Error parse CLEAN output: {e}")
            
    accuracy = 0.0
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100.0
        
    print(f"Iteration {args.iteration_num} CLEAN Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")

    # Iterate and compute/update logs
    for index, row in df[mask].iterrows():
        sequence = row['sequence']
        name = row['name'] # "4.2.1.1_0"
        
        # Calculate Reward (Combined)
        plddt = float(row.get('plddt', 0.0))
        aln_len = int(row.get('aln_len', 0))
        
        raw_seq = sequence
        if "<start>" in sequence:
            try:
                raw_seq = sequence.split("<start>")[1].split("<end>")[0]
            except:
                pass

        clean_cosine = 0.0
        clean_reward_val = 0.0
        
        # Helper: Get CLEAN prediction stats
        pred_ec = "None"
        confidence = 0.0
        if name in clean_preds:
            pred_ec, confidence, _ = clean_preds[name]
        
        # Embedding Reward Logic
        if esm1b_model and clean_model and reference_emb is not None:
             try:
                 seq_emb = get_esm_embedding(esm1b_model, esm_batch_converter, raw_seq, device)
                 if seq_emb is not None:
                     seq_emb = clean_model(seq_emb.to(device).unsqueeze(0))
                     similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)(seq_emb, reference_emb)
                     clean_cosine = float(similarity.item())
                     
                     if len(raw_seq) > 0:
                         ratio = aln_len / len(raw_seq)
                         length_rew = math.exp(-(((ratio - 1)**2)/(0.5**2)))
                     else:
                         length_rew = 0.0
                         
                     clean_reward_val = clean_cosine * plddt * length_rew
             except Exception as e:
                 pass 
        
        df.at[index, 'clean_cosine'] = clean_cosine
        df.at[index, 'clean_reward'] = clean_reward_val
        df.at[index, 'clean_accuracy'] = accuracy 
        df.at[index, 'clean_predicted_ec'] = pred_ec
        df.at[index, 'clean_confidence'] = confidence

    # Save back
    df.to_csv(logs_file, index=False)
    print(f"Updated rewards for iteration {args.iteration_num}")

if __name__ == "__main__":
    main()
