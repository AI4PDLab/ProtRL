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

    # Iterate and compute
    for index, row in df[mask].iterrows():
        sequence = row['sequence']
        # Extract raw sequence (remove tokens if present)
        # formatting_sequence adds <sep><start>...
        # But get_esm_embedding expects raw sequence?
        # CLEAN utils do not specify.
        # But in dataset_gen_tm.py we were passing `sequence` (the loop variable) which came from FASTA.
        # formatting_sequence stored it in CSV.
        # The CSV stores formatted sequence. We need to extract the actual amino acids?
        # Or did `sequence` in the loop hold the raw seq? Yes.
        # `fold_sequences.py` writes formatted sequence to CSV.
        # We need to strip "<sep><start>" etc.
        # Format: f"{ec_label}<sep><start>{sequence}<end><|endoftext|>"
        # We can split by <start> and <end>.
        raw_seq = sequence
        if "<start>" in sequence:
            try:
                raw_seq = sequence.split("<start>")[1].split("<end>")[0]
            except:
                pass
        
        plddt = float(row.get('plddt', 0.0))
        aln_len = int(row.get('aln_len', 0))
        
        clean_cosine, clean_reward = 0.0, 0.0
        
        if esm1b_model and clean_model:
             try:
                 seq_emb = get_esm_embedding(esm1b_model, esm_batch_converter, raw_seq, device)
                 if seq_emb is not None:
                     seq_emb = clean_model(seq_emb.to(device).unsqueeze(0))
                     similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)(seq_emb, reference_emb)
                     clean_cosine = float(similarity.item())
                     
                     # Gaussian length reward
                     if len(raw_seq) > 0:
                         ratio = aln_len / len(raw_seq)
                         length_rew = math.exp(-(((ratio - 1)**2)/(0.5**2)))
                     else:
                         length_rew = 0.0
                         
                     clean_reward = clean_cosine * plddt * length_rew
             except Exception as e:
                 print(f"Error computing reward for {index}: {e}")
        
        df.at[index, 'clean_cosine'] = clean_cosine
        df.at[index, 'clean_reward'] = clean_reward

    # Save back
    df.to_csv(logs_file, index=False)
    print(f"Updated rewards for iteration {args.iteration_num}")

if __name__ == "__main__":
    main()
