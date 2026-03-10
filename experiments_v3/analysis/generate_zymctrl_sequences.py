import argparse
import torch
import random
import numpy as np
from transformers import GPT2LMHeadModel, AutoTokenizer
from tqdm import tqdm

def remove_characters(sequence, char_list):
    columns = sequence.split('<sep>')
    seq = columns[1] if len(columns) > 1 else columns[0]
    for char in char_list:
        seq = seq.replace(char, '')
    return seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, default="generated_sequences.fasta")
    parser.add_argument("--model_path", type=str, default="/users/nferruz/fstocco/Desktop/ProtRL_paper_v3/models/ZymCTRL")
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Read labels
    with open(args.input_file, "r") as f:
        ec_labels = [l.strip() for l in f.read().strip().split(",") if l.strip()]

    # Sample
    if args.sample_size and args.sample_size < len(ec_labels):
        ec_labels = random.sample(ec_labels, args.sample_size)
    
    print(f"Processing {len(ec_labels)} labels...")

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path).to(device)
    model.eval()

    special_tokens = ['<start>', '<end>', '<|endoftext|>','<pad>',' ', '<sep>']
    output_sequences = []

    for label in tqdm(ec_labels):
        input_ids = tokenizer.encode(label, return_tensors='pt').to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                top_k=9,
                repetition_penalty=1.2,
                max_length=1024,
                eos_token_id=1,
                pad_token_id=0,
                do_sample=True,
                num_return_sequences=3
            )
        
        for i, output in enumerate(outputs):
            seq = remove_characters(tokenizer.decode(output), special_tokens)
            output_sequences.append(f">{label}_{i+1}\n{seq}\n")

    with open(args.output_file, "w") as f:
        f.writelines(output_sequences)

if __name__ == "__main__":
    main()
