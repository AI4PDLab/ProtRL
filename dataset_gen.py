import numpy as np
import os
import random
import argparse
import pandas as pd
import statistics
import csv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save results")

    args = parser.parse_args()

    generate_dataset(args.iteration_num, args.label, args.output_dir)

def append_to_csv(name, sequence, iteration_num, output_file):
    file_exists = os.path.exists(output_file) and os.stat(output_file).st_size > 0
    with open(output_file, "a", newline="") as csvfile:
        fieldnames = [
            "name",
            "sequence",
            "length",
            "iteration_num",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "name" : name,
            "sequence" : sequence,
            "length" : len(sequence),
            "iteration_num": iteration_num,
        })

def generate_dataset(iteration_num, ec_label, output_dir):

    output_file = os.path.join(output_dir, "logs.csv")
    fasta_file = os.path.join(output_dir, f"seq_gen_{ec_label}_iteration{iteration_num}.fasta")
     
    if not os.path.exists(fasta_file):
        # Fallback to current directory if not found (legacy)
        if os.path.exists(f"seq_gen_{ec_label}_iteration{iteration_num}.fasta"):
            fasta_file = f"seq_gen_{ec_label}_iteration{iteration_num}.fasta"
        else:
            print(f"Error: FASTA file not found: {fasta_file}")
            return

    existing_names = set()
    if os.path.exists(output_file):
        with open(output_file, newline="") as csvfile:
            for row in csv.DictReader(csvfile):
                if int(row["iteration_num"]) == iteration_num:
                    existing_names.add(row["name"])

    with open(fasta_file, "r") as f:
        rep_seq = f.readlines()

    for line in rep_seq:
            if ">" in line:
                name = line.split("\t")[0].replace(">", "").replace("\n", "")
            else:
                sequence = line.strip()
                if name in existing_names:
                    continue
                append_to_csv(name, sequence, iteration_num, output_file)
                existing_names.add(name)

if __name__ == "__main__":
    main()