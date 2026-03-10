import argparse
import os
import sys
import csv

# Add parent directory to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils import ESMFolder, run_foldseek, run_mmseqs_clustering, run_mmseqs_search



def formatting_sequence(sequence, ec_label):
    sequence = str(f"{ec_label}<sep><start>{sequence}<end><|endoftext|>")
    return sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--target_pdb", type=str, required=True)
    parser.add_argument("--target_pdb_aux", type=str, required=True)
    args = parser.parse_args()
    
    FOLDSEEK_BIN = "../../../models/foldseek/bin/foldseek"
    MMSEQS_BIN = "../../../models/mmseqs/bin/mmseqs"
    ESMFOLD_PATH = "../../../models/esm_fold"
    BRENDA_DIR = "../brenda_dataset"
    
    # Clustering
    fasta_file = os.path.join(args.output_dir, f"seq_gen_{args.label}_iteration{args.iteration_num}.fasta")
    if not os.path.exists(fasta_file):
        return

    clustering_dir = os.path.join(args.output_dir, "clustering")
    os.makedirs(clustering_dir, exist_ok=True)
    
    run_mmseqs_clustering(fasta_file, os.path.join(clustering_dir, f"clustResult_0.9_iter{args.iteration_num}"), MMSEQS_BIN, 0.9)
    run_mmseqs_clustering(fasta_file, os.path.join(clustering_dir, f"clustResult_0.5_iter{args.iteration_num}"), MMSEQS_BIN, 0.5)

    # Brenda Search
    brenda_db = os.path.join(BRENDA_DIR, f"database_{args.label}.fasta")
    mmseqs_scores = {}
    if os.path.exists(brenda_db):
        alignment_dir = os.path.join(args.output_dir, "alignment")
        os.makedirs(alignment_dir, exist_ok=True)
        aln_file = os.path.join(alignment_dir, f"alnResult_seq_gen_{args.label}_iteration{args.iteration_num}.m8")
        
        if run_mmseqs_search(fasta_file, brenda_db, aln_file, MMSEQS_BIN):
            if os.path.exists(aln_file):
                with open(aln_file, 'r') as f:
                    for line in f:
                        parts = line.split('\t')
                        if len(parts) >= 3:
                            query = parts[0]
                            pident = float(parts[2])
                            # Keep max identity
                            if query not in mmseqs_scores or pident > mmseqs_scores[query]:
                                mmseqs_scores[query] = pident

    # Fold & Score
    folder = ESMFolder(model_path=ESMFOLD_PATH)
    logs_file = os.path.join(args.output_dir, "logs.csv")
    file_exists = os.path.exists(logs_file) and os.stat(logs_file).st_size > 0
    
    with open(logs_file, "a", newline="") as csvfile:
        fieldnames = ["name", "sequence", "length", "iteration_num", "tm_score", "tm_score_2vvb", "mmseqs_identity"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        with open(fasta_file, "r") as f:
            lines = f.readlines()
            
        pdb_dir = os.path.join(args.output_dir, "pdb")
        
        for line in lines:
            if ">" in line:
                name = line.split("\t")[0].replace(">", "").replace("\n", "")
            else:
                sequence = line.strip()
                if not sequence: continue
                
                pdb_path = folder.fold_sequence(sequence, name, pdb_dir)
                tm_score = 0.0
                
                if pdb_path:
                    # Score against reward target (1I6P)
                    score_file = os.path.join(args.output_dir, f"{name}_score.m8")
                    if run_foldseek(pdb_path, args.target_pdb, score_file, FOLDSEEK_BIN):
                         if os.path.exists(score_file) and os.path.getsize(score_file) > 0:
                             with open(score_file, 'r') as sf:
                                 first_result = sf.readline().strip().split('\t')
                                 if len(first_result) > 2:
                                     tm_score = float(first_result[2])
                    if os.path.exists(score_file):
                        os.remove(score_file)

                    # Score against auxiliary target (2VVB)
                    score_file_aux = os.path.join(args.output_dir, f"{name}_score_aux.m8")
                    tm_score_aux = 0.0
                    if run_foldseek(pdb_path, args.target_pdb_aux, score_file_aux, FOLDSEEK_BIN):
                         if os.path.exists(score_file_aux) and os.path.getsize(score_file_aux) > 0:
                             with open(score_file_aux, 'r') as sf:
                                 first_result = sf.readline().strip().split('\t')
                                 if len(first_result) > 2:
                                     tm_score_aux = float(first_result[2])
                    if os.path.exists(score_file_aux):
                        os.remove(score_file_aux)
                
                # Get max identity for this sequence (default 0.0)
                mmseqs_identity = mmseqs_scores.get(name, 0.0)

                writer.writerow({
                    "name": name,
                    "sequence": formatting_sequence(sequence,args.label),
                    "length": len(sequence),
                    "iteration_num": args.iteration_num,
                    "tm_score": tm_score,
                    "tm_score_2vvb": tm_score_aux,
                    "mmseqs_identity": mmseqs_identity
                })

if __name__ == "__main__":
    main()
