import argparse
import os
import sys
import csv
from utils import ESMFolder, run_foldseek, run_mmseqs_clustering, run_mmseqs_search, \
                  extract_mean_plddt, formatting_sequence

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration_num", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--target_pdb", type=str, required=True)
    args = parser.parse_args()
    
    FOLDSEEK_BIN = "../../../models/foldseek/bin/foldseek"
    MMSEQS_BIN = "../../../models/mmseqs/bin/mmseqs"
    ESMFOLD_PATH = "../../../models/esm_fold"
    BRENDA_DIR = "../brenda_dataset"

    fasta_file = os.path.join(args.output_dir, f"seq_gen_{args.label}_iteration{args.iteration_num}.fasta")
    if not os.path.exists(fasta_file): return

    # Clustering
    clustering_dir = os.path.join(args.output_dir, "clustering")
    os.makedirs(clustering_dir, exist_ok=True)
    
    # run_mmseqs_clustering now returns the number of clusters
    num_clusters_09 = run_mmseqs_clustering(fasta_file, os.path.join(clustering_dir, f"clustResult_0.9_iter{args.iteration_num}"), MMSEQS_BIN, 0.9)
    num_clusters_05 = run_mmseqs_clustering(fasta_file, os.path.join(clustering_dir, f"clustResult_0.5_iter{args.iteration_num}"), MMSEQS_BIN, 0.5)



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
                            if query not in mmseqs_scores or pident > mmseqs_scores[query]:
                                mmseqs_scores[query] = pident

    # Fold & Score
    folder = ESMFolder(model_path=ESMFOLD_PATH)
    logs_file = os.path.join(args.output_dir, "logs.csv")
    file_exists = os.path.exists(logs_file) and os.stat(logs_file).st_size > 0
    
    with open(logs_file, "a", newline="") as csvfile:
        fieldnames = ["name", "sequence", "length", "iteration_num", "tm_score", "mmseqs_identity", "plddt", "clean_cosine", "clean_reward", "aln_len", "num_clusters_0.9", "num_clusters_0.5"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists: writer.writeheader()

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
                tm_score, plddt, aln_len = 0.0, 0.0, 0
                
                if pdb_path:
                    plddt = extract_mean_plddt(pdb_path)
                    
                    # Score against target
                    score_file = os.path.join(args.output_dir, f"{name}_score.m8")
                    if run_foldseek(pdb_path, args.target_pdb, score_file, FOLDSEEK_BIN):
                         if os.path.exists(score_file) and os.path.getsize(score_file) > 0:
                             with open(score_file, 'r') as sf:
                                 parts = sf.readline().strip().split('\t')
                                 if len(parts) >= 3: tm_score = float(parts[2])
                                 if len(parts) >= 6: aln_len = int(parts[5])
                    if os.path.exists(score_file): os.remove(score_file)
                
                mmseqs_identity = mmseqs_scores.get(name, 0.0)

                # CLEAN Reward placeholders
                clean_cosine, clean_reward = 0.0, 0.0

                writer.writerow({
                    "name": name,
                    "sequence": formatting_sequence(sequence,args.label),
                    "length": len(sequence),
                    "iteration_num": args.iteration_num,
                    "tm_score": tm_score,
                    "mmseqs_identity": mmseqs_identity,
                    "plddt": plddt,
                    "clean_cosine": clean_cosine,
                    "clean_reward": clean_reward,
                    "aln_len": aln_len,
                    "num_clusters_0.9": num_clusters_09, # Log cluster info
                    "num_clusters_0.5": num_clusters_05
                })

if __name__ == "__main__":
    main()
