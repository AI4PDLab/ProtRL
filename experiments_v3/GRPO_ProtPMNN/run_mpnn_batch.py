import os
import argparse
import subprocess
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Find PDB files
    pdb_files = glob.glob(os.path.join(args.pdb_dir, "*.pdb"))
    
    print(f"Found {len(pdb_files)} PDB files in {args.pdb_dir}")
    
    for pdb_file in pdb_files:
        # Construct command
        # This script runs INSIDE the container, so we access /app/ProteinMPNN
        cmd = [
            "python", "/app/ProteinMPNN/protein_mpnn_run.py",
            "--pdb_path", pdb_file,
            "--pdb_path_chains", "A", # Assuming chain A as per typical use, or make generic? DPO_weight2 used "A"
            "--score_only", "1",
            "--save_score", "1",
            "--out_folder", args.out_dir,
            "--batch_size", "1"
        ]
        
        try:
            # We don't want to print stdout for every file unless error
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {pdb_file}: {e.stderr.decode()}")
            
    print("Batch processing complete.")

if __name__ == "__main__":
    main()
