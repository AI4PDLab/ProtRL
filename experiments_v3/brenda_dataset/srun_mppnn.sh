#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=DPO_proteinMPNN
#SBATCH --time=24:00:00

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80



set -e
set -u
set -o pipefail


###################
# set environment #
###################

module load python
source /home/woody/b114cb/b114cb23/.test_env/bin/activate

python ESM_Fold.py --iteration_num 0  --label "5.4.99.5"

python "/home/woody/b114cb/b114cb23/ProteinMPNN/protein_mpnn_run.py" --pdb_path /home/woody/b114cb/b114cb23/brenda_dataset/PDB_5.4.99.5 --pdb_path_chains A --score_only 1 --save_score 1 --out_folder /home/woody/b114cb/b114cb23/brenda_dataset --batch_size 1
