#!/bin/bash -l

#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH --job-name=train
#SBATCH --account=ehpc450
#SBATCH --qos=acc_ehpc
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20


set -e
set -u
set -o pipefail


module load singularity
module load cuda/12.6

label="3.2.1.1"

model_directory="../../../models/ZymCTRL" 
singularity_path="/gpfs/scratch/ehpc450/models/rlevolution_2.sif"
max_iteration_num=30

PYTHON_EXEC_CLEAN="singularity exec --nv --bind /apps/ACC/CUDA:/apps/ACC/CUDA ../../../apptainers/clean.sif python"
PYTHON_EXEC_GEN="singularity exec --nv  --bind /apps/ACC/CUDA:/apps/ACC/CUDA /gpfs/scratch/ehpc450/models/rlevolution_2.sif python"

TARGET_PDB_REWARD="data/3BAJ.pdb"
max_length=500
num_sequences=2

timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="results/${timestamp}"
mkdir -p "$results_dir"

echo "Starting RL for label: $label"

for i in $(seq 1 $max_iteration_num)
do
    echo "Iteration $i"
    
    $PYTHON_EXEC_GEN seq_gen.py --iteration_num $i --label $label --model_dir "$model_directory" --output_dir "$results_dir" --num_sequences $num_sequences --max_length $max_length

    $PYTHON_EXEC_GEN fold_sequences.py --iteration_num $i --label $label --model_dir "$model_directory" --output_dir "$results_dir" --target_pdb "$TARGET_PDB_REWARD"
    
    $PYTHON_EXEC_CLEAN calc_reward.py --iteration_num $i --label $label --output_dir "$results_dir"

    $PYTHON_EXEC_GEN train.py --iteration_num $i --label $label --model_dir "$model_directory" --max_iteration_num $max_iteration_num --output_dir "$results_dir"
    
    $PYTHON_EXEC_GEN plot.py --output_dir "$results_dir"
done

echo "Finished on $(hostname)"
