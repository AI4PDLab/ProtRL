#!/bin/bash -l 

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=./logs/dataset_gen%j.out
#SBATCH --error=./logs/dataset_gen%j.err
#SBATCH --job-name=dataset_gen_PW
#SBATCH --time=03:00:00

# change this configuration to run on your GPU (80GB) configuration
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40
#SBATCH --array=0-18

i="$1"
label="$2" # use lower caps pls 
model_directory="$3" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
DPO_mode="$4" # choose between paired, ranked and weighted 



##################################
# make bash behave more robustly #
##################################
set -e
set -u
set -o pipefail

module load python/3.12-conda
module load cuda/12.6.1
export PATH=~/envs/myenv/bin:$PATH
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/hpc/b114cb/b114cb23/envs
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PATH=/home/woody/b114cb/b114cb23/foldx:$PATH

label="1.3.3.18" # use lower caps pls 
model_directory="/home/woody/b114cb/b114cb23/models/FT_ZymCTRL_altals_2/output_3" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
DPO_mode="weighted" # choose between paired, ranked and weighted 
hotspot_residues=""

python ./scripts/dataset_gen.py  --iteration_num $i --label ${label} --model_dir $model_directory --mode $DPO_mode --array ${SLURM_ARRAY_TASK_ID}

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
