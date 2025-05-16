#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=./logs/dataset_gen%j.out
#SBATCH --error=./logs/dataset_gen%j.err
#SBATCH --job-name=dataset_gen_PW
#SBATCH --time=12:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=18

#SBATCH --array=0-19

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
conda activate BindCraft
export PATH=/home/woody/b114cb/b114cb23/foldx:$PATH


label="1_3_3_18" # use lower caps pls 
model_directory="/home/woody/b114cb/b114cb23/models/FT_ZymCTRL_altals_2/output_3" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
DPO_mode="weighted" # choose between paired, ranked and weighted 
hotspot_residues=""
i=0

python dataset_gen.py  --iteration_num $i --label ${label} --model_dir $model_directory --mode $DPO_mode --array ${SLURM_ARRAY_TASK_ID}

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
