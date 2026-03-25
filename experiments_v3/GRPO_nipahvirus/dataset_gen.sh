#!/bin/bash -l 

##################
# slurm settings #
##################

i="$1"
label="$2" # use lower caps pls 
model_directory="$3" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
d_path="$4"


##################################
# make bash behave more robustly #
##################################
set -e
set -u
set -o pipefail

module load singularity
module load cuda/12.6
export PATH=/gpfs/scratch/crg77/models/foldx:$PATH

singularity exec --nv  --env PATH=/gpfs/scratch/crg77/models/foldx:\$PATH /gpfs/scratch/crg77/models/bindcraft.sif python ${d_path}scripts/dataset_gen.py --num_arrays 1 --iteration_num $i --label ${label} --model_dir $model_directory --array 0 --d_path ${d_path}

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
