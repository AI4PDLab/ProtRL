#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=./logs/folding_%j.out
#SBATCH --error=./logs/folding_%j.err
#SBATCH --job-name=folding_PW
#SBATCH --time=05:00:00

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

echo "Folding started"
apptainer exec \
        --nv \
        --bind /home/woody/b114cb/b114cb23/ProtWrap_2/:/root/af_input \
        --bind /home/woody/b114cb/b114cb23/ProtWrap_2/:/root/af_output \
        --bind /home/woody/b114cb/b114cb23/Filippo/AF3_model/:/root/models \
        --bind /anvme/data/alphafold3/databases/:/root/public_databases \
        /anvme/data/alphafold3/alphafold3-20250125.sif\
        python /home/woody/b114cb/b114cb23/alphafold3/run_alphafold.py\
        --input_dir=./alphafold3_input_iteration_"$i"/alphafold3_input_iteration_"$i"_array"${SLURM_ARRAY_TASK_ID}" \
        --output_dir=./alphafold_output_iteration"$i" \
        --run_data_pipeline=False \
        --model_dir=/root/models


###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
