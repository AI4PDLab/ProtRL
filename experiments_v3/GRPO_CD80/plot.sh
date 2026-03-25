#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=./logs/plot%j.out
#SBATCH --error=./logs/plot%j.err
#SBATCH --job-name=plot_protwrap
#SBATCH --time=00:30:00

#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40

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

python ./scripts/plot.py 

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
