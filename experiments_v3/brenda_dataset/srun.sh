#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=DPO_esmv1
#SBATCH --time=24:00:00

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100


set -e
set -u
set -o pipefail


###################
# set environment #
###################

module load python

cd /home/woody/b114cb/b114cb23/protein_gibbs_sampler
    module load cuda/12.4.1
    source  .gibbs_sampler/bin/activate
    cd /home/woody/b114cb/b114cb23/protein_gibbs_sampler/src/pgen
    python likelihood_esm.py -i /home/woody/b114cb/b114cb23/brenda_dataset/database_4.2.1.1.fasta -o /home/woody/b114cb/b114cb23/brenda_dataset/metrics_esm1v_4.2.1.1_train.txt --model esm1v --csv --device gpu --score_name esm1v  
    module unload cuda
    source deactivate
    source /home/woody/b114cb/b114cb23/.test_env/bin/activate
