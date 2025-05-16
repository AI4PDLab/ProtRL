#!/bin/bash -l 

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --job-name=EGFR_ZymCTRL
#SBATCH --time=4:00:00

# change this configuration to run on your GPU (80GB) configuration
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100

##################################
# make bash behave more robustly #
##################################
set -e
set -u
set -o pipefail


###################
# set environment #
###################

module load cuda/12.6.1
source /home/woody/b114cb/b114cb23/.test_env/bin/activate

###############
# run command #
###############

label="1.3.3.18" # EC label we want to prompt ZymCTRL with 
model_directory="/home/woody/b114cb/b114cb23/Filippo/DPO_EGFR_/DPO_EGFR_exp/round_1/ranked_v2/ranked_v2_tm/output_iteration1" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)
DPO_mode="ranked" # choose between paired, ranked and weighted 

echo DPO_pLM for the enzyme class $label, with $DPO_mode mode

# establish the number of iterations you want to do with DPO_pLM
i=1


      echo Train started
      python "DPO_pLM.py" --iteration_num $i --label $label --mode $DPO_mode --model_dir $model_directory
    

    echo Sequence generation started
    # Generate the sequences
    python seq_gen.py --iteration_num $i --label $label  --model_dir $model_directory




###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
