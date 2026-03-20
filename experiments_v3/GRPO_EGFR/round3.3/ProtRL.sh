#!/bin/bash -l

##################
# slurm settings #
##################

# where to put stdout / stderr
#SBATCH --output=./logs/%j.out
#SBATCH --error=./logs/%j.err
#SBATCH --job-name=ProtRL 
#SBATCH --time=10:00:00

#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --constraint=a100_80

set -e
set -u
set -o pipefail


module load python
module load cuda/12.6.1
source /home/woody/b114cb/b114cb23/.test_env/bin/activate


## -- hyperparameters for training -- ##
label="$1" # use lower caps pls 
d_path="$2" # put the path to your local model or a Huggingface's repository (to be called with transformer's API)

echo RL for the enzyme class $label

readonly model_directory="/home/woody/b114cb/b114cb23/Filippo/DPO_EGFR_/DPO_EGFR_exp/round_2.1/FT_egf_zymctrl"
readonly num_epochs=5 
readonly beta=0.5 
readonly ref_model="/home/woody/b114cb/b114cb23/Filippo/DPO_EGFR_/DPO_EGFR_exp/round_2.1/FT_egf_zymctrl" 
csv1="/home/woody/b114cb/b114cb23/Filippo/DPO_EGFR_/DPO_EGFR_exp/round_2.3.2/datasets_egfr_rounds/training_dataset_round1.csv"
csv2="/home/woody/b114cb/b114cb23/Filippo/DPO_EGFR_/DPO_EGFR_exp/round_2.3.2/datasets_egfr_rounds/training_dataset_round2.csv"

readonly output=$d_path"/output_"
readonly output2=$d_path"/output2_"

readonly learning_rate=2e-6
readonly d_path=$d_path




python train_exp.py --model_dir $model_directory \
                     --csv $csv1 \
                     --output $output\
                     --learning_rate $learning_rate\
                     --beta $beta\
                     --ref_model $ref_model\
                     --num_epochs $num_epochs\

python train_exp.py --model_dir $output \
                     --csv $csv2 \
                     --output $output2\
                     --learning_rate $learning_rate\
                     --beta $beta\
                     --ref_model $ref_model\
                     --num_epochs $num_epochs\



python seq_gen.py --label $label --d_path $d_path --model_dir $output2

#sort by perplexity
awk 'BEGIN{RS=">";FS="\n";ORS=""}
NR>1{
  split($1,H,"\t"); id=H[1]; per=H[2]; reward=H[3]
  seq=""; for(i=2;i<=NF;i++) seq=seq $i
  if(length(seq)>49)
    print per "\t" id "\t" reward "\t" seq "\n"
}' fasta_out_10k.fasta \
| sort -k1,1n \
| head -n1000 \
| awk -F"\t" '{ print ">" $2 "\t" $1 "\t" $3 "\n" $4 }' \
> top1000_lowest_perplexity.fasta


#sort by internal reward
awk 'BEGIN{RS=">";FS="\n";ORS=""}
NR>1{
  split($1,H,"\t"); id=H[1]; per=H[2]; reward=H[3]
  seq=""; for(i=2;i<=NF;i++) seq=seq $i
  if(length(seq)>49)
    # emit: per<TAB>id<TAB>reward<TAB>seq
    print per "\t" id "\t" reward "\t" seq "\n"
}' fasta_out_10k.fasta \
| sort -t$'\t' -k3,3nr \
| head -n1000 \
| awk -F"\t" '{ print ">" $2 "\t" $1 "\t" $3 "\n" $4 }' \
> top1000_highest_reward.fasta

#merge the two fasta
awk '
  BEGIN {
    RS=">"     # each record is a FASTA entry (sans the leading ">")
    FS="\n"    # split record into lines: $1=header, $2…=sequence lines
    ORS=""     # we’ll print our own newlines
  }
  NR>1 {
    header = $1
    split(header, H, "\t")
    id = H[1]  # grab just the sequence ID
    if (!seen[id]++) {
      # print it out (re-add the leading ">")
      printf(">%s\n", header)
      # print the seq lines exactly as they were
      for (i=2; i<=NF; i++)
        printf("%s\n", $i)
    }
  }
' top1000_highest_reward.fasta \
  top1000_lowest_perplexity.fasta \
> merged_top1000.fasta


python gen_template.py


###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
