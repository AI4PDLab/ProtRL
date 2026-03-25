#!/usr/bin/env bash
#SBATCH --output=/home/woody/b114cb/b114cb23/ProtWrap_2/logs/out_loop_%j.out
#SBATCH --error=/home/woody/b114cb/b114cb23/ProtWrap_2/logs/out_loop_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=a40

set -euo pipefail

# ——————————————————————————————
# CONFIG
# ——————————————————————————————
ITERATION=${1:-0}
MAX_ITERATIONS=100

label="1.3.3.18"
model_directory="/home/woody/b114cb/b114cb23/models/FT_ZymCTRL_altals_2/output_3"
DPO_mode="weighted"
path="/home/woody/b114cb/b114cb23/ProtWrap_2"
loop_script="$path/scripts/t_loop.sh"

# ——————————————————————————————
# PREP
# ——————————————————————————————
mkdir -p "$path/logs"



echo "[$(date +"%Y-%m-%d %H:%M:%S")] Starting iteration $ITERATION on $(hostname)"

# ——————————————————————————————
# 1) TRAIN
# ——————————————————————————————
jobid_train=$(sbatch --parsable \
    "$path/scripts/train_seq_gen.sh" \
    "$ITERATION" "$label" "$model_directory" "$DPO_mode")
echo "→ train job submitted as $jobid_train"

# ——————————————————————————————
# 2) FOLD
# ——————————————————————————————
jobid_fold=$(sbatch --parsable \
    --dependency=afterok:"$jobid_train" \
    "$path/scripts/fold_seq.sh" \
    "$ITERATION" "$label" "$model_directory" "$DPO_mode")
echo "→ fold job submitted as $jobid_fold"

# ——————————————————————————————
# 3) DATASET
# ——————————————————————————————
jobid_db=$(sbatch --parsable \
    --dependency=afterok:"$jobid_fold" \
    "$path/scripts/dataset_gen.sh" \
    "$ITERATION" "$label" "$model_directory" "$DPO_mode")
echo "→ dataset job submitted as $jobid_db"


jobid_plot=$(sbatch --parsable \
    --dependency=afterok:"$jobid_db" \
    "$path/scripts/plot.sh" )
echo "→ plot job submitted as $jobid_plot"



# ——————————————————————————————
# 4) RESUBMIT NEXT ITERATION
# ——————————————————————————————
if (( ITERATION < MAX_ITERATIONS )); then
    NEXT_ITER=$(( ITERATION + 1 ))
    echo "[Iteration $ITERATION] scheduling iteration $NEXT_ITER after job $jobid_db"
    sbatch --parsable \
        --dependency=afterok:"$jobid_db" \
        --job-name="DPO_outloop_iter${NEXT_ITER}" \
        "$loop_script" "$NEXT_ITER"
else
    echo "[Iteration $ITERATION] reached MAX_ITERATIONS ($MAX_ITERATIONS); stopping."
fi

echo "[Iteration $ITERATION] all submissions done."
