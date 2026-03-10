#!/bin/bash

EXP_NAME="GRPO_e1_experiment"
ITERATIONS=5
NUM_SEQUENCES=100
LABEL="enzyme"
EXP_DIR="/users/nferruz/fstocco/Desktop/ProtRL_paper_v3/ProtRL/experiments_v3/GRPO_e1"
APPTAINER_DIR="/users/nferruz/fstocco/Desktop/ProtRL_paper_v3/apptainers"
MODEL_CACHE_DIR="/users/nferruz/fstocco/Desktop/ProtRL_paper_v3/models/huggingface"

mkdir -p "$EXP_DIR/results"
cd "$EXP_DIR"

# Link dependencies
[ ! -f seq_gen.py ] && ln -s ../DPO_TM_maximization/seq_gen.py seq_gen.py
[ ! -f train.py ] && ln -s ../DPO_TM_maximization/train.py train.py
[ ! -f plot.py ] && ln -s ../DPO_TM_maximization/plot.py plot.py

for i in $(seq 1 $ITERATIONS); do
    echo "Iteration $i"
    
    python seq_gen.py --iteration_num $i --num_sequences $NUM_SEQUENCES --output_dir "$EXP_DIR/results"
    
    sudo apptainer exec --nv \
        --bind "$MODEL_CACHE_DIR:/models_cache" \
        "$APPTAINER_DIR/e1.sif" \
        python3 calc_e1_reward.py \
        --iteration_num $i \
        --output_dir "$EXP_DIR/results" \
        --label "$LABEL"
        
    python train.py --iteration_num $i --output_dir "$EXP_DIR/results"
    python plot.py --output_dir "$EXP_DIR/results"
done
