#!/bin/bash -l

# Benchmark Configuration
label="M"
model_directory="test"
max_iteration_num=50
PYTHON_EXEC="/users/nferruz/fstocco/Desktop/venv/bin/python"

# Root directory (assuming script is in benchmarking/)
ROOT_DIR=".."
cd "$(dirname "$0")" # Go to script directory

# Setup logging
timestamp=$(date +"%Y%m%d_%H%M%S")
benchmark_dir="results/${timestamp}"
mkdir -p "$benchmark_dir"
echo "Saving benchmark results to $benchmark_dir"

# Setup tiny model if needed
if [[ "$model_directory" == "test" ]]; then
    if [ ! -d "$ROOT_DIR/test/tiny" ]; then
        echo "Generating tiny model for testing..."
        $PYTHON_EXEC "$ROOT_DIR/setup_tiny_model.py" --output_dir "$ROOT_DIR/test/tiny"
    else
        echo "Tiny model already exists at $ROOT_DIR/test/tiny"
    fi
    actual_model_dir="$ROOT_DIR/test/tiny"
else
    actual_model_dir="$model_directory"
fi

# Function to run an experiment
run_experiment() {
    local method=$1
    local beta=$2
    local lr=$3
    local is_level=$4
    local run_id="${method}_beta${beta}_lr${lr}_is${is_level}"
    
    echo "----------------------------------------------------------------"
    echo "Starting Experiment: $run_id"
    echo "Method: $method, Beta: $beta, LR: $lr, IS: $is_level"
    echo "----------------------------------------------------------------"
    
    local run_dir="$benchmark_dir/$run_id"
    mkdir -p "$run_dir"
    
    for i in $(seq 1 $max_iteration_num)
    do
        echo "Run $run_id - Iteration $i"
        
        # Sequence Generation
        $PYTHON_EXEC "$ROOT_DIR/seq_gen.py" \
            --iteration_num $i \
            --label $label \
            --model_dir "$actual_model_dir" \
            --output_dir "$run_dir"
            
        # Dataset Generation
        $PYTHON_EXEC "$ROOT_DIR/dataset_gen.py" \
            --iteration_num $i \
            --label $label \
            --model_dir "$actual_model_dir" \
            --output_dir "$run_dir"
            
        # Training
        $PYTHON_EXEC train_benchmark.py \
            --iteration_num $i \
            --label $label \
            --model_dir "$actual_model_dir" \
            --max_iteration_num $max_iteration_num \
            --output_dir "$run_dir" \
            --method "$method" \
            --beta "$beta" \
            --learning_rate "$lr" \
            --importance_sampling "$is_level"
            
    done
    
    echo "Experiment $run_id completed."
}

# Define Experiments

# 1. pLM_GRPO (LRs: 1e-4, 1e-5, 1e-6)
# Assuming default beta 0.01 for pLM_GRPO as not specified otherwise
for lr in 1e-4 1e-5 1e-6; do
    run_experiment "pLM_GRPO" 0.01 "$lr" "sequence"
done

# 2. weighted_DPO (Betas: 0.1, 0.01, 0.001; LRs: 1e-4, 1e-5, 1e-6)
for beta in 0.1 0.01 0.001; do
    for lr in 1e-4 1e-5 1e-6; do
        run_experiment "weighted_DPO" "$beta" "$lr" "sequence"
    done
done

# 3. trl_GRPO (LRs: 1e-4, 1e-5, 1e-6; IS: token, sequence)
# Assuming default beta 0.01 for trl_GRPO
for lr in 1e-4 1e-5 1e-6; do
    for is_level in "token" "sequence"; do
        run_experiment "trl_GRPO" 0.01 "$lr" "$is_level"
    done
done

echo "All benchmarks completed."
