#!/bin/bash -l

# Benchmark Configuration
label="M"
model_directory="test"
max_iteration_num=10
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

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
        $PYTHON_EXEC "$ROOT_DIR/src/setup_tiny_model.py" --output_dir "$ROOT_DIR/test/tiny"
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

    if [[ "$method" == "trl_GRPO" ]]; then
        echo "Running single training session for trl_GRPO"
        $PYTHON_EXEC train_benchmark.py \
            --iteration_num 1 \
            --label $label \
            --model_dir "$actual_model_dir" \
            --max_iteration_num $max_iteration_num \
            --output_dir "$run_dir" \
            --method "$method" \
            --beta "$beta" \
            --learning_rate "$lr" \
            --importance_sampling "$is_level"
    else
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
    fi

    echo "Experiment $run_id completed."
}

# Define Experiments

# 1. ProtRL_GRPO
for beta in 0.1; do
    for lr in 2e-3; do
        run_experiment "ProtRL_GRPO" "$beta" "$lr" "sequence"
    done
done

# 2. ProtRL_wDPO
for beta in 0.1; do
    for lr in 2e-3; do
        run_experiment "ProtRL_wDPO" "$beta" "$lr" "sequence"
    done
done

# 3. trl_GRPO (online, IS: token and sequence)
for beta in 0.1; do
    for lr in 2e-3; do
        for is_level in "token" "sequence"; do
            run_experiment "trl_GRPO" "$beta" "$lr" "$is_level"
        done
    done
done

# 4. ProtRL_REINFORCE
for beta in 0.1; do
    for lr in 2e-3; do
        run_experiment "ProtRL_REINFORCE" "$beta" "$lr" "sequence"
    done
done

$PYTHON_EXEC plot_benchmark.py --results_dir $benchmark_dir

echo "All benchmarks completed."