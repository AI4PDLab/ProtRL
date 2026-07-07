#!/bin/bash -l
set -e

label="M"
model_directory="AI4PD/ProtGPT3-112M" # HuggingFace repo or local path; use "test" to auto-generate a tiny model for debugging
max_iteration_num=10
output_dir=""
PYTHON_EXEC="${PYTHON_EXEC:-python3}" # override with PYTHON_EXEC=/path/to/python bash ProtRL.sh ...

usage() {
    echo "Usage: bash ProtRL.sh [--model_dir DIR] [--output_dir DIR] [--label LABEL] [--max_iteration_num N]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model_dir) model_directory="$2"; shift 2 ;;
        --output_dir) output_dir="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --max_iteration_num) max_iteration_num="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown option: $1" >&2; usage ;;
    esac
done

# Setup logging directory
if [[ -z "$output_dir" ]]; then
    timestamp=$(date +"%Y%m%d_%H%M%S")
    results_dir="results/${timestamp}"
else
    results_dir="$output_dir"
fi
mkdir -p "$results_dir"
echo "Saving results to $results_dir"

# Check if model directory is test and generate model if needed
if [[ "$model_directory" == "test" ]]; then
    if [ ! -d "./test/tiny" ]; then
        echo "Generating tiny model for testing..."
        $PYTHON_EXEC ./src/setup_tiny_model.py --output_dir "./test/tiny"
    else
        echo "Tiny model already exists at ./test/tiny"
    fi
    # Update model directory to point to the actual model
    actual_model_dir="./test/tiny"
else
    actual_model_dir="$model_directory"
fi

echo RL for the enzyme class $label

for i in $(seq 1 $max_iteration_num)
do
    echo "Iteration $i"

    if [ $i -eq 1 ]
    then
        echo "start"
    fi

    echo Sequence generation started
    $PYTHON_EXEC seq_gen.py --iteration_num $i --label $label --model_dir "$actual_model_dir" --output_dir "$results_dir"

    echo dataset generation
    $PYTHON_EXEC dataset_gen.py --iteration_num $i --label $label --model_dir "$actual_model_dir" --output_dir "$results_dir"

    echo training started
    $PYTHON_EXEC train.py --iteration_num $i --label $label --model_dir "$actual_model_dir" --max_iteration_num $max_iteration_num --output_dir "$results_dir"

    $PYTHON_EXEC plot.py --output_dir "$results_dir"

    echo "Iteration $i completed"
done

###############
# end message #
###############
echo [$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)
