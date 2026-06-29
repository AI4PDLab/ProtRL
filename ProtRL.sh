#!/bin/bash -l

label="M"
model_directory="AI4PD/ProtGPT3-112M" # HuggingFace repo or local path; use "test" to auto-generate a tiny model for debugging
max_iteration_num=50
PYTHON_EXEC="/users/nferruz/fstocco/Desktop/venv/bin/python"

# Setup logging directory
timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="results/${timestamp}"
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
