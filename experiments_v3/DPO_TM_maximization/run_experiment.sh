#!/bin/bash -l

label="4.2.1.1"

model_directory="../../../models/ZymCTRL" 
max_iteration_num=30
PYTHON_EXEC="singularity exec --nv ../../../apptainers/clean.sif python"
TARGET_PDB_REWARD="data/1I6P.pdb"
TARGET_PDB_AUX="data/2VVB.pdb"
max_length=500
num_sequences=200

timestamp=$(date +"%Y%m%d_%H%M%S")
results_dir="results/${timestamp}"
mkdir -p "$results_dir"

if [[ "$model_directory" == "test" ]]; then
    if [ ! -d "./test/tiny" ]; then
        echo "Generating tiny model for testing..."
    fi
    actual_model_dir="$model_directory"
else
    actual_model_dir="$model_directory"
fi

echo "Starting RL for TM Maximization (Label: $label)"

for i in $(seq 1 $max_iteration_num)
do
    echo "Iteration $i"
    
    $PYTHON_EXEC seq_gen.py --iteration_num $i --label $label --model_dir "$actual_model_dir" --output_dir "$results_dir" --num_sequences $num_sequences --max_length $max_length

    $PYTHON_EXEC dataset_gen_tm.py --iteration_num $i --label $label --model_dir "$actual_model_dir" --output_dir "$results_dir" --target_pdb "$TARGET_PDB_REWARD" --target_pdb_aux "$TARGET_PDB_AUX"

    $PYTHON_EXEC train_tm.py --iteration_num $i --label $label --model_dir "$actual_model_dir" --max_iteration_num $max_iteration_num --output_dir "$results_dir"
    
    $PYTHON_EXEC plot_tm.py --output_dir "$results_dir"
done

echo "Finished on $(hostname)"
