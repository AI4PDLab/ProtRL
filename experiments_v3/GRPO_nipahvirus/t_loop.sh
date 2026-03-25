#!/bin/bash -l
#SBATCH --output=/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/ProtWrap_EpirinB/logs/t_loop_%j.out
#SBATCH --error=/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/ProtWrap_EpirinB/logs/t_loop_%j.err
#SBATCH --job-name=t_loop_egfr
#SBATCH --account=crg77
#SBATCH --qos=acc_resc
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00         
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20

set -euo pipefail

readonly max_iterations=100
readonly start_iteration="${1:-0}"  # default to 0 if not provided

if (( start_iteration > max_iterations )); then
  echo "Start iteration ($start_iteration) > max ($max_iterations). Exiting."
  exit 0
fi

readonly d_path="/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/ProtWrap_EpirinB/"
readonly training_job_script="${d_path}scripts/train_seq_gen.sh"
readonly folding_job_script="${d_path}scripts/fold_seq.sh"
readonly fritz_job_script="${d_path}scripts/dataset_gen.sh"
readonly log_dir="${d_path}logs"
readonly label="epirinb"
readonly model_directory="/gpfs/scratch/crg77/fstocco/Nipah_binder_competition/results/models/FT_epirinB_8_140_ProtGPT3"

mkdir -p "$log_dir" "${d_path}algn"

echo "================================================="
echo " Starting Iterations: $start_iteration .. $max_iterations"
echo "================================================="

for (( iteration=start_iteration; iteration<=max_iterations; iteration++ )); do
  echo ">>> Iteration $iteration"

  # All steps run sequentially on the same allocation
  bash  \
    "$training_job_script" "$iteration" "$label" "$model_directory" "$max_iterations" "$d_path"

  bash  \
    "$folding_job_script" "$iteration" "$d_path"

  bash  \
    "$fritz_job_script" "$iteration" "$label" "$model_directory" "$d_path"
done

echo "All iterations completed."
