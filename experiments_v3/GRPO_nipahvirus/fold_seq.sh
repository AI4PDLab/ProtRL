#!/bin/bash -l

set -euo pipefail

module load singularity
# If your site *requires* CUDA/GCC modules for driver visibility, uncomment:
# module load cuda/12.6
# module load gcc/13.2.0-nvidia-hpc-sdk

i="$1"
d_path="${2%/}/"
SLURM_ARRAY_TASK_ID=0

echo "Node: $(hostname)"
echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES (batch shell): ${CUDA_VISIBLE_DEVICES:-unset}"



singularity exec --nv /gpfs/scratch/crg77/models/boltz-2.sif \
      boltz predict \
        "${d_path}boltz_input_iteration_${i}/boltz_input_iteration_${i}_array${SLURM_ARRAY_TASK_ID}" \
        --cache /gpfs/scratch/crg77/fstocco/models/.boltz \
        --out_dir="${d_path}boltz_output_iteration${i}" \
        --output_format pdb \
        --use_potentials

SOURCE_BASE="${d_path}boltz_output_iteration${i}"
DESTINATION="${d_path}boltz_output_iteration${i}"
ARRAY_PATH="${SOURCE_BASE}/boltz_results_boltz_input_iteration_${i}_array${SLURM_ARRAY_TASK_ID}"
PREDICTIONS_PATH="${ARRAY_PATH}/predictions"

if [ -d "$PREDICTIONS_PATH" ]; then
  shopt -s nullglob
  for folder in "$PREDICTIONS_PATH"/*; do
    [ -d "$folder" ] && mv "$folder" "$DESTINATION" && echo "Moved: $(basename "$folder")"
  done
  rm -rf "${ARRAY_PATH}" && echo "Deleted array folder: ${ARRAY_PATH}"
fi

echo "[$(date +'%Y-%m-%d %H:%M:%S')] finished on $(hostname)"
