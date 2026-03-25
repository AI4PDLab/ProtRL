#!/bin/bash -l
set -euo pipefail

i="$1"
label="$2"            # lower caps per your note
model_directory="$3"  # local path or HF repo
MAX_ITERATION_NUM="$4"
d_path="$5"

echo "$d_path"

###################
# set environment #
###################
module load singularity
module load cuda/12.8

export PATH="/gpfs/scratch/crg77/models/mmseqs/bin:$PATH"

# Ensure dirs exist
mkdir -p "${d_path}algn"



###############
# run command #
###############

if [ "$i" -ne 0 ]; then
  echo "Train started"
  singularity exec --nv /gpfs/scratch/crg77/models/torch.sif \
    python "${d_path}scripts/train.py" \
      --iteration_num "$i" \
      --label "$label" \
      --model_dir "$model_directory" \
      --max_iteration_num "$MAX_ITERATION_NUM" \
      --d_path "$d_path"

  singularity exec --nv /gpfs/scratch/crg77/models/torch.sif \
    python "${d_path}scripts/plot.py" --d_path "$d_path"
fi

echo "Sequence generation started"
singularity exec --nv /gpfs/scratch/crg77/models/bindcraft.sif \
  python "${d_path}scripts/seq_gen.py" \
    --iteration_num "$i" \
    --label "$label" \
    --model_dir "$model_directory" \
    --d_path "$d_path"

   
mmseqs easy-search  "${d_path}seq_gen_"$label"_iteration${i}.fasta"  /gpfs/scratch/crg77/fstocco/Nipah_binder_competition/inputs/atlas_cluster_03_rep_seq.fasta ${d_path}algn/algn_results.txt $TMPDIR

mmseqs easy-cluster "${d_path}seq_gen_"$label"_iteration${i}.fasta" ${d_path}algn/clusterRes.txt $TMPDIR --min-seq-id 0.5 -c 0.8 --cov-mode 1


# Boltz dir
boltz_dir="${d_path}boltz_output_iteration${i}"
mkdir -p "$boltz_dir"

echo "[$(date +"%Y-%m-%d %H:%M:%S")] finished on $(hostname)"
