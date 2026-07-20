#!/bin/bash -l
set -e
set -o pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${EXAMPLE_DIR}/models/base/tiny"
MAX_ITERATIONS=30
LABEL="MDEMKAYVAL"

# One-time setup: build amino-acid tokenizer and tiny LLaMA config
if [ ! -d "${EXAMPLE_DIR}/models/tokenizer" ]; then
    echo "Building tokenizer..."
    python3 "${EXAMPLE_DIR}/build_llama_tokenizer.py"
fi
if [ ! -f "${EXAMPLE_DIR}/models/size_config/tiny/llama_config.json" ]; then
    echo "Creating LLaMA config..."
    python3 "${EXAMPLE_DIR}/create_llama_config.py" -s tiny -p 1024
fi

echo "Starting RL loop for label: ${LABEL}"

for i in $(seq 0 $MAX_ITERATIONS); do
    echo "=== Iteration ${i} ==="

    # Train on sequences from the previous iteration (skip at iteration 0)
    if [ $i -gt 0 ]; then
        python3 "${EXAMPLE_DIR}/train.py" \
            --iteration_num $i \
            --label        $LABEL \
            --model_dir    $MODEL_DIR
    fi

    # Generate sequences with the current model (also creates base model at iteration 0)
    python3 "${EXAMPLE_DIR}/seq_gen.py" \
        --iteration_num $i \
        --label         $LABEL

    python3 "${EXAMPLE_DIR}/plot_len_stats.py"
done
