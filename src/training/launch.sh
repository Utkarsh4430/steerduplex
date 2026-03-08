#!/bin/bash
# Launch SteerDuplex training using moshi-finetune
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "${SCRIPT_DIR}")"
MOSHI_FT_DIR="${SRC_DIR}/vendor/moshi-finetune"

# Default config
CONFIG="${1:-${SRC_DIR}/configs/pilot_training.yaml}"
NUM_GPUS="${2:-1}"

echo "=== SteerDuplex Training ==="
echo "Config: ${CONFIG}"
echo "GPUs: ${NUM_GPUS}"
echo "moshi-finetune: ${MOSHI_FT_DIR}"
echo ""

# Ensure we're in the moshi-finetune directory so relative imports work
cd "${MOSHI_FT_DIR}"

# Apply any patches if present
if [ -d "${SCRIPT_DIR}/patches" ] && [ "$(ls -A ${SCRIPT_DIR}/patches/*.py 2>/dev/null)" ]; then
    echo "Applying training patches..."
    for patch in "${SCRIPT_DIR}/patches/"*.py; do
        echo "  - $(basename ${patch})"
    done
    echo ""
fi

# Launch training
if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Launching distributed training on ${NUM_GPUS} GPUs..."
    torchrun \
        --nproc-per-node "${NUM_GPUS}" \
        --master_port $((RANDOM + 10000)) \
        -m train "${CONFIG}"
else
    echo "Launching single-GPU training..."
    torchrun \
        --nproc-per-node 1 \
        -m train "${CONFIG}"
fi
