#!/bin/bash
# Launch SteerDuplex training via moshi-finetune
# Usage:
#   bash training/launch.sh                                    # 8 GPUs, pilot config
#   bash training/launch.sh configs/pilot_training.yaml 4      # 4 GPUs
#   bash training/launch.sh configs/full_training.yaml 8       # full training
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "${SCRIPT_DIR}")"
MOSHI_FT_DIR="${SRC_DIR}/vendor/moshi-finetune"

CONFIG="${1:-${SRC_DIR}/configs/pilot_training.yaml}"
NUM_GPUS="${2:-8}"

echo "=== SteerDuplex Training ==="
echo "Config:    ${CONFIG}"
echo "GPUs:      ${NUM_GPUS}"
echo "moshi-ft:  ${MOSHI_FT_DIR}"
echo ""

# Validate
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

cd "${MOSHI_FT_DIR}"

# Launch with torchrun
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
