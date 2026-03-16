#!/bin/bash
# Launch SteerDuplex training.
#
# Automatically merges datasets from config before training starts.
# Just edit configs/full_training.yaml to toggle datasets and set hours.
#
# Usage:
#   bash training/launch.sh                                    # 8 GPUs, full config
#   bash training/launch.sh configs/full_training.yaml 4       # 4 GPUs
#   bash training/launch.sh configs/full_training.yaml 8       # 8 GPUs
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(dirname "${SCRIPT_DIR}")"

CONFIG="${1:-${SRC_DIR}/configs/full_training.yaml}"
NUM_GPUS="${2:-8}"

echo "=== SteerDuplex Training ==="
echo "Config:    ${CONFIG}"
echo "GPUs:      ${NUM_GPUS}"
echo ""

# Validate
if [ ! -f "${CONFIG}" ]; then
    echo "ERROR: Config not found: ${CONFIG}"
    exit 1
fi

cd "${SRC_DIR}"

# Auto-merge datasets if config has a datasets section
if grep -q "^datasets:" "${CONFIG}" 2>/dev/null; then
    echo "=== Merging datasets ==="
    python -m pipeline.merge_manifests --config "${CONFIG}"
    echo ""
fi

# Ensure CUDA_VISIBLE_DEVICES is set
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_GPUS - 1)))}"

# Disable torch.compile/dynamo — causes triton race conditions in multi-GPU finetuning
export TORCH_COMPILE_DISABLE=1
export TORCHDYNAMO_DISABLE=1

# NCCL tuning for H100/A100 multi-GPU
export OMP_NUM_THREADS=4
export NCCL_ASYNC_ERROR_HANDLING=1

# Launch with torchrun
if [ "${NUM_GPUS}" -gt 1 ]; then
    # Use hostname hash for deterministic unique port (avoids collision across nodes)
    PORT=$(python3 -c "import hashlib, os; print(10000 + int(hashlib.md5(os.uname().nodename.encode()).hexdigest(), 16) % 30000)")
    echo "Launching distributed training on ${NUM_GPUS} GPUs (port ${PORT})..."
    torchrun \
        --nproc-per-node "${NUM_GPUS}" \
        --master_port "${PORT}" \
        -m training.train "${CONFIG}"
else
    echo "Launching single-GPU training..."
    torchrun \
        --nproc-per-node 1 \
        -m training.train "${CONFIG}"
fi
