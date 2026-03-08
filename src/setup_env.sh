#!/bin/bash
# SteerDuplex environment setup
# Usage: bash setup_env.sh
set -euo pipefail

ENV_NAME="steerduplex"

echo "=== Setting up SteerDuplex environment ==="

if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Creating conda env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.11 -y
else
    echo "Conda env '${ENV_NAME}' already exists"
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch ==="
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing Qwen3-TTS ==="
pip install -U qwen-tts
pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARN: flash-attn failed, using default attention"

echo "=== Installing moshi + moshi-finetune ==="
pip install "moshi @ git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi"
pip install "git+https://github.com/kyutai-labs/moshi-finetune.git"

echo "=== Installing pipeline dependencies ==="
pip install \
    litellm \
    openai \
    soundfile \
    librosa \
    scipy \
    numpy \
    pyyaml \
    tqdm \
    jiwer \
    fire \
    huggingface_hub \
    wandb \
    whisper-timestamped \
    sphn==0.1.12 \
    auditok==0.2 \
    safetensors \
    simple-parsing \
    submitit

echo "=== Done ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Then run: python run_pipeline.py --help"
