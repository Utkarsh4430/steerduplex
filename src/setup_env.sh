#!/bin/bash
# Setup conda environment for SteerDuplex training pipeline
set -euo pipefail

ENV_NAME="steerduplex"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Setting up SteerDuplex environment ==="

# Create conda env if it doesn't exist
if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Creating conda env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.11 -y
else
    echo "Conda env '${ENV_NAME}' already exists"
fi

# Activate
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch ==="
pip install torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing moshi-finetune ==="
pip install -e "${SCRIPT_DIR}/vendor/moshi-finetune"

echo "=== Installing pipeline dependencies ==="
pip install \
    vllm \
    openai \
    soundfile \
    librosa \
    scipy \
    numpy \
    pyyaml \
    tqdm \
    pandas \
    resemblyzer \
    pyannote.audio \
    utmos \
    jiwer \
    fire \
    huggingface_hub

echo "=== Installing Qwen3-TTS dependencies ==="
pip install transformers accelerate

echo "=== Installing Whisper for annotation ==="
pip install whisper_timestamped

echo "=== Done ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Then run: python run_pipeline.py --help"
