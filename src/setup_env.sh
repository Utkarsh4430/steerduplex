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
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

echo "=== Installing sox (required by qwen-tts) ==="
conda install -y -c conda-forge sox

echo "=== Installing Qwen3-TTS ==="
pip install -U qwen-tts
pip install flash-attn --no-build-isolation 2>/dev/null || echo "WARN: flash-attn failed, using default attention"

echo "=== Installing moshi + moshi-finetune ==="
pip install "moshi @ git+https://github.com/kyutai-labs/moshi.git#subdirectory=moshi"
# Install finetune with --no-deps to avoid sphn version conflict
# (finetune pins sphn==0.1.12 but moshi requires sphn>=0.2.0)
# Clone + fix pyproject.toml to include subpackages (upstream bug: only lists top-level)
_FT_TMP=$(mktemp -d)
git clone --depth 1 https://github.com/kyutai-labs/moshi-finetune.git "${_FT_TMP}"
sed -i 's/packages = \["finetune"\]/packages = {find = {}}/' "${_FT_TMP}/pyproject.toml"
pip install --no-deps "${_FT_TMP}"
rm -rf "${_FT_TMP}"

echo "=== Installing pipeline dependencies ==="
pip install \
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
    sphn \
    auditok==0.2 \
    safetensors \
    simple-parsing \
    submitit \
    tensorboard

echo "=== Done ==="
echo "Activate with: conda activate ${ENV_NAME}"
echo "Then run: python run_pipeline.py --help"
