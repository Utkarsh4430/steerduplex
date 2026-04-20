#!/bin/bash
# Full Duplex Bench v1.0 / v1.5 evaluation environment setup.
# Creates a conda env that can run:
#   - the vendored FDB scoring scripts (scoring/evaluate.py, scoring/get_timing.py,
#     scoring/significance_test.py) via subprocess from run_eval.py
#   - the default Parakeet-TDT-0.6B-v2 ASR backend in run_asr.py
#
# Usage:
#   bash src/eval/fdb_v1/setup_env.sh
#
# This env is *separate* from the steerduplex inference env (which runs the
# Moshi WebSocket servers + client) and from raman_whisperx (only needed if
# you flip ASR_BACKEND=whisperx in run_fdb_eval.sh).

set -euo pipefail

ENV_NAME="raman_fdb_v1"

echo "=== Setting up FDB v1/v1.5 evaluation environment: ${ENV_NAME} ==="

if ! conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Creating conda env: ${ENV_NAME}"
    conda create -n "${ENV_NAME}" python=3.10 -y
else
    echo "Conda env '${ENV_NAME}' already exists"
fi

eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

echo "=== Installing PyTorch ==="
# UTMOSv2 / silero_vad / nemo_toolkit all need torch+torchaudio+torchvision.
# torchvision MUST come from the same index as torch, otherwise a later pip
# install will pull a mismatched PyPI torchvision wheel and you'll see
# `operator torchvision::nms does not exist` at import time.
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
pip install torch torchaudio torchvision --index-url "${TORCH_INDEX}"

echo "=== Installing FDB requirements ==="
# Mirrors Full-Duplex-Bench/v1_v1.5/requirements.txt, minus google-genai and
# flask (we don't run the model-serving side from this env, only scoring).
pip install \
    cryptography \
    datasets \
    numpy \
    scipy \
    soundfile \
    tqdm \
    transformers \
    pyyaml \
    "accelerate>=0.26.0" \
    statsmodels \
    python-dotenv \
    openai \
    silero_vad \
    "sphn==0.1.12"

echo "=== Installing NeMo (ASR) ==="
# nemo_toolkit[asr] pulls a large dep tree; retry on transient network errors.
pip install "nemo_toolkit[asr]" || {
    echo "WARN: nemo_toolkit[asr] install failed; retrying once"
    pip install "nemo_toolkit[asr]"
}

echo "=== Installing UTMOSv2 ==="
# Required by eval_general_before_after.py. Installed from GitHub per the
# official FDB requirements.
pip install "git+https://github.com/sarulab-speech/UTMOSv2.git"

echo "=== Installing Praat (for pitch via parselmouth) ==="
# eval_general_before_after imports praat-parselmouth lazily; not in FDB's
# requirements.txt but needed by the pitch/intensity code paths.
pip install praat-parselmouth

echo "=== Realigning torch / torchvision / torchaudio ==="
# nemo_toolkit and UTMOSv2 often pull mismatched torchvision wheels from
# PyPI, producing `operator torchvision::nms does not exist` at import. Force
# the trio back to the same cu128 build. --force-reinstall is cheap here
# because the wheels are already cached.
pip install --force-reinstall --no-deps \
    torch torchaudio torchvision --index-url "${TORCH_INDEX}"

echo "=== Verifying critical imports ==="
python - <<'PY'
import torch
print(f"  torch        {torch.__version__} (cuda={torch.version.cuda})")
try:
    import torchvision
    print(f"  torchvision  {torchvision.__version__}")
    # Force-load the torchvision C++ ops that utmosv2/nemo rely on.
    from torchvision.ops import nms  # noqa: F401
    print("  ok   torchvision::nms")
except Exception as exc:
    raise SystemExit(f"FAIL torchvision: {exc}")

missing = []
for mod in ["openai", "silero_vad", "nemo.collections.asr", "statsmodels.api",
            "soundfile", "torchaudio", "parselmouth"]:
    try:
        __import__(mod)
        print(f"  ok   {mod}")
    except Exception as exc:
        print(f"  FAIL {mod}: {exc}")
        missing.append(mod)
try:
    import utmosv2  # noqa: F401
    print("  ok   utmosv2")
except Exception as exc:
    print(f"  FAIL utmosv2: {exc}")
    missing.append("utmosv2")
if missing:
    raise SystemExit(f"Missing: {missing}")
PY

echo ""
echo "=== Done ==="
echo "Env name: ${ENV_NAME}"
echo ""
echo "run_fdb_eval.sh already defaults FDB_ENV=${ENV_NAME}; no further config needed."
echo "Launch the full pipeline with:"
echo "  bash src/eval/fdb_v1/run_fdb_eval.sh"
