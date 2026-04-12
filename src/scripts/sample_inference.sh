#!/bin/bash
set -euo pipefail

# Sample inference demonstrating usage of generate functionality
# Set up Hugging Face cache directories
# export HF_TOKEN=${HF_TOKEN:?'HF_TOKEN env var must be set'}
export HF_HOME="/fs/gamma-projects/audio/raman/steerd/cache"
export TRANSFORMERS_CACHE="/fs/gamma-projects/audio/raman/steerd/cache"
export HUGGINGFACE_HUB_CACHE="/fs/gamma-projects/audio/raman/steerd/cache"

python -m inference.generate \
  --hf_repo nvidia/personaplex-7b-v1 \
  --user_audio /fs/gamma-projects/audio/raman/steerd/personaplex/assets/test/input_assistant.wav \
  --seed 42424242 \
  --output output.wav