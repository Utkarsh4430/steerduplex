#!/bin/bash
set -euo pipefail

# BigBench Audio Evaluation Script
# Usage: bash run_bigbench_audio_eval.sh [optional_args...]
# Example: bash run_bigbench_audio_eval.sh --device cuda --limit 5
# Multi-GPU example: bash run_bigbench_audio_eval.sh --device cuda:0 cuda:1 --limit 100

# Set up Hugging Face cache directories
export HF_TOKEN=${HF_TOKEN:?'HF_TOKEN env var must be set'}
export HF_HOME="/fs/gamma-projects/audio/raman/steerd/cache"
export TRANSFORMERS_CACHE="/fs/gamma-projects/audio/raman/steerd/cache"
export HUGGINGFACE_HUB_CACHE="/fs/gamma-projects/audio/raman/steerd/cache"

# Run BigBench Audio evaluation with any provided arguments
# Default max_duration is 60 seconds (can be overridden with --max_duration flag)
CUDA_VISIBLE_DEVICES=0,1,2 python3 -m eval.bigbench_audio.eval --device cuda:0 cuda:1 cuda:2 --max_duration 60 "$@" --hf_repo nvidia/personaplex-7b-v1 --output_dir personaplex-7b-v1-bigbench
