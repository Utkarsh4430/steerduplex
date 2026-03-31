#!/bin/bash
set -euo pipefail

EVAL_NAME="${1:-}"

if [ -z "$EVAL_NAME" ]; then
    echo "Usage: bash scripts/run_eval.sh <eval_name> [eval_args...]"
    echo "Example: bash scripts/run_eval.sh bigbench_audio --device cuda --limit 5"
    exit 1
fi

shift

export HF_HOME="/fs/gamma-projects/audio/raman/steerd/cache"
export TRANSFORMERS_CACHE="/fs/gamma-projects/audio/raman/steerd/cache"
export HUGGINGFACE_HUB_CACHE="/fs/gamma-projects/audio/raman/steerd/cache"

case "$EVAL_NAME" in
    bigbench_audio)
        python3 -m eval.eval_bigbench "$@"
        ;;
    *)
        echo "Unknown eval: $EVAL_NAME"
        echo "Available evals: bigbench_audio"
        exit 1
        ;;
esac
