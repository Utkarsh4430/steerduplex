#!/bin/bash
set -euo pipefail

export HF_HOME="/fs/gamma-projects/audio/raman/steerd/cache"
export TRANSFORMERS_CACHE="/fs/gamma-projects/audio/raman/steerd/cache"
export HUGGINGFACE_HUB_CACHE="/fs/gamma-projects/audio/raman/steerd/cache"

python3 -m eval.fdb_v2.moshi_server

