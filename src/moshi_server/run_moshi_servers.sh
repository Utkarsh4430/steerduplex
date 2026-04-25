#!/bin/bash
# Launch N Moshi WebSocket servers across G GPUs via launch_moshi_servers.py.
#
# This is a thin wrapper: edit the MANDATORY block below, then run:
#
#     bash src/moshi_server/run_moshi_servers.sh
#
# The script stays in the foreground holding all child processes. Press
# Ctrl+C to terminate all spawned servers.
#
# Run it AFTER you've decided on a checkpoint and persona — servers are
# immutable once up (persona is set at startup). Point your eval driver at
# the URLs either from --output-json or by hard-coding the port range.

set -euo pipefail

# =============================================================================
# MANDATORY — edit these before running
# =============================================================================

# Model selection. See README.md for which checkpoints support persona.
# HF_REPO="nvidia/personaplex-7b-v1"            # HF repo id
HF_REPO="kyutai/moshiko-pytorch-bf16"
CHECKPOINT=""                                 # optional local checkpoint (dir or file); "" = use HF weights

# Pool layout.
NUM_INSTANCES=8                               # total server instances to spawn
GPU_IDS="4,5,6,7"                             # comma-separated GPU indices; instances are round-robined
BASE_PORT=9100                                # ports are BASE_PORT, BASE_PORT+1, … BASE_PORT+NUM_INSTANCES-1

# Persona — baked into every instance. Leave empty for the zero-persona
# path (moshiko / moshi-2 base models).
# TEXT_PROMPT="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." # PersonaPlex
TEXT_PROMPT=""                                # "" = no text prompt
VOICE_PROMPT=""                               # path to a .pt or .wav; "" = no voice prompt /tmp/personaplex_voices/voices/NATM1.pt

# =============================================================================
# OPTIONAL — defaults below work out of the box
# =============================================================================

CONDA_ENV="raman_steerduplex"                 # conda env with torch + moshi installed
HEALTH_TIMEOUT=420                            # seconds to wait per server for the 0x00 handshake
LOG_DIR="logs/moshi_servers"                  # per-instance stdout/stderr
OUTPUT_JSON=""                                # if set, write {"servers": [...]} to this path once all healthy
CFG_COEF=""                                   # back-compat only; moshi_server ignores it but warns

# =============================================================================
# End of configuration — everything below is plumbing.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_SRC="$(cd "$SCRIPT_DIR/.." && pwd)"

# Build argv for launch_moshi_servers.py.
ARGS=(
    --num-instances "$NUM_INSTANCES"
    --gpu-ids "$GPU_IDS"
    --base-port "$BASE_PORT"
    --hf-repo "$HF_REPO"
    --log-dir "$LOG_DIR"
    --health-timeout "$HEALTH_TIMEOUT"
)
if [[ -n "$CHECKPOINT" ]]; then
    ARGS+=(--moshi-weight "$CHECKPOINT")
fi
if [[ -n "$CFG_COEF" ]]; then
    ARGS+=(--cfg-coef "$CFG_COEF")
fi
if [[ -n "$TEXT_PROMPT" ]]; then
    ARGS+=(--text-prompt "$TEXT_PROMPT")
fi
if [[ -n "$VOICE_PROMPT" ]]; then
    ARGS+=(--voice-prompt "$VOICE_PROMPT")
fi
if [[ -n "$OUTPUT_JSON" ]]; then
    ARGS+=(--output-json "$OUTPUT_JSON")
fi

echo "======================================================================"
echo "Launching Moshi server pool"
echo "======================================================================"
echo "  HF_REPO       : $HF_REPO"
echo "  CHECKPOINT    : ${CHECKPOINT:-<HF weights>}"
echo "  NUM_INSTANCES : $NUM_INSTANCES"
echo "  GPU_IDS       : $GPU_IDS"
echo "  BASE_PORT     : $BASE_PORT  (ports $BASE_PORT – $((BASE_PORT + NUM_INSTANCES - 1)))"
PROMPT_PREVIEW="${TEXT_PROMPT:0:72}"
if (( ${#TEXT_PROMPT} > 72 )); then
    PROMPT_PREVIEW+="..."
fi
echo "  TEXT_PROMPT   : ${PROMPT_PREVIEW:-<none>}"
echo "  VOICE_PROMPT  : ${VOICE_PROMPT:-<none>}"
echo "  CONDA_ENV     : $CONDA_ENV"
echo "  LOG_DIR       : $LOG_DIR"
echo "  OUTPUT_JSON   : ${OUTPUT_JSON:-<not written>}"
echo "======================================================================"

# cd into src/ so that Python's sys.path[0] (= CWD) contains the
# `moshi_server/` *package* rather than a bare `moshi_server.py` *module*.
# Without this, invoking the script from inside src/moshi_server/ itself
# makes Python pick up moshi_server.py as a top-level module and fail with
# "__path__ attribute not found on 'moshi_server'".
cd "$STEER_SRC"
PYTHONPATH="$STEER_SRC${PYTHONPATH:+:$PYTHONPATH}" \
exec conda run --no-capture-output -n "$CONDA_ENV" python \
    -m moshi_server.launch_moshi_servers "${ARGS[@]}"
