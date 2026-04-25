#!/bin/bash
# End-to-end Full Duplex Bench v1.0 / v1.5 evaluation driver.
#
# Sequences three stages:
#   1. Inference  (raman_steerduplex env, async client → already-running Moshi servers)
#   2. ASR        (raman_fdb_v1 env w/ Parakeet, or raman_whisperx env w/ WhisperX)
#   3. Eval       (raman_fdb_v1 env, vendored FDB scoring scripts via subprocess)
#
# Server lifecycle is decoupled: launch moshi_server.py instances yourself
# (e.g. via launch_moshi_servers.py) then point SERVER_PORTS at them.
#
# Edit the MANDATORY block below before running. See README.md for details.
#
# Usage:
#   bash src/eval/fdb_v1/run_fdb_eval.sh

set -euo pipefail

# =============================================================================
# MANDATORY — edit these before running
# =============================================================================
MODEL_NAME="full_v3_20260406_235311"                                  # label stored in run_meta.json
OUT_ROOT="/mnt/efs/ramaneswaranselvakumar/projects/steerduplex/artifacts/evaluation_outputs/fdb_v1/full_v3_20260406_235311" # where all outputs go
LITELLM_BASE_URL="https://litellm-proxy.ml-serving-internal.scale.com/v1"         # LiteLLM proxy base URL
LITELLM_API_KEY="<LITELLM_API_KEY>"                            # LiteLLM API key

# Moshi server URLs. Required for the inference stage. Accepts:
#   - Port ranges     : "9001-9010"                              (10 servers, host=$SERVER_HOST)
#   - Comma list      : "9001,9003,9005-9007"                    (5 servers, host=$SERVER_HOST)
#   - Full URLs       : "ws://gpu-host-1:9001/api/chat,ws://gpu-host-2:9001/api/chat"
#   - Mix             : "9001,ws://other-host:9100/api/chat"
# Launch servers first via:
#   conda run -n raman_steerduplex python -m eval.fdb_v1.launch_moshi_servers \
#       --num-instances 10 --gpu-ids 0,1,2,3 --base-port 9001 --hf-repo "$HF_REPO"
SERVER_PORTS="9100-9113"                                        # e.g. "9001-9010"
SERVER_HOST="localhost"                                # host applied when SERVER_PORTS uses bare ports

# =============================================================================
# OPTIONAL — defaults below work out of the box
# =============================================================================
HF_REPO="kyutai/moshiko-pytorch-bf16"     # base Moshi checkpoint
CHECKPOINT=""                             # path to local checkpoint dir/file; "" = use HF_REPO weights
CFG_COEF=""                               # CFG coefficient (forwarded to moshi_server.py); "" = server default
VERSION="both"                            # 1.0 | 1.5 | both
TASKS=""                                  # space-separated task filter; "" = all tasks
SEED=42                                   # provenance only (server controls sampling)
STAGES="inference asr eval"               # any subset, space-separated
OVERWRITE_INFERENCE=0                     # 1 = regenerate output.wav even if it exists
OVERWRITE_ASR=0                           # 1 = regenerate output.json even if it exists

DATASET_ROOT="/mnt/efs/utkarshtyagi/personal_projects/steerduplex/data/benchmarks/fd_bench_v1"          # v1.0 source dirs at top level. v1.5 zips ship vendored next to this script (data/v1.5/) and are auto-extracted.
STEER_ENV="raman_steerduplex"             # conda env that runs the moshi server + client
ASR_ENV="raman_whisperx"                  # conda env with whisperx (only needed if ASR_BACKEND=whisperx)
FDB_ENV="raman_fdb_v1"                    # conda env with FDB scoring deps + nemo Parakeet
STEER_PY="conda run --no-capture-output -n ${STEER_ENV} python"   # python for inference orchestrator + servers
FDB_PY="conda run --no-capture-output -n ${FDB_ENV} python"       # python for FDB scoring + Parakeet ASR

# ASR backend. Parakeet matches the official FDB pipeline; WhisperX is a fallback.
ASR_BACKEND="parakeet"                    # parakeet | whisperx
ASR_MODEL="large-v3"                      # whisperx ASR model (ignored when ASR_BACKEND=parakeet)
ASR_DEVICE="cuda:0"                       # single GPU for ASR
ASR_COMPUTE_TYPE="float16"                # whisperx CTranslate2 compute type (ignored for parakeet)
CUDA_VISIBLE_DEVICES="0,1,2,3"            # GPU indices exposed to subprocesses

# =============================================================================
# End of configuration — everything below this line is plumbing.
# =============================================================================

export CUDA_VISIBLE_DEVICES LITELLM_BASE_URL LITELLM_API_KEY

# Resolve paths: script lives at src/eval/fdb_v1/run_fdb_eval.sh → steerduplex/src is 2 levels up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"

mkdir -p "$OUT_ROOT"

# --- Wall-clock timing helpers ------------------------------------------------
SECONDS_AT_START=$SECONDS

fmt_duration() {
    local t=$1
    if (( t < 60 )); then
        echo "${t}s"
    elif (( t < 3600 )); then
        echo "$(( t / 60 ))m$(( t % 60 ))s"
    else
        echo "$(( t / 3600 ))h$(( (t % 3600) / 60 ))m$(( t % 60 ))s"
    fi
}

stage_banner() {
    local name=$1
    local sep
    sep=$(printf '=%.0s' {1..72})
    echo ""
    echo "$sep"
    echo ">>> [$(date '+%H:%M:%S')] $name"
    echo "$sep"
}

stage_footer() {
    local name=$1
    local start=$2
    local elapsed=$(( SECONDS - start ))
    echo ""
    echo "<<< [$(date '+%H:%M:%S')] $name — done in $(fmt_duration $elapsed)"
}

# Expand a SERVER_PORTS spec into one ws://… URL per line on stdout.
# Accepts a comma-separated list whose elements are any of:
#   - a port range  "9001-9010"   (inclusive)
#   - a single port "9001"
#   - a full URL    "ws://host:port/api/chat" (passes through unchanged)
# Single-port and range entries are formatted as ws://${SERVER_HOST}:PORT/api/chat.
expand_server_spec() {
    local spec="$1"
    local host="${2:-localhost}"
    if [[ -z "$spec" ]]; then
        return 0
    fi
    local IFS=','
    # shellcheck disable=SC2206
    local parts=( $spec )
    for p in "${parts[@]}"; do
        # trim whitespace
        p="${p#"${p%%[![:space:]]*}"}"
        p="${p%"${p##*[![:space:]]}"}"
        [[ -z "$p" ]] && continue
        if [[ "$p" == ws://* || "$p" == wss://* ]]; then
            echo "$p"
        elif [[ "$p" =~ ^[0-9]+-[0-9]+$ ]]; then
            local start="${p%-*}"
            local end="${p#*-}"
            if (( start > end )); then
                echo "ERROR: invalid port range '$p' (start > end)" >&2
                return 1
            fi
            local port
            for (( port=start; port<=end; port++ )); do
                echo "ws://${host}:${port}/api/chat"
            done
        elif [[ "$p" =~ ^[0-9]+$ ]]; then
            echo "ws://${host}:${p}/api/chat"
        else
            echo "ERROR: cannot parse SERVER_PORTS entry '$p' (expected port, port-range, or ws:// URL)" >&2
            return 1
        fi
    done
}

echo "======================================================================"
echo "Full Duplex Bench v1.0 / v1.5 evaluation"
echo "======================================================================"
echo "  started     : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  MODEL_NAME  : $MODEL_NAME"
echo "  OUT_ROOT    : $OUT_ROOT"
echo "  HF_REPO     : $HF_REPO"
echo "  CHECKPOINT  : ${CHECKPOINT:-<none>}"
echo "  VERSION     : $VERSION"
echo "  TASKS       : ${TASKS:-<all>}"
echo "  STAGES      : $STAGES"
echo "  SERVER_PORTS: ${SERVER_PORTS:-<unset>} (host=$SERVER_HOST)"
echo "  DATASET     : $DATASET_ROOT"
echo "  STEER_ENV   : $STEER_ENV"
echo "  FDB_ENV     : $FDB_ENV"
echo "  ASR_BACKEND : $ASR_BACKEND ($ASR_DEVICE)"
echo "======================================================================"

# --- Build optional flags ----------------------------------------------------
TASKS_FLAG=()
if [[ -n "$TASKS" ]]; then
    # shellcheck disable=SC2206
    TASKS_FLAG=(--tasks $TASKS)
fi

CHECKPOINT_FLAG=()
SERVER_CKPT_FLAG=()
if [[ -n "$CHECKPOINT" ]]; then
    CHECKPOINT_FLAG=(--checkpoint "$CHECKPOINT")
    SERVER_CKPT_FLAG=(--moshi-weight "$CHECKPOINT")
fi

CFG_COEF_FLAG=()
SERVER_CFG_FLAG=()
if [[ -n "$CFG_COEF" ]]; then
    CFG_COEF_FLAG=(--cfg_coef "$CFG_COEF")
    SERVER_CFG_FLAG=(--cfg-coef "$CFG_COEF")
fi

OVERWRITE_INF_FLAG=()
[[ "$OVERWRITE_INFERENCE" == "1" ]] && OVERWRITE_INF_FLAG=(--overwrite)

OVERWRITE_ASR_FLAG=()
[[ "$OVERWRITE_ASR" == "1" ]] && OVERWRITE_ASR_FLAG=(--overwrite)

# --- Vendored v1.5 zip extraction --------------------------------------------
# v1.5 sources ship as zips under data/v1.5/ alongside this script. Extract
# them (idempotent) so dataset_utils._find_source_dir can resolve them
# without depending on DATASET_ROOT.
V15_LOCAL_DIR="$SCRIPT_DIR/data/v1.5"
if compgen -G "$V15_LOCAL_DIR/*.zip" > /dev/null; then
    for zf in "$V15_LOCAL_DIR"/*.zip; do
        name="$(basename "$zf" .zip)"
        if [[ ! -d "$V15_LOCAL_DIR/$name" ]]; then
            echo "Extracting $(basename "$zf") -> $V15_LOCAL_DIR/$name"
            unzip -q "$zf" -d "$V15_LOCAL_DIR/"
        fi
    done
fi

# --- 1. Inference -------------------------------------------------------------
# Connects to already-running Moshi servers. Launch them yourself first, e.g.:
#   conda run --no-capture-output -n raman_steerduplex python \
#       -m eval.fdb_v1.launch_moshi_servers \
#       --num-instances 10 --gpu-ids 0,1,2,3 --base-port 9001 --hf-repo "$HF_REPO"
if [[ " $STAGES " == *" inference "* ]]; then
    if [[ -z "$SERVER_PORTS" ]]; then
        echo "ERROR: SERVER_PORTS is empty but STAGES includes 'inference'." >&2
        echo "  Launch Moshi servers first (see launch_moshi_servers.py) then set" >&2
        echo "  SERVER_PORTS to e.g. \"9001-9010\" or a comma list." >&2
        exit 1
    fi

    # Resolve the spec into an array of URLs, one per server. Bail out early
    # if expansion fails so we don't pass garbage to run_inference.
    if ! mapfile -t SERVER_URLS < <(expand_server_spec "$SERVER_PORTS" "$SERVER_HOST"); then
        echo "ERROR: failed to parse SERVER_PORTS=\"$SERVER_PORTS\"" >&2
        exit 1
    fi
    if (( ${#SERVER_URLS[@]} == 0 )); then
        echo "ERROR: SERVER_PORTS=\"$SERVER_PORTS\" expanded to zero URLs." >&2
        exit 1
    fi

    stage_banner "Stage 1/3: Inference (${#SERVER_URLS[@]} Moshi server URL(s))"
    echo "  servers: ${SERVER_URLS[*]}"
    STAGE1_START=$SECONDS

    cd "$STEER_SRC"
    PYTHONPATH="$STEER_SRC${PYTHONPATH:+:$PYTHONPATH}" \
    $STEER_PY -m eval.fdb_v1.run_inference \
        --dataset_root "$DATASET_ROOT" \
        --output_root "$OUT_ROOT" \
        --model_name "$MODEL_NAME" \
        --hf_repo "$HF_REPO" \
        "${CHECKPOINT_FLAG[@]}" \
        "${CFG_COEF_FLAG[@]}" \
        --version "$VERSION" \
        "${TASKS_FLAG[@]}" \
        --server "${SERVER_URLS[@]}" \
        --seed "$SEED" \
        "${OVERWRITE_INF_FLAG[@]}"

    stage_footer "Stage 1/3: Inference" $STAGE1_START
fi

# --- 2. ASR -------------------------------------------------------------------
# Parakeet runs in raman_fdb_v1 (already has nemo_toolkit[asr]); WhisperX runs
# in raman_whisperx. Both produce identical JSON shape so downstream eval is
# backend-agnostic.
if [[ " $STAGES " == *" asr "* ]]; then
    if [[ "$ASR_BACKEND" == "parakeet" ]]; then
        stage_banner "Stage 2/3: ASR (parakeet-tdt-0.6b-v2, env=$FDB_ENV)"
        ASR_PY="$FDB_PY"
        ASR_BACKEND_ARGS=(--asr_backend parakeet --device "$ASR_DEVICE")
    elif [[ "$ASR_BACKEND" == "whisperx" ]]; then
        stage_banner "Stage 2/3: ASR (whisperx $ASR_MODEL, env=$ASR_ENV)"
        ASR_PY="conda run --no-capture-output -n ${ASR_ENV} python"
        ASR_BACKEND_ARGS=(--asr_backend whisperx --asr_model "$ASR_MODEL" \
                          --device "$ASR_DEVICE" --compute_type "$ASR_COMPUTE_TYPE")
    else
        echo "ERROR: unknown ASR_BACKEND=$ASR_BACKEND (expected parakeet|whisperx)" >&2
        exit 1
    fi
    STAGE2_START=$SECONDS
    PYTHONPATH="$STEER_SRC${PYTHONPATH:+:$PYTHONPATH}" \
    $ASR_PY -m eval.fdb_v1.run_asr \
        --output_root "$OUT_ROOT" \
        --version "$VERSION" \
        "${TASKS_FLAG[@]}" \
        "${ASR_BACKEND_ARGS[@]}" \
        "${OVERWRITE_ASR_FLAG[@]}"
    stage_footer "Stage 2/3: ASR" $STAGE2_START
fi

# --- 3. Eval (FDB env via subprocess) ----------------------------------------
if [[ " $STAGES " == *" eval "* ]]; then
    stage_banner "Stage 3/3: FDB scoring ($FDB_PY)"
    STAGE3_START=$SECONDS
    cd "$STEER_SRC"
    PYTHONPATH="$STEER_SRC${PYTHONPATH:+:$PYTHONPATH}" \
    $STEER_PY -m eval.fdb_v1.run_eval \
        --output_root "$OUT_ROOT" \
        --fdb_python "$FDB_PY" \
        --version "$VERSION" \
        "${TASKS_FLAG[@]}"
    stage_footer "Stage 3/3: FDB scoring" $STAGE3_START
fi

TOTAL_ELAPSED=$(( SECONDS - SECONDS_AT_START ))
echo ""
echo "======================================================================"
echo "All stages complete in $(fmt_duration $TOTAL_ELAPSED)"
echo "  finished : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  summary  : $OUT_ROOT/summary.md"
echo "  logs     : $OUT_ROOT/logs/"
echo "  meta     : $OUT_ROOT/run_meta.json"
echo "======================================================================"
