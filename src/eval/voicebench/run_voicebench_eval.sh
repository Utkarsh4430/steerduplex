#!/bin/bash
# End-to-end VoiceBench evaluation driver.
#
# Wraps the single command `python -m eval.voicebench.eval` which runs all 4
# stages with per-stage resume:
#   A. Inference (Moshi/PersonaPlex → per-split audio outputs)
#   B. ASR       (local Parakeet → plain-text transcriptions in output.json)
#   C. Judge     (gpt-4o-mini for open/qa splits; programmatic for the rest)
#   D. Summary   (paper-style leaderboard → summary.md + summary.json)
#
# Re-running this script is idempotent: each stage skips work that's already
# done. Use it to resume interrupted runs or to re-emit summary.md after fixes.
#
# Edit the MANDATORY block below, then:
#   bash src/eval/voicebench/run_voicebench_eval.sh
#
# When it finishes, open $OUT_ROOT/summary.md — that's all you need.

set -euo pipefail

# =============================================================================
# MANDATORY — edit these before running
# =============================================================================
MODEL_NAME="full_v3_20260406_235311"                                                                    # free-form label, appears in summary.md header
OUT_ROOT="/mnt/efs/ramaneswaranselvakumar/projects/steerduplex/artifacts/evaluation_outputs/voicebench/personaplex"
LITELLM_BASE_URL="https://litellm-proxy.ml-serving-internal.scale.com/v1"               # LiteLLM proxy base URL (used by the judge only)
LITELLM_API_KEY="<LITELLM_API_KEY>"                                             # LiteLLM API key (used by the judge only)

# Which model to evaluate. Uncomment one block:
#   - Moshi:       HF_REPO="kyutai/moshiko-pytorch-bf16"   ; SYSTEM_PROMPT=""
#       Moshi has NO system-prompt support. Kyutai's FAQ states changing Moshi's
#       personality "would require fine tuning, which is not currently supported"
#       (https://github.com/kyutai-labs/moshi/blob/main/FAQ.md). The official
#       inference & server code have no --system_prompt / --text_prompt flag;
#       only an internal ConditionAttributes("very_good"/"very_bad") quality
#       token. Leave SYSTEM_PROMPT empty for faithful Moshi eval.
#   - PersonaPlex: HF_REPO="nvidia/personaplex-7b-v1"      ; SYSTEM_PROMPT="Answer the question concisely."
#       PersonaPlex WAS trained with text system conditioning and auto-wraps
#       SYSTEM_PROMPT in <system>…<system> tags (see personaplex/moshi/moshi/server.py:wrap_with_system_tags).
#       Use "" to disable the system conditioning entirely.
HF_REPO="kyutai/moshiko-pytorch-bf16"
SYSTEM_PROMPT="You enjoy having a good conversation. Listen well, respond thoughtfully, and be natural."
VOICE_PROMPT="/tmp/personaplex_voices/voices/NATM1.pt"                                                                         # path to a voice-prompt .wav or .pt; "" = no voice conditioning. Moshi has no voice-prompt support — leave empty.
CHECKPOINT="/mnt/efs/utkarshtyagi/personal_projects/steerduplex/src/runs/full_v3_20260406_235311"                                                                           # path to a local finetuned checkpoint dir; "" = use HF_REPO weights

# =============================================================================
# OPTIONAL — defaults below work out of the box
# =============================================================================
# Splits to run. "" = all supported (alpacaeval_full commoneval wildvoice sd-qa ifeval bbh advbench).
# MMSU / OpenBookQA are MCQ and deliberately unsupported — they appear as "—" in summary.md.
SPLITS=""
SDQA_REGIONS="usa"                          # sd-qa region(s), space-separated. Paper reports usa only; leave as "usa".

# Sampling / decoding.
MAX_DURATION="60.0"                         # seconds — safety cap on Moshi's generation; inference also early-stops after 5s of silence
GREEDY=0                                    # 1 = greedy decode; 0 = sampling with SEED
SEED=42

# GPU placement — single knob, literal torch device strings.
# Splits always run sequentially (enforced by eval.py), so serial execution is
# the norm. Workers *within* a split do load in parallel — INSTANCES_PER_GPU=2
# keeps 2 Moshi copies resident on one GPU (~22 GB; fits comfortably on 80 GB).
# Run `nvidia-smi` to confirm the chosen GPU is free before launching.
# For PersonaPlex (~2× larger) drop INSTANCES_PER_GPU to 1.
DEVICES="cuda:4 cuda:5 cuda:6 cuda:7"                            # space-separated list of cuda:N device strings; one entry = one GPU
INSTANCES_PER_GPU=4                         # Moshi copies per GPU

# Smoke / partial runs.
SANITY=0                                    # 1 = first 5 examples per split (fast smoke test)
LIMIT=""                                    # positive int = cap per-split examples; overrides SANITY; "" = full split

# Stage B (Parakeet ASR) — runs in its own conda env so the Moshi env doesn't
# need nemo_toolkit. ASR_ENV_PYTHON must point at a python binary with
# `import nemo.collections.asr` working.
ASR_ENV_PYTHON="/mnt/efs/ramaneswaranselvakumar/miniconda3/envs/raman_fdb_v1/bin/python"
ASR_DEVICE="cuda:4"                         # GPU for Parakeet (independent of inference DEVICES)
ASR_BATCH_SIZE=32                           # batch size for parakeet.transcribe(); ~0.7 GB/item
FORCE_ASR=0                                 # 1 = re-transcribe everything (wipes downstream scores too)

# Stage C (judge) knobs — pure HTTP fan-out to LiteLLM, unrelated to GPU.
WORKERS=32                                  # parallel LLM-judge requests via ThreadPoolExecutor
JUDGE_MODEL="gpt-4o-mini"                   # LiteLLM model for open/qa evaluators

# Advanced: restrict to a single stage (inference / asr / judge / summary).
# Useful for re-rendering the summary after a manual edit, or re-running ASR
# after a Parakeet upgrade. Leave empty to run all stages.
ONLY="judge"
SKIP_ASR=0                                  # 1 = skip Stage B even when ONLY is empty
SKIP_JUDGE=0                                # 1 = skip Stage C even when ONLY is empty

# Conda env that has moshi + this repo installed.
STEER_ENV="raman_steerduplex"

# =============================================================================
# End of configuration — everything below this line is plumbing.
# =============================================================================

export LITELLM_BASE_URL LITELLM_API_KEY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEER_SRC="$(cd "$SCRIPT_DIR/../.." && pwd)"      # steerduplex/src
STEER_PY="conda run --no-capture-output -n ${STEER_ENV} python"

mkdir -p "$OUT_ROOT"

echo "======================================================================"
echo "VoiceBench evaluation"
echo "======================================================================"
echo "  started        : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  MODEL_NAME     : $MODEL_NAME"
echo "  OUT_ROOT       : $OUT_ROOT"
echo "  HF_REPO        : $HF_REPO"
echo "  CHECKPOINT     : ${CHECKPOINT:-<none>}"
echo "  SYSTEM_PROMPT  : ${SYSTEM_PROMPT:-<empty>}"
echo "  VOICE_PROMPT   : ${VOICE_PROMPT:-<none>}"
echo "  DEVICES        : $DEVICES (instances/gpu=$INSTANCES_PER_GPU)"
echo "  ASR_DEVICE     : $ASR_DEVICE (batch=$ASR_BATCH_SIZE, force=$FORCE_ASR)"
echo "  SPLITS         : ${SPLITS:-<all>}"
echo "  SDQA_REGIONS   : $SDQA_REGIONS"
echo "  WORKERS        : $WORKERS (judge model: $JUDGE_MODEL)"
echo "  SANITY=$SANITY  LIMIT=${LIMIT:-<full>}  ONLY=${ONLY:-<all stages>}"
echo "======================================================================"

# --- Build optional flags ----------------------------------------------------
CHECKPOINT_FLAG=()
[[ -n "$CHECKPOINT" ]] && CHECKPOINT_FLAG=(--checkpoint "$CHECKPOINT")

SPLITS_FLAG=()
if [[ -n "$SPLITS" ]]; then
    # shellcheck disable=SC2206
    SPLITS_FLAG=(--splits $SPLITS)
fi

SDQA_FLAG=()
if [[ -n "$SDQA_REGIONS" ]]; then
    # shellcheck disable=SC2206
    SDQA_FLAG=(--sdqa_regions $SDQA_REGIONS)
fi

# shellcheck disable=SC2206
DEVICES_ARR=($DEVICES)

GREEDY_FLAG=()
[[ "$GREEDY" == "1" ]] && GREEDY_FLAG=(--greedy)

SANITY_FLAG=()
[[ "$SANITY" == "1" ]] && SANITY_FLAG=(--sanity)

LIMIT_FLAG=()
[[ -n "$LIMIT" ]] && LIMIT_FLAG=(--limit "$LIMIT")

VOICE_PROMPT_FLAG=()
[[ -n "$VOICE_PROMPT" ]] && VOICE_PROMPT_FLAG=(--voice_prompt "$VOICE_PROMPT")

FORCE_ASR_FLAG=()
[[ "$FORCE_ASR" == "1" ]] && FORCE_ASR_FLAG=(--force_asr)

SKIP_ASR_FLAG=()
[[ "$SKIP_ASR" == "1" ]] && SKIP_ASR_FLAG=(--skip_asr)

SKIP_JUDGE_FLAG=()
[[ "$SKIP_JUDGE" == "1" ]] && SKIP_JUDGE_FLAG=(--skip_judge)

ONLY_FLAG=()
[[ -n "$ONLY" ]] && ONLY_FLAG=(--only "$ONLY")

# --- Run eval (all 4 stages) -------------------------------------------------
cd "$STEER_SRC"
PYTHONPATH="$STEER_SRC${PYTHONPATH:+:$PYTHONPATH}" \
$STEER_PY -m eval.voicebench.eval \
    --output_dir "$OUT_ROOT" \
    --hf_repo "$HF_REPO" \
    "${CHECKPOINT_FLAG[@]}" \
    --system_prompt "$SYSTEM_PROMPT" \
    "${VOICE_PROMPT_FLAG[@]}" \
    --device "${DEVICES_ARR[@]}" \
    --instances_per_gpu "$INSTANCES_PER_GPU" \
    --max_duration "$MAX_DURATION" \
    --seed "$SEED" \
    "${GREEDY_FLAG[@]}" \
    "${SPLITS_FLAG[@]}" \
    "${SDQA_FLAG[@]}" \
    "${SANITY_FLAG[@]}" \
    "${LIMIT_FLAG[@]}" \
    --asr_env_python "$ASR_ENV_PYTHON" \
    --asr_device "$ASR_DEVICE" \
    --asr_batch_size "$ASR_BATCH_SIZE" \
    "${FORCE_ASR_FLAG[@]}" \
    --workers "$WORKERS" \
    --judge_model "$JUDGE_MODEL" \
    --litellm-base "$LITELLM_BASE_URL" \
    "${SKIP_ASR_FLAG[@]}" \
    "${SKIP_JUDGE_FLAG[@]}" \
    "${ONLY_FLAG[@]}" \
    2>&1 | tee "$OUT_ROOT/run.log"

echo ""
echo "======================================================================"
echo "Done."
echo "  finished : $(date '+%Y-%m-%d %H:%M:%S')"
echo "  summary  : $OUT_ROOT/summary.md"
echo "  metrics  : $OUT_ROOT/summary.json"
echo "  logs     : $OUT_ROOT/run.log"
echo "======================================================================"
