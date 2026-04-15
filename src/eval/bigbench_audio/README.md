# BigBench Audio Evaluation

Two-phase pipeline: **Phase 1** runs Moshi inference on the BigBench Audio dataset, **Phase 2** judges the outputs with an LLM.

---

## Phase 1 — Moshi Inference (`eval.py`)

### Sequential (single GPU)

```bash
python -m eval.bigbench_audio.eval --device cuda:0
```

### Parallel (multiple GPUs)

Pass multiple devices; one worker process is spawned per GPU and examples are distributed round-robin.

```bash
python -m eval.bigbench_audio.eval --device cuda:0 cuda:1 cuda:2 cuda:3
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--device` | `cuda` | One or more devices. Multiple devices → parallel inference. |
| `--checkpoint` | None | Path to a finetuned checkpoint directory. Omit to use the base model. |
| `--hf_repo` | `kyutai/moshiko-pytorch-bf16` | HuggingFace model repo for the base model. |
| `--output_dir` | auto-generated | Directory to write `output.json`. Auto-resumes if the directory already contains results. |
| `--max_duration` | `30.0` | Maximum audio duration (seconds) fed to Moshi. |
| `--greedy` | off | Pass to disable sampling (deterministic decoding). |
| `--skip_transcription` | off | Skip Whisper post-transcription of Moshi outputs. |
| `--limit` | None | Process only the first N examples (useful for debugging). |
| `--seed` | `42` | Random seed. |

**Note:** `eval.py` auto-resumes from an existing `output.json` — already-processed examples are skipped.

---

## Phase 2 — LLM Judge (`llm_judge.py`)

Reads the `output.json` produced by Phase 1 and judges each Moshi transcription as `CORRECT` or `INCORRECT` against the official answer using a LiteLLM-proxied model.

### Environment variables

| Variable | Description |
|---|---|
| `LITELLM_BASE_URL` | Base URL of the LiteLLM proxy. |
| `LITELLM_API_KEY` | API key for the LiteLLM proxy. |

### Run

```bash
export LITELLM_BASE_URL=https://your-litellm-proxy
export LITELLM_API_KEY=your-key

# Auto-detects the latest moshi_results_* directory
python llm_judge.py

# Or point to a specific output.json
python llm_judge.py --input ./moshi_results_20240101_120000/output.json
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | latest `moshi_results_*/output.json` | Path to `output.json` from Phase 1. |
| `--output` | `<input_dir>/llm_judge_output.json` | Where to write judged results. |
| `--model` | `gpt-5.4-mini` | LiteLLM model identifier to use for judging. |
| `--workers` | `min(16, cpu_count)` | Number of parallel worker processes. |
| `--candidate-field` | `transcription` | Field to judge: `transcription` (Whisper-transcribed audio) or `moshi_text` (Moshi text stream). |
| `--litellm-base` | `$LITELLM_BASE_URL` | Override the proxy URL on the command line. |
| `--limit` | None | Judge only the first N examples. |

**Note:** `llm_judge.py` auto-resumes — entries already present in `llm_judge_output.json` are skipped.

---

## Full pipeline example

```bash
# 1. Run inference on 4 GPUs
python eval.py --device cuda:0 cuda:1 cuda:2 cuda:3 --output_dir ./results

# 2. Judge outputs
export LITELLM_BASE_URL=https://your-litellm-proxy
export LITELLM_API_KEY=your-key
python llm_judge.py --input ./results/output.json
```
