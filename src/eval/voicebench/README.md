# VoiceBench Evaluation

Three-phase pipeline for running SteerDuplex / Moshi on the
[VoiceBench](https://huggingface.co/datasets/hlt-lab/voicebench) dataset,
restricted to single-turn, non-MCQ subsets.

| Phase | Script | What it does |
|---|---|---|
| 1 | [eval.py](eval.py) | Moshi inference on every split (multi-GPU, multi-instance-per-GPU). |
| 2 | [llm_judge.py](llm_judge.py) | Whisper-transcribe outputs + gpt-4o-mini judging for `open` / `qa` splits. |
| 3 | [summarize.py](summarize.py) | Aggregate all splits via VoiceBench's own scorers → `summary.md`. |

`eval.py` runs all three phases by default; pass `--skip_judge` to stop after Phase 1.

## Supported splits (default: run all)

| Subset | # | Scorer | Notes |
|---|---:|---|---|
| `alpacaeval_full` | 636 | `open` (gpt-4o-mini 1–5 mean) | |
| `commoneval` | 200 | `open` | |
| `wildvoice` | 1000 | `open` | |
| `sd-qa` (usa) | 553 | `qa` (gpt-4o-mini yes %, PEDANT %) | `usa` region by default; see `--sdqa_regions`. |
| `ifeval` | 345 | `ifeval` (programmatic) | |
| `bbh` | 1000 | `bbh` (programmatic) | |
| `advbench` | 520 | `harm` (programmatic refusal rate) | |

Excluded (by design): `mmsu`, `openbookqa` (MCQ), `mtbench` (multi-turn).

## Usage

### Full 8×H100 run

```bash
export LITELLM_BASE_URL=https://your-litellm-proxy
export LITELLM_API_KEY=...

python -m eval.voicebench.eval \
    --output_dir /path/to/output \
    --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --instances_per_gpu 2 \
    --hf_repo kyutai/moshiko-pytorch-bf16
```

### Sanity smoke (5 examples per split, 1 GPU)

```bash
python -m eval.voicebench.eval \
    --output_dir /tmp/vb_smoke \
    --device cuda:0 \
    --instances_per_gpu 1 \
    --sanity
```

### Finetuned checkpoint

```bash
python -m eval.voicebench.eval \
    --output_dir /path/to/output \
    --checkpoint runs/full_v3_.../checkpoint_005000/consolidated \
    --device cuda:0 cuda:1 cuda:2 cuda:3 \
    --instances_per_gpu 2
```

### Run Phase 2 / 3 separately

```bash
python -m eval.voicebench.llm_judge --output_dir /path/to/output
python -m eval.voicebench.summarize --output_dir /path/to/output
```

## Key flags

| Flag | Default | Notes |
|---|---|---|
| `--output_dir` | *(required)* | Holds all outputs. Auto-resumable. |
| `--splits` | all 7 | Subset names. |
| `--sdqa_regions` | `usa` (warning logged) | One or more of `aus gbr ind_n ind_s irl kenya nga nzl phl usa zaf`. |
| `--device` | `cuda:0` | Physical GPUs to use. |
| `--instances_per_gpu` | `2` | Moshiko is ~14GB bf16, so 2 fits comfortably on H100. |
| `--hf_repo` / `--checkpoint` | `kyutai/moshiko-pytorch-bf16` / none | Base model + optional finetune. |
| `--sanity` | off | Cap every split to 5 examples (smoke test). |
| `--limit N` | none | Per-split cap (overrides `--sanity` if set). |
| `--skip_judge` | off | Stop after Phase 1. |
| `--judge_model` | `gpt-4o-mini` | LiteLLM model for `open` / `qa` judging. |
| `--workers` | `16` | Threads/processes for Phase 2. |
| `--greedy` | off | Deterministic decoding. |

## Output layout

```
<output_dir>/
├── run_manifest.json           # records all CLI args / config
├── alpacaeval_full/
│   ├── output.json             # inference records
│   ├── output_audios/*.wav     # Moshi outputs
│   ├── _inputs/*.wav           # pre-extracted user audio (safe to delete post-run)
│   └── llm_judge_output.json   # adds `transcription` + `score`
├── commoneval/…
├── wildvoice/…
├── sd-qa/usa/…                 # per-region subdir(s)
├── ifeval/…
├── bbh/…
├── advbench/…
├── summary.md                  # human-readable table
└── summary.json                # same data, machine-readable
```

Re-running with the same `--output_dir` auto-resumes: completed `unique_id`s
are skipped in Phase 1, and entries with `score` populated are skipped in
Phase 2.

## Environment

- `.env` at repo root (loaded via `python-dotenv`) supplies:
  - `LITELLM_BASE_URL`, `LITELLM_API_KEY` (Phase 2)
  - `HF_TOKEN` (not required for the public VoiceBench dataset)
- `summarize.py` expects the official VoiceBench repo at
  `/mnt/efs/ramaneswaranselvakumar/projects/steerduplex/VoiceBench/` and
  `qa_metrics`, `loguru` installed. If that path changes, edit
  `VOICEBENCH_SRC` in [summarize.py](summarize.py).

## How it differs from bigbench_audio

- Dataset lives on HF as Arrow (not snapshot MP3s) → audio is pre-extracted
  into `<split>/_inputs/*.wav` once per split for workers to share.
- Multiple model instances per GPU via `--instances_per_gpu`. Each worker pins
  `CUDA_VISIBLE_DEVICES` to one physical GPU; multiple workers on the same
  device share VRAM.
- Split-specific scoring: `open` and `qa` use the gpt-4o-mini judge; `ifeval`
  / `bbh` / `harm` are programmatic. Unified `summary.md` at the end.
