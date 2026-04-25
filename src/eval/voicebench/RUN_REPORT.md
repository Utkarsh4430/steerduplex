# VoiceBench Harness — Implementation + First-Run Report

**Date:** 2026-04-20
**Author:** Claude (Opus 4.7) for @ramaneswaran.selvakumar
**Status:** Module implemented; end-to-end sanity run launched in background on GPU 0.

---

## 1. What got built

New module at [steerduplex/src/eval/voicebench/](.):

| File | Purpose |
|---|---|
| [splits.py](splits.py) | Registry of the 7 target subsets + sd-qa region handling. Exposes `build_specs()` that expands `sd-qa` into one `SplitSpec` per region. |
| [eval.py](eval.py) | Phase 1. Multi-GPU, multi-instance-per-GPU inference. Single-worker fast path + spawn-based workers for N>1. Pre-extracts audio to `<split>/_inputs/*.wav` so workers read from disk (no IPC bloat). |
| [llm_judge.py](llm_judge.py) | Phase 2. Whisper-1 transcription via LiteLLM (ThreadPool) for all splits, then gpt-4o-mini judging (fork-based multiprocessing) for `open` / `qa` evaluators only. Programmatic scorers (ifeval / bbh / harm) don't need this phase. |
| [summarize.py](summarize.py) | Phase 3. Delegates scoring to **VoiceBench's official evaluator classes** (imported by adding `VoiceBench/src` to `sys.path`) — leaderboard-comparable. Writes `summary.md` + `summary.json`. |
| [README.md](README.md) | User-facing docs. |

Default splits run (all single-turn, non-MCQ):

`alpacaeval_full` (636), `commoneval` (200), `wildvoice` (1000), `sd-qa/usa` (553), `ifeval` (345), `bbh` (1000), `advbench` (520).

sd-qa defaults to `usa` only; a loud `WARNING` is logged listing the 10 skipped regions and pointing at `--sdqa_regions` for opting into more.

## 2. Dependencies installed

Installed into `raman_steerduplex` conda env (it already had `moshi`; it was missing the HF loading + scorer deps):

```
pip install datasets qa_metrics loguru python-dotenv torchcodec nltk
```

Notes:
- **numpy downgrade**: `qa_metrics` pinned `numpy<2`, so 2.2.6 → 1.26.4. Moshi + torch verified to still import OK (`moshi 0.2.13`, `torch 2.9.1+cu128`, `cuda.is_available()=True`).
- **torchcodec import fails** on this box: missing `libavutil.so.59/58/57` + binary mismatch with torch 2.9.
  - **Workaround shipped in [eval.py:130-163](eval.py#L130-L163)**: load VoiceBench via `Audio(decode=False)` and decode raw bytes with `soundfile` ourselves. Zero torchcodec dependency at runtime.
- `qa_metrics.pedant.PEDANT` imports cleanly; first call at summarize time may trigger NLTK/POS-tagger downloads — NLTK already pip-installed, but if corpora are missing you may need one-off `nltk.download('averaged_perceptron_tagger')`, etc. No action needed unless Phase 3 errors out.

## 3. Sanity run (in progress — background task ID `bqyyzct1r`)

Launched with:
```bash
cd /mnt/efs/ramaneswaranselvakumar/projects/steerduplex/steerduplex/src
CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n raman_steerduplex \
    python -u -m eval.voicebench.eval \
    --output_dir /mnt/efs/ramaneswaranselvakumar/projects/steerduplex/artifacts/voicebench_smoke \
    --device cuda:0 --instances_per_gpu 1 --sanity \
    > /mnt/efs/ramaneswaranselvakumar/projects/steerduplex/artifacts/voicebench_smoke/run.log 2>&1
```

- `--sanity` = 5 examples per split. Totals ~35 inferences (sd-qa has 1 region).
- End-to-end: Phase 1 (inference) → Phase 2 (Whisper + gpt-4o-mini judge) → Phase 3 (summary.md).
- Live log: `artifacts/voicebench_smoke/run.log`.
- Results: `artifacts/voicebench_smoke/summary.md` + `summary.json` once the run finishes.

**How to check when you're back:**
```bash
tail -50 /mnt/efs/ramaneswaranselvakumar/projects/steerduplex/artifacts/voicebench_smoke/run.log
cat /mnt/efs/ramaneswaranselvakumar/projects/steerduplex/artifacts/voicebench_smoke/summary.md
```

`ps -ef | grep voicebench.eval` tells you if it's still alive.

## 4. Pre-launch verification done

- [x] `from eval.voicebench import splits, eval, llm_judge, summarize` — clean imports.
- [x] `build_specs(['commoneval', 'sd-qa', 'ifeval'], ['usa'])` — correct expansion, all 3 specs built.
- [x] HF dataset probe: confirmed every target subset loads (`len`, `keys`, audio sr=16000). See `artifacts/voicebench_smoke/.probe.txt` style reference is in the run history; fields carried forward match each evaluator's expectations:
  - `open`: `prompt`, `response` (transcription)
  - `qa`: + `reference`
  - `ifeval`: + `key`, `instruction_id_list`, `kwargs`
  - `bbh`: + `reference`, `id`
  - `harm`: `response` only
- [x] Moshi `MoshiInference.generate(user_audio_path, system_prompt, max_duration_sec)` signature matches our call site.
- [x] Initial smoke (prior iteration, killed after model load) produced the first 5 `alpacaeval_full` Moshi audios successfully and started `commoneval` — this confirms Phase 1 core path works. Killed and restarted as the full end-to-end run.

## 5. Known things to watch on the first real run

1. **NLTK corpora**: PEDANT may call NLTK downloaders on first use. If Phase 3 fails with `LookupError: Resource averaged_perceptron_tagger not found`, run:
   ```python
   import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')
   ```
2. **LiteLLM credentials**: Loaded from `src/.env` via `python-dotenv`. Verified the two keys exist (`LITELLM_API_KEY`, `LITELLM_BASE_URL`). If Phase 2 errors, double-check the proxy is reachable.
3. **Per-split logs**: tqdm progress bars may print garbled when interleaved with `conda run` stdout buffering. The `-u` flag is set to force unbuffered output. If bars still look ugly, redirect through `stdbuf -oL` in future runs.
4. **VoiceBench/src import**: `summarize.py` hardcodes `VOICEBENCH_SRC = "/mnt/efs/ramaneswaranselvakumar/projects/steerduplex/VoiceBench/src"`. If you ever move that repo, update the constant.

## 6. Next steps (once the sanity run confirms green)

1. Launch the real all-splits run across 8×H100 with `--instances_per_gpu 2`:
   ```bash
   python -m eval.voicebench.eval \
       --output_dir artifacts/evaluation_outputs/voicebench/moshiko \
       --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
       --instances_per_gpu 2
   ```
   Expected runtime: a few hours (largest splits are `wildvoice` 1000 and `bbh` 1000).
2. Compare `summary.md` against the [VoiceBench leaderboard](https://matthewcym.github.io/VoiceBench/) — numbers should be in the right ballpark for Moshi. Large deviations → inspect individual `output.json` entries.
3. If you want sd-qa across all 11 regions: `--sdqa_regions aus gbr ind_n ind_s irl kenya nga nzl phl usa zaf`.

## 7. Design decisions worth flagging

- **No `__init__.py`** in this module — matches `bigbench_audio/` convention.
- **Resumability** mirrors bigbench: re-running with the same `--output_dir` skips examples already in `output.json` / entries already scored in `llm_judge_output.json`.
- **Transcription is the canonical response** (not `moshi_text`). Matches bigbench convention and is what VoiceBench expects via the `response` field.
- **Audio pre-extraction** writes `<split>/_inputs/*.wav` once before spawning workers — workers then read from disk instead of receiving ndarrays via IPC. Safe to delete the `_inputs/` subdirs post-run to reclaim disk (~1-2 GB on a full run).
- **VoiceBench scorer imports over vendoring**: we add their `src/` to `sys.path` in `summarize.py` rather than copy-pasting evaluators. Keeps leaderboard parity if they update their scoring.
