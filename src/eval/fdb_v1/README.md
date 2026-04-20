# Full Duplex Bench v1.0 / v1.5 integration

Runs steerduplex Moshi-based models against [Full Duplex Bench](https://github.com/DanielLin94144/Full-Duplex-Bench) and produces the full set of v1.0 / v1.5 metrics for comparison against published numbers.

This pipeline is **self-contained**: the FDB scoring scripts live vendored under [`scoring/`](scoring/), the v1.5 dataset zips ship under [`data/v1.5/`](data/) (gitignored), and Moshi runs through the same WebSocket server stack that the official baseline uses. No external `Full-Duplex-Bench/` checkout required at runtime.

## What this evaluates

**FDB v1.0** — four turn-taking tasks: `pause_handling`, `backchannel`, `smooth_turn_taking`, `user_interruption`. Metrics: Turn-Over Rate (TOR), response latency, backchannel frequency / Jensen-Shannon divergence, and a GPT-4-turbo contextual-relevance judge for interruptions.

**FDB v1.5** — overlap robustness with paired inputs (`input.wav` + `clean_input.wav`). Four tasks: `user_interruption`, `user_backchannel`, `talking_to_other`, `background_speech`. Metrics: behavior categorization (`C_RESPOND` / `C_RESUME` / `C_UNCERTAIN_HANDLING` / `C_UNKNOWN`) via GPT-4o, stop/response latencies via Silero VAD, and speech-adaptation metrics (UTMOSv2, WPM, pitch, intensity) comparing pre-overlap vs. post-overlap segments — with paired t-tests for significance.

## Workflow

```
[ launched separately ]
   moshi_server.py × N   (one Moshi WebSocket session per process; spawn via launch_moshi_servers.py)

FDB dataset/{task}/{id}/input.wav [+ clean_input.wav + annotations]
     │  build_mirror()  (symlinks; v1.5 input.json carried through)
     ▼
$OUT_ROOT/{task}/{id}/
     │  run_inference.py  (async client → already-running Moshi server URLs)
     ▼
     + output.wav [+ clean_output.wav]      (24 kHz PCM_16, len(input) samples, SKIP_FRAMES pad)
     │  run_asr.py  (Parakeet-TDT-0.6B-v2 by default; WhisperX optional)
     ▼
     + output.json [+ clean_output.json]    (input.json / clean_input.json ship with v1.5 data)
     │  run_eval.py  (subprocess → vendored scoring/ scripts)
     ▼
     + rating.json / content_tag.json / latency_intervals.json / general_split.json
$OUT_ROOT/
     + run_meta.json, logs/*.log, summary.json, summary.md
```

## How to run

### Dataset layout

- **v1.0** (at `DATASET_ROOT/`): `candor_pause_handling/`, `synthetic_pause_handling/`, `candor_turn_taking/`, `icc_backchannel/`, `synthetic_user_interruption/`. Multi-source tasks (`pause_handling`) are unioned automatically with `candor_…` / `synthetic_…` ID prefixes; single-source tasks keep the original IDs.
- **v1.5** ships as zips vendored under [`data/v1.5/`](data/) (`*.zip`). [`run_fdb_eval.sh`](run_fdb_eval.sh) extracts them on first run; subsequent runs reuse the extracted folders.

### Step 1: Launch Moshi servers (separate process)

Server lifecycle is decoupled from the eval pipeline so a single server pool can be reused across many evaluations and so server crashes don't take the eval down with them.

```bash
conda run --no-capture-output -n raman_steerduplex python \
    -m eval.fdb_v1.launch_moshi_servers \
    --num-instances 10 \
    --gpu-ids 0,1,2,3 \
    --base-port 9001 \
    --hf-repo kyutai/moshiko-pytorch-bf16
```

This spawns 10 `moshi_server.py` instances on ports 9001–9010, round-robin across GPUs 0–3, and stays foregrounded until you Ctrl-C. Add `--moshi-weight /path/to/checkpoint` for a fine-tuned weight, `--cfg-coef X` for CFG. Use a separate terminal / tmux pane.

### Step 2: Run the evaluation

In another terminal, edit the MANDATORY block at the top of [`run_fdb_eval.sh`](run_fdb_eval.sh) — set `SERVER_PORTS` to the launched range, plus `MODEL_NAME`, `OUT_ROOT`, and the LiteLLM credentials — then:

```bash
bash src/eval/fdb_v1/run_fdb_eval.sh
```

`SERVER_PORTS` accepts:

| Form | Example | Expands to |
|------|---------|------------|
| Port range | `9001-9010` | 10 URLs `ws://${SERVER_HOST}:9001…9010/api/chat` |
| Single port | `9001` | one URL on `${SERVER_HOST}` |
| Comma list | `9001,9003,9005-9007` | mix of singles + ranges |
| Full URL  | `ws://gpu-host:9001/api/chat` | passes through unchanged (mix with bare ports) |

`SERVER_HOST` defaults to `localhost`; override when servers run on another machine.

All other config lives in the OPTIONAL block. Common knobs:

| Variable | Default | Purpose |
|----------|---------|---------|
| `HF_REPO` | `kyutai/moshiko-pytorch-bf16` | Recorded in `run_meta.json` for provenance (the live config is whatever you launched the servers with) |
| `CHECKPOINT` | _(empty)_ | Recorded in `run_meta.json` (server-side flag is `--moshi-weight`) |
| `CFG_COEF` | _(empty)_ | Recorded in `run_meta.json` (server-side flag is `--cfg-coef`) |
| `VERSION` | `both` | `1.0`, `1.5`, or `both` |
| `TASKS` | _(empty = all)_ | Space-separated subset, e.g. `"backchannel pause_handling"`. `"pause_handling"` expands to two evals (Synthetic + Candor) so the headline table reports each subset separately, matching the published v1.0 table; you can also pass `pause_handling_synthetic` or `pause_handling_candor` directly. |
| `STAGES` | `inference asr eval` | Any subset to re-run a partial pipeline |
| `ASR_BACKEND` | `parakeet` | `parakeet` (matches official) or `whisperx` (fallback) |
| `ASR_DEVICE` | `cuda:0` | GPU for ASR (single-GPU stage) |
| `DATASET_ROOT` | _(EFS path)_ | Root of v1.0 source dirs (v1.5 lives in `data/v1.5/` regardless) |
| `STEER_ENV` / `FDB_ENV` / `ASR_ENV` | conda env names | Inference / scoring / WhisperX-fallback envs |

Resuming: re-run the launcher — every stage skips work already on disk. Set `OVERWRITE_INFERENCE=1` or `OVERWRITE_ASR=1` (or pass `--overwrite` to a python module directly) to force regeneration.

## File-by-file roles

- [`dataset_utils.py`](dataset_utils.py) — sample discovery + mirror-tree construction via symlinks. Defines `TASKS_V1` / `TASKS_V15` registries. Single-source tasks keep raw sample IDs (so `eval_backchannel.py`'s `spk.isdigit()` filter works against the ICC ground-truth distribution). v1.5 mirror also carries through the dataset's pre-shipped `input.json` / `clean_input.json` (Parakeet-aligned).
- [`moshi_server.py`](moshi_server.py) — vendored Kyutai Moshi WebSocket server (one streaming session per process). Loads the model, exposes `/api/chat` for opus-framed audio.
- [`launch_moshi_servers.py`](launch_moshi_servers.py) — spawns N `moshi_server.py` instances across the requested GPU IDs, health-checks each, writes the URL list to `--output-json`.
- [`moshi_client.py`](moshi_client.py) — port of upstream `model_inference/moshi/inference.py`. Reads a WAV, streams 1920-sample @ 24 kHz frames, writes `output.wav` clipped to `len(input)` with the `SKIP_FRAMES` zero-pad at the start (parity with the published baseline).
- [`run_inference.py`](run_inference.py) — async driver. Reads server URLs from `--servers_json`, mirrors the dataset, distributes pending samples across servers (one in-flight stream per URL), records `run_meta.json`.
- [`run_asr.py`](run_asr.py) — Parakeet-TDT-0.6B-v2 (default, mirrors official) or WhisperX large-v3 (fallback). Writes the JSON shape `{"text": ..., "chunks": [{"text": word, "timestamp": [s, e]}]}` the FDB scoring scripts expect. Crops before `interrupt.json` end for v1.0 `user_interruption`. v1.5 transcribes only `output.wav` / `clean_output.wav`; user-side transcripts ship with the dataset.
- [`run_eval.py`](run_eval.py) — subprocess orchestrator. `cwd`s to [`scoring/`](scoring/) (required for relative imports + `./instruction/*.txt` access), shells out to `evaluate.py`, `get_timing.py`, `significance_test.py`. Captures per-task logs into `$OUT_ROOT/logs/` and aggregates into `summary.md` + `summary.json`.
- [`scoring/`](scoring/) — vendored upstream FDB evaluation scripts (do not edit).
- [`run_fdb_eval.sh`](run_fdb_eval.sh) — top-level launcher. Sequences zip-extraction → server-launch → inference → ASR → scoring.

## How to interpret results

Headline numbers land in `$OUT_ROOT/summary.md`. Per-metric guide:

- **TOR (Turn-Over Rate)**: fraction of samples where the model takes a conversational turn.
  - *Lower is better* for `pause_handling` (model should stay quiet during user pauses) and `backchannel` (model should produce listener cues, not full turns).
  - *Higher is better* for `smooth_turn_taking` and `user_interruption` (model should take turn when user yields).
- **Response latency (seconds)**: time between user yielding and model starting to speak. Lower = more natural. `<500 ms` feels human-like, `>1.5 s` feels laggy.
- **Backchannel frequency / JSD**: Jensen-Shannon divergence between the model's backchannel distribution and ground-truth (`scoring/icc_gt_distribution.json`). Lower JSD = closer to human.
- **GPT-4-turbo rating (v1.0 `user_interruption`)**: 0–5 contextual relevance. ≥3 = acceptable, 5 = explicitly addresses the interruption.
- **Behavior tags (v1.5, `content_tag.json`)**: which of `C_RESPOND` / `C_RESUME` / `C_UNCERTAIN_HANDLING` / `C_UNKNOWN` the model produced. The "correct" tag depends on the task:
  - `user_interruption` → want `C_RESPOND`
  - `background_speech`, `talking_to_other` → want `C_RESUME`
  - `user_backchannel` → typically want `C_RESUME` (don't be derailed by listener noise)
- **Stop latency (v1.5)**: time between overlap onset and model going silent. Relevant when the model should yield (user_interruption).
- **Resume latency (v1.5)**: time between overlap end and model resuming speech.
- **UTMOSv2 (v1.5)**: predicted MOS (1–5) on model speech quality. Pre- vs. post-overlap drops indicate disfluent/jittery recovery.
- **WPM / pitch / intensity (v1.5)**: speaking-rate and prosody before vs. after overlap. `pair_t_{task}.txt` contains paired t-test p-values — `p < 0.05` flags that the overlap measurably changed the model's speech statistics (for samples tagged `C_RESPOND`).

## Environments

Three conda envs:

1. **`raman_steerduplex`** — runs `moshi_server.py`, `launch_moshi_servers.py`, `run_inference.py`, and the `run_eval.py` orchestrator. Already has the moshi stack; the server-based pipeline additionally needs `websockets` and a working `torchaudio`. If your env lacks them:
   ```bash
   conda run -n raman_steerduplex pip install websockets
   # If `import torchaudio.functional` raises an OSError on _torchaudio.abi3.so
   # (mismatch with the installed torch), realign:
   conda run -n raman_steerduplex pip install --force-reinstall --no-deps \
       torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
2. **`raman_fdb_v1`** — runs the vendored FDB scoring scripts via `$FDB_PY` and (by default) Parakeet ASR via `nemo_toolkit[asr]`. Set up via [`setup_env.sh`](setup_env.sh).
3. **`raman_whisperx`** — only needed if you flip `ASR_BACKEND=whisperx`. Has `whisperx` + `faster-whisper`.

Setup once:
```bash
bash src/eval/fdb_v1/setup_env.sh
```

## Known caveats

- **Inference parity** with the published Moshi numbers requires running the WebSocket server stack used here (the `SKIP_FRAMES` start-pad and `len(input)` output clip are encoded in `moshi_client.py`). Output durations match input durations within ~1 frame (~80 ms).
- **ASR parity**: Parakeet-TDT-0.6B-v2 is the default and matches `Full-Duplex-Bench/v1_v1.5/get_transcript/asr.py`. Switching to WhisperX shifts every timestamp-driven metric (TOR cutoffs, latency, JSD, WPM, behavior judge inputs) — keep WhisperX as a fallback only.
- **FDB upstream bug** (no action needed): `scoring/eval_user_interruption.py` appends each rating twice. `sum/len` is invariant so the reported "Average rating" is correct.
- **Vendored scoring scripts write log files to cwd**: `run_eval.py` `cwd`s to `scoring/` and captures `{dir}_behavior.log`, `{dir}_general.log`, `pair_t_{dir}.txt` into `$OUT_ROOT/logs/`. Two concurrent runs in the same checkout can race.
- **GPT cost**: v1.0 `user_interruption` = 1 call/sample × N samples; v1.5 `behavior` = 1 call/sample × 4 tasks × N samples. Budget via LiteLLM dashboard.
