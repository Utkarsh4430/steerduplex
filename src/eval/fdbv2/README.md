# FDB v2 Revamped

## Stage 1 — Conversation Generation

Runs GPT Realtime (Examiner) against Gemini Live (Examinee) across benchmark tasks in parallel, recording each conversation to audio.

### 1. Install dependencies

```bash
npm install
pip install rich pyyaml python-dotenv
```

### 2. Configure

Edit [`configs/default.yaml`](configs/default.yaml) for your experiment:

```yaml
parallel_sessions: 4     # how many conversations run at once
num_samples: 10          # 10 for a sanity run, -1 for all tasks
examiner_vad_mode: slow  # slow (patient) or fast (strict, shorter silences)
litellm:
  base_url: http://localhost:4000
```

### 3. Set your API key

```bash
cp .env.example .env
# then set LITELLM_API_KEY in .env
```

### 4. Run

```bash
python run_sessions.py
# or point to a different config:
python run_sessions.py --config configs/my_experiment.yaml
```

Outputs land in `outputs/default/{split}/{task_id}/` with `A.wav`, `B.wav`, `combined.wav`, and `cost_A.json` (GPT token usage + estimated USD cost) per session.

---

## Running with Moshi / PersonaPlex (self-hosted examinee)

Moshi runs as a local WebSocket server. Because each server instance handles **one conversation at a time**, you need as many instances as you want parallel sessions. `launch_moshi_servers.py` handles this automatically, spreading instances round-robin across your GPUs.

### 1. Launch the Moshi server pool

```bash
# Example: 6 instances on GPUs 0, 1, 2 (2 per GPU), starting at port 9100
python launch_moshi_servers.py \
    --num-instances 8 \
    --gpu-ids 4,5,6,7 \
    --base-port 9100 \
    --hf-repo kyutai/moshiko-pytorch-bf16
```

The script prints a `✓` for each server once the model has loaded and is ready to accept connections (this can take a few minutes). **Keep this terminal open** — it holds the server processes alive.

Optional flags forwarded to `moshi_server.py`:

| Flag | Purpose |
|------|---------|
| `--moshi-weight /path/to/ckpt` | Load weights from a local checkpoint instead of HuggingFace |
| `--mimi-weight /path/to/ckpt` | Override Mimi codec weights |
| `--cfg-coef 1.0` | CFG coefficient |
| `--half` | Use float16 instead of bfloat16 |
| `--output-json servers.json` | Write the server URL list to a JSON file |

To use a **PersonaPlex** model, pass its HuggingFace repo via `--hf-repo` (the loader detects it automatically).

### 2. Configure the run

Edit [`configs/moshi.yaml`](configs/moshi.yaml) — particularly `parallel_sessions` (should be ≤ number of server instances) and `pool.urls` (must match the ports used above).

```yaml
parallel_sessions: 6          # one worker per server instance

examinee_adapter:
  script: adapters/moshi_adapter.js
  args: {}
  pool:
    urls:
      - ws://localhost:9100/api/chat
      - ws://localhost:9101/api/chat
      # … one entry per launched instance
    url_arg: moshiUrl
```

### 3. Run

```bash
# In a second terminal
python run_sessions.py --config configs/moshi.yaml
```

Workers will acquire a server URL from the pool before each conversation and release it when done, so no two conversations hit the same server instance simultaneously.

---

## Adding other self-hosted adapters (FreezeOmni, etc.)

The examinee adapter is fully config-driven. To use any other self-hosted model:

1. Copy or write an adapter JS file into `adapters/`
2. Add an `examinee_adapter` block to your YAML config with `pool.urls` pointing to your server instances and `pool.url_arg` set to the CLI flag the adapter uses for its server URL
