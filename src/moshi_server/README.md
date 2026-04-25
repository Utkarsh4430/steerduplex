# moshi_server — shared Moshi WebSocket server

One source of truth for the full-duplex Moshi inference server used across
steerduplex evals (FDB v1, FDB v2, future consumers). Replaces the two
previously-duplicated copies under [src/eval/fdb_v1/](../eval/fdb_v1/) and
[src/eval/fdbv2/](../eval/fdbv2/) — those copies are left in place
untouched; new work should import from here.

## What's in this directory

| File | Role |
|---|---|
| [`moshi_server.py`](moshi_server.py) | Single-instance aiohttp WebSocket server. One GPU, one conversation at a time. Uses the steerduplex-ported [`inference.lm_gen.LMGen`](../inference/lm_gen.py) — persona-aware (text + voice prompt) via the 4-phase `step_system_prompts` injection. |
| [`launch_moshi_servers.py`](launch_moshi_servers.py) | Spawns N `moshi_server.py` instances across G GPUs (round-robin), health-checks each via WS handshake, writes URLs to JSON, stays alive to keep children running. |
| [`client_utils.py`](client_utils.py) | Tiny logging helper shared with the server. |
| [`run_moshi_servers.sh`](run_moshi_servers.sh) | Bash wrapper with a config block at the top — edit the fields and run to spawn the pool. |

The canonical [`inference.personaplex_loader`](../inference/personaplex_loader.py)
(weight patching for PersonaPlex `dep_q=8→16`) lives under `src/inference/` and
is imported directly — it is also used by
[`inference/generate.py`](../inference/generate.py).

---

## Quick start — single server

```bash
conda activate raman_steerduplex
export PYTHONPATH=/mnt/efs/ramaneswaranselvakumar/projects/steerduplex/steerduplex/src

# Moshiko (base model, no persona)
CUDA_VISIBLE_DEVICES=0 python -m moshi_server.moshi_server \
    --port 9100 \
    --hf-repo kyutai/moshiko-pytorch-bf16

# PersonaPlex with a voice prompt + teacher text prompt
CUDA_VISIBLE_DEVICES=0 python -m moshi_server.moshi_server \
    --port 9100 \
    --hf-repo nvidia/personaplex-7b-v1 \
    --voice-prompt /tmp/personaplex_voices/voices/NATM1.pt \
    --text-prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
```

Wait for the log line `loaders INFO: Access the Web UI directly at …`, then
connect via WebSocket at `ws://localhost:9100/api/chat`. The server sends
`b"\x00"` as a handshake, accepts `b"\x01" + opus_bytes` as user audio, and
emits `b"\x01" + opus_bytes` (audio) and `b"\x02" + utf8` (text tokens).

## Pool — many servers across many GPUs

[`launch_moshi_servers.py`](launch_moshi_servers.py) handles the pool:

```bash
conda activate raman_steerduplex
export PYTHONPATH=/mnt/efs/ramaneswaranselvakumar/projects/steerduplex/steerduplex/src

python -m moshi_server.launch_moshi_servers \
    --num-instances 8 \
    --gpu-ids 0,1,2,3 \
    --base-port 9100 \
    --hf-repo nvidia/personaplex-7b-v1 \
    --voice-prompt /tmp/personaplex_voices/voices/NATM1.pt \
    --text-prompt "You are a wise and friendly teacher..." \
    --output-json /tmp/moshi_servers.json
```

- Instances are distributed round-robin across `--gpu-ids`. `--num-instances 8`
  with `--gpu-ids 0,1,2,3` yields 2 instances per GPU.
- Per-instance logs go to `logs/moshi_servers/moshi_<port>.log`.
- `--output-json` writes `{"servers": ["ws://localhost:9100/api/chat", …]}`
  once every instance has returned the `0x00` handshake.
- The launcher stays in the foreground holding the child processes. `Ctrl+C`
  terminates all of them.

All `moshi_server.py` flags with a `help="…(forwarded to moshi_server.py)"`
string are pass-through — `--hf-repo`, `--moshi-weight`, `--mimi-weight`,
`--tokenizer`, `--cfg-coef`, `--half`, `--text-prompt`, `--voice-prompt`.

## `run_moshi_servers.sh` — wrapped launch

Edit the `MANDATORY` block at the top of the script, then run it:

```bash
bash src/moshi_server/run_moshi_servers.sh
```

It activates the conda env and calls `launch_moshi_servers.py` with the
config block values. Stays in the foreground until `Ctrl+C`. See the script
for the full list of tunable vars.

---

## Default system prompts by model

The server's `--text-prompt` (wrapped with `<system> … <system>`) is injected
once per connection via the 4-phase `step_system_prompts` pattern
(voice → silence → text → silence). **Whether the model actually conditions
on the prompt depends on whether it was trained for that.**

### `kyutai/moshiko-pytorch-bf16` (base Moshi 7B)
- **Trained with system prompts?** No. Passing `--text-prompt` is a no-op in
  practice — the base model was never exposed to `<system>` tokens during
  training. Same for `--voice-prompt`.
- **Default to use**: `""` (empty).
- **Notes**: CFG support (`--cfg-coef`) exists in the upstream LMGen; our
  custom LMGen ignores it and logs a warning if non-default.

### `kyutai/moshi-2-*` (base Moshi-2)
- **Trained with system prompts?** No (same as moshiko). Persona args are a
  no-op.
- **Default to use**: `""`.
- **Notes**: moshi-2 checkpoints use conditioning + CFG for control. Our
  custom LMGen doesn't implement those — if you need them, use the upstream
  `moshi.models.LMGen` path (not in this server).

### `nvidia/personaplex-7b-v1` (PersonaPlex)
- **Trained with system prompts?** Yes — [`<system> … <system>`-wrapped text
  prompts and voice prompts are core to the training data](../../../personaplex/moshi/moshi/offline.py).
- **Built-in default** (from upstream `personaplex/offline.py`):
  > `"You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."`
- **Other in-distribution templates**: customer-service roles, casual
  conversation roles, named personas. Anything stylistically similar to the
  training data above will work.
- **Voice prompts**: use `.pt` files from
  [`nvidia/personaplex-7b-v1` voices.tgz](https://huggingface.co/nvidia/personaplex-7b-v1).
  18 pre-packaged voices: `NATF0..NATF3`, `NATM0..NATM3`, `VARF0..VARF4`,
  `VARM0..VARM4`. The first `hf_hub_download` call fetches and caches
  them locally; extract once via `tar -xzf voices.tgz`.

### Steerduplex-trained checkpoints (moshiko base + steerduplex training)
- **Trained with system prompts?** Yes — on a mix of
  [`NATURAL_PROMPTS`](../pipeline/add_prompts_external.py) (for Fisher /
  Annutacon natural-conversation corpora) plus LLM-generated / LLM-rephrased
  prompts (for the synthetic A-series and B-series).
- **No single built-in default.** Pick one of the eight natural prompts
  below, or craft a custom prompt consistent with the training distribution.
- **Natural prompt pool** (randomly sampled per conversation during
  training on Fisher/Annutacon):
  1. "You are a friendly, natural conversationalist. Speak naturally and engage with what the other person says."
  2. "You are having a casual conversation. Be yourself, be natural, and respond to what you hear."
  3. "You are a warm and attentive conversation partner. Listen actively and respond naturally."
  4. "You are chatting naturally with someone. Keep the conversation flowing and be genuine."
  5. "You are an easygoing conversationalist. Respond naturally, be yourself, and keep things friendly."
  6. "You enjoy having a good conversation. Listen well, respond thoughtfully, and be natural."
  7. "You are having an everyday conversation. Be relaxed, genuine, and engaged."
  8. "You are a good listener who responds naturally. Keep the conversation comfortable and flowing."
- **Voice prompts**: matched to the training distribution under
  [`src/data/voices/`](../data/voices/) — see the presets/pool files there.

---

## Argument reference

See `python -m moshi_server.moshi_server --help` for the full list. Highlights:

| Flag | Purpose |
|---|---|
| `--hf-repo` | HF repo ID (default `kyutai/moshiko-pytorch-bf16`). |
| `--moshi-weight` | Path to a local checkpoint (dir or single file). |
| `--text-prompt` | System-prompt text, `<system>`-wrapped and injected each connection. |
| `--voice-prompt` | WAV or `.pt` voice prompt file. `.pt` files must contain both `embeddings` and `cache` keys (PersonaPlex format); pure WAVs are encoded on load with −24 LUFS normalization. |
| `--cfg-coef` | Accepted for back-compat, **ignored** (our LMGen doesn't implement CFG). |
| `--half` | float16 instead of bfloat16. |
| `--port` | WebSocket port (default 8998). |
| `--static none` | Disable serving the dist/ static UI (useful for headless eval pools). |

---

## Testing

```bash
conda activate raman_steerduplex

# 1. Quick smoke — base moshiko, no persona
CUDA_VISIBLE_DEVICES=0 python -m moshi_server.moshi_server \
    --port 9100 --hf-repo kyutai/moshiko-pytorch-bf16 --static none &
# wait for handshake, then POST a WAV via any client that speaks the
# b"\x01" + opus_bytes wire protocol. fdb_v1's moshi_client.MoshiFileClient
# works as-is.

# 2. PersonaPlex persona — should produce voice-conditioned, in-distribution
#    output. Expect ~3-5s startup (model + warmup).
CUDA_VISIBLE_DEVICES=0 python -m moshi_server.moshi_server \
    --port 9100 --hf-repo nvidia/personaplex-7b-v1 --static none \
    --voice-prompt /tmp/personaplex_voices/voices/NATM1.pt \
    --text-prompt "You are a wise and friendly teacher..."
```

Server log shows `text prompt set (N tokens)` and
`voice prompt loaded from embeddings NATM1.pt` at startup when persona args
are supplied — absence of those lines means the args didn't reach the
`ServerState` constructor.

---

## Relationship to the old eval/ copies

[`src/eval/fdb_v1/moshi_server.py`](../eval/fdb_v1/moshi_server.py),
[`src/eval/fdb_v1/launch_moshi_servers.py`](../eval/fdb_v1/launch_moshi_servers.py),
[`src/eval/fdbv2/moshi_server.py`](../eval/fdbv2/moshi_server.py), and
[`src/eval/fdbv2/launch_moshi_servers.py`](../eval/fdbv2/launch_moshi_servers.py)
are byte-identical to the files in this directory (post the persona fix).
They remain in place for now; new code should invoke
`-m moshi_server.moshi_server` and `-m moshi_server.launch_moshi_servers`
instead.
