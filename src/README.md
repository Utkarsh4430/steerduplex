# SteerDuplex

Steerable full-duplex speech-to-speech model based on Moshi 7B. Supports dynamic, user-directed mid-conversation steering with compositional attribute control and principled boundaries.

## Quick Start

```bash
# 1. Create environment
bash setup_env.sh
conda activate steerduplex

# 2. Prepare voice pool (LibriSpeech + Common Voice)
python -m pipeline.prepare_voicepool \
    --add_common_voice data/voicepool/common_voice_audios.tar.gz

# 3. Set your API key
export OPENAI_API_KEY="your-api-key"

# 4. Generate synthetic data (run on N nodes simultaneously)
python run_pipeline.py --scale full

# 5. Import external datasets
python -m pipeline.import_external --dataset annutacon
python -m pipeline.import_external --dataset fisher

# 6. Merge all datasets for training
python -m pipeline.merge_manifests --config configs/full_training.yaml

# 7. Train
bash training/launch.sh configs/full_training.yaml 8
```

## Datasets

Three data sources, toggleable via `configs/full_training.yaml`:

| Dataset | Type | Est. Hours | Description |
|---------|------|-----------|-------------|
| **synthetic** | Generated | ~1,000h | 66K conversations with steerable styles (A1-A10) |
| **annutacon** | Natural | ~1,497h | Scale natural conversation data |
| **fisher** | Natural | TBD | Fisher telephone conversations |

### Managing datasets

```bash
# Check status of all datasets
python -m pipeline.import_external --list

# Import with hour cap
python -m pipeline.import_external --dataset annutacon --max_hours 500

# Preview merge without writing
python -m pipeline.merge_manifests --config configs/full_training.yaml --stats_only

# Full merge
python -m pipeline.merge_manifests --config configs/full_training.yaml
```

Toggle datasets in `configs/full_training.yaml`:
```yaml
datasets:
  synthetic:
    enabled: true
    max_hours: "all"
  annutacon:
    enabled: true
    max_hours: 500  # cap at 500 hours
  fisher:
    enabled: false  # skip for this run
```

## Multi-Node Pipeline

All pipeline stages use atomic file claiming on shared EFS — safe to run on N nodes:

```bash
# Same command on every node:
python run_pipeline.py --scale full
```

- GPU workers auto-planned based on available VRAM (skips busy GPUs)
- Stale claims expire after 2 hours
- Every stage is resumable — restart anytime

## Data Generation Pipeline

| Phase | Module | Description |
|-------|--------|-------------|
| 1 | `generate_transcripts` | LLM transcript generation (64 parallel workers) |
| 2 | `assign_voices` | Map speakers to voice pool |
| 3 | `synthesize_tts` | Qwen3-TTS synthesis (multi-GPU, memory-aware) |
| QF | `quality_filter` | Whisper WER + duration checks (multi-GPU) |
| 4 | `assemble_channels` | Stereo assembly with role-aware gaps |
| 5 | `format_dataset` | Manifest + Whisper annotation |

### TTS setup

| Role | Model | Voice | Style Control |
|------|-------|-------|---------------|
| Assistant | Qwen3-TTS CustomVoice | Preset (Ryan/Vivian/Dylan/Serena) | `instruct` param |
| User | Qwen3-TTS Base | Voice cloning from ref audio | None |

### Channel assembly

- **User→Assistant gap**: 100-400ms (trains fast response)
- **Assistant→User gap**: 300-1200ms (natural think time)
- **Barge-in**: 5% probability, assistant fades out in 150ms
- **User silence**: 8% probability, 2-6s pause (trains patience)

## Training

```bash
# Pilot (synthetic only)
bash training/launch.sh configs/pilot_training.yaml 8

# Full (all datasets)
python -m pipeline.merge_manifests --config configs/full_training.yaml
bash training/launch.sh configs/full_training.yaml 8
```

Follows PersonaPlex guidelines:
- System prompt loss masking
- Non-semantic audio tokens downweighted 0.02x
- Padded text tokens downweighted 0.3x
- LoRA rank 128, bf16, gradient checkpointing

## Project Structure

```
src/
├── configs/
│   ├── generation.yaml          # Data pipeline config
│   ├── pilot_training.yaml      # Pilot training (synthetic only)
│   └── full_training.yaml       # Full training (all datasets, toggleable)
├── data/
│   ├── voices/assistant/        # Assistant voice presets
│   ├── voices/user/             # User voice pool (22K+ voices)
│   ├── voicepool/               # Raw voice reference audio
│   ├── external/                # External datasets (annutacon, fisher)
│   ├── transcripts/ → tts_audio/ → assembled/ → formatted/
│   └── tts_comparison/          # TTS engine comparison samples
├── pipeline/
│   ├── generate_transcripts.py  # Phase 1: LLM transcripts
│   ├── assign_voices.py         # Phase 2: Voice mapping
│   ├── synthesize_tts.py        # Phase 3: TTS synthesis
│   ├── quality_filter.py        # Quality checks (Whisper WER)
│   ├── assemble_channels.py     # Phase 4: Stereo assembly
│   ├── format_dataset.py        # Phase 5: Training format
│   ├── import_external.py       # External dataset import
│   ├── merge_manifests.py       # Combine datasets for training
│   ├── prepare_voicepool.py     # Voice pool preparation
│   └── distributed.py           # Multi-node coordination
├── training/
│   ├── train.py                 # Training with loss masking
│   ├── launch.sh                # torchrun launcher
│   └── annotate.py              # Whisper annotation
├── inference/
│   └── generate.py              # Offline inference
└── run_pipeline.py              # End-to-end orchestrator
```

Training (Terminal 1 — 8 GPUs)

cd /mnt/efs/utkarshtyagi/personal_projects/steerduplex/src
bash training/launch.sh configs/full_training.yaml 8

FD-Bench Checkpoint Watcher (Terminal 2 — 1 GPU)

cd /mnt/efs/utkarshtyagi/personal_projects/steerduplex/src

# Start after training begins and run_dir is created
python -m eval.checkpoint_watcher \
    --run_dir runs/full_v3_20260322_223728 \
    --data_dir data/benchmarks/fd_bench_v1 \
    --device cuda:1 \
    --max_samples 0 \
    --wandb_project steerduplex

Note: Replace <RUN_NAME_TIMESTAMP> with the actual directory name printed by training (e.g., full_v3_20260322_143000).

Post-Training: Evaluate Best Checkpoint

# After training, pick the checkpoint with the best FD-Bench aggregate from wandb
python -m eval.fd_bench_v1 \
    --checkpoint runs/<RUN_NAME>/checkpoints/checkpoint_XXXXXX/consolidated \
    --data_dir data/benchmarks/fd_bench_v1 \
    --device cuda:0

# Interactive chat with best checkpoint
python -m inference.chat \
    --checkpoint runs/<RUN_NAME>/checkpoints/checkpoint_XXXXXX

    full_v3_20260322_223728


python -m eval.checkpoint_watcher \
      --run_dir runs/full_v3_20260322_223728 \
      --data_dir data/benchmarks fd_bench_v1\                                                                                
      --device cuda:0 \
      --max_samples 0     ,