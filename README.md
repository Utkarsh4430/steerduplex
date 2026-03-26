# SteerDuplex

Steerable full-duplex speech-to-speech model based on Moshi (7B), with dynamic mid-conversation steering (user says "speak slower" and the model adapts). Competes with PersonaPlex (NVIDIA, ICASSP 2026).

## Setup

```bash
conda activate steerduplex
cd src
```

Requires `moshi` and `moshi-finetune` installed via pip (not vendored).

## Data Pipeline

Five-phase pipeline: transcripts -> voice assignment -> TTS -> channel assembly -> formatting.

```bash
# Pilot scale (~620 conversations)
python -m run_pipeline --scale pilot

# Full scale (~132K conversations, multi-node)
python -m run_pipeline --scale full
```

## Training

Full finetuning (PersonaPlex methodology) on 8x H100 GPUs:

```bash
cd src

# Full training (~4,702h data, 15 epochs, ~16K steps)
bash training/launch.sh configs/full_training.yaml 8

# Single GPU (testing)
bash training/launch.sh configs/full_training.yaml 1

# Resume from checkpoint (set moshi_paths.moshi_path in config first)
bash training/launch.sh configs/full_training.yaml 8
```

Key hyperparameters (PersonaPlex-aligned):
- Temporal transformer LR: 2e-6, Depth transformer LR: 4e-6
- Linear warmup (1000 steps) + cosine annealing
- System prompt loss masking
- All checkpoints saved (every 1000 steps) for post-hoc benchmark selection

## Evaluation

### FD-Bench v1.0 Checkpoint Watcher (run during training)

Automatically evaluates new checkpoints on Full-Duplex-Bench v1.0 and logs to wandb:

```bash
cd src

# Run alongside training in a separate terminal/tmux pane
python -m eval.checkpoint_watcher \
    --run_dir runs/<RUN_NAME> \
    --data_dir data/benchmarks/fd_bench_v1 \
    --device cuda:0 \
    --max_samples 50
```

### Standalone Checkpoint Evaluation

```bash
cd src

# Evaluate a specific checkpoint
python -m eval.fd_bench_v1 \
    --checkpoint runs/<RUN_NAME>/checkpoints/step_XXXX/consolidated \
    --data_dir data/benchmarks/fd_bench_v1 \
    --max_samples 50
```

### Interactive Chat

```bash
cd src

# Base Moshi model
python -m inference.chat

# With finetuned checkpoint
python -m inference.chat --checkpoint runs/<RUN_NAME>/checkpoints/step_XXXX

# Fallback mode (upload audio)
python -m inference.chat --fallback --checkpoint runs/<RUN_NAME>/checkpoints/step_XXXX
```

## Checkpoint Selection Strategy

Following PersonaPlex and Moshi, we do NOT use eval loss to select checkpoints. Both papers trained for a fixed number of steps and evaluated post-hoc with downstream benchmarks.

Our approach:
1. Save ALL checkpoints during training (every 1000 steps)
2. FD-Bench watcher runs concurrently, evaluating each checkpoint on duplex metrics (TOR, latency, backchannel frequency)
3. Select the checkpoint with best FD-Bench aggregate score
4. Run full Tier 1+2 benchmark suite on the selected checkpoint

## Project Structure

```
src/
  configs/           Training configs (full_training.yaml, pilot_training.yaml)
  pipeline/          Data generation pipeline (5 phases)
  training/          Training loop, loss masking, launch script
  eval/              FD-Bench evaluator + checkpoint watcher
  inference/         Offline generation + interactive chat
  data/              Generated data (gitignored)
  run_pipeline.py    End-to-end orchestrator
docs/
  beyond_steerability_plan.md   Execution plan with benchmarks + B-series categories
data_categories/    YAML definitions for B-series data categories
```

## Datasets

| Source | Conversations | Hours |
|--------|--------------|-------|
| A-Series (Steerability) | 66,000 | ~1,000h |
| B-Series (Capabilities, v4) | 53,600 | ~1,230h |
| Annutacon (Natural) | 7,503 | ~1,497h |
| Fisher (Telephone) | 5,849 | ~975h |
| **Total** | **~132,952** | **~4,702h** |
