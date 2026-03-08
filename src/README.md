# SteerDuplex: Data Generation & Training Pipeline

End-to-end pipeline for generating steerable full-duplex speech training data and fine-tuning Moshi 7B.

## Quick Start

```bash
# 1. Setup environment
bash setup_env.sh
conda activate steerduplex

# 2. Populate voice database (see Voice Database section below)

# 3. Start your LLM endpoint (vLLM example)
vllm serve Qwen/Qwen3-32B --port 8000

# 4. Run full pipeline
python run_pipeline.py --config configs/generation.yaml

# 5. Train
bash training/launch.sh configs/pilot_training.yaml 8
```

## Pipeline Phases

| Phase | Script | Input | Output |
|-------|--------|-------|--------|
| 1. Transcripts | `pipeline/generate_transcripts.py` | data_categories/*.yaml | data/transcripts/{cat}/*.json |
| 2. Voice Assignment | `pipeline/assign_voices.py` | transcripts + voice DB | data/voice_assignments/{cat}/*.json |
| 3. TTS Synthesis | `pipeline/synthesize_tts.py` | assignments | data/tts_audio/{cat}/{conv}/turn_*.wav |
| (opt) Quality Filter | `pipeline/quality_filter.py` | TTS audio | adds quality_passed to synth JSONs |
| 4. Channel Assembly | `pipeline/assemble_channels.py` | TTS audio | data/assembled/{conv}.wav + _meta.json |
| 5. Format Dataset | `pipeline/format_dataset.py` | assembled WAVs | data/formatted/manifest_{train,eval}.jsonl |

### Run specific phases
```bash
python run_pipeline.py --phase 1                          # transcripts only
python run_pipeline.py --from_phase 3 --to_phase 4       # TTS + assembly
python run_pipeline.py --category A9_dynamic_steering     # single category
python run_pipeline.py --skip_quality                     # skip quality filter
```

### Resumability
All phases are resumable. Rerunning skips already-completed work. If the process crashes mid-way, just rerun the same command.

## Data Categories

| Category | # Pilot | Description |
|----------|---------|-------------|
| A1: Customer Service | 60 | 20 domains × 10 scenarios |
| A2: QA/Assistant | 60 | Knowledge QA, tutoring, reasoning |
| A3: Tone-Controlled | 60 | 33 tones (user requests tone at start) |
| A4: Persona-Controlled | 60 | 70+ personas |
| A5: Style & Accent | 40 | 15 styles + 15 accents |
| A6: Speed & Length | 40 | 5×5 speed/length matrix |
| A7: Emotional Adaptation | 40 | User emotion detection + adaptation |
| A8: Edge Cases | 20 | Spelling, sarcasm, interruptions |
| A9: Dynamic Steering | 200 | **Core novelty** — mid-conversation steering |
| A10: Graceful Failure | 40 | Clarification, refusal, repair |

## Data Types

Each conversation is one of 5 types:

| Type | % | Description |
|------|---|-------------|
| Standard | 50% | User sets style at start, model maintains |
| Dynamic | 25% | User changes style mid-conversation |
| Counterfactual | 10% | Same text, different style (paired) |
| Long-form | 10% | 10-15 turns, multi-turn coherence |
| Graceful failure | 5% | Misunderstanding, refusal, repair |

## Voice Database

### Assistant voices (Qwen3-TTS CustomVoice presets)

Defined in `data/voices/assistant/presets.yaml`:
- 2 male (Ryan, Dylan) + 2 female (Vivian, Serena)
- Style controlled per-turn via `instruct` parameter
- Voice identity is fixed per conversation

### User voices (Qwen3-TTS Base model, voice cloning)

Populated in `data/voices/user/pool.jsonl` (one JSON per line):
```json
{"id": "user_001", "ref_path": "data/voices/user/audio/user_001/ref.wav", "ref_text": "Reference transcript.", "gender": "male", "accent": "american", "age_range": "25-35"}
```

Place reference WAV files (3-10s clean speech) in `data/voices/user/audio/{id}/ref.wav`.

**If no user voices are provided**, the pipeline falls back to using CustomVoice presets for user turns (limited diversity but works for pilot).

## TTS Architecture

Two Qwen3-TTS models run simultaneously:

| Role | Model | Voice | Style Control |
|------|-------|-------|---------------|
| Assistant | CustomVoice (1.7B) | Preset speaker name | `instruct` param per turn |
| User | Base (1.7B) | `ref_audio` cloning | None (natural speech) |

**Key constraint**: Qwen3-TTS `ref_audio` and `instruct` cannot be used simultaneously — they're separate model types. Our architecture naturally handles this by splitting assistant/user onto different models.

## Training

### Pilot (LoRA)
```bash
bash training/launch.sh configs/pilot_training.yaml 8
```
- 8xH100, LoRA rank 128, ~2-3 hours
- Checkpoints every 500 steps, keeps 5
- WandB logging enabled
- Eval every 500 steps on 5% held-out data

### Full (future)
- Full fine-tuning, all ~4,700h data
- Add loss masking on system prompt region (`training/loss_masking.py`)
- ~12-24 hours on 8xH100

## Config Files

- `configs/generation.yaml` — Full pipeline config (LLM, TTS, assembly, quality)
- `configs/pilot_training.yaml` — moshi-finetune training config

## Project Structure

```
src/
├── configs/
│   ├── generation.yaml          # Pipeline config
│   └── pilot_training.yaml      # Training config
├── data/
│   ├── voices/
│   │   ├── assistant/presets.yaml
│   │   └── user/pool.jsonl
│   ├── transcripts/             # Phase 1 output
│   ├── voice_assignments/       # Phase 2 output
│   ├── tts_audio/               # Phase 3 output
│   ├── assembled/               # Phase 4 output
│   └── formatted/               # Phase 5 output (training data)
├── pipeline/
│   ├── generate_transcripts.py  # Phase 1: LLM transcript generation
│   ├── assign_voices.py         # Phase 2: Voice mapping
│   ├── synthesize_tts.py        # Phase 3: Qwen3-TTS synthesis
│   ├── quality_filter.py        # Optional: WER + duration checks
│   ├── assemble_channels.py     # Phase 4: Stereo WAV assembly
│   ├── format_dataset.py        # Phase 5: moshi-finetune formatting
│   └── utils.py                 # Shared utilities
├── training/
│   ├── launch.sh                # Training launcher
│   └── loss_masking.py          # System prompt loss masking
├── run_pipeline.py              # End-to-end orchestrator
├── setup_env.sh                 # Environment setup
└── vendor/
    ├── moshi-finetune/          # Training framework
    └── personaplex/             # Reference (not used in training)
```
