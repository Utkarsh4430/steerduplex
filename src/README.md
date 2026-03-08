# SteerDuplex

Steerable full-duplex speech-to-speech model based on Moshi 7B. Supports dynamic, user-directed mid-conversation steering with compositional attribute control and principled boundaries.

## Setup

```bash
# 1. Create environment
bash setup_env.sh
conda activate steerduplex

# 2. Populate voices (see "Managing Voices" below)

# 3. Start LLM proxy for transcript generation
litellm --model Qwen/Qwen3-32B --port 4000
# Set llm_base_url: "http://localhost:4000/v1" in configs/generation.yaml

# 4. Run data generation pipeline
python run_pipeline.py --config configs/generation.yaml

# 5. Train
bash training/launch.sh configs/pilot_training.yaml 8

# 6. Inference
python -m inference.generate \
    --user_audio input.wav \
    --output output.wav \
    --system_prompt "You are a helpful assistant." \
    --checkpoint runs/pilot_v1/checkpoint_3000
```

Or install dependencies manually:
```bash
pip install -r requirements.txt
```

---

## Managing Voices

### Assistant Voices (Qwen3-TTS CustomVoice)

Assistant voices use Qwen3-TTS **CustomVoice** model with preset speakers. Style is controlled per-turn via the `instruct` parameter while voice identity stays fixed per conversation.

**Current presets** in `data/voices/assistant/presets.yaml`:

| ID | Speaker | Gender | Description |
|----|---------|--------|-------------|
| male_A | Ryan | Male | Clear, neutral, versatile |
| male_B | Dylan | Male | Warmer, deeper |
| female_A | Vivian | Female | Clear, neutral, versatile |
| female_B | Serena | Female | Warmer |

**Available CustomVoice speakers** (can be added): Eric, Aiden, Uncle_Fu, Ono_Anna, Sohee

#### Adding a new assistant voice

1. Edit `data/voices/assistant/presets.yaml`:
   ```yaml
   voices:
     # ... existing voices ...
     - id: male_C
       speaker: Eric        # Must be a valid Qwen3-TTS CustomVoice speaker name
       gender: male
       description: "Casual, energetic male voice"
   ```

2. That's it. The pipeline will randomly sample from all listed presets during data generation. No audio files needed — CustomVoice speakers are built into the model.

#### Deploying specific voices

For deployment, you may want a subset. Edit the `voices` list to include only the voices you want, or filter at inference time:
```bash
python -m inference.generate \
    --user_audio input.wav \
    --output output.wav \
    --system_prompt "You are a helpful assistant."
    # Voice identity comes from the trained model + system prompt
```

### User Voices (Qwen3-TTS Base, Voice Cloning)

User voices use Qwen3-TTS **Base** model with voice cloning from reference audio. No style control — natural speech only.

**Voice pool** in `data/voices/user/pool.jsonl` (one JSON object per line):

```json
{"id": "user_001", "ref_path": "data/voices/user/audio/user_001/ref.wav", "ref_text": "Hello, my name is John and I'm calling about my account.", "gender": "male", "accent": "american", "age_range": "25-35", "energy": "medium"}
```

#### Adding a new user voice

1. Record or obtain a clean speech clip (3-10 seconds, 16kHz+ WAV, minimal background noise).

2. Place it at `data/voices/user/audio/<id>/ref.wav`:
   ```bash
   mkdir -p data/voices/user/audio/user_042
   cp /path/to/clip.wav data/voices/user/audio/user_042/ref.wav
   ```

3. Add an entry to `data/voices/user/pool.jsonl`:
   ```bash
   echo '{"id": "user_042", "ref_path": "data/voices/user/audio/user_042/ref.wav", "ref_text": "Transcript of the reference audio.", "gender": "female", "accent": "british", "age_range": "30-40", "energy": "medium"}' >> data/voices/user/pool.jsonl
   ```

4. The `ref_text` field should be an accurate transcript of the reference audio. This improves voice cloning quality.

#### Bulk adding from VoxCeleb/Fisher

Use the import tool for large-scale voice pool population:
```bash
# Import VoxCeleb speakers as user voice pool
python -m pipeline.import_external \
    --input_dir /data/voxceleb1/wav \
    --dataset_name voxceleb_voices \
    --format voice_pool \
    --max_clips 1000

# Then append to main pool
cat data/external/voxceleb_voices/pool_import.jsonl >> data/voices/user/pool.jsonl
```

Or manually:
```bash
for spk_dir in /path/to/voxceleb1/wav/id*; do
    spk_id=$(basename "$spk_dir")
    ref_wav=$(find "$spk_dir" -name "*.wav" | head -1)
    mkdir -p "data/voices/user/audio/${spk_id}"
    cp "$ref_wav" "data/voices/user/audio/${spk_id}/ref.wav"
    echo "{\"id\": \"${spk_id}\", \"ref_path\": \"data/voices/user/audio/${spk_id}/ref.wav\", \"ref_text\": \"\", \"gender\": \"unknown\", \"accent\": \"unknown\", \"age_range\": \"unknown\", \"energy\": \"medium\"}" >> data/voices/user/pool.jsonl
done
```

**If no user voices are provided**, the pipeline falls back to using CustomVoice presets for user turns. This works for the pilot but limits voice diversity.

---

## Data Generation Pipeline

| Phase | Module | Input | Output |
|-------|--------|-------|--------|
| 1. Transcripts | `pipeline.generate_transcripts` | data_categories/*.yaml | data/transcripts/{cat}/*.json |
| 2. Voice Assignment | `pipeline.assign_voices` | transcripts + voice DB | data/voice_assignments/{cat}/*.json |
| 3. TTS Synthesis | `pipeline.synthesize_tts` | assignments | data/tts_audio/{cat}/{conv}/turn_*.wav |
| (opt) Quality Filter | `pipeline.quality_filter` | TTS audio | adds quality_passed to synth JSONs |
| 4. Channel Assembly | `pipeline.assemble_channels` | TTS audio | data/assembled/{conv}.wav + _meta.json |
| 5. Format Dataset | `pipeline.format_dataset` | assembled WAVs | data/formatted/manifest_{train,eval}.jsonl |

### Running specific phases

```bash
python run_pipeline.py --phase 1                          # transcripts only
python run_pipeline.py --from_phase 3 --to_phase 4       # TTS + assembly
python run_pipeline.py --category A9_dynamic_steering     # single category
python run_pipeline.py --skip_quality                     # skip quality filter
```

All phases are **resumable** — rerunning skips completed work.

### Data categories

| Category | # Pilot | Description |
|----------|---------|-------------|
| A1: Customer Service | 60 | 20 domains x 10 scenarios |
| A2: QA/Assistant | 60 | Knowledge QA, tutoring, reasoning |
| A3: Tone-Controlled | 60 | 33 tones (user requests at start) |
| A4: Persona-Controlled | 60 | 70+ personas |
| A5: Style & Accent | 40 | 15 styles + 15 accents |
| A6: Speed & Length | 40 | 5x5 speed/length matrix |
| A7: Emotional Adaptation | 40 | User emotion detection + adaptation |
| A8: Edge Cases | 20 | Spelling, sarcasm, interruptions |
| **A9: Dynamic Steering** | **200** | **Mid-conversation steering (core novelty)** |
| A10: Graceful Failure | 40 | Clarification, refusal, repair |

### Data types

| Type | % | Description |
|------|---|-------------|
| Standard | 50% | User sets style at start, model maintains |
| Dynamic | 25% | User changes style mid-conversation |
| Counterfactual | 10% | Same text, different style (paired) |
| Long-form | 10% | 10-15 turns, multi-turn coherence |
| Graceful failure | 5% | Misunderstanding, refusal, repair |

---

## Importing External Datasets

Use `pipeline.import_external` to convert existing audio datasets into training format.

### Supported formats

| Format | Description | Example datasets |
|--------|-------------|-----------------|
| `stereo` | Stereo WAV (left=assistant, right=user) | Fisher, pre-assembled conversations |
| `mono_pairs` | Paired mono files (agent + caller) | Call center recordings |
| `voice_pool` | Single-speaker clips → user voice pool | VoxCeleb, LibriSpeech |

### Import Fisher conversations

```bash
python -m pipeline.import_external \
    --input_dir /data/fisher/audio \
    --dataset_name fisher \
    --format stereo \
    --system_prompt "You are a helpful voice assistant." \
    --max_duration 100
```

### Import call center data (separate agent/caller files)

```bash
python -m pipeline.import_external \
    --input_dir /data/call_center \
    --dataset_name call_center \
    --format mono_pairs \
    --assistant_suffix "_agent.wav" \
    --user_suffix "_caller.wav"
```

### Import VoxCeleb as user voice pool

```bash
python -m pipeline.import_external \
    --input_dir /data/voxceleb1/wav \
    --dataset_name voxceleb_voices \
    --format voice_pool \
    --max_clips 1000
```

### Merge with existing training data

moshi-finetune supports multi-source training with weights:
```yaml
# In training config
data:
  train_data: "data/formatted/manifest_train.jsonl:1.0,data/external/fisher/manifest_train.jsonl:0.5"
```

### moshi-finetune data format reference

```
dataset_dir/
├── audio/
│   ├── 00001.wav          # Stereo WAV: left=assistant (ch0), right=user (ch1)
│   │                      # Sample rate: 24kHz, max duration: 100s
│   ├── 00001.json         # Word-level alignments (Whisper-generated)
│   ├── 00002.wav
│   └── 00002.json
├── manifest_train.jsonl   # One JSON per line: {"path": "audio/00001.wav", "duration": 45.2}
└── manifest_eval.jsonl
```

**Alignment JSON format** (generated by Whisper, enriched with system prompt):
```json
{
  "alignments": [
    ["<system>", [0.0, 0.1], "SPEAKER_MAIN"],
    ["You", [0.1, 0.2], "SPEAKER_MAIN"],
    ["are", [0.2, 0.3], "SPEAKER_MAIN"],
    ["a", [0.3, 0.35], "SPEAKER_MAIN"],
    ["helpful", [0.35, 0.5], "SPEAKER_MAIN"],
    ["assistant.", [0.5, 0.7], "SPEAKER_MAIN"],
    ["<system>", [0.7, 0.8], "SPEAKER_MAIN"],
    ["Hello", [5.2, 5.5], "SPEAKER_MAIN"],
    ["how", [5.5, 5.7], "SPEAKER_MAIN"],
    ["can", [5.7, 5.9], "SPEAKER_MAIN"],
    ["I", [5.9, 6.0], "SPEAKER_MAIN"],
    ["help?", [6.0, 6.3], "SPEAKER_MAIN"]
  ],
  "text_conditions": null,
  "_metadata": {
    "system_prompt": "<system> You are a helpful assistant. <system>",
    "prompt_end_sec": 5.0,
    "category": "A1_customer_service",
    "data_type": "standard"
  }
}
```

### Generating Audio MultiChallenge-style data

Audio MultiChallenge tests inference memory, instruction retention, self-coherence, and voice editing. To generate this type of data:

1. Create a new category YAML (e.g., `data_categories/A11_multi_challenge.yaml`) following the same structure as A1-A10
2. Define scenarios that test: recall across turns, instruction persistence, mid-utterance corrections
3. Run the pipeline on just that category:
   ```bash
   python run_pipeline.py --category A11_multi_challenge
   ```

The existing A9 (dynamic steering) and A2 (QA) categories already cover many Audio MultiChallenge capabilities. See `data_categories/` for the YAML schema.

---

## Training

### Training objective (follows PersonaPlex)

Our training follows the PersonaPlex guidelines:
1. **System prompt loss masking**: Loss is zeroed out in the voice_prompt + text_prompt region (the model shouldn't be penalized for predictions during conditioning)
2. **Non-semantic audio token downweighting**: Codebooks 1-7 weighted at 0.02x relative to codebook 0 (`first_codebook_weight_multiplier: 50`)
3. **Padded text token downweighting**: Padding tokens weighted at 0.3x (`text_padding_weight: 0.3`)
4. **FSDP**: Fully Sharded Data Parallel across GPUs with per-transformer-layer sharding
5. **Gradient checkpointing**: Enabled to reduce memory footprint
6. **bf16 mixed precision**: Parameters in bfloat16, optimizer states in float32
7. **OneCycleLR scheduler**: 5% warmup, cosine decay

### Pilot (LoRA)

```bash
bash training/launch.sh configs/pilot_training.yaml 8
```

- 8xH100, LoRA rank 128
- Effective batch size: 4 per GPU x 8 GPUs = 32
- `duration_sec: 100` (each sample up to 100s of audio)
- Checkpoints every 500 steps in `runs/pilot_v1/`
- WandB logging: project=steerduplex, run=pilot_v1_lora128
- Eval every 500 steps on 5% held-out data

### Full fine-tuning (future)

```bash
bash training/launch.sh configs/full_training.yaml 8
```

- Full parameter fine-tuning on all ~4,700h data
- Same loss masking + token weighting as pilot
- `num_microbatches: 2` for gradient accumulation (larger effective batch)
- ~12-24 hours on 8xH100

### Training config

See `configs/pilot_training.yaml` for all options:

```yaml
lora:
  enable: true
  rank: 128
  scaling: 2.0

# PersonaPlex token weighting
first_codebook_weight_multiplier: 50.0   # non-semantic tokens get 1/50 = 0.02 weight
text_padding_weight: 0.3                  # padded text tokens at 0.3x

duration_sec: 100          # max audio length per sample
batch_size: 4              # per GPU
max_steps: 3000
gradient_checkpointing: true
param_dtype: "bfloat16"

optim:
  lr: 2e-6
  weight_decay: 0.1
  pct_start: 0.05          # 5% warmup
```

---

## Inference

### Offline inference

Generate assistant audio response given user audio input:

```bash
# Base model (no finetuning)
python -m inference.generate \
    --user_audio input.wav \
    --output output.wav \
    --system_prompt "You are a helpful assistant."

# With finetuned checkpoint
python -m inference.generate \
    --user_audio input.wav \
    --output output.wav \
    --system_prompt "You are a friendly tutor. Speak slowly and clearly." \
    --checkpoint runs/pilot_v1/checkpoint_3000

# With voice prompt (3-10s reference audio)
python -m inference.generate \
    --user_audio input.wav \
    --output output.wav \
    --system_prompt "You are a helpful assistant." \
    --voice_prompt data/voices/assistant/audio/custom_voice.wav \
    --checkpoint runs/pilot_v1/checkpoint_3000
```

### Inference options

| Flag | Default | Description |
|------|---------|-------------|
| `--user_audio` | (required) | Path to user audio WAV |
| `--output` | output.wav | Output WAV path |
| `--system_prompt` | "" | System prompt text |
| `--voice_prompt` | None | Voice conditioning WAV (3-10s) |
| `--checkpoint` | None | Finetuned checkpoint directory |
| `--hf_repo` | kyutai/moshiko-pytorch-bf16 | Base model HF repo |
| `--device` | cuda | Device (cuda/cpu) |
| `--max_duration` | 30.0 | Max output duration (seconds) |
| `--temperature` | 0.8 | Sampling temperature |
| `--top_k` | 250 | Top-k sampling |

### System prompt design

System prompts set the assistant's **role and boundaries** — NOT style/tone/speed:

```
You are a helpful voice assistant. Your voice and identity are fixed.
Stay kind and constructive. Follow the user's instructions about speaking style.
```

```
You are a customer service agent for a bank. Your voice and identity are fixed.
Always maintain a professional, respectful tone. Follow the user's instructions
about speaking speed, formality, and detail level.
```

Style/tone/speed is controlled dynamically by the user during the conversation.

### Attribute hierarchy

| Type | Examples | Can user change? |
|------|----------|------------------|
| **Immutable** | Voice gender/age, base timbre, politeness floor, safety | No — model politely declines |
| **Mutable** | Tone, speed, length, formality, energy, persona | Yes — model adapts immediately |
| **Conditionally mutable** | Accent, language register | Sometimes — depends on context |

---

## TTS Architecture

Two Qwen3-TTS models run in the data generation pipeline:

| Role | Model | Voice | Style Control |
|------|-------|-------|---------------|
| Assistant | CustomVoice (1.7B) | Preset speaker name | `instruct` param per turn |
| User | Base (1.7B) | `ref_audio` cloning | None (natural speech) |

**Key constraint**: Qwen3-TTS `ref_audio` and `instruct` cannot be used simultaneously — they're separate model types. The dual-model architecture handles this naturally.

---

## Project Structure

```
src/
├── configs/
│   ├── generation.yaml          # Data pipeline config
│   └── pilot_training.yaml      # Training config
├── data/
│   ├── voices/
│   │   ├── assistant/presets.yaml   # Assistant voice presets
│   │   └── user/pool.jsonl          # User voice pool
│   ├── transcripts/             # Phase 1 output
│   ├── voice_assignments/       # Phase 2 output
│   ├── tts_audio/               # Phase 3 output
│   ├── assembled/               # Phase 4 output
│   ├── formatted/               # Phase 5 output (training data)
│   └── external/                # Imported external datasets
├── pipeline/
│   ├── generate_transcripts.py  # Phase 1: LLM transcript generation
│   ├── assign_voices.py         # Phase 2: Voice mapping
│   ├── synthesize_tts.py        # Phase 3: Qwen3-TTS synthesis
│   ├── quality_filter.py        # Optional: WER + duration checks
│   ├── assemble_channels.py     # Phase 4: Stereo WAV assembly
│   ├── format_dataset.py        # Phase 5: moshi-finetune formatting
│   ├── import_external.py       # Import Fisher/VoxCeleb/custom datasets
│   └── utils.py                 # Shared utilities
├── training/
│   ├── train.py                 # Training loop with system prompt loss masking
│   ├── annotate.py              # Whisper annotation for alignments
│   ├── loss_masking.py          # System prompt region loss masking
│   └── launch.sh               # Training launcher
├── inference/
│   └── generate.py              # Offline inference with voice/prompt selection
├── run_pipeline.py              # End-to-end orchestrator
├── setup_env.sh                 # Environment setup
└── requirements.txt             # Python dependencies
```

### External dependencies (installed via pip, not vendored)

- **[moshi](https://github.com/kyutai-labs/moshi)**: Moshi model code (LM, Mimi codec)
- **[moshi-finetune](https://github.com/kyutai-labs/moshi-finetune)**: Training framework (`finetune` package — args, data loading, FSDP, checkpointing)
- **[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)**: TTS synthesis (CustomVoice + Base models)

---

## Evaluation Benchmarks

### Primary benchmarks

| Benchmark | What it tests | Our categories | Priority |
|-----------|---------------|----------------|----------|
| **[Service-Duplex-Bench](https://arxiv.org/abs/2602.06053)** | Role adherence, customer service, speaker similarity, turn-taking | A1, A4, A10 | High |
| **[Full-Duplex-Bench v2](https://github.com/DanielLin94144/Full-Duplex-Bench)** | Turn-taking, backchanneling, interruption handling, multi-turn dialogue | A1, A6, A9, A10 | High |
| **[Audio MultiChallenge](https://arxiv.org/abs/2512.14865)** | Inference memory, instruction retention, self-coherence, voice editing | A2, A4, A7, A9 | High |
| **[Game-Time](https://arxiv.org/abs/2509.26388)** | Tempo adherence, timing, synchronized responses | A5, A6, A9 | Medium |

### Secondary benchmarks

| Benchmark | What it tests | Our categories |
|-----------|---------------|----------------|
| **[SOVA-Bench](https://arxiv.org/abs/2506.02457)** | Knowledge, speech recognition/understanding/generation | A1, A2, A3, A4, A7 |
| **[VocalBench](https://arxiv.org/abs/2505.15727)** | Semantic abilities, acoustic quality, chat, robustness | A2, A3, A4, A7, A8 |
| **[WildSpeech-Bench](https://arxiv.org/abs/2506.21875)** | Paralinguistic features, acoustic robustness, real-world speech | A5, A7, A8, A10 |
| **[MULTI-Bench](https://arxiv.org/abs/2511.00850)** | Emotional intelligence (recognition, reasoning, support) | A3, A4, A7, A9 |
| **[HumDial Challenge](https://arxiv.org/abs/2601.05564)** (ICASSP 2026) | Emotional trajectories, full-duplex interaction | A1, A3, A7, A9, A10 |
| **[VoiceAssistant-Eval](https://arxiv.org/abs/2509.22651)** | Multi-turn dialogue, role-play, consistency | A1, A4, A5, A7 |

### SteerDuplex-specific metrics (novel)

These are NOT covered by existing benchmarks — they're our contribution:

| Metric | Description | How to measure |
|--------|-------------|----------------|
| **Edit Latency** | Frames until output changes after a steering instruction | Compare pre/post steering codec outputs |
| **Edit Retention** | Turns the edit persists through topic changes/interruptions | Track attribute values across subsequent turns |
| **Composition Independence** | Cross-talk between dimensions when one changes | Change tone, measure speed/accent drift |
| **Control Leakage** | Voice identity shift when style changes | WavLM speaker similarity before/after steering |
| **Refused Steering Rate** | Model correctly declines immutable attribute changes | % correct refusals on immutable requests |

### Coverage matrix

| Category | Service-Duplex | Full-Duplex v2 | Audio MC | Game-Time | SOVA | VocalBench | WildSpeech | MULTI |
|----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| A1 Customer Service | x | x | | | x | x | x | |
| A2 QA/Assistant | | | x | | x | x | | |
| A3 Tone Control | | | | x | x | x | | x |
| A4 Persona | x | x | x | | x | x | | x |
| A5 Style/Accent | | | | x | | | x | |
| A6 Speed/Length | | x | | x | | | | |
| A7 Emotional | | | x | | x | x | x | x |
| A8 Edge Cases | | | | | | x | x | |
| **A9 Dynamic Steering** | | | **x** | **x** | | | | **x** |
| A10 Graceful Failure | x | x | | | | x | x | |

---

## Config Files

- `configs/generation.yaml` — Full data pipeline config (LLM, TTS, assembly, quality filtering)
- `configs/pilot_training.yaml` — Training config (base model, LoRA, optimizer, checkpointing)

See the YAML files for all configurable parameters.
