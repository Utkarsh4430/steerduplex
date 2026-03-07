# SteerDuplex: Execution Plan

## Locked Decisions

- **Base model**: Moshi 7B (NOT PersonaPlex — training from Moshi weights for fair comparison)
- **Architecture**: Unchanged Moshi (user audio + agent text + agent audio streams)
- **Assistant voice**: Fixed per session via voice cloning from 3-10s reference. Style varies per turn via TTS `instruct`.
- **User voice**: Random per session. May or may not follow style characteristics — realistic variation.
- **Assistant voice pool**: 3-4 curated reference voices for assistant
- **User voice pool**: Remaining voices from VoxCeleb/LibriTTS/CommonAccent/Fisher
- **TTS engine**: Qwen3-TTS (style control via `instruct`, voice cloning via `ref_audio`)
- **LLMs for transcript generation**: Qwen-3-32B / Llama-3.1-70B
- **Language**: English only for v1
- **Approach**: Pilot first (500 conversations), then scale
- **Training**: moshi-finetune repo (LoRA initially, full FT later)

---

## Phase 0: Characteristic Database

Build structured YAML/JSON database for A1-A7 categories defining all possible values, TTS instruct templates, and combinability rules.

**Output**: `data_categories/` directory with per-category YAML files
**Status**: IN PROGRESS

---

## Phase 1: LLM Transcript Generation

### What the LLM generates per conversation:

```json
{
  "category": "A3_tone_controlled",
  "sampled_traits": {"tone": "sarcastic", "domain": "general", "topic": "diet advice"},
  "system_prompt": "You are a fitness enthusiast who gives advice with a sarcastic edge...",
  "system_prompt_tts_instruct": "confident, slightly smug, dry humor, moderate pace",
  "assistant_voice_id": "assistant_voice_02",
  "user_voice_id": "user_voice_random_147",
  "turns": [
    {
      "role": "user",
      "text": "So I've been eating pizza every day...",
      "tts_instruct": "genuine, slightly confused, casual tone"
    },
    {
      "role": "assistant",
      "text": "Wow, what a mystery...",
      "tts_instruct": "dry sarcasm, deadpan delivery, slight pause"
    }
  ]
}
```

### Key design points:
- Per-turn `tts_instruct` for BOTH user and assistant
- User tts_instruct is for realism during TTS generation, NOT a training target
- Assistant tts_instruct must be consistent with system_prompt but can vary turn-by-turn
- 3 granularity levels for system prompts: minimal, topic-specific, highly detailed
- Non-verbal cues embedded in text: (laughs), (sighs), (pauses)

### LLM meta-prompt structure:
1. Sample characteristics from Phase 0 DB
2. Feed to LLM with structured output format
3. Validate output (correct roles, reasonable turn count, tts_instruct present)
4. Store as JSON

---

## Phase 2: Voice Assignment

### Assistant voices (3-4 curated):
- Pick high-quality, distinct reference samples (male/female, different timbres)
- These become the voice prompts in hybrid system prompts
- Fixed per conversation, all assistant turns use same reference

### User voices (large pool):
- Source from VoxCeleb, LibriTTS, CommonAccent, Fisher
- Tag each with: estimated age, gender, accent, energy level
- Random selection per conversation
- For A5 (accent), match user voice to specified accent using CommonAccent

### Voice pool prep:
- Download datasets
- Extract 3-10s clean single-speaker clips
- Auto-tag using speaker verification models
- Manual QC on a sample

---

## Phase 3: TTS Synthesis (Qwen3-TTS)

```python
# Per conversation
for turn in conversation.turns:
    if turn.role == "assistant":
        wav = qwen3_tts.generate(
            text=turn.text,
            ref_audio=assistant_voice_ref,   # same every turn
            instruct=turn.tts_instruct       # varies per turn
        )
    else:  # user
        wav = qwen3_tts.generate(
            text=turn.text,
            ref_audio=user_voice_ref,        # same every turn
            instruct=turn.tts_instruct       # varies or None
        )
```

### Quality filtering pipeline:
- UTMOS score > threshold for naturalness
- Whisper ASR → WER check (generated speech matches intended text)
- Speaker similarity check (cosine sim between ref and generated)
- Discard and regenerate failures

### Qwen3-TTS models to use:
- `Qwen3-TTS-12Hz-1.7B-Base`: For voice cloning (ref_audio based)
- `Qwen3-TTS-12Hz-1.7B-CustomVoice`: For premium voices + instruct
- `Qwen3-TTS-12Hz-1.7B-VoiceDesign`: For creating voices from description

### Open question:
- Can we use Base model (voice cloning) AND instruct simultaneously? Need to test.
- If not, fallback: clone voice first, then post-process for style (speed/pitch via sox/praat)

---

## Phase 4: 2-Channel Assembly

### Channel layout (moshi-finetune convention):
- **Left channel (ch0)**: Assistant/Moshi audio
- **Right channel (ch1)**: User audio

### Assembly process:
```
For each conversation:
  1. Collect all turn audio files (from Phase 3)
  2. Calculate timeline:
     - Insert 200-800ms silence between turns (sampled from distribution)
     - 5-10% of turn transitions: negative overlap (barge-in simulation)
  3. Build two mono tracks:
     - assistant_track: assistant audio at correct positions, silence elsewhere
     - user_track: user audio at correct positions, silence elsewhere
  4. Inject backchannels (for A2, A7):
     - "uh-huh", "mm-hmm" in assistant channel during long user turns
  5. Merge into stereo WAV (left=assistant, right=user)
  6. Truncate/chunk to ≤100 seconds (moshi-finetune duration_sec limit)
```

### Hybrid system prompt prepending:
```
Assistant channel: [voice_prompt_3-10s][440Hz_sine_200ms][silence_during_text]...[dialog]
User channel:      [silence           ][silence         ][silence            ]...[dialog]
```

For pilot: bake system prompt into audio directly (no training code changes).
For full run: modify moshi-finetune to support loss masking on prompt region.

---

## Phase 5: Format for moshi-finetune

### Expected format:
```
data/
├── manifest.jsonl           # {"path": "convos/0001.wav", "duration": 87.3}
├── convos/
│   ├── 0001.wav             # stereo WAV, left=assistant, right=user
│   ├── 0001.json            # transcript + timestamps (from annotate.py)
│   └── ...
```

### Steps:
1. Place all stereo WAVs in directory
2. Run `annotate.py` from moshi-finetune to generate `.json` transcripts
3. Generate `manifest.jsonl`
4. Configure YAML (duration_sec=100, batch_size=16, lr=2e-6)

---

## Phase 6: Training

### Pilot training config:
```yaml
data:
  train_data: 'data/manifest.jsonl'
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"
full_finetuning: false
lora:
  enable: true
  rank: 128
  scaling: 2.0
duration_sec: 100
batch_size: 16
max_steps: 2000
optim:
  lr: 2e-6
```

### Full training (after pilot validation):
- Increase data to full ~4000h
- Consider full finetuning instead of LoRA
- Increase max_steps proportionally
- Add system prompt loss masking (requires code modification)

---

## Phase 7: Evaluation

### Automated:
- Full-Duplex-Bench (turn taking, backchanneling, interruption)
- Service-Duplex-Bench (role adherence)
- Speaker similarity (WavLM cosine sim between voice prompt and generated)
- UTMOS for naturalness
- GPT-4o judge for instruction following

### Steerability-specific:
- Does the model adopt the correct tone? (A3)
- Does the model maintain persona across turns? (A4)
- Does the model match the requested accent/style? (A5)
- Does the model follow speed/length instructions? (A6)
- Does the model respond appropriately to user emotion? (A7)

---

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Qwen3-TTS accent/emotion control is weak | A3, A5, A7 data quality | Test early in pilot; fallback to voice cloning from accent-specific datasets + post-processing |
| moshi-finetune has no system prompt support | Model can't learn prompt following | Pilot: bake into audio. Full: modify training code for loss masking |
| Voice consistency degrades across turns | Unnatural conversations | Use `x_vector_only_mode` in Qwen3-TTS; test in pilot |
| 100s duration too short for long conversations | Truncated multi-turn | Most conversations 5-8 turns ~60-80s; chunk longer ones with repeated system prompt |
| LLM generates repetitive transcripts | Low diversity | Multiple LLMs, high temperature, varied seeds, post-hoc dedup |
| Qwen3-TTS 12Hz → Mimi 12.5Hz re-encoding | Audio quality loss | Test quality; if bad, consider matching codec rates |

---

## Pilot Plan (Week 1-2)

### Week 1:
- Day 1-2: Build characteristic DB (A1-A7 YAML files) ✓
- Day 2-3: Write + test LLM meta-prompt → generate 50-100 convos per category
- Day 3-4: Test Qwen3-TTS (accent, emotion, speed, cloning quality)
- Day 4-5: Build TTS pipeline script + 2-channel stitching

### Week 2:
- Day 1-2: Generate all pilot audio (~500 conversations)
- Day 2-3: Format for moshi-finetune, run annotate.py
- Day 3-4: Train LoRA on pilot data
- Day 5: Evaluate — does model follow prompts? voice cloning work? tone audible?

---

## Data Scale Targets

| Category | Pilot | Full | Est. Hours (Full) |
|----------|-------|------|--------------------|
| A1: Customer Service | 70 | ~50K | ~800h |
| A2: QA/Assistant | 70 | ~40K | ~400h |
| A3: Tone-Controlled | 70 | ~15K | ~250h |
| A4: Persona-Controlled | 70 | ~15K | ~250h |
| A5: Style & Accent | 70 | ~10K | ~150h |
| A6: Speed & Length | 70 | ~5K | ~80h |
| A7: Emotional | 70 | ~10K | ~150h |
| A8: Failure Cases | 10 | ~2K | ~30h |
| **Total** | **500** | **~147K** | **~2,110h** |

---

## References

- PersonaPlex: ICASSP 2026, NVIDIA (Roy et al.)
- Moshi: Defossez et al., 2024
- moshi-finetune: https://github.com/kyutai-labs/moshi-finetune
- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- Dia TTS: https://github.com/nari-labs/dia (backup option)
- Chatterbox TTS: https://github.com/resemble-ai/chatterbox (backup option)
