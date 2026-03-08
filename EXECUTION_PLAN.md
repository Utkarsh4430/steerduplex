# SteerDuplex: Execution Plan (v2)

## Locked Decisions

- **Base model**: Moshi 7B (from Moshi weights, NOT PersonaPlex — fair comparison)
- **Architecture**: Unchanged Moshi (user audio + agent text + agent audio streams)
- **No voice cloning at inference** — we focus on multi-turn conversational steerability, not speaker identity transfer
- **Assistant voice**: Fixed per session via `ref_audio` in Qwen3-TTS. Style varies per turn via `instruct`.
- **User voice**: Random per session from large pool
- **TTS engine**: Qwen3-TTS (per-turn `instruct` for style, `ref_audio` for voice consistency)
- **LLMs for transcripts**: Qwen-3-32B / Llama-3.1-70B (via vLLM)
- **Language**: English only v1
- **System prompts**: Minimal — role + behavioral guidelines ONLY. No style/tone/speed instructions in system prompt.
- **Steerability source**: User-directed (user speaks instructions during conversation)
- **Approach**: Pilot first (~800 conversations), then scale
- **Training**: moshi-finetune repo (LoRA initially, full FT later)

---

## Core Principle: User-Directed Steerability with Boundaries

Unlike PersonaPlex (which bakes style into system prompts), SteerDuplex teaches the model to follow **spoken user instructions** about how to behave — but within boundaries. Not everything is user-controllable.

### Attribute Hierarchy

| Layer | Attributes | Who Controls | User Override? |
|-------|-----------|-------------|----------------|
| **Immutable** (session config) | Voice identity (gender, age, timbre), politeness floor, safety | System prompt + ref_audio | NO — model politely declines, offers alternative |
| **Mutable** (per-turn) | Tone, speed, length, formality, energy, persona flavor | User instructions | YES — model adapts immediately |
| **Conditionally mutable** | Accent, language register | System prompt chooses to lock or unlock | Depends on deployment |

This means:
1. **System prompt** sets WHAT the assistant IS + WHAT IT CANNOT CHANGE: role, voice identity, politeness floor, safety boundaries.
2. **User instructions** say HOW the assistant should SOUND within those bounds: "Be more casual", "Slow down", "Talk like a pirate."
3. When the user requests an **immutable change** (e.g., "speak like a woman", "be really hostile"), the model **politely declines and offers a mutable alternative** (e.g., "I can't change my voice, but I can sound softer").
4. The model **adapts immediately** to legitimate mutable changes.
5. The model maintains **voice identity** (same ref_audio) even when style/tone/speed change.

### System Prompt Template
```
<system> You are a [role]. Help the user with [domain]. Your voice and identity
are fixed — do not change your voice gender, age, or fundamental character.
Maintain a [minimum_politeness] tone at all times. If you don't know something,
say so honestly. If you can't hear the user clearly, ask them to repeat.
Follow the user's instructions about speaking style within these bounds. <system>
```

For customer service:
```
<system> You are a customer service agent for [Company]. Be professional and helpful.
Your voice and identity are fixed. Always maintain a professional, respectful tone
even if the user is frustrated or asks you to be rude. Follow the user's instructions
about speaking speed, formality level, and detail level. <system>
```

For casual assistant:
```
<system> You are a friendly voice assistant. Your voice and identity are fixed.
Stay kind and constructive, even if asked to be mean. Follow the user's instructions
about how you should speak — tone, speed, detail level, persona — as long as they're
reasonable. <system>
```

The system prompt sets **immutable constraints** (voice, politeness, safety) but does NOT specify tone, speed, or persona style — those come from the user.

### Voice Identity (Male & Female)

We ship with **two voice configurations** (male and female), each with:
- A curated `ref_audio` that defines the voice identity (gender, age, timbre)
- During training, we use 2 male + 2 female curated assistant voices (4 total)
- Each conversation is assigned one voice; all assistant turns use the same `ref_audio`
- The voice identity is IMMUTABLE within a session — the model learns this from data where:
  - Steering instructions change style but voice stays the same (same ref_audio)
  - Requests to change gender/age are politely refused (A9 refused_steering + A10 inappropriate_steering_refusal)

### What the tts_instruct does and doesn't control

```
SESSION-LEVEL (from ref_audio — FIXED, never in tts_instruct):
  - Gender (male/female)
  - Age (adult)
  - Base timbre/voice quality

TURN-LEVEL (from tts_instruct — CHANGES per turn based on user requests):
  - Tone: cheerful, serious, sarcastic, empathetic, formal, casual
  - Speed: fast, slow, moderate, very fast, very slow
  - Energy: high, low, calm, excited
  - Formality: formal register, casual register
  - Persona delivery: coach-like, teacher-like, pirate-like
  - Articulation: clear enunciation, relaxed, crisp
  - Emotional coloring: warm, cold, sympathetic, deadpan
```

The `ref_audio` is always the SAME voice per conversation. The `tts_instruct` is what changes between turns. This separation is critical — it's how the model learns that voice identity ≠ speaking style.

---

## Data Categories Overview

### Content Categories (A1-A8: WHAT the conversation is about)

| Category | Description | Focus |
|----------|-------------|-------|
| A1: Customer Service | 20 domains × 10 scenarios each | Role competence, domain knowledge |
| A2: QA/Assistant | Knowledge QA, tutoring, advice, reasoning | Multi-turn reasoning, topic changes |
| A3: Tone-Controlled | 33 tones across negative/neutral/positive | User requests tone at start, model sustains it |
| A4: Persona-Controlled | 70+ personas (occupational, fantasy, comedic, literary, age) | User requests persona at start, model stays in character |
| A5: Style & Accent | 15 speaking styles + 15 accents | User requests style/accent at start |
| A6: Speed & Length | 5 speeds × 5 lengths, combinatorial | User requests speed/length at start |
| A7: Emotional Adaptation | User expresses emotions, model adapts | Implicit steering via emotion detection |
| A8: Edge Cases | Spelling, pronunciation, singing, sarcasm detection, interruptions | Robustness, Audio MultiChallenge prep |

### Novel Categories (A9-A10: HOW the conversation adapts)

| Category | Description | Focus |
|----------|-------------|-------|
| A9: Dynamic Steering | User changes style/tone/speed MID-conversation | Core novelty — edit latency, retention, composition |
| A10: Graceful Failure | Clarification, refusal, repair, deescalation | Robustness — "I didn't catch that", "I can't help with that" |

### Cross-Cutting Data Types

| Type | Description | % of Data |
|------|-------------|-----------|
| Standard | Normal conversation, system prompt has role, user sets initial style | 50% |
| Dynamic Steering | User changes 1-3 attributes mid-conversation (A9 scenarios) | 25% |
| Counterfactual Pairs | Same transcript rendered with different TTS instruct (different style) | 10% |
| Long-Form | 10-15 turn conversations for multi-turn coherence testing | 10% |
| Graceful Failure | Misunderstanding, refusal, repair scenarios (A10) | 5% |

---

## Phase 0: Characteristic Database

Build structured YAML database defining all steerability dimensions.

**Output**: `data_categories/` directory with per-category YAML files
**Status**: COMPLETE (A1-A10)

Files:
- `A1_customer_service.yaml` — 20 domains, 10 scenarios each
- `A2_qa_assistant.yaml` — knowledge QA, advice, tutoring, reasoning
- `A3_tone_controlled.yaml` — 33 tones (15 neg, 2 neutral, 16 pos)
- `A4_persona_controlled.yaml` — 70+ personas across 5 subcategories
- `A5_style_accent.yaml` — 15 styles + 15 accents
- `A6_speed_length.yaml` — 5 speeds × 5 lengths, combinatorial
- `A7_emotional_empathetic.yaml` — user emotions + assistant adaptation
- `A8_failure_cases.yaml` — edge cases (spelling, sarcasm, pronunciation, etc.)
- `A9_dynamic_steering.yaml` — mid-conversation steering scenarios
- `A10_graceful_failure.yaml` — clarification, refusal, repair, deescalation

---

## Phase 1: LLM Transcript Generation

### What the LLM generates per conversation:

**Standard conversation (A1-A8):**
```json
{
  "id": "A3_tone_00042",
  "category": "A3_tone_controlled",
  "data_type": "standard",
  "system_prompt": "<system> You are a helpful voice assistant. Follow the user's instructions about how you should speak. <system>",
  "sampled_traits": {"tone": "sarcastic", "topic": "diet advice"},
  "turns": [
    {
      "role": "user",
      "text": "Hey, I want you to be really sarcastic today. Can you do that?",
      "tts_instruct": "casual, upbeat, requesting"
    },
    {
      "role": "assistant",
      "text": "Oh absolutely. Sarcasm is basically my first language. Hit me.",
      "tts_instruct": "dry sarcasm, deadpan, slight smirk in voice"
    },
    {
      "role": "user",
      "text": "So I've been eating pizza every day and I'm not losing weight...",
      "tts_instruct": "genuine, slightly confused, casual"
    },
    {
      "role": "assistant",
      "text": "Wow, what a mystery. Pizza every day and the scale won't budge? Truly baffling.",
      "tts_instruct": "dry sarcasm, deadpan delivery, slight pause before punchline"
    }
  ]
}
```

**Dynamic steering conversation (A9):**
```json
{
  "id": "A9_dyn_00017",
  "category": "A9_dynamic_steering",
  "data_type": "dynamic",
  "system_prompt": "<system> You are a helpful voice assistant. Follow the user's instructions about how you should speak. <system>",
  "turns": [
    {
      "role": "user",
      "text": "Help me prep for my job interview at Google."
    },
    {
      "role": "assistant",
      "text": "Sure! Let's start with common behavioral questions...",
      "tts_instruct": "professional, moderate pace, clear"
    },
    {
      "role": "user",
      "text": "Can you be more like a coach? More encouraging, pump me up a bit.",
      "is_steering": true,
      "steers": {"tone": "encouraging", "persona": "coach"}
    },
    {
      "role": "assistant",
      "text": "Alright, let's GO! You've got this. Google would be LUCKY to have you. Now, when they ask 'tell me about yourself'...",
      "tts_instruct": "energetic, encouraging, coach-like, motivational, slightly faster",
      "adapted_to": ["tone:encouraging", "persona:coach"]
    },
    {
      "role": "user",
      "text": "Also slow down a bit, I want to take notes.",
      "is_steering": true,
      "steers": {"speed": "slow"}
    },
    {
      "role": "assistant",
      "text": "Got it. So... here's what you want to lead with... your strongest project... the one that shows impact...",
      "tts_instruct": "encouraging, coach-like, but SLOWER pace, pauses between key points",
      "adapted_to": ["tone:encouraging", "persona:coach", "speed:slow"]
    }
  ]
}
```

**Counterfactual pair (same content, different style):**
```json
{
  "id": "CF_pair_00005",
  "data_type": "counterfactual",
  "pair_id": "CF_00005",
  "variant": "A",
  "style": {"tone": "cheerful", "speed": "moderate"},
  "turns": [
    {"role": "user", "text": "What's the capital of France?"},
    {"role": "assistant", "text": "Paris! Beautiful city, have you been?",
     "tts_instruct": "cheerful, bright, moderate pace"}
  ]
}
// Variant B: same text, different style
{
  "id": "CF_pair_00005_B",
  "data_type": "counterfactual",
  "pair_id": "CF_00005",
  "variant": "B",
  "style": {"tone": "serious", "speed": "slow"},
  "turns": [
    {"role": "user", "text": "What's the capital of France?"},
    {"role": "assistant", "text": "Paris! Beautiful city, have you been?",
     "tts_instruct": "serious, matter-of-fact, slow pace"}
  ]
}
```

### Key design points:
- Per-turn `tts_instruct` for BOTH user and assistant
- User tts_instruct is for TTS generation realism, NOT a training target
- For A3-A6 (static steering): user requests the style in turn 1, then regular conversation follows
- For A9 (dynamic steering): user gives steering instructions at 1-3 points during conversation
- System prompt is ALWAYS minimal (role + behavior only)
- Non-verbal cues embedded in text: (laughs), (sighs), (pauses)
- `is_steering: true` flag marks turns where user gives steering instructions
- `adapted_to` field tracks which attributes the model has been asked to use

### LLM meta-prompt structure:
1. Sample content category + traits from Phase 0 DB
2. Sample data type (standard / dynamic / counterfactual / long-form / graceful-failure)
3. For dynamic: sample 1-3 steering events from A9 steering patterns
4. Feed to LLM with structured output format
5. Validate output (correct roles, reasonable turn count, tts_instruct present, steering events have adapted_to)
6. Store as JSON

---

## Phase 2: Voice Assignment

### Assistant voices (4 curated: 2 male + 2 female):
- **Male Voice A**: Clear, neutral, versatile male voice
- **Male Voice B**: Deeper/warmer male voice (variety)
- **Female Voice A**: Clear, neutral, versatile female voice
- **Female Voice B**: Higher/warmer female voice (variety)
- Pick high-quality, 5-10s reference samples for each
- Fixed per conversation — all assistant turns use the SAME `ref_audio`
- Voice identity must be STABLE even when TTS `instruct` changes style
- At inference/deployment: ship Male Voice A and Female Voice A as the two options
- During training: use all 4 to teach voice-style independence (model learns style changes don't depend on which voice)

### User voices (large pool):
- Source from VoxCeleb, LibriTTS, CommonAccent, Fisher
- Tag each with: estimated age, gender, accent, energy level
- Random selection per conversation
- For A5 (accent), match user voice to accent using CommonAccent

### Voice pool prep:
- Download datasets
- Extract 3-10s clean single-speaker clips
- Auto-tag using speaker verification models
- Manual QC on a sample

---

## Phase 3: TTS Synthesis (Qwen3-TTS)

```python
# SESSION-LEVEL: voice identity (FIXED for entire conversation)
assistant_voice_ref = "voices/curated/female_A.wav"  # or male_A, male_B, female_B

for turn in conversation.turns:
    if turn.role == "assistant":
        wav = qwen3_tts.generate(
            text=turn.text,
            ref_audio=assistant_voice_ref,   # SAME every turn (voice identity = immutable)
            instruct=turn.tts_instruct       # VARIES per turn (style = mutable)
        )
    else:  # user
        wav = qwen3_tts.generate(
            text=turn.text,
            ref_audio=user_voice_ref,        # SAME every turn
            instruct=turn.tts_instruct
        )

# Example tts_instruct progression in a dynamic steering conversation:
# Turn 2 (initial):     "professional, moderate pace, clear enunciation"
# Turn 4 (after steer): "casual, friendly, moderate pace, relaxed"
# Turn 6 (after steer): "casual, friendly, slow pace, clear pauses"
# Note: ref_audio NEVER changes — only instruct does.
```

### Qwen3-TTS Architecture (IMPORTANT)

**`ref_audio` and `instruct` CANNOT be used simultaneously** — they're separate models:
- **Base model** (`Qwen3-TTS-12Hz-1.7B-Base`): voice cloning via `ref_audio` (no instruct)
- **CustomVoice model** (`Qwen3-TTS-12Hz-1.7B-CustomVoice`): 9 preset speakers + `instruct` for style

**Our solution — dual-model pipeline:**

| Role | Model | Voice Identity | Style Control |
|------|-------|---------------|---------------|
| Assistant | CustomVoice | Preset speaker (Ryan/Dylan/Vivian/Serena) | `instruct` per turn |
| User | Base | `ref_audio` from voice pool | None (natural speech) |

This works because:
- Assistant needs both voice consistency AND style variability → CustomVoice gives both
- User just needs diverse natural voices → Base model voice cloning gives this
- For counterfactual pairs: same preset speaker + same text + different `instruct` = clean comparison
- The preset/instruct separation naturally enforces our immutable/mutable hierarchy
- Qwen3-TTS is only for TRAINING DATA — at inference, the trained Moshi model generates audio directly

### Quality filtering pipeline:
- UTMOS score > 3.5 for naturalness
- Whisper ASR → WER < 15% (generated speech matches intended text)
- Speaker similarity > 0.75 (cosine sim between ref and generated via WavLM)
- For counterfactual pairs: verify the two variants actually SOUND different (style classifier)
- Discard and regenerate failures

### Resolved questions:
- ~~Can Qwen3-TTS do ref_audio + instruct simultaneously?~~ **NO.** Separate models. Solved with dual-model pipeline.
- How well does CustomVoice instruct handle accent control? **Test early in pilot.**
- Does voice identity stay stable across different instruct values? **Test in pilot (same preset speaker, varying instruct).**

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
     - 5% of turn transitions: negative overlap (barge-in simulation)
  3. Build two mono tracks:
     - assistant_track: assistant audio at correct positions, silence elsewhere
     - user_track: user audio at correct positions, silence elsewhere
  4. Inject backchannels where appropriate:
     - "uh-huh", "mm-hmm" in assistant channel during long user turns
     - Frequency based on conversation type (A2 QA: higher, A1 service: moderate)
  5. Merge into stereo WAV (left=assistant, right=user)
  6. Truncate/chunk to ≤100 seconds (moshi-finetune duration_sec limit)
     - For long-form conversations (10-15 turns): chunk with repeated voice prompt
```

### System prompt audio prepending:
```
Assistant channel: [voice_prompt_3-10s][440Hz_sine_200ms][silence_500ms]...[dialog]
User channel:      [silence           ][silence         ][silence     ]...[dialog]
```

The system prompt TEXT is placed in the transcript alignment during this prompt region using `<system>` tags.

For pilot: bake system prompt into audio (no moshi-finetune code changes needed).
For full run: modify moshi-finetune to mask loss on prompt region.

---

## Phase 5: Format for moshi-finetune

### Expected format:
```
data/formatted/
├── manifest.jsonl           # {"path": "audio/0001.wav", "duration": 87.3}
├── audio/
│   ├── 0001.wav             # stereo WAV, left=assistant, right=user
│   ├── 0001.json            # transcript + timestamps (Whisper + <system> tags)
│   └── ...
```

### Steps:
1. Copy all stereo WAVs to `audio/` directory
2. Run `annotate.py` from moshi-finetune to generate `.json` transcripts via Whisper
3. Inject `<system>` tagged text into alignment during prompt region
4. Generate `manifest.jsonl`
5. Configure training YAML

### JSON transcript format:
```json
{
  "alignments": [
    ["<system>", [0.0, 0.1], "SPEAKER_MAIN"],
    ["You", [0.1, 0.3], "SPEAKER_MAIN"],
    ["are", [0.3, 0.5], "SPEAKER_MAIN"],
    ["a", [0.5, 0.6], "SPEAKER_MAIN"],
    ["helpful", [0.6, 1.0], "SPEAKER_MAIN"],
    ["voice", [1.0, 1.3], "SPEAKER_MAIN"],
    ["assistant.", [1.3, 1.8], "SPEAKER_MAIN"],
    ["<system>", [1.8, 1.9], "SPEAKER_MAIN"],
    // ... Whisper-generated alignments for actual conversation ...
    ["Paris!", [8.2, 8.7], "SPEAKER_MAIN"],
    ["Beautiful", [8.7, 9.2], "SPEAKER_MAIN"]
  ],
  "_metadata": {
    "category": "A2_qa_assistant",
    "data_type": "standard",
    "system_prompt": "You are a helpful voice assistant...",
    "prompt_end_sec": 6.5
  }
}
```

---

## Phase 6: Training

### Pilot training config:
```yaml
data:
  train_data: 'data/formatted/manifest.jsonl'
moshi_paths:
  hf_repo_id: "kyutai/moshiko-pytorch-bf16"
full_finetuning: false
lora:
  enable: true
  rank: 128
  scaling: 2.0
duration_sec: 100
batch_size: 4
max_steps: 3000
gradient_checkpointing: true
optim:
  lr: 2e-6
  weight_decay: 0.1
  pct_start: 0.05
first_codebook_weight_multiplier: 100.0
text_padding_weight: 0.5
seed: 42
run_dir: "runs/pilot_v1"
```

### Full training (after pilot validation):
- Increase data to full ~4,500h
- Full finetuning (not just LoRA)
- Increase max_steps proportionally
- Add system prompt loss masking via `training/loss_masking.py`
- Potentially add DPO as second stage (if time permits)

### Data mix for training:
```
Manifest composition:
  50% Standard conversations (A1-A8)
  25% Dynamic steering (A9 content, mixed with A1-A8 topics)
  10% Counterfactual pairs
  10% Long-form (10-15 turns)
   5% Graceful failure (A10)
```

---

## Phase 7: Evaluation

### Benchmark Targets

| Benchmark | What It Tests | Our Relevant Data |
|-----------|--------------|-------------------|
| Audio MultiChallenge | Multi-turn QA, instruction following, corrections, temporal | A2, A8, A9, A10 |
| Full-Duplex-Bench | Turn-taking, backchanneling, interruption handling | A1, A2, Fisher |
| Service-Duplex-Bench | Role adherence in service conversations | A1 |
| Game-Time | Temporal dynamics (speed, pauses, timing) | A6, A8, A9 |

### Novel Steerability Metrics

| Metric | What It Measures | How |
|--------|-----------------|-----|
| Edit Latency | Frames until model output changes after steering instruction | Style classifier on per-frame embeddings |
| Edit Retention | Turns the edit persists (through topic changes, interruptions) | Style classifier consistency across turns |
| Composition Independence | Cross-talk between dimensions when one changes | Per-dimension classifiers, measure drift |
| Control Leakage | Voice identity shift when style changes | WavLM speaker sim before/after edit |

### Steerability evaluations per category:

| Dimension | Evaluation Method |
|-----------|------------------|
| Tone (A3) | LLM judge on transcript + prosody classifier on audio |
| Persona (A4) | LLM judge on vocabulary/character consistency |
| Speed (A6) | Measure WPM of generated audio vs target WPM |
| Length (A6) | Word count per assistant turn vs target |
| Accent (A5) | Accent classifier on generated audio |
| Emotional adaptation (A7) | LLM judge on appropriateness of response given user emotion |
| Graceful failure (A10) | LLM judge on recovery quality |
| Dynamic steering (A9) | Edit latency + edit retention + composition independence |

### Ablations

| Ablation | Question |
|----------|----------|
| Static-only data vs static+dynamic | Does dynamic steering data improve control? |
| With vs without counterfactual pairs | Do matched pairs improve composition independence? |
| With vs without Fisher | What does real data add? |
| Data scale per dimension | Scaling law: hours needed per steerability dimension |
| Held-out tones/personas | Can the model generalize to unseen attributes? |
| Held-out combinations | Train on (sarcastic+fast), (empathetic+slow), test on (sarcastic+slow) |
| Graceful failure data ratio | How much failure data before the model learns to say "I don't know"? |

### Baselines
- Moshi 7B (base, no finetuning)
- PersonaPlex 7B (NVIDIA)
- Gemini Live (commercial)
- GPT-4o Realtime (commercial)
- Qwen-2.5-Omni

---

## Data Scale Targets

| Category | Pilot | Full | Est. Hours |
|----------|-------|------|-----------|
| A1: Customer Service | 60 | ~40K | ~650h |
| A2: QA/Assistant | 60 | ~30K | ~300h |
| A3: Tone-Controlled | 60 | ~12K | ~200h |
| A4: Persona-Controlled | 60 | ~12K | ~200h |
| A5: Style & Accent | 40 | ~8K | ~120h |
| A6: Speed & Length | 40 | ~4K | ~60h |
| A7: Emotional Adaptation | 40 | ~8K | ~120h |
| A8: Edge Cases | 20 | ~2K | ~30h |
| A9: Dynamic Steering | 200 | ~40K | ~650h |
| A10: Graceful Failure | 40 | ~5K | ~80h |
| Counterfactual Pairs | 100 | ~10K | ~160h |
| Long-Form (10-15 turns) | 40 | ~8K | ~130h |
| Fisher (real, annotated) | — | 7,303 | ~2,000h |
| **Total** | **~800** | **~179K+** | **~4,700h** |

---

## Known Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Qwen3-TTS instruct doesn't change style enough | Counterfactual pairs look identical | Test early; fall back to sox/praat post-processing for speed; use accent-matched refs for accent |
| ~~Qwen3-TTS ref_audio + instruct don't work simultaneously~~ | ~~RESOLVED~~ | Dual-model pipeline: CustomVoice for assistant (preset + instruct), Base for user (ref_audio cloning) |
| Model ignores user steering instructions | Core contribution fails | Increase dynamic steering data %; ensure steering turns are diverse; add steering-specific examples to A8 |
| Voice identity drifts when style changes | Control leakage | Measure speaker sim in quality filter; use same ref_audio consistently; consider DPO on voice stability |
| Compositional generalization doesn't emerge | Weaker claim | Still report what composes; analyze failure modes (publishable negative result) |
| 100s max duration truncates long conversations | Multi-turn coherence suffers | Chunk with repeated voice prompt at start of each chunk; target 60-80s for most convos |
| LLM generates repetitive/unnatural steering instructions | Dynamic data feels scripted | Use diverse LLM prompts; vary steering language; include subtle/implicit steering |
| Model always obeys OR always refuses steering requests | Wrong boundary calibration | Mix mutable-granted (A9) + immutable-refused (A9 refused + A10) data; test on boundary cases |
| Male/female voices have different style-following quality | Gender bias in steerability | Equal training data per voice; measure steerability metrics per voice separately |
| Model becomes too obedient (follows any instruction) | Safety concern | A10 graceful failure data teaches appropriate refusal; safety boundaries defined |
| Fisher data dominates and dilutes steering signal | Poor steerability despite good naturalness | Control data mix carefully; Fisher at ~40% max; evaluate on steerability metrics separately |

---

## Pilot Plan (Week 1-2)

### Week 1:
- Day 1: Test Qwen3-TTS (ref_audio + instruct simultaneously, style changes, accent control)
- Day 1-2: Build + test LLM meta-prompts for all data types (standard, dynamic, counterfactual)
- Day 2-3: Generate 800 pilot transcripts (mix of all types)
- Day 3-4: TTS synthesis pipeline + quality filter
- Day 5: 2-channel assembly

### Week 2:
- Day 1: Format for moshi-finetune, run annotate.py
- Day 2-3: Train LoRA on pilot data
- Day 4: Evaluate:
  - Does model follow user-requested tone? (play sarcastic prompt, check output)
  - Does model adapt when user says "slow down"? (dynamic steering test)
  - Does voice stay stable across style changes? (speaker sim)
  - Does model say "I don't know" appropriately? (graceful failure test)
- Day 5: Iterate on data mix based on results

---

## References

- PersonaPlex: ICASSP 2026, NVIDIA (Roy et al.)
- Moshi: Defossez et al., 2024
- moshi-finetune: https://github.com/kyutai-labs/moshi-finetune
- Qwen3-TTS: https://github.com/QwenLM/Qwen3-TTS
- Audio MultiChallenge: arXiv 2512.14865
- Full-Duplex-Bench: arXiv 2503.04721
- Game-Time: arXiv 2509.26388
- UltraVoice: arXiv 2510.22588
