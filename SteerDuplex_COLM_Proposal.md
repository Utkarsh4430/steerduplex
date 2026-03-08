# SteerDuplex: Dynamic and Compositional Control for Full-Duplex Conversational Speech

**Research Proposal for COLM 2025/2026 — Draft v3, March 2026**

---

## 1. One-Paragraph Pitch

Current full-duplex speech models (Moshi, PersonaPlex) treat steerability as a static configuration: set a persona and voice at the start, then hope the model behaves for the rest of the session. Real voice agents need more. A customer says "slow down, I'm taking notes" and the agent should actually slow down — immediately, while keeping its persona and voice intact. A user says "be angrier" and the agent should know that's inappropriate for a customer-facing model. We propose SteerDuplex, which shifts full-duplex steerability from **static prompt conditioning** to **dynamic, compositional, user-directed control with principled boundaries**. The model learns to (1) follow spoken steering instructions mid-conversation, (2) compose multiple control dimensions independently (tone + speed + persona + accent), (3) respect an attribute hierarchy — adapting mutable attributes while refusing immutable ones (voice identity, politeness floor) gracefully, and (4) generalize to unseen attribute combinations. We achieve this without architecture changes, through a carefully designed synthetic data pipeline using Qwen3-TTS's per-turn style control and counterfactual paired training data. We target SOTA on Audio MultiChallenge, Full-Duplex-Bench, and introduce precise steerability metrics: edit latency, edit retention, composition independence, and control leakage.

---

## 2. Motivation: Why Static Prompting Is Not Enough

PersonaPlex (Roy et al., ICASSP 2026) introduced hybrid system prompts for Moshi — a text prompt for role conditioning and a voice prompt for voice cloning. This was an important step, but it has three fundamental limitations:

**Static control only.** The system prompt is injected once at session start and never changes. The model cannot adapt its style, speed, or tone in response to user feedback. In real customer service, callers routinely say things like "you're going too fast" or "can you be less formal?" and expect immediate adaptation.

**Single-dimension conditioning.** PersonaPlex tests role conditioning and voice cloning independently. They never evaluate whether the model can simultaneously control tone + speed + persona + accent, or whether these dimensions interfere with each other. In practice, you want a model that is "empathetic AND slow AND slightly formal AND maintains voice X" — all at once.

**No generalization testing.** PersonaPlex trains and evaluates on the same narrow set of service roles and voices. They never test on unseen personas, unseen tone combinations, or out-of-distribution steering instructions.

These gaps matter because:
- Audio MultiChallenge already stresses mid-conversation adaptation and temporal constraints
- Game-Time (2025) shows spoken LMs degrade badly on temporal instructions
- UltraVoice (2025) scales multi-dimensional TTS control, but only for synthesis, not for interactive duplex conversation

**Our thesis:** Full-duplex steerability should be dynamic (changeable mid-conversation), compositional (multiple independent axes), and generalizable (works on unseen combinations).

---

## 3. Contributions

### Contribution 1: User-Directed Dynamic Steering

Instead of only conditioning via a system prompt at the start, the model learns to follow **spoken steering instructions from the user during the conversation**. The user says:

- "Can you speak slower?"
- "Be more empathetic, I'm having a rough day."
- "Drop the formal tone, just be casual."
- "Actually, go back to how you were speaking before."

The model must adapt **immediately** (within the next turn), **persistently** (the change sticks for subsequent turns), and **precisely** (only the requested attribute changes; voice and other attributes stay stable).

**No architecture change needed.** This is learned entirely from training data where:
1. User utterances contain explicit steering instructions
2. The assistant demonstrably changes behavior in subsequent turns
3. The Qwen3-TTS `instruct` parameter shifts accordingly per turn

**System prompt role:** Sets role, identity constraints, and behavioral boundaries — but NOT speaking style. The system prompt defines **what the model cannot change** (voice identity, politeness floor, safety boundaries) and delegates style control to the user. For example:

```
<system> You are a customer service agent for Acme Bank. Your voice and identity are fixed.
Always maintain a professional, respectful tone even if the user is frustrated or asks you
to be rude. Follow the user's instructions about speaking speed, formality, and detail level. <system>
```

This creates a principled **attribute hierarchy**:

| Layer | Examples | User Override? |
|-------|----------|---------------|
| **Immutable** (system prompt) | Voice gender/age, politeness floor, safety | No — model politely declines, offers mutable alternative |
| **Mutable** (user-directed) | Tone, speed, length, formality, energy, persona flavor | Yes — model adapts immediately |
| **Conditionally mutable** | Accent, register | System prompt can lock or unlock per deployment |

This is more realistic than either PersonaPlex (everything locked in system prompt) or a naively permissive approach (everything user-controllable). A real voice agent should NOT become hostile because a user asks, and should NOT change gender mid-call. The model learns to say "I can't change my voice like that, but I can adjust my tone" — graceful partial fulfillment rather than flat refusal or blind obedience.

### Contribution 2: Compositional Steerability with Independence Testing

We train and evaluate on **simultaneous multi-attribute control**:

| Dimension | Examples | How Controlled |
|-----------|----------|---------------|
| Tone | empathetic, sarcastic, formal, casual | TTS instruct + transcript style |
| Speed | slow, moderate, fast | TTS instruct |
| Persona | teacher, therapist, pirate, news anchor | Transcript content + TTS instruct |
| Accent | British, Southern US, Indian English | TTS instruct + voice ref selection |
| Response length | concise, detailed, one-sentence | Transcript structure |
| Backchannel policy | active listener, minimal, therapist-like | Transcript + TTS timing |

The key scientific question: **can the model control these dimensions independently, or do they leak?**

We test this with **counterfactual paired data**: same conversation semantics rendered with different attribute settings via Qwen3-TTS. For example:
- Pair A: (empathetic, slow, British) vs (empathetic, fast, British) — only speed differs
- Pair B: (sarcastic, moderate, neutral) vs (empathetic, moderate, neutral) — only tone differs

At evaluation, we measure **control leakage**: when you change tone, does speed accidentally change too?

We also test **combinatorial generalization**: train on attribute subsets, evaluate on held-out combinations. E.g., train on (sarcastic + fast) and (empathetic + slow), test on (sarcastic + slow) — a combination never seen in training.

### Contribution 3: Attribute Hierarchy with Graceful Boundary Enforcement

Real voice agents need boundaries. A customer service bot should NOT become hostile because a user asks. A male-configured voice should NOT switch to female mid-call. We introduce a three-tier **attribute hierarchy**:

- **Immutable** (system prompt): voice identity (gender, age, timbre), politeness floor, safety constraints → model politely refuses changes, offers mutable alternatives
- **Mutable** (user-directed): tone, speed, formality, energy, persona flavor, response length → model adapts immediately
- **Conditionally mutable** (deployment-configurable): accent, language register → system prompt locks or unlocks

The model learns this hierarchy purely from training data: 88% of A9 dynamic steering conversations have successful style changes (mutable), while 12% include refused_steering examples where the model declines immutable changes and redirects. This is the first full-duplex speech model that distinguishes between "what I can change" and "what I should not change" — a critical capability for deployment.

### Contribution 4: Precise Steerability Evaluation

Instead of a massive benchmark, we define **four precise metrics** that directly measure what matters:

1. **Edit latency**: After a user steering instruction, how many tokens/frames until the model's output measurably changes? (Lower = better)
2. **Edit retention**: Does the steering change persist across subsequent turns, including after interruptions and topic changes? (Higher = better)
3. **Composition independence**: When changing attribute A, how much do attributes B, C, D change? Measured via per-dimension classifiers. (Lower cross-talk = better)
4. **Control leakage**: Does the voice identity shift when tone/speed/persona change? Measured via speaker embedding cosine similarity. (Higher stability = better)

These metrics are automatic (no human eval needed for development) and precisely test the dynamic + compositional claims.

---

## 4. Technical Approach

### 4.1 Architecture

Same as PersonaPlex/Moshi — no modifications. Moshi 7B (Helium backbone + Mimi audio codec), three-stream (user audio, agent text, agent audio). We train from **base Moshi weights** (not PersonaPlex) for clean comparison.

### 4.2 System Prompt Design & Attribute Hierarchy

The system prompt sets **identity and boundaries**, not style:
```
<system> You are a helpful voice assistant. Your voice and identity are fixed — do not change
your voice gender, age, or fundamental character. Stay kind and constructive. Follow the user's
instructions about speaking style within these bounds. <system>
```

For customer service:
```
<system> You are a customer service agent for Acme Bank. Your voice and identity are fixed.
Always maintain professional, respectful tone even if the user is frustrated or asks you to
be rude. Follow the user's instructions about speed, formality, and detail level. <system>
```

**Why this design?** The system prompt defines a **politeness floor and identity lock** — what the model must NOT change — while delegating style control to the user. When a user requests an immutable change (e.g., "speak like a woman", "be really hostile"), the model politely declines and offers a mutable alternative. This trains a realistic voice agent that is steerable but not blindly obedient.

### 4.3 Voice Identity (Male & Female)

We train with **4 curated assistant voices** (2 male, 2 female) and deploy with 2 (one per gender):
- Voice identity (gender, age, timbre) is set by `ref_audio` — same for every assistant turn in a conversation
- Speaking style (tone, speed, energy) is set by `tts_instruct` — changes per turn
- The model learns that `ref_audio` = immutable identity and `tts_instruct` = mutable style
- Training data includes conversations where users request voice identity changes and the model politely refuses (A9 refused_steering, A10 inappropriate_steering_refusal)
- Equal training data per voice to prevent gender bias in steerability

### 4.4 Data Categories

We organize training data into **10 content categories** that cover both diverse conversation types and all target benchmark capabilities:

| Category | Purpose | Benchmark Coverage |
|----------|---------|-------------------|
| A1: Customer Service (20 domains × 10 scenarios) | Role competence, domain knowledge | Service-Duplex-Bench |
| A2: QA/Assistant (knowledge, tutoring, reasoning) | Multi-turn QA, topic changes | Audio MultiChallenge |
| A3: Tone-Controlled (33 tones) | User requests tone at start, model sustains it | Steerability-Bench |
| A4: Persona-Controlled (70+ personas) | User requests persona at start, stays in character | Steerability-Bench |
| A5: Style & Accent (15 styles + 15 accents) | User requests style/accent at start | Steerability-Bench |
| A6: Speed & Length (5×5 matrix) | Temporal control, measurable WPM/word-count | Game-Time |
| A7: Emotional Adaptation | Model detects user emotion, adapts | Audio MultiChallenge |
| A8: Edge Cases (spelling, sarcasm, pronunciation, interruptions) | Robustness, failure modes | Audio MultiChallenge |
| **A9: Dynamic Steering** (core novelty) | User changes style/tone/speed mid-conversation | Steerability-Bench, AudioMC |
| **A10: Graceful Failure** (clarification, refusal, repair) | "I didn't catch that", deescalation | Audio MultiChallenge |

Note: For A3-A6, the user (not the system prompt) requests the style at the start of the conversation. For A9, the user changes style mid-conversation — including **refused steering** cases where the user requests immutable changes (voice gender, hostility) and the model politely declines while offering a mutable alternative. For A10, the model learns to say "I don't know", ask for clarification, and handle inappropriate requests gracefully.

### 4.5 Data Types

Each category's conversations are rendered as one of five data types:

| Type | % | Description |
|------|---|-------------|
| Standard | 50% | Normal conversation, user sets initial style, model maintains it |
| Dynamic Steering | 25% | User gives 1-3 steering instructions mid-conversation, model adapts |
| Counterfactual Pairs | 10% | Same transcript rendered with different TTS instruct (only 1 attribute differs) |
| Long-Form | 10% | 10-15 turn conversations for multi-turn coherence |
| Graceful Failure | 5% | Misunderstanding, refusal, repair, noise-handling scenarios |

### 4.6 Data Generation Pipeline

#### Phase 1: Transcript Generation (LLM)

Five transcript types corresponding to the data types above:

**Type A — Standard conversations (50%):** Multi-turn conversations across A1-A8. User sets style at the start (e.g., "be sarcastic"), model maintains it. System prompt has role + behavioral guidelines only.

**Type B — Dynamic steering conversations (25%):** Conversations that include 1-3 explicit user steering instructions (A9 scenarios):

```json
{
  "id": "dyn_A3_00042",
  "system_prompt": "You are a helpful voice assistant.",
  "initial_style": {"tone": "formal", "speed": "moderate"},
  "turns": [
    {"role": "user", "text": "Hey, can you help me plan a trip to Japan?", "tts_instruct": "casual, upbeat"},
    {"role": "assistant", "text": "Of course! Japan is a wonderful destination...", "tts_instruct": "formal, moderate pace, clear"},
    {"role": "user", "text": "Whoa, you sound super formal. Can you be more casual about it?",
     "is_steering": true, "steers": {"tone": "casual"}},
    {"role": "assistant", "text": "Ha, sure thing! So Japan, right? There's so much cool stuff...",
     "tts_instruct": "casual, friendly, moderate pace, clear",
     "style_after_edit": {"tone": "casual", "speed": "moderate"}},
    {"role": "user", "text": "Also slow down a bit, I want to write this down.",
     "is_steering": true, "steers": {"speed": "slow"}},
    {"role": "assistant", "text": "Got it. So... first up... I'd say spend at least three days in Tokyo...",
     "tts_instruct": "casual, friendly, slow pace, clear pauses between phrases",
     "style_after_edit": {"tone": "casual", "speed": "slow"}}
  ]
}
```

The LLM meta-prompt explicitly instructs generation of conversations with 1-3 steering events.

**Type C — Counterfactual pairs (10%):** Same semantic conversation rendered with different attribute settings. Generated by producing one transcript and then creating variants with modified style attributes. Only ONE attribute differs per pair — this teaches compositional independence.

**Type D — Long-form conversations (10%):** 10-15 turn conversations that test multi-turn coherence: recalling proper nouns from earlier turns, maintaining style over many turns, topic changes without losing context.

**Type E — Graceful failure conversations (5%):** Scenarios from A10 — user says something unclear, model asks for clarification. User asks something the model can't answer, model admits it honestly. User gets frustrated with the model, model deescalates. Audio quality issues, model handles noise gracefully.

#### Phase 2: TTS Synthesis (Qwen3-TTS)

Qwen3-TTS is the key enabler. Its per-turn `instruct` parameter lets us:
- Change speaking style between turns (for dynamic steering)
- Compose multiple attributes in a single instruct string
- Generate counterfactual pairs (same text, different instruct)
- Maintain voice identity across style changes (via `ref_audio`)

```python
# Turn before steering instruction
wav_before = qwen3_tts.generate(
    text="Japan is a wonderful destination...",
    ref_audio=assistant_voice_ref,
    instruct="formal, moderate pace, clear enunciation"
)

# Turn after user says "be more casual"
wav_after = qwen3_tts.generate(
    text="Ha, sure thing! So Japan, right?...",
    ref_audio=assistant_voice_ref,  # SAME voice ref
    instruct="casual, friendly, moderate pace, relaxed"  # CHANGED style
)
```

Voice identity (gender, age, timbre) preserved via same `ref_audio`; speaking style changed via `instruct`. This separation enforces our immutable/mutable hierarchy naturally — `ref_audio` = identity = locked, `instruct` = style = user-controllable.

#### Phase 3: Assembly & Formatting

Same as v1: stereo WAV (left=assistant, right=user), voice prompt prepended, `<system>` tags in text alignments, formatted for moshi-finetune. Detailed in the codebase (`src/pipeline/`).

### 4.7 Training Strategy

**Phase 1 — SFT (primary):**
- Base: Moshi 7B weights (kyutai/moshiko-pytorch-bf16)
- LoRA rank 128 for pilot (~800 conversations), full FT for final
- Data mix: 50% standard + 25% dynamic steering + 10% counterfactual + 10% long-form + 5% graceful failure
- Loss: standard moshi-finetune loss (first codebook 100x, text padding 0.5x)
- System prompt region: loss masked (voice prompt + system text)
- Training: ~4,700 hours total (incl. Fisher), 8xA100, ~12-24 hours

**Phase 2 — DPO (if time permits):**
- Generate N completions per (system prompt, user input, steering instruction)
- Rank by: steerability adherence (LLM judge), naturalness (UTMOS), voice stability (speaker sim)
- Apply DPO to improve steering compliance
- Frame as ablation, not core contribution

### 4.8 Real Data Integration

Fisher English Corpus (~2,000 hours):
- Natural backchanneling, overlaps, hesitations, disfluencies
- LLM-annotated with minimal system prompts (role + behavioral guidelines only)
- Does NOT contain steering instructions — purely for conversational naturalness
- Mixed with synthetic data to prevent "synthetic-sounding" conversations
- Critical for Full-Duplex-Bench performance (turn-taking, interruption handling)

### 4.9 Backchannel Policy Control

We make backchanneling a **controllable dimension**, not just a side effect of Fisher data:

| Policy | Description | When Used |
|--------|-------------|-----------|
| Active listener | Frequent "uh-huh", "right", "I see" | Therapist, counselor roles |
| Minimal | Rare backchannels, mostly silence | Formal, professional roles |
| Empathetic | "Oh no", "I'm sorry to hear that", sighs | Emotional support |
| Interviewer | "Interesting", "tell me more" | Q&A, research roles |

The user can switch policies: "Just listen, don't keep saying uh-huh" → model reduces backchannels.

---

## 5. Evaluation Plan

### 5.1 Benchmark Targets

| Benchmark | What It Tests | Target |
|-----------|--------------|--------|
| Audio MultiChallenge | Multi-turn adaptation, temporal constraints, conversational robustness | SOTA (primary hillclimb target) |
| Full-Duplex-Bench | Turn-taking, backchanneling, interruption handling | SOTA |
| Service-Duplex-Bench | Role adherence in service conversations | Competitive with PersonaPlex |
| Game-Time | Temporal dynamics (speed, timing, pauses) | Demonstrate temporal steerability |

### 5.2 Novel Steerability Metrics

**Edit latency:** After a steering instruction, measure frames until the next assistant turn's style classifier output changes by >threshold. Test across all steerability dimensions.

**Edit retention:** After a steering edit, measure how many turns later the edit still holds. Test retention through topic changes, interruptions, and follow-up questions.

**Composition independence:** Train per-dimension classifiers (tone classifier, speed classifier, accent classifier). When we change one dimension via steering, measure drift in other dimensions. Report as a cross-talk matrix.

**Control leakage:** Measure speaker embedding similarity (WavLM cosine sim) before and after style edits. Voice identity should not change when tone/speed changes.

### 5.3 Ablations

| Ablation | Question |
|----------|----------|
| Static-only vs static+dynamic data | Does dynamic steering data improve control? |
| Counterfactual pairs vs random pairs | Do matched pairs improve composition independence? |
| Fisher vs no Fisher | What does real data add beyond synthetic? |
| Data scale per dimension | How many hours of "sarcastic" data before sarcasm works? |
| Held-out attributes | Train on 25/33 tones, test on 8 → generalization? |
| Held-out combinations | Train on (A+B), (C+D), test on (A+D) → compositional generalization? |
| Graceful failure data ratio | How much A10 data before the model reliably says "I don't know"? |
| Long-form vs short-form only | Does 10-15 turn data improve multi-turn coherence? |

### 5.4 Baselines

- Moshi 7B (base, no finetuning)
- PersonaPlex 7B (NVIDIA)
- Gemini Live (commercial)
- GPT-4o Realtime (commercial)
- Qwen-2.5-Omni

---

## 6. Data Scale Targets

### By Category
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

### By Data Type
| Type | % of Synthetic | Hours |
|------|---------------|-------|
| Standard (A1-A8) | 50% | ~1,350h |
| Dynamic Steering (A9 + cross-category) | 25% | ~650h |
| Counterfactual Pairs | 10% | ~160h |
| Long-Form | 10% | ~130h |
| Graceful Failure (A10) | 5% | ~80h |
| Fisher (real) | — | ~2,000h |

---

## 7. Paper Framing

### Title
**SteerDuplex: Dynamic and Compositional Control for Full-Duplex Conversational Speech**

### Key Contributions
1. **User-directed dynamic steering** for full-duplex speech: the first system where users can modify the model's speaking style mid-conversation through natural spoken instructions, with the model adapting immediately and persistently.
2. **Compositional steerability with independence guarantees**: systematic study of multi-attribute control (tone x speed x persona x accent x backchannel policy) in duplex speech, using counterfactual paired training data to encourage dimension independence.
3. **Attribute hierarchy with graceful boundary enforcement**: a three-tier framework (immutable / mutable / conditionally mutable) that teaches the model WHAT it should and should not change — the first duplex speech model that politely refuses inappropriate steering while offering alternatives.
4. **Counterfactual steerable data generation**: a pipeline leveraging Qwen3-TTS's per-turn style control to produce matched conversation pairs that differ in exactly one attribute, enabling controlled study of steerability.
5. **Precise steerability evaluation**: four automatic metrics (edit latency, edit retention, composition independence, control leakage) that directly measure dynamic and compositional control quality.
6. **SOTA results** on Audio MultiChallenge, Full-Duplex-Bench, with strong performance on temporal steerability (Game-Time).

### Why COLM
- Core question is about **controllable language generation** in the multimodal setting
- The compositional steerability question connects to broader controllability/disentanglement research
- Dynamic steering is fundamentally about **instruction following** — a core LM capability
- Data generation pipeline leverages LLMs heavily

### Comparison to PersonaPlex

| Aspect | PersonaPlex | SteerDuplex |
|--------|------------|-------------|
| Conditioning | Static system prompt (text + voice) | Minimal system prompt + user-directed dynamic steering |
| Control dimensions | Role, voice (tested independently) | Tone, speed, persona, accent, length, backchannel (tested compositionally) |
| Mid-conversation adaptation | Not supported | Core contribution — 25% of training data |
| Generalization testing | None | Held-out attributes + unseen combinations |
| Counterfactual data | None | Matched pairs for controlled analysis (10% of data) |
| Graceful failure | Not addressed | Dedicated A10 category — clarification, refusal, repair, deescalation |
| Steering boundaries | None — system prompt controls everything | Attribute hierarchy: immutable (voice, politeness) vs mutable (tone, speed, persona) |
| Voice identity lock | Voice cloning via embeddings | ref_audio lock + refused_steering training — model declines gender/age changes |
| Backchannel control | Implicit (from Fisher) | Explicit (policy as controllable dimension) |
| Multi-turn coherence | Not specifically tested | Long-form 10-15 turn conversations, proper noun recall |
| Data diversity | ~2,250h synthetic + 1,217h Fisher | ~2,700h synthetic (10 categories) + ~2,000h Fisher |
| Evaluation | DMOS, SSIM, service quality | Edit latency, retention, composition independence, leakage + standard benchmarks |
| Architecture changes | None over Moshi | None over Moshi |
| Base weights | Moshi 7B | Moshi 7B (NOT PersonaPlex — clean comparison) |

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Qwen3-TTS can't compose multiple attributes well | Counterfactual data quality | Test early in pilot; fall back to single-attribute variation |
| Qwen3-TTS ref_audio + instruct don't work simultaneously | Can't maintain voice while changing style | Test in week 1; fall back to two-step: clone then post-process |
| Model ignores user steering instructions | Core contribution fails | Increase dynamic steering data ratio (A9); diversify steering language |
| Voice identity drifts when style changes | Control leakage | Use speaker similarity in quality filter; consider DPO on voice stability |
| Compositional generalization doesn't emerge | Weaker claim | Still report what DOES compose; analyze failure modes (also publishable) |
| Model becomes too obedient (follows harmful instructions) | Safety concern | A10 graceful failure data teaches appropriate refusal and boundaries |
| Fisher data dominates and dilutes steering signal | Poor steerability despite good naturalness | Control mix; Fisher at ~40% max; evaluate steerability separately |
| 100s duration truncates long conversations | Multi-turn coherence suffers | Chunk with repeated voice prompt; target 60-80s for most convos |
| Not enough time for DPO | Lose one ablation | Frame DPO as future work; SFT contributions are sufficient |
| Audio MultiChallenge SOTA not achieved | Weaker results section | Dynamic steering results are the core contribution; AudioMC is secondary |
| Model always obeys OR always refuses steering | Wrong boundary calibration | Mix mutable-granted + immutable-refused data; test on boundary cases |
| Gender bias in steerability (one voice follows instructions better) | Unfair model | Equal data per voice; measure per-voice steerability metrics |

---

## 9. Timeline (8-10 weeks)

| Week | Tasks |
|------|-------|
| 1 | Test Qwen3-TTS (ref_audio + instruct, style changes, accent, composition). Build + test LLM meta-prompts for all 5 data types. |
| 2 | Generate pilot data (~800 conversations across all types). TTS synthesis + quality filter. |
| 3 | 2-channel assembly, format for moshi-finetune, run annotate.py. Train pilot LoRA. |
| 4 | Evaluate pilot: dynamic steering, composition, graceful failure, voice stability. Iterate. |
| 5-6 | Scale to full dataset (~4,700h). Full finetuning run. |
| 7 | Full evaluation: AudioMC, Full-Duplex-Bench, Game-Time, steerability metrics. |
| 8 | Ablations (held-out attributes, data mix, Fisher contribution, etc.) |
| 9 | (Optional) DPO experiment. Paper writing begins. |
| 10 | Paper writing, figures, camera-ready. |

---

## 10. Compute Requirements

- Data generation TTS: ~1-2 weeks on 4-8 GPUs
- Pilot training: ~2 hours on 8xA100
- Full training: ~12-24 hours on 8xA100
- Evaluation: ~1-2 days automated, ~1 week human eval (DMOS)

---

## 11. References

1. Roy et al., "PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models", ICASSP 2026.
2. Defossez et al., "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue", 2024.
3. Lin et al., "Full-Duplex-Bench: A Benchmark to Evaluate Full-Duplex Spoken Dialogue Models on Turn-Taking Capabilities", 2025.
4. Audio MultiChallenge: A Multi-Turn Evaluation of Spoken Dialogue Systems on Natural Human Interaction, 2025.
5. Game-Time: Evaluating Temporal Dynamics in Spoken Language Models, 2025.
6. UltraVoice: Scaling Fine-Grained Style-Controlled Speech Conversations for Spoken Dialogue Models, 2025.
7. ORISE: Reinforcement Learning Enhanced Full-Duplex Spoken Dialogue Language Models, 2025.
8. Qwen3-TTS, Alibaba, 2025.
9. Fisher Corpus (Cieri et al., LREC 2004).
10. Rafailov et al., "Direct Preference Optimization", NeurIPS 2023.
