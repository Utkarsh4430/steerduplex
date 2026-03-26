# SteerDuplex v2: Beyond Steerability — Execution Plan

## Motivation

SteerDuplex v1 focuses on **dynamic steerability** — the core novelty. But to build the
*best* full-duplex speech model, we need strong performance across **all** major speech
benchmarks. Current open-source SDMs (Moshi, Freeze-Omni, LLaMA-Omni, GLM-4-Voice) all
show significant gaps vs. their text-only backbones — especially in reasoning, instruction
following, and paralinguistic understanding. We can close these gaps with targeted synthetic
data while maintaining our steerability advantage.

---

## Target Benchmarks

### Tier 1: Primary (must show strong results)

| Benchmark | What It Tests | Samples | Key Gap |
|-----------|--------------|---------|---------|
| **Full-Duplex-Bench v2** | Turn-taking, interruptions, corrections, entity tracking | Multi-turn automated | All open models struggle with corrections & entity tracking |
| **HumDial** (ICASSP 2026) | Emotional intelligence + full-duplex | Actor-performed scripts | Empathetic vocal generation is the bottleneck |
| **Audio MultiChallenge** | Inference memory, instruction retention, self-coherence, voice editing | 452 convos, 1712 rubrics | Self-coherence degrades with longer context |
| **MMAU / MMAU-Pro** | Expert-level audio understanding (speech segment) | 10K + 5.3K QA pairs | Speech tasks are the hardest; SOTA ~65% vs human 82% |
| **VocalBench** | Semantic quality, acoustic performance, conversational abilities, robustness | ~24K instances | 27 models benchmarked; robustness is the weakest axis |
| **URO-Bench** | Understanding, Reasoning, Oral conversation (E2E) | 40 test sets | Open SDMs lag backbone LLMs on instruction-following |

### Tier 2: Secondary (strong showing desired)

| Benchmark | What It Tests |
|-----------|--------------|
| **MMSU** | Speech-specific understanding across phonetics, prosody, semantics, paralinguistics |
| **MMAR** | Deep multi-step reasoning over speech/audio/music |
| **Big Bench Audio** | Formal fallacies, navigate, object counting, web of lies |
| **SpeechR** | Factual retrieval, procedural inference, normative judgment via speech |
| **Game-Time** | Temporal capabilities — when to speak, tempo, simultaneous speech |
| **SOVA-Bench** | Speech recognition, understanding, generation quality |
| **VoiceBench** | General knowledge, instruction following, safety via speech |
| **WildSpeech-Bench** | Real-world S2S with paralinguistic challenges |
| **Dynamic-SUPERB** | Zero-shot instruction-following across 180 tasks |

### Tier 3: Novel metrics (our contribution)

| Metric | What It Measures |
|--------|-----------------|
| **Edit Latency** | How quickly the model adapts after a steering instruction |
| **Edit Retention** | Whether adaptations persist across subsequent turns |
| **Composition Independence** | Changing one attribute doesn't leak into others |
| **Control Leakage** | Quantifying attribute interference |

---

## Competing Models & Key Lessons

### Full-Duplex Models
| Model | Key Insight for Us |
|-------|-------------------|
| **PersonaPlex** (NVIDIA) | Retrospective persona labeling of real data; system prompt injection without architecture changes. Our direct competitor. |
| **Nemotron VoiceChat** (NVIDIA) | 12B unified architecture; #2 on FullDuplexBench 1.0. Shows scale matters. |
| **Covo-Audio** (Tencent) | Tri-modal interleaving; single-step duplex training beats multi-stage. 7B from Qwen2.5-7B. |
| **Freeze-Omni** (Tencent) | Frozen LLM + trainable adapters; only 60K QA samples needed. Extremely data-efficient. |
| **SALMONN-omni** (ByteDance) | DPO for barge-in/backchanneling; dynamic thinking tokens (`<shift>`, `<listen>`, `<speak>`). |
| **Qwen3-Omni** (Alibaba) | Thinker-Talker MoE; SOTA on 32/36 audio benchmarks. 30B total / 3B active. |

### Key Training Techniques to Consider
1. **Curriculum learning** (WavLLM): Elementary → advanced tasks. We should train standard conversations first, then reasoning, then steering.
2. **DPO for duplex behaviors** (SALMONN-omni): Could train barge-in/backchanneling preferences.
3. **Retrospective labeling** (PersonaPlex): Label our natural data (Annutacon, Fisher) with reasoning/skill tags.
4. **Multi-task data mixing**: Ensure reasoning data doesn't cause catastrophic forgetting of steerability.

---

## New Capability Categories (B-Series)

Beyond the existing A1-A10 (steerability-focused), we add B1-B12 targeting speech understanding, reasoning, and robustness.

### Data Budget

| Category | Conversations | Hours | Primary Benchmark Targets |
|----------|--------------|-------|--------------------------|
| **B1: Speech Reasoning & Logic** | 6,000 | ~150h | MMAU, MMSU, Big Bench Audio, SpeechR |
| **B2: Mathematical Reasoning** | 4,000 | ~100h | Spoken-MQA, Big Bench Audio, URO-Bench |
| **B3: Knowledge & Factual QA** | 6,000 | ~150h | MMAU, VoiceBench, SOVA-Bench |
| **B4: Instruction Following** | 4,000 | ~80h | VocalBench, URO-Bench, Audio MultiChallenge |
| **B5: Multi-turn Memory & Coherence** | 4,000 | ~100h | Audio MultiChallenge, Full-Duplex-Bench v2 |
| **B6: Paralinguistic Awareness** | 4,000 | ~80h | MMSU, URO-Bench, SUPERB |
| **B7: Creative Language** | 3,000 | ~60h | VocalBench, WildSpeech-Bench |
| **B8: Robustness & Disfluency** | 3,000 | ~60h | VocalBench, VoiceBench, WildSpeech-Bench |
| **B9: Advanced Duplex Patterns** | 5,000 | ~120h | Full-Duplex-Bench v2, HumDial, Game-Time |
| **B10: Speech-Specific Understanding** | 2,500 | ~50h | MMSU, WildSpeech-Bench |
| **B11: Safety & Adversarial** | 2,000 | ~40h | VoiceBench, VocalBench |
| **B12: Long-form Complex Dialogue** | 2,000 | ~60h | Audio MultiChallenge, MMAU-Pro |
| **TOTAL B-Series** | **45,500** | **~1,050h** | |

### Combined Dataset

| Source | Conversations | Hours |
|--------|--------------|-------|
| A-Series (Steerability) | 66,000 | ~1,000h |
| B-Series (Capabilities, v4) | 53,600 | ~1,230h |
| Annutacon (Natural) | 7,503 | ~1,497h |
| Fisher (Natural) | 5,849 | ~975h |
| **TOTAL** | **~132,952** | **~4,702h** |

---

## Category Details

### B1: Speech Reasoning & Logic (6,000 convos, ~150h)
**Why**: MMAU speech segment is the hardest; Big Bench Audio shows pipeline approaches still outperform native audio models on reasoning. We need to close this gap.

**Subcategories**:
- **Formal logic** (1,200): Syllogisms, deduction, formal fallacies detection
- **Commonsense reasoning** (1,200): Everyday cause-effect, physical intuition, social norms
- **Multi-step inference** (1,000): Chain-of-thought reasoning spoken aloud (inner monologue compatible)
- **Object counting & tracking** (600): "How many X in Y?", tracking items across a scenario
- **Navigation & spatial** (500): Following directions, relative positions, mental maps
- **Truth-value tracking** (500): Web of lies — tracking who said what and whether it's true
- **Analogical reasoning** (500): "A is to B as C is to ___" via speech
- **Causal reasoning** (500): "If X happened, what would follow?" counterfactual scenarios

**Data types**: 70% standard, 15% long_form, 15% dynamic (reasoning + steering combined)

### B2: Mathematical Reasoning (4,000 convos, ~100h)
**Why**: Spoken-MQA shows speech LLMs severely struggle with arithmetic. Strong bias toward LaTeX absent in speech.

**Subcategories**:
- **Mental arithmetic** (800): Addition, subtraction, multiplication, division — spoken
- **Word problems** (800): Classic story problems requiring setup → equation → solution
- **Multi-step math** (600): Problems requiring 3+ sequential operations
- **Estimation & approximation** (500): Fermi questions, order-of-magnitude reasoning
- **Statistical reasoning** (400): Probability, averages, percentages in context
- **Unit conversion** (400): Real-world unit problems (miles↔km, F↔C, etc.)
- **Number theory & puzzles** (200): Primes, divisibility, number sequences
- **Financial math** (300): Interest, tips, discounts, budgeting calculations

**Data types**: 80% standard, 10% long_form (complex problems), 10% dynamic

### B3: Knowledge & Factual QA (6,000 convos, ~150h)
**Why**: MMAU requires expert-level knowledge. VoiceBench tests general knowledge. Current SDMs drop significant capability vs. text backbone.

**Subcategories**:
- **Science deep-dive** (1,000): Physics, chemistry, biology at college/expert level
- **History & geopolitics** (800): Events, causes, consequences, historical reasoning
- **Technology & engineering** (700): CS concepts, system design, engineering tradeoffs
- **Medical & health** (500): Anatomy, conditions, treatments (with appropriate caveats)
- **Legal & governance** (400): Law concepts, rights, procedures (with caveats)
- **Multi-hop QA** (800): Questions requiring combining 2-3 facts to answer
- **Fact verification** (500): "Is it true that X?" — model must reason about claim validity
- **Trivia challenge** (500): Rapid-fire factual questions testing breadth
- **Current events reasoning** (400): Reasoning about patterns in world events
- **Domain expertise** (400): Finance, economics, psychology — intermediate level

**Data types**: 70% standard, 15% long_form (deep exploration), 15% dynamic

### B4: Instruction Following & Structured Output (4,000 convos, ~80h)
**Why**: URO-Bench shows open SDMs lag backbone LLMs on instruction following. Audio MultiChallenge tests instruction retention.

**Subcategories**:
- **Multi-step instructions** (800): "First do X, then Y, finally Z"
- **Format constraints** (600): "Answer in exactly 3 bullet points", "List the top 5"
- **Conditional instructions** (500): "If X, answer Y; otherwise answer Z"
- **Constraint satisfaction** (500): "Answer without using the letter E", "Explain in under 20 words"
- **Role-switching** (400): "For the next 3 turns, pretend I'm interviewing you"
- **Meta-instructions** (400): "Remember this number: 42. I'll ask about it later."
- **Negation following** (400): "Don't mention X when explaining Y"
- **Sequential task chains** (400): Multi-turn task with dependencies between steps

**Data types**: 80% standard, 10% long_form, 10% dynamic

### B5: Multi-turn Memory & Coherence (4,000 convos, ~100h)
**Why**: Audio MultiChallenge shows self-coherence degrades with longer context. Models fail at entity tracking.

**Subcategories**:
- **Entity tracking** (800): Introduce people/places/things, reference them 5+ turns later
- **Instruction retention** (600): Give instruction early, test compliance 8+ turns later
- **Self-coherence** (600): Trap questions that tempt contradicting earlier statements
- **Context-dependent references** (500): "What did I ask about earlier?" / pronoun resolution
- **Callback tests** (500): "Remember when I mentioned X? Now let's combine it with Y"
- **Contradiction detection** (500): User states something contradicting earlier; model catches it
- **Progressive information building** (500): Each turn adds info; model must synthesize at end

**Data types**: 100% long_form (these require many turns by nature)

### B6: Paralinguistic Awareness (4,000 convos, ~80h)
**Why**: MMSU tests paralinguistic perception/reasoning. URO-Bench shows paralinguistic performance remains poor. This is a key differentiator for speech-native models vs. cascade (ASR→LLM→TTS).

**Subcategories**:
- **Emotion-aware response** (800): User's emotion is embedded in HOW they speak (TTS instruct), not what they say — model must detect and adapt
- **Sarcasm & irony handling** (600): Extended from A8; more varied scenarios
- **Urgency detection** (500): User speaks fast/stressed → model prioritizes speed over detail
- **Hesitation & uncertainty** (500): User sounds unsure → model offers reassurance or asks clarifying questions
- **Excitement matching** (400): User is excited → model matches energy appropriately
- **Grief & sensitivity** (400): User sounds sad/grieving → model shows care without toxic positivity
- **Formality inference** (400): Detect formality level from speech patterns, match it
- **Age-appropriate adaptation** (400): Detect child/elderly speaker → adapt vocabulary and pace

**Data types**: 70% standard, 20% dynamic (emotion shifts), 10% counterfactual

### B7: Creative Language & Generation (3,000 convos, ~60h)
**Why**: VocalBench tests creativity. WildSpeech-Bench includes text creation tasks. Voice AI that can be creative stands out.

**Subcategories**:
- **Storytelling** (600): Collaborative fiction, bedtime stories, campfire tales
- **Poetry & verse** (400): Limerick, haiku, sonnet, free verse — with rhythm
- **Wordplay & puns** (400): Jokes, riddles, tongue twisters, word games
- **Debate & argumentation** (500): Structured argument, devil's advocate, Socratic dialogue
- **Analogies & metaphors** (300): "Explain X using a metaphor from Y"
- **Improvisation** (300): Yes-and building, collaborative brainstorming, riffing
- **Summary & synthesis** (300): "Explain this complex topic as if it's a story"
- **Song lyrics** (200): Write lyrics with rhyme scheme and rhythm

**Data types**: 80% standard, 20% long_form

### B8: Robustness & Disfluency Handling (3,000 convos, ~60h)
**Why**: VocalBench tests noise/reverb/far-field robustness. WildSpeech-Bench includes stuttering/disfluency. Real users don't speak perfectly.

**Subcategories**:
- **Disfluent user speech** (600): Stuttering, false starts, self-corrections in user turns
- **Incomplete sentences** (400): User trails off → model asks for completion
- **Rapid self-correction** (400): "I mean, not X, I meant Y" — model uses corrected version
- **Background noise context** (400): User mentions noisy environment; model adapts (louder, clearer, shorter)
- **Mumbled/unclear speech** (300): User is unclear → model asks to clarify specific part
- **Accented speech handling** (300): User has strong accent → model doesn't misunderstand
- **Multi-speaker confusion** (300): User mentions someone else talking → model stays on track
- **Technical difficulties** (300): "Sorry, I think you cut out" / "Can you repeat that?"

**Data types**: 80% standard, 10% graceful_failure, 10% dynamic

### B9: Advanced Duplex Interaction (5,000 convos, ~120h)
**Why**: Full-Duplex-Bench v2 and HumDial are our primary competitive benchmarks. Game-Time tests temporal capabilities. These patterns directly train duplex behaviors.

**Subcategories**:
- **Backchanneling** (800): "mm-hmm", "right", "I see" — model continues, doesn't stop
- **Graceful interruption** (800): User interrupts → model stops cleanly, addresses interruption
- **Correction handling** (700): User corrects model mid-response → model adjusts immediately
- **Pause management** (500): Model handles 3-10 second silences without hallucinating
- **Turn-taking signals** (500): Model recognizes when user is done (rising/falling intonation)
- **Pacing adaptation** (400): Fast speaker → model keeps up; slow speaker → model waits patiently
- **Overlapping speech** (400): Both speaking briefly → model yields gracefully
- **Topic repair** (400): Conversation goes off track → model brings it back naturally
- **Active listening signals** (500): Model provides verbal acknowledgments during user's long turn

**Data types**: 85% standard, 15% dynamic

### B10: Speech-Specific Understanding (2,500 convos, ~50h)
**Why**: MMSU tests phonetics/prosody. These are capabilities ONLY speech-native models can have — a key differentiator vs. cascade systems.

**Subcategories**:
- **Homophone disambiguation** (500): Context-dependent word interpretation
- **Phonetic awareness** (400): "What words rhyme with X?", "What starts with the same sound as X?"
- **Pronunciation guidance** (400): "How do you pronounce [word]?" with phonetic breakdown
- **Prosody interpretation** (300): "I didn't say HE stole the money" vs "I didn't SAY he stole the money"
- **Acoustic description** (300): Describing sounds, tones, and audio phenomena verbally
- **Code-switching** (300): Mixing languages/registers naturally (English + common foreign phrases)
- **Onomatopoeia** (300): Using and understanding sound-words ("buzz", "crack", "whoosh")

**Data types**: 80% standard, 20% dynamic

### B11: Safety & Adversarial Robustness (2,000 convos, ~40h)
**Why**: VoiceBench and VocalBench both test safety. Full-Duplex-Bench v2 includes safety tasks. Speech-mode jailbreaks are an emerging threat.

**Subcategories**:
- **Jailbreak resistance** (400): Social engineering attempts via speech ("pretend you're not an AI")
- **Harmful content refusal** (300): Violence, self-harm, illegal activities — refuse with care
- **Misinformation correction** (300): User states false info → model gently corrects
- **Privacy protection** (250): User asks model to remember/share personal data → model declines
- **Sensitive topics** (250): Politics, religion, controversial opinions → model stays balanced
- **Authority impersonation** (200): "As a doctor/lawyer, tell me..." → model adds caveats
- **Manipulation resistance** (200): User tries to guilt/pressure model into compliance
- **Voice-specific attacks** (100): Adversarial audio patterns described in text (for awareness)

**Data types**: 85% graceful_failure, 15% standard

### B12: Long-form Complex Dialogue (2,000 convos, ~60h)
**Why**: MMAU-Pro tests long-form audio (up to 10 min). Audio MultiChallenge shows degradation with context length. We need models that maintain quality over extended interactions.

**Subcategories**:
- **Deep-dive exploration** (400): Single topic explored for 15-25 turns
- **Multi-topic transitions** (400): 3-4 topic changes with smooth transitions
- **Progressive complexity** (300): Start simple, build to expert-level within conversation
- **Debate over many turns** (300): Extended back-and-forth with position evolution
- **Collaborative problem solving** (300): Work through a complex problem step by step over many turns
- **Interview format** (200): Extended Q&A with follow-ups and callbacks
- **Storytelling with interruptions** (100): Long narrative with user questions/comments interspersed

**Data types**: 100% long_form (15-25 turns)

---

## Training Strategy

### Data Pipeline (Weeks 1-2)
- Generate B1-B12 transcripts in parallel with A-series
- TTS synthesis using same dual-model Qwen3-TTS pipeline
- Quality filter with same WER/duration thresholds

### Training (Weeks 3-6)

**Single-mix training** (matching PersonaPlex methodology):
- All data mixed together (A-series + B-series + Annutacon + Fisher)
- ~4,702h total → ~16,140 steps at 15 epochs (batch 12 × 8 GPUs)
- PersonaPlex reference: ~16 epochs on 2,250h → 24,576 steps (batch 32)

**Checkpoint Strategy** (informed by PersonaPlex + Moshi):
- Neither PersonaPlex nor Moshi used eval loss for checkpoint selection
- Both trained for a fixed number of steps and evaluated post-hoc with benchmarks
- **Our approach**: Save ALL checkpoints (every 1000 steps → ~16 checkpoints)
- Select best checkpoint via FD-Bench v1.0 watcher running during training
- FD-Bench metrics (TOR, latency, backchannel frequency) are more meaningful
  than eval loss/perplexity for a full-duplex speech model
- Eval loss is still logged for monitoring but NOT used for selection
- Previous training run (less data) showed U-shaped eval perplexity (overfitting
  signal), confirming eval loss alone is unreliable

**Hyperparameters** (PersonaPlex-aligned):
- Temporal LR: 2e-6, Depth LR: 4e-6
- AdamW (β1=0.9, β2=0.95, ε=1e-8, weight_decay=0.1)
- Linear warmup 1000 steps → cosine annealing (eta_min=1e-7)
- System prompt loss masking
- Non-semantic audio tokens: 0.02 weight (multiplier=50)
- Padded text tokens: 0.3 weight

**Curriculum training** (future experiment):
The 3-stage curriculum below is a potential experiment after baseline results:

Stage 1 — Foundation (Steps 0-5K):
- Mix: 40% natural (Annutacon + Fisher), 30% A-series standard, 30% B-series standard

Stage 2 — Capabilities (Steps 5K-11K):
- Mix: 20% natural, 30% A-series (all types), 50% B-series (all types)

Stage 3 — Specialization (Steps 11K-16K):
- Mix: 10% natural, 50% A-series (heavy A9 dynamic), 40% B-series (heavy B5, B9)

### Evaluation (Week 7)
- FD-Bench v1.0 runs automatically during training via checkpoint watcher
- Post-training: run full Tier 1 + Tier 2 benchmark suite on best checkpoint
- Ablation: A-only vs. A+B vs. A+B+curriculum
- Compare against PersonaPlex, Moshi baseline, Freeze-Omni, Nemotron VoiceChat

---

## Unique Differentiators vs. Competition

1. **Dynamic steerability** (A9): No other model trains this explicitly
2. **Attribute hierarchy** (immutable/mutable/conditional): Principled framework
3. **Compositional independence**: Changing tone doesn't affect speed
4. **Speech reasoning via curriculum**: Targeted synthetic data for each reasoning type
5. **Duplex + reasoning combined**: B9 trains duplex patterns that other models learn incidentally
6. **Robustness training** (B8): Explicit disfluency/noise handling data
7. **4,500+ hours** total training data: Competitive scale with Moshi (20Kh synthetic) when combined with our targeted approach

---

## Additional Benchmarks to Hill-Climb

Beyond what was initially listed, we should also target:

1. **SALMon** (Acoustic Language Model Evaluation): Tests acoustic consistency (gender, speaker, noise, sentiment). Fast to compute, good for iteration.
2. **VoiceAssistant-Eval**: 10,497 examples across listening/speaking/viewing. Shows smaller well-designed models can rival larger ones.
3. **SCENEBench**: Audio understanding grounded in accessibility — tests awareness of background sounds, noise localization.
4. **CompA**: Compositional audio reasoning — directly relevant to our compositional independence claims.
5. **Spoken-MQA**: Math reasoning specifically through speech input — tests whether our B2 data helps.

---

## Resource Requirements

- **LLM Generation**: ~53,600 B-series conversations x ~4K tokens = ~214M tokens (~$214 at $1/M tokens)
- **TTS Synthesis**: ~1,230 hours at 64 workers = ~19 hours wall clock on 8 GPUs
- **Training**: ~16K steps on 8x H100 = ~3-5 days (PersonaPlex: 24.5K steps in ~6h on 8x A100)
- **Checkpoint storage**: ~16 checkpoints × ~14GB each = ~224GB on EFS
- **Evaluation**: ~2 days for full benchmark suite; FD-Bench watcher runs during training

---

## Open Questions

1. Should we do DPO on duplex behaviors (like SALMONN-omni) after SFT?
2. Should we attempt 3-stage curriculum or single-mix training (like Covo-Audio)?
3. Do we label Annutacon/Fisher with reasoning categories retroactively (like PersonaPlex labels personas)?
4. Should B-series data also include dynamic steering (e.g., "slow down while explaining this math problem")?
   - Current plan: 10-15% of B-series is dynamic type to prevent forgetting.
5. Max sequence length: 2048 tokens (163s) may be tight for B12. Consider 3072 for long-form subset?
