"""Phase 1: Generate conversation transcripts using LLMs via OpenAI SDK + LiteLLM proxy.

Supports all 5 data types: standard, dynamic, counterfactual, long_form, graceful_failure.
Supports all 10 categories: A1-A10.
Resumable: skips already-generated conversations.
Retries: 3 attempts per conversation, exponential backoff on rate limits.

Usage:
    python -m pipeline.generate_transcripts --config configs/generation.yaml
    python -m pipeline.generate_transcripts --config configs/generation.yaml --category A9_dynamic_steering
"""

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import openai
from openai import OpenAI

from pipeline.utils import ensure_dir, load_all_categories, load_yaml, save_json, set_seed

# Module-level client, initialized in main()
_client: OpenAI | None = None
# Weighted model list: [(model_name, cumulative_weight), ...]
_model_choices: list[tuple[str, float]] = []

# ---------------------------------------------------------------------------
# System prompt templates (role + boundaries, NEVER style/tone)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_TEMPLATES = {
    "generic": (
        "You are a helpful voice assistant. Your voice and identity are fixed. "
        "Stay kind and constructive. Follow the user's instructions about speaking style."
    ),
    "customer_service": (
        "You are a customer service agent for {domain}. Your voice and identity are fixed. "
        "Always maintain a professional, respectful tone. Follow the user's instructions "
        "about speaking speed, formality, and detail level."
    ),
    "qa": (
        "You are a knowledgeable assistant. Your voice and identity are fixed. "
        "Be accurate and helpful. If you don't know something, say so honestly."
    ),
    # B-series system prompts
    "reasoning": (
        "You are a helpful voice assistant skilled at logical reasoning and problem-solving. "
        "Think step-by-step and explain your reasoning aloud. If you're unsure, say so."
    ),
    "math": (
        "You are a patient math tutor. Work through problems step-by-step, explaining each "
        "operation aloud. Double-check your calculations. Be encouraging when the user struggles."
    ),
    "knowledge_expert": (
        "You are a knowledgeable assistant with deep expertise across many domains. "
        "Be accurate, cite reasoning, and add appropriate caveats for medical/legal topics. "
        "If you don't know something, say so honestly."
    ),
    "instruction_follower": (
        "You are a precise voice assistant that follows instructions exactly. "
        "Pay attention to format constraints, step ordering, and specific requirements. "
        "If an instruction is unclear, ask for clarification."
    ),
    "memory_coherent": (
        "You are a voice assistant with excellent conversational memory. "
        "Remember details from earlier in the conversation and maintain consistency. "
        "If the user contradicts something they said earlier, gently ask about it."
    ),
    "paralinguistic": (
        "You are an emotionally intelligent voice assistant. Pay attention to how "
        "the user speaks, not just what they say. Adapt your tone and approach to "
        "match their emotional state."
    ),
    "creative": (
        "You are a creative and engaging voice assistant. You enjoy wordplay, "
        "storytelling, poetry, and creative collaboration. Be expressive and playful."
    ),
    "robust": (
        "You are a patient voice assistant. When the user's speech is unclear, "
        "disfluent, or interrupted, stay calm, ask for clarification on the specific "
        "part you missed, and never comment negatively on their speech patterns."
    ),
    "duplex": (
        "You are a natural conversationalist. Handle interruptions gracefully, "
        "provide backchannels when appropriate, and manage turn-taking smoothly. "
        "If interrupted, stop cleanly and address what the user said."
    ),
    "speech_expert": (
        "You are a voice assistant with strong awareness of language and speech. "
        "You understand phonetics, pronunciation, homophones, and prosody. "
        "Help users with pronunciation and language questions."
    ),
    "safety": (
        "You are a helpful voice assistant with clear safety boundaries. "
        "Decline harmful requests kindly and offer alternatives. Never be preachy "
        "or lecture the user — be brief, warm, and redirect to something helpful."
    ),
}

# ---------------------------------------------------------------------------
# Meta-prompt for transcript generation
# ---------------------------------------------------------------------------
META_PROMPT_STANDARD = """\
Generate a realistic spoken conversation between a USER and an ASSISTANT.

## Context
- Category: {category}
- Traits: {traits_json}
- System prompt (for the assistant's role): {system_prompt}
- Number of turns: {num_turns} (alternating user/assistant, user speaks first)

## CRITICAL Rules
1. User speaks FIRST.
2. The user requests the style/tone/persona at the START of the conversation (e.g., "Can you be really sarcastic?" or "Talk like a pirate").
3. The assistant FOLLOWS the user's request and maintains it throughout.
4. Each turn has `tts_instruct` describing HOW it should sound (for TTS synthesis).
5. Assistant tts_instruct must match the style the user requested.
6. User tts_instruct should be natural (casual, varied, realistic).
7. Include natural filler words: "um", "uh", "like", "you know", "I mean". Do NOT include non-spoken stage directions like (laughs), (sighs), (pauses) — only speakable words.
8. Keep turns 1-4 sentences. Natural, not scripted.
9. Assistant tts_instruct should be SHORT — max 3 comma-separated descriptors. Keep it subtle, e.g., "warm, steady pace" not "warm, gentle tone, soft delivery, caring inflection, heartfelt".

## Output: ONLY valid JSON, no markdown
{{
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_DYNAMIC = """\
Generate a spoken conversation where the USER changes the ASSISTANT's speaking style MID-CONVERSATION.

## Context
- Category: {category}
- Traits: {traits_json}
- System prompt: {system_prompt}
- Number of turns: {num_turns} (user speaks first)
- Steering events: {num_steering} time(s) the user changes the assistant's style

## CRITICAL Rules
1. User speaks first with a normal request.
2. At {num_steering} point(s) during the conversation, the user gives an explicit instruction to change how the assistant speaks (e.g., "slow down", "be more casual", "drop the formal tone").
3. The assistant ADAPTS IMMEDIATELY in its next turn — both text content and tts_instruct must change.
4. Changes PERSIST — the assistant keeps the new style for all subsequent turns.
5. Mark steering turns with "is_steering": true and "steers": {{"dimension": "new_value"}}.
6. After adaptation, mark assistant turns with "adapted_to": ["dimension:value", ...].
7. Include tts_instruct for every turn. Keep instruct SHORT — max 3 comma-separated descriptors.
8. The assistant should acknowledge the style change briefly (not ignore it).
9. Only speakable words in text — no stage directions like (laughs), (sighs), (pauses). Filler words like "um", "uh" are fine.

## Output: ONLY valid JSON, no markdown
{{
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}},
    {{"role": "user", "text": "Can you slow down?", "tts_instruct": "...", "is_steering": true, "steers": {{"speed": "slow"}}}},
    {{"role": "assistant", "text": "Sure, I'll take it slower...", "tts_instruct": "slow, deliberate, clear pauses", "adapted_to": ["speed:slow"]}}
  ]
}}"""

META_PROMPT_COUNTERFACTUAL = """\
Generate a SHORT spoken conversation (3-4 turns) that can be rendered in two different styles.

## Context
- Category: {category}
- Traits: {traits_json}
- Style A: {style_a}
- Style B: {style_b}
- Differing dimension: {diff_dimension}

## CRITICAL Rules
1. Generate ONE conversation with neutral text that works for BOTH styles.
2. Provide TWO sets of tts_instruct — one for style_a, one for style_b.
3. ONLY the {diff_dimension} should differ between variants.
4. Keep it 3-4 turns, simple topic.
5. Only speakable words in text — no (laughs), (sighs), (pauses). Keep tts_instruct SHORT (max 3 descriptors).

## Output: ONLY valid JSON, no markdown
{{
  "diff_dimension": "{diff_dimension}",
  "style_a": "{style_a}",
  "style_b": "{style_b}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "casual, natural"}},
    {{"role": "assistant", "text": "...", "tts_instruct_a": "...", "tts_instruct_b": "..."}}
  ]
}}"""

META_PROMPT_LONG_FORM = """\
Generate a LONG spoken conversation (10-15 turns) testing multi-turn coherence.

## Context
- Category: {category}
- Traits: {traits_json}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. User speaks first.
2. Include topic changes (at least 1-2).
3. Reference proper nouns from earlier turns (test recall).
4. The assistant must maintain its style/role consistently over many turns.
5. Include natural filler words ("um", "uh") but NO stage directions like (laughs), (sighs), (pauses).
6. Each turn has tts_instruct — keep it SHORT, max 3 comma-separated descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_GRACEFUL_FAILURE = """\
Generate a spoken conversation where the ASSISTANT must handle a difficult situation gracefully.

## Context
- Failure type: {failure_type}
- Scenario: {scenario}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. User speaks first.
2. At some point, the user says something the assistant can't fulfill, doesn't understand, or that's inappropriate.
3. The assistant must handle it GRACEFULLY — not go silent, not give a robotic refusal.
4. The assistant should: acknowledge, briefly explain the limit, offer an alternative or redirect.
5. The conversation should feel NATURAL, not like a test.
6. Each turn has tts_instruct — keep it SHORT, max 3 comma-separated descriptors.
7. Only speakable words in text — no (laughs), (sighs), (pauses).

## Failure type guidance
- clarification: user is unclear, assistant asks to repeat/clarify
- knowledge_limit: user asks something assistant doesn't know, assistant admits it honestly
- capability_limit: user asks assistant to do something it can't (call, email, etc.)
- misunderstanding_repair: assistant gets it wrong, user corrects, assistant recovers
- frustration_deescalation: user is frustrated, assistant stays calm and helpful
- safety_boundary: user asks for something inappropriate, assistant declines kindly
- inappropriate_steering: user asks assistant to change voice/be hostile, assistant offers alternative
- noise_issue: audio quality problem, assistant asks to repeat

## Output: ONLY valid JSON, no markdown
{{
  "failure_type": "{failure_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""


# ---------------------------------------------------------------------------
# B-series meta-prompts (capabilities beyond steerability)
# ---------------------------------------------------------------------------
META_PROMPT_REASONING = """\
Generate a spoken conversation where the USER asks questions that require LOGICAL REASONING.

## Context
- Reasoning type: {reasoning_type}
- Subcategory: {subcategory}
- Difficulty: {difficulty}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. User speaks first with a question or puzzle that requires reasoning.
2. The assistant must THINK STEP-BY-STEP aloud — say each reasoning step as it works through the problem.
3. The user should ask follow-up questions like "wait, how did you get that?" or "can you explain that step?"
4. Include natural filler words ("um", "let me think", "so basically"). NO stage directions.
5. Assistant tts_instruct: SHORT, max 3 descriptors. E.g., "thoughtful, measured pace, clear"
6. The ANSWER MUST BE CORRECT. Double-check all logic.

## Output: ONLY valid JSON, no markdown
{{
  "reasoning_type": "{reasoning_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_MATH = """\
Generate a spoken conversation involving MATHEMATICAL REASONING.

## Context
- Math type: {math_type}
- Difficulty: {difficulty}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. User asks a math question or problem.
2. Assistant works through it STEP-BY-STEP aloud: "Let me work through this. First... then... so the answer is..."
3. All numbers must be spoken as words: "three hundred forty-seven", not "347".
4. The ANSWER MUST BE MATHEMATICALLY CORRECT. Verify before outputting.
5. User should ask follow-ups or give new problems.
6. Include filler: "hmm", "let me think", "okay so". NO stage directions.
7. tts_instruct: SHORT, max 3 descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "math_type": "{math_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_KNOWLEDGE = """\
Generate a spoken conversation requiring EXPERT-LEVEL KNOWLEDGE.

## Context
- Domain: {domain}
- Topic: {topic}
- Depth: {depth}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. User asks questions requiring real, accurate knowledge — not surface-level.
2. Assistant provides ACCURATE information. If medical/legal, add "I'm not a doctor/lawyer" caveat.
3. Multi-turn: user drills deeper with follow-ups. Later questions build on earlier answers.
4. For fact-verification, clearly state whether claims are true/false/partially-true with reasoning.
5. Include natural speech patterns. NO stage directions. tts_instruct: SHORT, max 3 descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "domain": "{domain}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_INSTRUCTION = """\
Generate a spoken conversation testing INSTRUCTION FOLLOWING.

## Context
- Instruction type: {instruction_type}
- Specific constraint: {constraint}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. User gives a clear instruction with specific constraints (format, count, conditional, etc.).
2. Assistant must follow the instruction EXACTLY — not approximately.
3. If the user catches a violation: "Hey, you didn't follow the rule about X" — assistant acknowledges and corrects.
4. Include at least one turn where the constraint is actively tested.
5. Natural speech. NO stage directions. tts_instruct: SHORT, max 3 descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "instruction_type": "{instruction_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_MEMORY = """\
Generate a LONG spoken conversation (12-20 turns) testing MEMORY AND COHERENCE.

## Context
- Memory test type: {memory_type}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. User introduces specific details (names, numbers, facts) in early turns.
2. These details are referenced or tested in LATER turns (5+ turns apart).
3. Assistant must correctly recall earlier information without contradictions.
4. Include at least one "memory test" where the user asks about earlier content.
5. For contradiction detection: user introduces contradicting info, assistant catches it gently.
6. Natural speech. NO stage directions. tts_instruct: SHORT, max 3 descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "memory_type": "{memory_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_PARALINGUISTIC = """\
Generate a spoken conversation where the assistant must respond to the USER'S EMOTIONAL STATE.

## Context
- Emotion scenario: {scenario}
- User's actual emotion: {user_emotion}
- User's text: may be NEUTRAL — the emotion is conveyed through HOW they speak (tts_instruct), not what they say
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. The user's TEXT can be neutral/ambiguous, but their tts_instruct encodes the real emotion.
2. The assistant must DETECT the emotion from context and respond appropriately.
3. A BAD response ignores or misreads the emotion. A GOOD response shows awareness.
4. The assistant's tone should MATCH the emotional context (empathetic for sad, celebratory for happy, etc.).
5. Natural speech. NO stage directions. tts_instruct: SHORT, max 3 descriptors.
6. User tts_instruct MUST reflect the actual emotion: "{user_tts}"

## Output: ONLY valid JSON, no markdown
{{
  "scenario": "{scenario}",
  "user_emotion": "{user_emotion}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "{user_tts}"}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_CREATIVE = """\
Generate a spoken conversation involving CREATIVE LANGUAGE.

## Context
- Creative type: {creative_type}
- Specific task: {task}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. The conversation involves creative output: stories, poetry, wordplay, debate, etc.
2. If poetry: maintain proper rhythm and rhyme scheme. If story: maintain narrative arc.
3. If debate: both sides present genuine arguments. Assistant may play devil's advocate.
4. The output should be genuinely creative and engaging, not generic.
5. Natural speech with creative flair. NO stage directions. tts_instruct: SHORT, max 3 descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "creative_type": "{creative_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_ROBUSTNESS = """\
Generate a spoken conversation where the USER speaks IMPERFECTLY (disfluent, unclear, self-correcting).

## Context
- Disfluency type: {disfluency_type}
- Scenario: {scenario}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. User turns include realistic speech imperfections: stuttering, false starts, self-corrections, fillers, trailing off.
2. The assistant must EXTRACT THE INTENDED MEANING despite disfluencies.
3. The assistant NEVER comments negatively on the user's speech. Stay patient and helpful.
4. For self-corrections: always use the FINAL corrected value.
5. For trailing off: gently ask for completion.
6. User tts_instruct must reflect the disfluency pattern.
7. NO stage directions. tts_instruct: SHORT, max 3 descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "disfluency_type": "{disfluency_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_DUPLEX = """\
Generate a spoken conversation showcasing NATURAL DUPLEX INTERACTION PATTERNS.

## Context
- Pattern type: {pattern_type}
- Scenario: {scenario}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. This conversation simulates full-duplex speech interaction patterns.
2. Pattern type "{pattern_type}" must be clearly demonstrated.
3. For BACKCHANNELING: include brief "mm-hmm", "right", "I see" interjections from the assistant during user's long turns. Mark with "is_backchannel": true.
4. For INTERRUPTION: user cuts off assistant mid-sentence. Mark with "is_interruption": true. Assistant stops and addresses the interruption.
5. For CORRECTION: user corrects model mid-response. Mark with "is_correction": true. Model adjusts immediately.
6. For PAUSE: mark extended silences with "pause_sec": N.
7. Natural speech. NO stage directions. tts_instruct: SHORT, max 3 descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "pattern_type": "{pattern_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""

META_PROMPT_SPEECH_UNDERSTANDING = """\
Generate a spoken conversation about SPEECH AND LANGUAGE PHENOMENA.

## Context
- Topic: {speech_topic}
- Specific focus: {focus}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## CRITICAL Rules
1. The conversation involves speech-specific understanding: homophones, phonetics, pronunciation, prosody, etc.
2. The assistant must demonstrate genuine understanding of HOW language sounds, not just what it means.
3. When explaining pronunciation, use approximate phonetic spelling: "kuh-NEL" for "colonel".
4. For homophones: demonstrate correct disambiguation from context.
5. Natural speech. NO stage directions. tts_instruct: SHORT, max 3 descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "speech_topic": "{speech_topic}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""


# ---------------------------------------------------------------------------
# Trait sampling per category
# ---------------------------------------------------------------------------
def sample_traits(category_data: dict, category_id: str) -> dict:
    """Sample random traits from a category definition."""
    traits = {"category": category_id}

    if category_id.startswith("A1"):
        domains = list(category_data.get("domains", {}).keys())
        if domains:
            domain = random.choice(domains)
            traits["domain"] = domain
            scenarios = category_data["domains"][domain].get("scenarios", [])
            if scenarios:
                traits["scenario"] = random.choice(scenarios)

    elif category_id.startswith("A2"):
        subcats = list(category_data.get("subcategories", {}).keys())
        if subcats:
            subcat = random.choice(subcats)
            traits["subcategory"] = subcat
            topics = category_data["subcategories"][subcat].get("topics", [])
            if topics:
                traits["topic"] = random.choice(topics)

    elif category_id.startswith("A3"):
        all_tones = []
        for group in ["negative", "neutral", "positive"]:
            tones = category_data.get("tones", {}).get(group, {})
            for tone_name, tone_data in tones.items():
                all_tones.append({"name": tone_name, "group": group, **(tone_data if isinstance(tone_data, dict) else {})})
        if all_tones:
            tone = random.choice(all_tones)
            traits["tone"] = tone["name"]
            traits["tts_instruct"] = tone.get("tts_instruct", f"{tone['name']} tone")
            traits["description"] = tone.get("description", "")
            topics = tone.get("suitable_topics", [])
            if topics:
                traits["topic"] = random.choice(topics)

    elif category_id.startswith("A4"):
        all_personas = []
        for subcat_name, subcat_data in category_data.get("subcategories", {}).items():
            personas = subcat_data.get("personas", {}) if isinstance(subcat_data, dict) else {}
            for persona_name, persona_data in personas.items():
                all_personas.append({"name": persona_name, "subcategory": subcat_name, **(persona_data if isinstance(persona_data, dict) else {})})
        if all_personas:
            persona = random.choice(all_personas)
            traits["persona"] = persona["name"]
            traits["description"] = persona.get("description", "")
            traits["tts_instruct"] = persona.get("tts_instruct", f"{persona['name']} persona")

    elif category_id.startswith("A5"):
        styles = category_data.get("speaking_styles", category_data.get("styles", {}))
        accents = category_data.get("accents", {})
        if styles:
            style = random.choice(list(styles.keys()))
            style_data = styles[style] if isinstance(styles[style], dict) else {}
            traits["style"] = style
            traits["tts_instruct"] = style_data.get("tts_instruct", f"{style} style")
        if accents and random.random() < 0.5:
            accent = random.choice(list(accents.keys()))
            traits["accent"] = accent

    elif category_id.startswith("A6"):
        speeds = category_data.get("speed_controls", category_data.get("speeds", {}))
        lengths = category_data.get("length_controls", category_data.get("lengths", {}))
        if speeds:
            speed = random.choice(list(speeds.keys()))
            traits["speed"] = speed
        if lengths:
            length = random.choice(list(lengths.keys()))
            traits["length"] = length

    elif category_id.startswith("A7"):
        emotions = category_data.get("user_emotions", category_data.get("emotions", {}))
        if emotions:
            emotion = random.choice(list(emotions.keys()))
            traits["user_emotion"] = emotion
            emo_data = emotions[emotion] if isinstance(emotions[emotion], dict) else {}
            traits["adaptation"] = emo_data.get("assistant_adaptation", "respond appropriately")

    elif category_id.startswith("A8"):
        cases = category_data.get("cases", category_data.get("categories", {}))
        if cases:
            case = random.choice(list(cases.keys()))
            traits["case"] = case

    elif category_id.startswith("A9"):
        dims = list(category_data.get("steering_dimensions", {}).keys())
        if dims:
            traits["steering_dimension"] = random.choice(dims)
        patterns = list(category_data.get("steering_patterns", {}).keys())
        if patterns:
            traits["steering_pattern"] = random.choice(patterns)

    elif category_id.startswith("A10"):
        cats = list(category_data.get("categories", {}).keys())
        if cats:
            failure_type = random.choice(cats)
            traits["failure_type"] = failure_type
            cat_data = category_data["categories"][failure_type]
            if isinstance(cat_data, dict):
                scenarios = cat_data.get("scenarios", [])
                if scenarios:
                    traits["scenario"] = random.choice(scenarios) if isinstance(scenarios[0], str) else random.choice(list(scenarios.keys()))

    # --- B-series categories ---
    elif category_id.startswith("B1"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            subcat = random.choice(subcats)
            traits["reasoning_type"] = subcat
            traits["subcategory"] = subcat
        traits["difficulty"] = random.choice(["easy", "medium", "hard"])

    elif category_id.startswith("B2"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            traits["math_type"] = random.choice(subcats)
        traits["difficulty"] = random.choice(["easy", "medium", "hard"])

    elif category_id.startswith("B3"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            subcat = random.choice(subcats)
            traits["domain"] = subcat
            subcat_data = category_data["categories"][subcat]
            if isinstance(subcat_data, dict):
                domains = subcat_data.get("domains", subcat_data.get("topics", {}))
                if isinstance(domains, dict) and domains:
                    traits["topic"] = random.choice(list(domains.keys()))
                elif isinstance(domains, list) and domains:
                    traits["topic"] = random.choice(domains)
                # Fallback: try examples list for subcats like multi_hop_qa
                if "topic" not in traits:
                    examples = subcat_data.get("examples", [])
                    if examples:
                        traits["topic"] = random.choice(examples) if isinstance(examples[0], str) else subcat
                    else:
                        traits["topic"] = subcat
        traits["depth"] = random.choice(["intermediate", "advanced", "expert"])

    elif category_id.startswith("B4"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            traits["instruction_type"] = random.choice(subcats)
        constraints = [
            "exactly 5 items", "one sentence only", "no jargon",
            "numbered list", "start each point with a verb",
            "under 20 words", "alphabetical order", "if X then Y format",
        ]
        traits["constraint"] = random.choice(constraints)

    elif category_id.startswith("B5"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            traits["memory_type"] = random.choice(subcats)

    elif category_id.startswith("B6"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            subcat = random.choice(subcats)
            traits["scenario"] = subcat
        emotions_map = {
            "emotion_aware_response": ("hidden_sadness", "flat, subdued, low energy"),
            "sarcasm_irony_extended": ("sarcasm", "flat, sarcastic emphasis, exaggerated"),
            "urgency_detection": ("urgency", "fast, stressed, clipped"),
            "hesitation_uncertainty": ("uncertainty", "hesitant, trailing off, quiet"),
            "excitement_matching": ("excitement", "ecstatic, fast, high energy"),
            "grief_sensitivity": ("grief", "quiet, choked up, slow"),
            "formality_inference": ("formality", "varies per scenario"),
            "age_appropriate_adaptation": ("adaptation", "varies per scenario"),
        }
        emo_info = emotions_map.get(subcat, ("neutral", "natural, conversational"))
        traits["user_emotion"] = emo_info[0]
        traits["user_tts"] = emo_info[1]

    elif category_id.startswith("B7"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            traits["creative_type"] = random.choice(subcats)
        tasks = [
            "Tell a bedtime story about a brave fox",
            "Write a limerick about programming",
            "Let's debate: is social media good or bad?",
            "Give me 5 puns about food",
            "Explain quantum physics as a fairy tale",
            "Write a haiku about the ocean",
            "Let's play word association",
            "Write a rap verse about Monday mornings",
        ]
        traits["task"] = random.choice(tasks)

    elif category_id.startswith("B8"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            traits["disfluency_type"] = random.choice(subcats)
        scenarios = [
            "asking for directions with lots of fillers",
            "ordering food while self-correcting",
            "describing a problem with false starts",
            "asking a question in a noisy environment",
            "speaking with a heavy accent about technology",
            "trailing off while explaining something complex",
        ]
        traits["scenario"] = random.choice(scenarios)

    elif category_id.startswith("B9"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            traits["pattern_type"] = random.choice(subcats)
        scenarios = [
            "casual conversation with natural turn-taking",
            "user telling a long story with backchannels",
            "user interrupting to change topic",
            "user correcting a factual error",
            "long pause while user thinks",
            "both speakers accidentally overlapping",
        ]
        traits["scenario"] = random.choice(scenarios)

    elif category_id.startswith("B10"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            traits["speech_topic"] = random.choice(subcats)
        focuses = [
            "homophones in context", "rhyming words", "pronunciation guidance",
            "stress and emphasis", "onomatopoeia", "code-switching with French phrases",
            "heteronyms", "syllable counting", "phonetic spelling",
        ]
        traits["focus"] = random.choice(focuses)

    elif category_id.startswith("B11"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            failure_type = random.choice(subcats)
            traits["failure_type"] = failure_type
        scenarios = [
            "user tries roleplay bypass", "user asks for harmful content",
            "user states misinformation confidently", "user shares sensitive data",
            "user asks about controversial topic", "user claims authority for advice",
            "user tries emotional manipulation",
        ]
        traits["scenario"] = random.choice(scenarios)

    elif category_id.startswith("B12"):
        subcats = list(category_data.get("categories", {}).keys())
        if subcats:
            traits["dialogue_type"] = random.choice(subcats)
        topics = [
            "artificial intelligence", "climate change", "philosophy of consciousness",
            "modern economics", "space exploration", "history of music",
            "startup planning", "ethical dilemma", "quantum computing",
        ]
        traits["topic"] = random.choice(topics)

    return traits


def build_system_prompt(traits: dict) -> str:
    """Build minimal system prompt from traits (role + boundaries only)."""
    cat = traits.get("category", "")
    if cat.startswith("A1") and "domain" in traits:
        return SYSTEM_PROMPT_TEMPLATES["customer_service"].format(domain=traits["domain"])
    elif cat.startswith("A2"):
        return SYSTEM_PROMPT_TEMPLATES["qa"]
    elif cat.startswith("B1"):
        return SYSTEM_PROMPT_TEMPLATES["reasoning"]
    elif cat.startswith("B2"):
        return SYSTEM_PROMPT_TEMPLATES["math"]
    elif cat.startswith("B3"):
        return SYSTEM_PROMPT_TEMPLATES["knowledge_expert"]
    elif cat.startswith("B4"):
        return SYSTEM_PROMPT_TEMPLATES["instruction_follower"]
    elif cat.startswith("B5"):
        return SYSTEM_PROMPT_TEMPLATES["memory_coherent"]
    elif cat.startswith("B6"):
        return SYSTEM_PROMPT_TEMPLATES["paralinguistic"]
    elif cat.startswith("B7"):
        return SYSTEM_PROMPT_TEMPLATES["creative"]
    elif cat.startswith("B8"):
        return SYSTEM_PROMPT_TEMPLATES["robust"]
    elif cat.startswith("B9"):
        return SYSTEM_PROMPT_TEMPLATES["duplex"]
    elif cat.startswith("B10"):
        return SYSTEM_PROMPT_TEMPLATES["speech_expert"]
    elif cat.startswith("B11"):
        return SYSTEM_PROMPT_TEMPLATES["safety"]
    elif cat.startswith("B12"):
        return SYSTEM_PROMPT_TEMPLATES["knowledge_expert"]
    elif cat.startswith("A"):
        return SYSTEM_PROMPT_TEMPLATES["generic"]
    else:
        return SYSTEM_PROMPT_TEMPLATES["generic"]


def sample_data_type(weights: dict, category_id: str, category_data: dict | None = None) -> str:
    """Sample a data type, respecting category constraints and overrides."""
    # A9 should mostly be dynamic, A10 should mostly be graceful_failure
    if category_id.startswith("A9"):
        return random.choices(
            ["dynamic", "standard"], weights=[0.85, 0.15], k=1
        )[0]
    if category_id.startswith("A10"):
        return random.choices(
            ["graceful_failure", "standard"], weights=[0.85, 0.15], k=1
        )[0]

    # B-series: check for data_type_override in category YAML
    if category_data and "data_type_override" in category_data:
        override = category_data["data_type_override"]
        types = list(override.keys())
        probs = [override[t] for t in types]
        return random.choices(types, weights=probs, k=1)[0]

    types = list(weights.keys())
    probs = [weights[t] for t in types]
    return random.choices(types, weights=probs, k=1)[0]


# ---------------------------------------------------------------------------
# Build the LLM prompt based on data type
# ---------------------------------------------------------------------------
COUNTERFACTUAL_DIMENSIONS = {
    "tone": [("cheerful", "serious"), ("sarcastic", "sincere"), ("formal", "casual"), ("empathetic", "matter-of-fact")],
    "speed": [("fast", "slow"), ("very_fast", "moderate")],
    "energy": [("high_energy", "calm"), ("excited", "subdued")],
}


def build_llm_prompt(traits: dict, data_type: str, system_prompt: str, turns_range: tuple[int, int]) -> str:
    """Build the appropriate meta-prompt based on data type and category."""
    clean_traits = {k: v for k, v in traits.items() if isinstance(v, (str, int, float, bool, list))}
    traits_json = json.dumps(clean_traits, indent=2)
    category = traits.get("category", "unknown")

    # --- B-series category-specific prompts ---
    # These use specialized meta-prompts regardless of data_type
    # (unless data_type is dynamic/counterfactual/graceful_failure, which use A-series prompts)
    if category.startswith("B") and data_type in ("standard", "long_form"):
        return _build_b_series_prompt(traits, system_prompt, turns_range, category)

    # --- A-series and fallback data-type-based prompts ---
    if data_type == "standard":
        num_turns = random.randint(turns_range[0], turns_range[1])
        return META_PROMPT_STANDARD.format(
            category=category, traits_json=traits_json,
            system_prompt=system_prompt, num_turns=num_turns,
        )

    elif data_type == "dynamic":
        num_turns = random.randint(max(6, turns_range[0]), min(14, turns_range[1] + 4))
        num_steering = random.randint(1, 3)
        return META_PROMPT_DYNAMIC.format(
            category=category, traits_json=traits_json,
            system_prompt=system_prompt, num_turns=num_turns,
            num_steering=num_steering,
        )

    elif data_type == "counterfactual":
        dim = random.choice(list(COUNTERFACTUAL_DIMENSIONS.keys()))
        pair = random.choice(COUNTERFACTUAL_DIMENSIONS[dim])
        return META_PROMPT_COUNTERFACTUAL.format(
            category=category, traits_json=traits_json,
            style_a=pair[0], style_b=pair[1], diff_dimension=dim,
        )

    elif data_type == "long_form":
        num_turns = random.randint(10, 15)
        return META_PROMPT_LONG_FORM.format(
            category=category, traits_json=traits_json,
            system_prompt=system_prompt, num_turns=num_turns,
        )

    elif data_type == "graceful_failure":
        failure_type = traits.get("failure_type", "clarification_requests")
        scenario = traits.get("scenario", "user says something unclear")
        num_turns = random.randint(4, 8)
        return META_PROMPT_GRACEFUL_FAILURE.format(
            failure_type=failure_type, scenario=scenario,
            system_prompt=system_prompt, num_turns=num_turns,
        )

    return META_PROMPT_STANDARD.format(
        category=category, traits_json=traits_json,
        system_prompt=system_prompt, num_turns=random.randint(*turns_range),
    )


def _build_b_series_prompt(traits: dict, system_prompt: str, turns_range: tuple[int, int], category: str) -> str:
    """Build specialized prompts for B-series categories."""

    if category.startswith("B1"):
        num_turns = random.randint(max(4, turns_range[0]), min(10, turns_range[1]))
        return META_PROMPT_REASONING.format(
            reasoning_type=traits.get("reasoning_type", "commonsense_reasoning"),
            subcategory=traits.get("subcategory", "general"),
            difficulty=traits.get("difficulty", "medium"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B2"):
        num_turns = random.randint(max(3, turns_range[0]), min(8, turns_range[1]))
        return META_PROMPT_MATH.format(
            math_type=traits.get("math_type", "word_problems"),
            difficulty=traits.get("difficulty", "medium"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B3"):
        num_turns = random.randint(max(3, turns_range[0]), min(10, turns_range[1]))
        return META_PROMPT_KNOWLEDGE.format(
            domain=traits.get("domain", "science_deep_dive"),
            topic=traits.get("topic", "general"),
            depth=traits.get("depth", "intermediate"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B4"):
        num_turns = random.randint(max(4, turns_range[0]), min(10, turns_range[1]))
        return META_PROMPT_INSTRUCTION.format(
            instruction_type=traits.get("instruction_type", "multi_step_instructions"),
            constraint=traits.get("constraint", "follow instructions exactly"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B5"):
        num_turns = random.randint(12, 20)
        return META_PROMPT_MEMORY.format(
            memory_type=traits.get("memory_type", "entity_tracking"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B6"):
        num_turns = random.randint(max(4, turns_range[0]), min(10, turns_range[1]))
        return META_PROMPT_PARALINGUISTIC.format(
            scenario=traits.get("scenario", "emotion_aware_response"),
            user_emotion=traits.get("user_emotion", "neutral"),
            user_tts=traits.get("user_tts", "natural, conversational"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B7"):
        num_turns = random.randint(max(3, turns_range[0]), min(10, turns_range[1]))
        return META_PROMPT_CREATIVE.format(
            creative_type=traits.get("creative_type", "storytelling"),
            task=traits.get("task", "tell a creative story"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B8"):
        num_turns = random.randint(max(3, turns_range[0]), min(8, turns_range[1]))
        return META_PROMPT_ROBUSTNESS.format(
            disfluency_type=traits.get("disfluency_type", "disfluent_user_speech"),
            scenario=traits.get("scenario", "general conversation with disfluencies"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B9"):
        num_turns = random.randint(max(4, turns_range[0]), min(12, turns_range[1]))
        return META_PROMPT_DUPLEX.format(
            pattern_type=traits.get("pattern_type", "backchanneling"),
            scenario=traits.get("scenario", "natural conversation"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B10"):
        num_turns = random.randint(max(3, turns_range[0]), min(8, turns_range[1]))
        return META_PROMPT_SPEECH_UNDERSTANDING.format(
            speech_topic=traits.get("speech_topic", "homophone_disambiguation"),
            focus=traits.get("focus", "general speech understanding"),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    elif category.startswith("B11"):
        # Reuse graceful_failure prompt for safety
        failure_type = traits.get("failure_type", "harmful_content_refusal")
        scenario = traits.get("scenario", "user asks for something unsafe")
        num_turns = random.randint(3, 8)
        return META_PROMPT_GRACEFUL_FAILURE.format(
            failure_type=failure_type, scenario=scenario,
            system_prompt=system_prompt, num_turns=num_turns,
        )

    elif category.startswith("B12"):
        # Long-form with topic specified
        num_turns = random.randint(15, 25)
        clean_traits = {k: v for k, v in traits.items() if isinstance(v, (str, int, float, bool, list))}
        return META_PROMPT_LONG_FORM.format(
            category=category,
            traits_json=json.dumps(clean_traits, indent=2),
            system_prompt=system_prompt,
            num_turns=num_turns,
        )

    # Fallback
    num_turns = random.randint(turns_range[0], turns_range[1])
    clean_traits = {k: v for k, v in traits.items() if isinstance(v, (str, int, float, bool, list))}
    return META_PROMPT_STANDARD.format(
        category=category, traits_json=json.dumps(clean_traits, indent=2),
        system_prompt=system_prompt, num_turns=num_turns,
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Model selection (weighted random from configured model list)
# ---------------------------------------------------------------------------
def _pick_model() -> str:
    """Randomly select a model from the weighted model list."""
    if not _model_choices:
        raise RuntimeError("No models configured")
    r = random.random()
    for model_name, cum_weight in _model_choices:
        if r <= cum_weight:
            return model_name
    return _model_choices[-1][0]  # fallback to last


# ---------------------------------------------------------------------------
# LLM call via OpenAI SDK (pointing at LiteLLM proxy) with retry + backoff
# ---------------------------------------------------------------------------
def call_llm(
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    max_retries: int = 3,
    retry_wait_sec: float = 5.0,
) -> str | None:
    """Call the LLM via OpenAI SDK with retry logic. Returns raw text or None."""
    # GPT-5.x models only support temperature=1
    model_temp = 1.0 if "gpt-5" in model else temperature

    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=model_temp,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()

        except openai.RateLimitError:
            wait = retry_wait_sec * (2 ** attempt)
            print(f"  [RATE LIMIT] Waiting {wait:.0f}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  [RETRY {attempt + 1}] {e}")
                time.sleep(2)
            else:
                print(f"  [FAIL] {e}")
                return None
    return None


def parse_llm_response(content: str) -> dict | None:
    """Parse JSON from LLM response, handling markdown code blocks."""
    if not content:
        return None
    # Strip markdown
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*\n?", "", content)
        content = re.sub(r"\n?\s*```\s*$", "", content)
    # Strip any text before first { or after last }
    first_brace = content.find("{")
    last_brace = content.rfind("}")
    if first_brace >= 0 and last_brace > first_brace:
        content = content[first_brace:last_brace + 1]
    try:
        data = json.loads(content)
        if "turns" not in data:
            return None
        if len(data["turns"]) < 2:
            return None
        for turn in data["turns"]:
            if "role" not in turn or "text" not in turn:
                return None
        return data
    except (json.JSONDecodeError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def generate_single(
    model: str | None,
    category_id: str,
    category_data: dict,
    data_type: str,
    temperature: float,
    max_tokens: int,
    turns_range: tuple[int, int],
    max_retries: int,
    retry_wait_sec: float,
) -> dict | None:
    """Generate a single conversation transcript.

    If model is None, randomly selects from the configured model list.
    """
    # Pick model for this conversation (diverse generation)
    selected_model = model if model else _pick_model()

    traits = sample_traits(category_data, category_id)
    system_prompt = build_system_prompt(traits)
    prompt = build_llm_prompt(traits, data_type, system_prompt, turns_range)

    content = call_llm(selected_model, prompt, temperature, max_tokens, max_retries, retry_wait_sec)
    transcript = parse_llm_response(content)

    if transcript is None:
        return None

    transcript["category"] = category_id
    transcript["data_type"] = data_type
    transcript["system_prompt"] = f"<system> {system_prompt} <system>"
    transcript["sampled_traits"] = {k: v for k, v in traits.items() if isinstance(v, (str, int, float, bool, list))}
    transcript["llm_model"] = selected_model

    return transcript


# Thread-safe counter for assigning sequential IDs
_counter_lock = threading.Lock()


def _worker(
    model: str | None,
    category_id: str,
    category_data: dict,
    dt_weights: dict,
    cfg: dict,
) -> dict | None:
    """Worker function for ThreadPoolExecutor.

    If model is None, each call picks randomly from the configured model list.
    """
    data_type = sample_data_type(dt_weights, category_id, category_data)
    return generate_single(
        model=model,
        category_id=category_id,
        category_data=category_data,
        data_type=data_type,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        turns_range=tuple(cfg["turns_range"]),
        max_retries=cfg.get("max_retries", 3),
        retry_wait_sec=cfg.get("retry_wait_sec", 5),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate conversation transcripts")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument("--num_conversations", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None, help="Parallel LLM workers (default: from config)")
    parser.add_argument("--scale", choices=["pilot", "full"], default="pilot",
                        help="Use pilot_per_category or full_per_category counts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["transcript"]
    set_seed(args.seed)

    # Configure OpenAI client (pointing at LiteLLM proxy)
    global _client, _model_choices
    base_url = cfg.get("llm_base_url") or None
    api_key = cfg.get("llm_api_key") or None  # falls back to OPENAI_API_KEY env var
    _client = OpenAI(base_url=base_url, api_key=api_key)

    # Build weighted model list for diverse generation
    llm_models_cfg = cfg.get("llm_models")
    if llm_models_cfg and isinstance(llm_models_cfg, list):
        # Normalize weights and build cumulative distribution
        total_weight = sum(m.get("weight", 1.0) for m in llm_models_cfg)
        cum = 0.0
        _model_choices = []
        for m in llm_models_cfg:
            cum += m.get("weight", 1.0) / total_weight
            _model_choices.append((m["model"], cum))
        model = None  # signal to pick randomly per conversation
        model_names = [m["model"] for m in llm_models_cfg]
        weights = [m.get("weight", 1.0) for m in llm_models_cfg]
        print(f"LLM models: {list(zip(model_names, weights))}")
    else:
        # Legacy single-model mode
        model = cfg["llm_model"]
        _model_choices = [(model, 1.0)]
        print(f"LLM model: {model}")

    categories = load_all_categories(cfg["categories_dir"])
    output_dir = ensure_dir(cfg["output_dir"])

    if args.category:
        categories = {k: v for k, v in categories.items() if k == args.category}

    counts_key = "full_per_category" if args.scale == "full" else "pilot_per_category"
    count_map = cfg.get(counts_key, cfg.get("pilot_per_category", {}))
    dt_weights = cfg.get("data_type_weights", {"standard": 1.0})
    num_workers = args.num_workers or cfg.get("num_workers", 8)

    # Distributed claiming: each slot index is a work item.
    # Multiple nodes/processes can run this concurrently on shared FS.
    from pipeline.distributed import is_done, release_claim, try_claim

    for cat_id, cat_data in sorted(categories.items()):
        if isinstance(count_map, dict):
            num_target = args.num_conversations or count_map.get(cat_id, 60)
        else:
            num_target = args.num_conversations or count_map

        cat_dir = ensure_dir(output_dir / cat_id)
        claims_dir = ensure_dir(Path(cfg["output_dir"]) / ".claims" / cat_id)

        # Count what's already done
        done_count = sum(1 for i in range(num_target) if is_done(cat_dir / f"{i:05d}.json"))
        if done_count >= num_target:
            print(f"[SKIP] {cat_id}: all {num_target} done")
            continue

        remaining = num_target - done_count
        print(f"\n=== {cat_id}: {remaining} remaining ({done_count}/{num_target} done) | workers={num_workers} ===")

        # Build list of unclaimed, undone slot indices
        slots = [i for i in range(num_target) if not is_done(cat_dir / f"{i:05d}.json")]
        random.shuffle(slots)  # shuffle so different nodes don't race on the same end

        generated = 0
        failed = 0

        def _claim_and_generate(slot_idx: int) -> tuple[int, dict | None]:
            claim_path = claims_dir / f"{slot_idx:05d}.claim"
            out_path = cat_dir / f"{slot_idx:05d}.json"
            if is_done(out_path):
                return slot_idx, "skip"
            if not try_claim(claim_path):
                return slot_idx, "skip"
            try:
                transcript = _worker(model, cat_id, cat_data, dt_weights, cfg)
                if transcript is not None:
                    transcript["id"] = f"{cat_id}_{slot_idx:05d}"
                    save_json(transcript, out_path)
                    release_claim(claim_path)  # cleanup
                else:
                    release_claim(claim_path)  # let another node retry
                return slot_idx, transcript
            except Exception as e:
                release_claim(claim_path)
                print(f"  [ERROR] slot {slot_idx}: {e}")
                return slot_idx, None

        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_claim_and_generate, s): s for s in slots}

            for fut in as_completed(futures):
                try:
                    slot_idx, result = fut.result()
                except Exception as e:
                    print(f"  [ERROR] Worker exception: {e}")
                    failed += 1
                    continue

                if result == "skip":
                    continue
                elif result is not None:
                    generated += 1
                    if generated % 50 == 0:
                        print(f"  {cat_id}: {generated} generated so far")
                else:
                    failed += 1

        print(f"  {cat_id}: generated {generated}, failed {failed}")


if __name__ == "__main__":
    main()
