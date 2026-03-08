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
7. Include natural speech: "um", "uh", "(laughs)", "(sighs)", "(pauses)".
8. Keep turns 1-4 sentences. Natural, not scripted.

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
7. Include tts_instruct for every turn.
8. The assistant should acknowledge the style change briefly (not ignore it).

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
5. Include natural speech patterns and some backchanneling.
6. Each turn has tts_instruct.

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
6. Each turn has tts_instruct.

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

    return traits


def build_system_prompt(traits: dict) -> str:
    """Build minimal system prompt from traits (role + boundaries only)."""
    cat = traits.get("category", "")
    if cat.startswith("A1") and "domain" in traits:
        return SYSTEM_PROMPT_TEMPLATES["customer_service"].format(domain=traits["domain"])
    elif cat.startswith("A2"):
        return SYSTEM_PROMPT_TEMPLATES["qa"]
    else:
        return SYSTEM_PROMPT_TEMPLATES["generic"]


def sample_data_type(weights: dict, category_id: str) -> str:
    """Sample a data type, respecting category constraints."""
    # A9 should mostly be dynamic, A10 should mostly be graceful_failure
    if category_id.startswith("A9"):
        return random.choices(
            ["dynamic", "standard"], weights=[0.85, 0.15], k=1
        )[0]
    if category_id.startswith("A10"):
        return random.choices(
            ["graceful_failure", "standard"], weights=[0.85, 0.15], k=1
        )[0]

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
    """Build the appropriate meta-prompt based on data type."""
    clean_traits = {k: v for k, v in traits.items() if isinstance(v, (str, int, float, bool, list))}
    traits_json = json.dumps(clean_traits, indent=2)
    category = traits.get("category", "unknown")

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
    for attempt in range(max_retries):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
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
    model: str,
    category_id: str,
    category_data: dict,
    data_type: str,
    temperature: float,
    max_tokens: int,
    turns_range: tuple[int, int],
    max_retries: int,
    retry_wait_sec: float,
) -> dict | None:
    """Generate a single conversation transcript."""
    traits = sample_traits(category_data, category_id)
    system_prompt = build_system_prompt(traits)
    prompt = build_llm_prompt(traits, data_type, system_prompt, turns_range)

    content = call_llm(model, prompt, temperature, max_tokens, max_retries, retry_wait_sec)
    transcript = parse_llm_response(content)

    if transcript is None:
        return None

    transcript["category"] = category_id
    transcript["data_type"] = data_type
    transcript["system_prompt"] = f"<system> {system_prompt} <system>"
    transcript["sampled_traits"] = {k: v for k, v in traits.items() if isinstance(v, (str, int, float, bool, list))}

    return transcript


# Thread-safe counter for assigning sequential IDs
_counter_lock = threading.Lock()


def _worker(
    model: str,
    category_id: str,
    category_data: dict,
    dt_weights: dict,
    cfg: dict,
) -> dict | None:
    """Worker function for ThreadPoolExecutor."""
    data_type = sample_data_type(dt_weights, category_id)
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["transcript"]
    set_seed(args.seed)

    # Configure OpenAI client (pointing at LiteLLM proxy)
    global _client
    model = cfg["llm_model"]
    base_url = cfg.get("llm_base_url") or None
    api_key = cfg.get("llm_api_key") or "unused"
    _client = OpenAI(base_url=base_url, api_key=api_key)

    categories = load_all_categories(cfg["categories_dir"])
    output_dir = ensure_dir(cfg["output_dir"])

    if args.category:
        categories = {k: v for k, v in categories.items() if k == args.category}

    pilot_counts = cfg.get("pilot_per_category", {})
    dt_weights = cfg.get("data_type_weights", {"standard": 1.0})
    num_workers = args.num_workers or cfg.get("num_workers", 8)

    for cat_id, cat_data in sorted(categories.items()):
        if isinstance(pilot_counts, dict):
            num_target = args.num_conversations or pilot_counts.get(cat_id, 60)
        else:
            num_target = args.num_conversations or pilot_counts

        cat_dir = ensure_dir(output_dir / cat_id)

        # Resume: count already generated
        existing = sorted(cat_dir.glob("*.json"))
        start_idx = len(existing)

        if start_idx >= num_target:
            print(f"[SKIP] {cat_id}: already have {start_idx}/{num_target}")
            continue

        remaining = num_target - start_idx
        print(f"\n=== {cat_id}: generating {remaining} more (have {start_idx}/{num_target}) | workers={num_workers} ===")

        generated = start_idx
        failed = 0
        max_failures = remaining * 2  # give up after too many failures

        # Use ThreadPoolExecutor for parallel LLM calls
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            # Submit initial batch
            futures = {}
            batch_size = min(remaining, num_workers * 2)
            for _ in range(batch_size):
                fut = pool.submit(_worker, model, cat_id, cat_data, dt_weights, cfg)
                futures[fut] = True

            while generated < num_target and failed < max_failures:
                done_futures = []
                for fut in as_completed(futures):
                    done_futures.append(fut)
                    try:
                        transcript = fut.result()
                    except Exception as e:
                        print(f"  [ERROR] Worker exception: {e}")
                        transcript = None

                    if transcript is not None:
                        with _counter_lock:
                            transcript["id"] = f"{cat_id}_{generated:05d}"
                            save_json(transcript, cat_dir / f"{generated:05d}.json")
                            generated += 1
                            if generated % 10 == 0:
                                print(f"  {generated}/{num_target} done")
                    else:
                        failed += 1

                    # Submit replacement if still need more
                    if generated < num_target and failed < max_failures:
                        new_fut = pool.submit(_worker, model, cat_id, cat_data, dt_weights, cfg)
                        futures[new_fut] = True

                    # Stop if target reached
                    if generated >= num_target:
                        break

                # Remove completed futures
                for fut in done_futures:
                    futures.pop(fut, None)

                if generated >= num_target:
                    break

        print(f"  Completed: {generated}/{num_target} (failed: {failed})")


if __name__ == "__main__":
    main()
