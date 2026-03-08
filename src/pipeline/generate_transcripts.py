"""Phase 1: Generate conversation transcripts using LLMs.

Samples characteristics from the data_categories YAML files,
feeds them to an LLM with a structured meta-prompt, and produces
JSON transcripts ready for TTS synthesis.

Usage:
    python -m pipeline.generate_transcripts \
        --config configs/generation.yaml \
        --category A3_tone_controlled \
        --num_conversations 70 \
        --seed 42
"""

import argparse
import json
import random
import re
from pathlib import Path

from openai import OpenAI

from pipeline.utils import ensure_dir, load_all_categories, load_yaml, save_json, set_seed

# Meta-prompt template for transcript generation
META_PROMPT = """\
You are generating a realistic two-person conversation transcript for training a speech-to-speech AI model.

## Task
Generate a conversation between an ASSISTANT and a USER based on the following specifications.

## Specifications
- Category: {category}
- Sampled Traits: {traits_json}
- System Prompt for Assistant: {system_prompt}
- TTS Style Instruction for Assistant: {system_prompt_tts_instruct}
- Number of turns: {num_turns} (total, alternating user/assistant)

## Requirements
1. The conversation MUST start with the user speaking first.
2. Each turn must include the spoken text and a `tts_instruct` field describing HOW it should sound.
3. The assistant's `tts_instruct` must be consistent with: "{system_prompt_tts_instruct}"
4. The user's `tts_instruct` should reflect realistic human speech patterns (casual, natural variation).
5. Include natural speech elements where appropriate: (laughs), (sighs), (pauses), "um", "uh".
6. Keep each turn 1-4 sentences. Conversations should feel natural, not scripted.
7. The assistant MUST consistently follow the system prompt personality/style throughout.
8. Do NOT include greetings like "How can I help you?" unless contextually natural.

## Output Format
Return ONLY valid JSON (no markdown, no explanation) with this exact structure:
{{
  "category": "{category}",
  "sampled_traits": {traits_json},
  "system_prompt": "{system_prompt}",
  "system_prompt_tts_instruct": "{system_prompt_tts_instruct}",
  "turns": [
    {{
      "role": "user",
      "text": "...",
      "tts_instruct": "..."
    }},
    {{
      "role": "assistant",
      "text": "...",
      "tts_instruct": "..."
    }}
  ]
}}
"""

SYSTEM_PROMPT_TEMPLATES = {
    "minimal": "You are an assistant. {trait_description}",
    "detailed": "You are an AI assistant with the following characteristics: {trait_description}. "
    "Maintain this style consistently throughout the conversation.",
    "highly_detailed": "You are an AI voice assistant engaged in a real-time spoken conversation. "
    "Your personality and speaking style: {trait_description}. "
    "You must maintain this exact style in every response. "
    "Respond naturally as if speaking aloud. Keep responses concise and conversational.",
}


def sample_traits(category_data: dict, category_id: str) -> dict:
    """Sample random traits from a category definition."""
    traits = {"category": category_id}

    if category_id.startswith("A1"):
        # Customer service: sample domain + scenario
        domains = list(category_data.get("domains", {}).keys())
        if domains:
            domain = random.choice(domains)
            traits["domain"] = domain
            scenarios = category_data["domains"][domain].get("scenarios", [])
            if scenarios:
                traits["scenario"] = random.choice(scenarios)

    elif category_id.startswith("A2"):
        # QA assistant: sample subcategory
        subcats = list(category_data.get("subcategories", {}).keys())
        if subcats:
            subcat = random.choice(subcats)
            traits["subcategory"] = subcat
            topics = category_data["subcategories"][subcat].get("topics", [])
            if topics:
                traits["topic"] = random.choice(topics)

    elif category_id.startswith("A3"):
        # Tone controlled: sample tone
        all_tones = []
        for group in ["negative", "neutral", "positive"]:
            tones = category_data.get("tones", {}).get(group, {})
            for tone_name, tone_data in tones.items():
                all_tones.append(
                    {
                        "name": tone_name,
                        "group": group,
                        **tone_data,
                    }
                )
        if all_tones:
            tone = random.choice(all_tones)
            traits["tone"] = tone["name"]
            traits["tone_group"] = tone["group"]
            traits["tts_instruct"] = tone.get("tts_instruct", "")
            traits["description"] = tone.get("description", "")
            topics = tone.get("suitable_topics", [])
            if topics:
                traits["topic"] = random.choice(topics)

    elif category_id.startswith("A4"):
        # Persona controlled: sample persona
        all_personas = []
        for subcat_name, subcat_data in category_data.get("subcategories", {}).items():
            personas = subcat_data.get("personas", {})
            for persona_name, persona_data in personas.items():
                all_personas.append(
                    {
                        "name": persona_name,
                        "subcategory": subcat_name,
                        **persona_data,
                    }
                )
        if all_personas:
            persona = random.choice(all_personas)
            traits["persona"] = persona["name"]
            traits["subcategory"] = persona["subcategory"]
            traits["description"] = persona.get("description", "")
            traits["tts_instruct"] = persona.get("tts_instruct", "")

    elif category_id.startswith("A5"):
        # Style & accent
        styles = list(category_data.get("styles", {}).keys())
        accents = list(category_data.get("accents", {}).keys())
        if styles:
            style = random.choice(styles)
            traits["style"] = style
            traits["style_data"] = category_data["styles"][style]
        if accents:
            accent = random.choice(accents)
            traits["accent"] = accent
            traits["accent_data"] = category_data["accents"][accent]

    elif category_id.startswith("A6"):
        # Speed & length
        speeds = list(category_data.get("speeds", {}).keys())
        lengths = list(category_data.get("lengths", {}).keys())
        if speeds:
            traits["speed"] = random.choice(speeds)
        if lengths:
            traits["length"] = random.choice(lengths)

    elif category_id.startswith("A7"):
        # Emotional/empathetic
        emotions = list(category_data.get("user_emotions", {}).keys())
        if emotions:
            emotion = random.choice(emotions)
            traits["user_emotion"] = emotion
            traits["emotion_data"] = category_data["user_emotions"][emotion]

    elif category_id.startswith("A8"):
        # Failure cases
        cases = list(category_data.get("cases", {}).keys())
        if cases:
            case = random.choice(cases)
            traits["case"] = case
            traits["case_data"] = category_data["cases"][case]

    return traits


def build_system_prompt(traits: dict, granularity: str) -> tuple[str, str]:
    """Build system prompt and TTS instruct from sampled traits."""
    # Build trait description
    parts = []
    if "tone" in traits:
        parts.append(f"Speak with a {traits['tone']} tone. {traits.get('description', '')}")
    if "persona" in traits:
        parts.append(f"You are {traits['persona']}. {traits.get('description', '')}")
    if "style" in traits:
        style_data = traits.get("style_data", {})
        parts.append(f"Speak in a {traits['style']} style. {style_data.get('description', '')}")
    if "accent" in traits:
        accent_data = traits.get("accent_data", {})
        parts.append(f"Speak with a {traits['accent']} accent. {accent_data.get('description', '')}")
    if "speed" in traits:
        parts.append(f"Speak at a {traits['speed']} pace.")
    if "length" in traits:
        parts.append(f"Keep responses {traits['length']}.")
    if "domain" in traits:
        parts.append(f"You are a {traits['domain']} customer service agent.")
    if "scenario" in traits:
        parts.append(f"Handle this scenario: {traits['scenario']}.")
    if "user_emotion" in traits:
        emotion_data = traits.get("emotion_data", {})
        parts.append(
            f"The user is feeling {traits['user_emotion']}. "
            f"{emotion_data.get('assistant_response', 'Respond appropriately.')}"
        )
    if "case" in traits:
        case_data = traits.get("case_data", {})
        parts.append(f"Edge case: {traits['case']}. {case_data.get('description', '')}")

    trait_description = " ".join(parts) if parts else "Be helpful and conversational."

    template = SYSTEM_PROMPT_TEMPLATES.get(granularity, SYSTEM_PROMPT_TEMPLATES["detailed"])
    system_prompt = template.format(trait_description=trait_description)

    # TTS instruct
    tts_instruct = traits.get("tts_instruct", "natural, conversational, clear enunciation")

    return system_prompt, tts_instruct


def generate_single_transcript(
    client: OpenAI,
    model: str,
    category_id: str,
    category_data: dict,
    temperature: float,
    max_tokens: int,
    turns_range: tuple[int, int],
    granularity: str,
) -> dict | None:
    """Generate a single conversation transcript via LLM."""
    traits = sample_traits(category_data, category_id)
    system_prompt, tts_instruct = build_system_prompt(traits, granularity)
    num_turns = random.randint(turns_range[0], turns_range[1])

    # Clean traits for JSON serialization (remove nested dicts with non-serializable data)
    clean_traits = {}
    for k, v in traits.items():
        if isinstance(v, (str, int, float, bool, list)):
            clean_traits[k] = v

    prompt = META_PROMPT.format(
        category=category_id,
        traits_json=json.dumps(clean_traits, indent=2),
        system_prompt=system_prompt,
        system_prompt_tts_instruct=tts_instruct,
        num_turns=num_turns,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content.strip()

        # Extract JSON from response (handle markdown code blocks)
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\n?", "", content)
            content = re.sub(r"\n?```$", "", content)

        transcript = json.loads(content)

        # Validate structure
        if "turns" not in transcript or len(transcript["turns"]) < 2:
            print(f"  [WARN] Invalid transcript: missing/short turns")
            return None
        for turn in transcript["turns"]:
            if "role" not in turn or "text" not in turn:
                print(f"  [WARN] Invalid turn: missing role/text")
                return None

        return transcript

    except (json.JSONDecodeError, Exception) as e:
        print(f"  [ERROR] Failed to generate transcript: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Generate conversation transcripts")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--category", type=str, default=None, help="Specific category (e.g. A3_tone_controlled)")
    parser.add_argument("--num_conversations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["transcript"]
    set_seed(args.seed)

    client = OpenAI(base_url=cfg["llm_base_url"], api_key="unused")
    categories = load_all_categories(cfg["categories_dir"])
    output_dir = ensure_dir(cfg["output_dir"])

    # Filter to specific category if requested
    if args.category:
        categories = {k: v for k, v in categories.items() if k == args.category}

    num_per_cat = args.num_conversations or cfg["pilot_per_category"]
    granularities = cfg["system_prompt_granularities"]

    for cat_id, cat_data in categories.items():
        cat_dir = ensure_dir(output_dir / cat_id)
        print(f"\n=== Generating {num_per_cat} transcripts for {cat_id} ===")

        generated = 0
        attempts = 0
        max_attempts = num_per_cat * 3

        while generated < num_per_cat and attempts < max_attempts:
            attempts += 1
            granularity = random.choice(granularities)

            transcript = generate_single_transcript(
                client=client,
                model=cfg["llm_model"],
                category_id=cat_id,
                category_data=cat_data,
                temperature=cfg["temperature"],
                max_tokens=cfg["max_tokens"],
                turns_range=tuple(cfg["turns_range"]),
                granularity=granularity,
            )

            if transcript is not None:
                transcript["id"] = f"{cat_id}_{generated:05d}"
                transcript["granularity"] = granularity
                save_json(transcript, cat_dir / f"{generated:05d}.json")
                generated += 1
                if generated % 10 == 0:
                    print(f"  {generated}/{num_per_cat} done")

        print(f"  Completed: {generated}/{num_per_cat} (attempts: {attempts})")


if __name__ == "__main__":
    main()
