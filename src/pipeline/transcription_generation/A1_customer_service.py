"""Phase 1 (A1 customer service only): Generate customer-service transcripts.

Pulled out of generate_transcripts.py so the customer-service flow can be
iterated on independently. Same output schema and on-disk path as the shared
generator, so downstream pipeline phases (rephrase_system_prompts,
synthesize_tts, assemble_channels, format_dataset) work unchanged.

Mirrors the existing A1 flow in generate_transcripts.py with these deltas:
- Wires in user_traits.mood_distribution (weighted) and assistant_traits
  .tts_instruct_templates (uniform); both flow into the meta-prompt and pin
  per-turn tts_instruct.
- Wires in conversation_structure (greeting opener, clear resolution close,
  backchannel ~0.30) as explicit rules in the meta-prompt.
- For data_type="graceful_failure", samples failure_type from a CS-specific
  list instead of always falling back to "clarification_requests".
- Adds CS response-style guidance: confirm/restate/commit values, interim
  summary once details are gathered, resolution summary at the end, brevity
  (no full restating every turn).

Usage:
    cd src && python -m pipeline.transcription_generation.A1_customer_service \
        --config configs/generation.yaml --scale pilot
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

import pipeline.generate_transcripts as _gt
from pipeline.distributed import is_done, release_claim, try_claim
from pipeline.generate_transcripts import (
    SYSTEM_PROMPT_TEMPLATES,
    call_llm,
    parse_llm_response,
    sample_data_type,
)
from pipeline.utils import ensure_dir, load_yaml, save_json, set_seed

CATEGORY_ID = "A1_customer_service"


# ---------------------------------------------------------------------------
# CS-specific failure types (used only when data_type == "graceful_failure").
# Replaces the generic "clarification_requests" fallback that A1 falls into in
# generate_transcripts.py because no failure_type is ever set on A1 traits.
# ---------------------------------------------------------------------------
CS_FAILURE_TYPES: list[str] = [
    "cant_authenticate",         # user fails identity verification
    "system_outage",             # backend down, agent can't look up record
    "cant_resolve",              # issue exists but is beyond agent's authority
    "escalate_to_human",         # needs supervisor / specialist department
    "refund_denied",             # request reasonable but policy says no
    "knowledge_limit",           # agent doesn't have the answer
    "frustration_deescalation",  # user is angry, agent stays calm and redirects
    "clarification_requests",    # user request is unclear
]


# Counterfactual axis pairs — same table as generate_transcripts.py:847.
# Duplicated here so the module is self-contained.
COUNTERFACTUAL_DIMENSIONS: dict[str, list[tuple[str, str]]] = {
    "tone":   [("cheerful", "serious"), ("sarcastic", "sincere"),
               ("formal", "casual"), ("empathetic", "matter-of-fact")],
    "speed":  [("fast", "slow"), ("very_fast", "moderate")],
    "energy": [("high_energy", "calm"), ("excited", "subdued")],
}


# ---------------------------------------------------------------------------
# CS-specific blocks injected into every meta-prompt (regardless of data_type).
# ---------------------------------------------------------------------------
def _render_cs_context_block(traits: dict) -> str:
    return (
        "## Customer-service context\n"
        f"- Domain: {traits['domain']}\n"
        f"- Scenario: {traits['scenario']}\n"
        f"- User mood: {traits['user_mood']} — they sound: {traits['user_tts']}\n"
        f"- Assistant tone: {traits['assistant_tone']} — they sound: {traits['assistant_tts']}\n"
        "\n"
        "Every USER turn's tts_instruct must be a short variant of:\n"
        f"  \"{traits['user_tts']}\"\n"
        "Every ASSISTANT turn's tts_instruct must be a short variant of:\n"
        f"  \"{traits['assistant_tts']}\"\n"
        "Max 3 comma-separated descriptors per tts_instruct."
    )


def _render_structure_block(conv_structure: dict, domain: str) -> str:
    backchannel_pct = int(round(float(conv_structure.get("backchannel_probability", 0.3)) * 100))
    parts = ["## Conversation structure (CS-specific, REQUIRED)"]
    if conv_structure.get("includes_greeting", True):
        parts.append(
            f"- GREETING: the assistant's FIRST turn must open with a customer-service "
            f"greeting that names the company/{domain} and offers help — e.g. "
            f"\"Thanks for calling {domain.replace('_', ' ').title()} support, this is "
            f"[agent name], how can I help today?\""
        )
    if conv_structure.get("includes_resolution", True):
        parts.append(
            "- RESOLUTION: the FINAL assistant turn must close cleanly — confirm the "
            "issue is resolved, escalated, or scheduled for follow-up. Do NOT trail off. "
            "End with an offer to help with anything else, or a polite sign-off."
        )
    parts.append(
        f"- BACKCHANNELS: roughly {backchannel_pct}% of turns should include a short "
        "acknowledgement noise (\"mm-hmm\", \"I see\", \"right\", \"okay\") naturally "
        "embedded — these are speakable words, not stage directions."
    )
    return "\n".join(parts)


_RESPONSE_STYLE_BLOCK = """\
## Response style (CS-specific, REQUIRED)
- CONFIRM, DON'T NOD: the assistant must NOT just say bare "that sounds good", "sure", or "I can do that". When the user provides a value (date, time, name, address, account number, item, dollar amount), the assistant must briefly RESTATE that value back, ASK for confirmation where appropriate, or COMMIT it explicitly. Examples:
    BAD : "Sure, I can book that."
    GOOD: "Got it — Tuesday the eighth at three p.m., one adult, under the name Patel. Want me to lock that in?"
- INTERIM SUMMARY: once the assistant has gathered the key details needed to act (typically one or two turns before action), the assistant must produce a SHORT summary of those details and confirm before proceeding. Brief — only the load-bearing facts, not a full recap.
- RESOLUTION SUMMARY: in the final assistant turn (or the one just before sign-off), the assistant must summarize what was done and what happens next — e.g. "So I've cancelled the older charge and started the refund — you'll see it in three to five business days."
- BREVITY: do NOT re-state every detail every turn. After the interim summary, refer to gathered facts compactly ("the three-p.m. booking", "your refund") rather than repeating the full chain. Brevity over completeness, every turn."""


def _build_cs_blocks(traits: dict, conv_structure: dict) -> str:
    """The three CS-specific blocks, joined with blank lines, ready to inject."""
    return "\n\n".join([
        _render_cs_context_block(traits),
        _render_structure_block(conv_structure, traits["domain"]),
        _RESPONSE_STYLE_BLOCK,
    ])


# ---------------------------------------------------------------------------
# Data-type-specific meta-prompts — same 5 shapes as generate_transcripts.py
# (standard / dynamic / counterfactual / long_form / graceful_failure), each
# extended with the CS context, structure, and response-style blocks.
# ---------------------------------------------------------------------------
_META_STANDARD = """\
Generate a realistic spoken conversation between a USER and an ASSISTANT.

## Context
- Category: {category}
- Traits: {traits_json}
- System prompt (for the assistant's role): {system_prompt}
- Number of turns: {num_turns} (alternating user/assistant, user speaks first)

{cs_blocks}

## CRITICAL Rules
1. User speaks FIRST.
2. The assistant FOLLOWS the assigned tone and stays in-character throughout.
3. Each turn has `tts_instruct` describing HOW it should sound (for TTS synthesis).
4. Assistant tts_instruct must match the assigned assistant tone.
5. User tts_instruct must reflect the assigned user mood.
6. Include natural filler words: "um", "uh", "like", "you know", "I mean". Do NOT include non-spoken stage directions like (laughs), (sighs), (pauses) — only speakable words.
7. Keep turns 1-4 sentences. Natural, not scripted.
8. Assistant tts_instruct should be SHORT — max 3 comma-separated descriptors. Keep it subtle.

## Output: ONLY valid JSON, no markdown
{{
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""


_META_DYNAMIC = """\
Generate a spoken conversation where the USER changes the ASSISTANT's speaking style MID-CONVERSATION.

## Context
- Category: {category}
- Traits: {traits_json}
- System prompt: {system_prompt}
- Number of turns: {num_turns} (user speaks first)
- Steering events: {num_steering} time(s) the user changes the assistant's style

{cs_blocks}

## CRITICAL Rules
1. User speaks first with a normal customer-service request.
2. At {num_steering} point(s) during the conversation, the user gives an explicit instruction to change how the assistant speaks (e.g., "slow down", "be more casual", "drop the formal tone").
3. The assistant ADAPTS IMMEDIATELY in its next turn — both text content and tts_instruct must change. After adaptation, the assistant's tts_instruct may diverge from the assigned baseline tone — that's expected.
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


_META_COUNTERFACTUAL = """\
Generate a SHORT spoken conversation (3-4 turns) that can be rendered in two different styles.

## Context
- Category: {category}
- Traits: {traits_json}
- Style A: {style_a}
- Style B: {style_b}
- Differing dimension: {diff_dimension}

{cs_blocks}

## CRITICAL Rules
1. Generate ONE conversation with neutral text that works for BOTH styles.
2. Provide TWO sets of tts_instruct on assistant turns — one for style_a, one for style_b. (User turns get a single tts_instruct matching the assigned user mood.)
3. ONLY the {diff_dimension} should differ between variants.
4. Keep it 3-4 turns, simple customer-service topic — the conversation structure rules above (greeting, resolution) still apply, just compressed.
5. Only speakable words in text — no (laughs), (sighs), (pauses). Keep tts_instruct SHORT (max 3 descriptors).

## Output: ONLY valid JSON, no markdown
{{
  "diff_dimension": "{diff_dimension}",
  "style_a": "{style_a}",
  "style_b": "{style_b}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct_a": "...", "tts_instruct_b": "..."}}
  ]
}}"""


_META_LONG_FORM = """\
Generate a LONG spoken customer-service conversation ({num_turns} turns) testing multi-turn coherence.

## Context
- Category: {category}
- Traits: {traits_json}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

{cs_blocks}

## CRITICAL Rules
1. User speaks first.
2. Include topic changes (at least 1-2) — e.g., user adds a related request, or the agent surfaces an upsell/cross-issue.
3. Reference proper nouns from earlier turns (test recall) — e.g., the agent recalls the order number, address, or member tier mentioned earlier.
4. The assistant must maintain its assigned tone consistently over many turns.
5. Include natural filler words ("um", "uh") but NO stage directions like (laughs), (sighs), (pauses).
6. Each turn has tts_instruct — keep it SHORT, max 3 comma-separated descriptors.

## Output: ONLY valid JSON, no markdown
{{
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""


_META_GRACEFUL_FAILURE = """\
Generate a spoken customer-service conversation where the ASSISTANT must handle a difficult situation gracefully.

## Context
- Failure type: {failure_type}
- Scenario: {scenario}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

{cs_blocks}

## CRITICAL Rules
1. User speaks first.
2. At some point, the user says something the assistant can't fulfill, doesn't understand, or runs into a system/policy limit (per the failure type).
3. The assistant must handle it GRACEFULLY — not go silent, not give a robotic refusal.
4. The assistant should: acknowledge the issue, briefly explain the limit, and offer an alternative or redirect (transfer, callback, alternative process, escalation path).
5. The conversation should feel NATURAL, not like a test.
6. Each turn has tts_instruct — keep it SHORT, max 3 comma-separated descriptors.
7. Only speakable words in text — no (laughs), (sighs), (pauses).

## Failure type guidance
- cant_authenticate: user fails identity verification (wrong DOB / no account match) — agent declines safely, offers in-branch / callback verification.
- system_outage: backend is down — agent acknowledges, offers callback once systems restore, gives ETA if known.
- cant_resolve: issue is real but beyond agent's authority — agent escalates with named next step.
- escalate_to_human: routes to supervisor or specialist department — agent stays warm, transitions cleanly.
- refund_denied: request is reasonable but policy says no — agent declines empathetically, explains policy briefly, offers any partial alternative.
- knowledge_limit: agent doesn't have the answer — agent says so honestly, offers to follow up or transfer.
- frustration_deescalation: user is angry — agent stays calm, validates the frustration without parroting, redirects to action.
- clarification_requests: user request is unclear — agent asks one specific clarifying question, not a barrage.

## Output: ONLY valid JSON, no markdown
{{
  "failure_type": "{failure_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""


# ---------------------------------------------------------------------------
# Trait sampling
# ---------------------------------------------------------------------------
def _weighted_choice(weight_map: dict[str, float]) -> str:
    keys = list(weight_map.keys())
    weights = [float(weight_map[k]) for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def _sample_traits(category_data: dict) -> dict:
    # 1. Domain + scenario — same as generate_transcripts.py:530-537
    domains = list(category_data["domains"].keys())
    domain = random.choice(domains)
    scenarios = category_data["domains"][domain]["scenarios"]
    scenario = random.choice(scenarios)

    # 2. User mood — weighted by user_traits.mood_distribution
    user_traits = category_data.get("user_traits", {})
    moods = user_traits.get("mood_distribution", {"neutral": 1.0})
    user_mood = _weighted_choice(moods)
    user_tts_templates = user_traits.get("tts_instruct_templates", {})
    user_tts = user_tts_templates.get(user_mood, "conversational, normal pace, matter-of-fact")

    # 3. Assistant tone — uniform from tts_instruct_templates
    asst_traits = category_data.get("assistant_traits", {})
    tone_templates = asst_traits.get("tts_instruct_templates", {
        "professional": "clear, professional tone, moderate pace, confident delivery"
    })
    assistant_tone = random.choice(list(tone_templates.keys()))
    assistant_tts = tone_templates[assistant_tone]

    return {
        "category": CATEGORY_ID,
        "domain": domain,
        "scenario": scenario,
        "user_mood": user_mood,
        "user_tts": user_tts,
        "assistant_tone": assistant_tone,
        "assistant_tts": assistant_tts,
    }


def _sample_failure_type() -> str:
    return random.choice(CS_FAILURE_TYPES)


def _build_system_prompt(traits: dict) -> str:
    # Single template parameterized by domain — unchanged from
    # generate_transcripts.py:786-787.
    return SYSTEM_PROMPT_TEMPLATES["customer_service"].format(domain=traits["domain"])


def _sample_num_turns(data_type: str, conv_structure: dict) -> int:
    """Per-data-type turn count, mirroring generate_transcripts.py:867-905."""
    cs_lo = int(conv_structure.get("min_turns", 4))
    cs_hi = int(conv_structure.get("max_turns", 10))
    if data_type == "standard":
        return random.randint(cs_lo, cs_hi)
    if data_type == "dynamic":
        return random.randint(max(6, cs_lo), min(14, cs_hi + 4))
    if data_type == "long_form":
        return random.randint(10, 15)
    if data_type == "graceful_failure":
        return random.randint(4, 8)
    # counterfactual — handled inside that template (3-4 turns)
    return random.randint(cs_lo, cs_hi)


def _build_meta_prompt(
    traits: dict,
    data_type: str,
    system_prompt: str,
    conv_structure: dict,
) -> str:
    cs_blocks = _build_cs_blocks(traits, conv_structure)
    # Trim non-scalar fields out of traits_json (the meta-prompt only wants
    # human-readable values, not nested dicts).
    clean_traits = {k: v for k, v in traits.items()
                    if isinstance(v, (str, int, float, bool, list))}
    traits_json = json.dumps(clean_traits, indent=2)
    category = traits["category"]

    if data_type == "standard":
        num_turns = _sample_num_turns(data_type, conv_structure)
        return _META_STANDARD.format(
            category=category, traits_json=traits_json,
            system_prompt=system_prompt, num_turns=num_turns,
            cs_blocks=cs_blocks,
        )

    if data_type == "dynamic":
        num_turns = _sample_num_turns(data_type, conv_structure)
        num_steering = random.randint(1, 3)
        return _META_DYNAMIC.format(
            category=category, traits_json=traits_json,
            system_prompt=system_prompt, num_turns=num_turns,
            num_steering=num_steering, cs_blocks=cs_blocks,
        )

    if data_type == "counterfactual":
        dim = random.choice(list(COUNTERFACTUAL_DIMENSIONS.keys()))
        pair = random.choice(COUNTERFACTUAL_DIMENSIONS[dim])
        return _META_COUNTERFACTUAL.format(
            category=category, traits_json=traits_json,
            style_a=pair[0], style_b=pair[1], diff_dimension=dim,
            cs_blocks=cs_blocks,
        )

    if data_type == "long_form":
        num_turns = _sample_num_turns(data_type, conv_structure)
        return _META_LONG_FORM.format(
            category=category, traits_json=traits_json,
            system_prompt=system_prompt, num_turns=num_turns,
            cs_blocks=cs_blocks,
        )

    if data_type == "graceful_failure":
        num_turns = _sample_num_turns(data_type, conv_structure)
        return _META_GRACEFUL_FAILURE.format(
            failure_type=traits["failure_type"],
            scenario=traits["scenario"],
            system_prompt=system_prompt,
            num_turns=num_turns,
            cs_blocks=cs_blocks,
        )

    # Fallback — shouldn't hit
    num_turns = _sample_num_turns("standard", conv_structure)
    return _META_STANDARD.format(
        category=category, traits_json=traits_json,
        system_prompt=system_prompt, num_turns=num_turns,
        cs_blocks=cs_blocks,
    )


# ---------------------------------------------------------------------------
# Per-conversation runner
# ---------------------------------------------------------------------------
def generate_single_cs(
    category_data: dict,
    data_type: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_wait_sec: float,
) -> dict | None:
    selected_model = _gt._pick_model()

    traits = _sample_traits(category_data)
    if data_type == "graceful_failure":
        traits["failure_type"] = _sample_failure_type()

    system_prompt = _build_system_prompt(traits)
    conv_structure = category_data.get("conversation_structure", {})

    prompt = _build_meta_prompt(traits, data_type, system_prompt, conv_structure)

    content = call_llm(
        selected_model, prompt, temperature, max_tokens, max_retries, retry_wait_sec,
    )
    transcript = parse_llm_response(content) if content else None
    if transcript is None:
        return None

    transcript["category"] = CATEGORY_ID
    transcript["data_type"] = data_type
    transcript["system_prompt"] = f"<system> {system_prompt} <system>"
    transcript["sampled_traits"] = {
        k: v for k, v in traits.items()
        if isinstance(v, (str, int, float, bool, list)) or v is None
    }
    transcript["llm_model"] = selected_model
    return transcript


def _worker(category_data: dict, dt_weights: dict, cfg: dict) -> dict | None:
    data_type = sample_data_type(dt_weights, CATEGORY_ID, category_data)
    return generate_single_cs(
        category_data=category_data,
        data_type=data_type,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        max_retries=cfg.get("max_retries", 3),
        retry_wait_sec=cfg.get("retry_wait_sec", 5),
    )


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate A1 customer-service transcripts")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--num_conversations", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--scale", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["transcript"]
    set_seed(args.seed)

    base_url = cfg.get("llm_base_url") or None
    api_key = cfg.get("llm_api_key") or None
    _gt._client = OpenAI(base_url=base_url, api_key=api_key)

    llm_models_cfg = cfg.get("llm_models")
    if llm_models_cfg and isinstance(llm_models_cfg, list):
        total_weight = sum(m.get("weight", 1.0) for m in llm_models_cfg)
        cum = 0.0
        choices: list[tuple[str, float]] = []
        for m in llm_models_cfg:
            cum += m.get("weight", 1.0) / total_weight
            choices.append((m["model"], cum))
        _gt._model_choices = choices
        names = [m["model"] for m in llm_models_cfg]
        weights = [m.get("weight", 1.0) for m in llm_models_cfg]
        print(f"LLM models: {list(zip(names, weights))}")
    else:
        model = cfg["llm_model"]
        _gt._model_choices = [(model, 1.0)]
        print(f"LLM model: {model}")

    categories_dir = Path(cfg["categories_dir"])
    category_data = load_yaml(categories_dir / f"{CATEGORY_ID}.yaml")

    output_dir = ensure_dir(cfg["output_dir"])
    cat_dir = ensure_dir(output_dir / CATEGORY_ID)
    claims_dir = ensure_dir(Path(cfg["output_dir"]) / ".claims" / CATEGORY_ID)

    counts_key = "full_per_category" if args.scale == "full" else "pilot_per_category"
    count_map = cfg.get(counts_key, cfg.get("pilot_per_category", {}))
    if isinstance(count_map, dict):
        num_target = args.num_conversations or count_map.get(CATEGORY_ID, 60)
    else:
        num_target = args.num_conversations or count_map

    dt_weights = cfg.get("data_type_weights", {"standard": 1.0})
    num_workers = args.num_workers or cfg.get("num_workers", 8)

    done_count = sum(1 for i in range(num_target) if is_done(cat_dir / f"{i:05d}.json"))
    if done_count >= num_target:
        print(f"[SKIP] {CATEGORY_ID}: all {num_target} done")
        return

    remaining = num_target - done_count
    print(f"\n=== {CATEGORY_ID}: {remaining} remaining "
          f"({done_count}/{num_target} done) | workers={num_workers} ===")

    slots = [i for i in range(num_target) if not is_done(cat_dir / f"{i:05d}.json")]
    random.shuffle(slots)

    generated = 0
    failed = 0
    skipped = 0

    def _claim_and_generate(slot_idx: int):
        claim_path = claims_dir / f"{slot_idx:05d}.claim"
        out_path = cat_dir / f"{slot_idx:05d}.json"
        if is_done(out_path):
            return slot_idx, "skip"
        if not try_claim(claim_path):
            return slot_idx, "skip"
        try:
            transcript = _worker(category_data, dt_weights, cfg)
            if transcript is not None:
                transcript["id"] = f"{CATEGORY_ID}_{slot_idx:05d}"
                save_json(transcript, out_path)
                release_claim(claim_path)
            else:
                release_claim(claim_path)
            return slot_idx, transcript
        except Exception as e:
            release_claim(claim_path)
            print(f"  [ERROR] slot {slot_idx}: {e}")
            return slot_idx, None

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_claim_and_generate, s): s for s in slots}
        with tqdm(total=len(slots), desc=CATEGORY_ID, unit="conv") as pbar:
            for fut in as_completed(futures):
                try:
                    _, result = fut.result()
                except Exception as e:
                    pbar.write(f"  [ERROR] Worker exception: {e}")
                    failed += 1
                else:
                    if result == "skip":
                        skipped += 1
                    elif result is not None:
                        generated += 1
                    else:
                        failed += 1
                pbar.set_postfix(gen=generated, fail=failed, skip=skipped)
                pbar.update(1)

    print(f"  {CATEGORY_ID}: generated {generated}, failed {failed}, skipped {skipped}")


if __name__ == "__main__":
    main()
