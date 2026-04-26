"""Phase 1 (B4 instruction following only): Generate instruction-following transcripts.

Pulled out of generate_transcripts.py so instruction-following prompts and trait
sampling can be iterated on independently. Same output schema and on-disk path
as the shared generator, so downstream pipeline phases (rephrase_system_prompts,
synthesize_tts, assemble_channels, format_dataset) work unchanged.

Key differences from the shared generator:
- 10 categories: 6 single-turn primitives + 4 multi-turn shapes
- Weighted sampling of categories from yaml `sample_weight` per subcategory
- Each turn carries a `verification` field with concrete, mechanically-checkable
  predicates (or null when no rule applies)
- Sampled `has_slip` flag (multi-turn shapes only) for slip-and-recover patterns
- Sampled `num_rules` ∈ {1, 2, 3} for persistent_rule (stacked-constraint variant)
- Speech-only constraints — no markdown, capitalization, or visual layout
- Always-satisfiable constraints — assistant never refuses

Usage:
    cd src && python -m pipeline.transcription_generation.B4_instruction_following \
        --config configs/generation.yaml --scale pilot
"""

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

import pipeline.generate_transcripts as _gt
from pipeline.distributed import is_done, release_claim, try_claim
from pipeline.generate_transcripts import (
    call_llm,
    parse_llm_response,
    sample_data_type,
)
from pipeline.utils import ensure_dir, load_yaml, save_json, set_seed

CATEGORY_ID = "B4_instruction_following"


# ---------------------------------------------------------------------------
# Per-category turn ranges (lo, typical, hi) — overrides yaml defaults.
# Single-turn primitives need 3-4 turns (instruction, answer, light follow-up).
# Multi-turn shapes need 6-8 turns to actually exercise the pattern.
# ---------------------------------------------------------------------------
_TURN_RANGES: dict[str, tuple[int, int, int]] = {
    # Single-turn primitives
    "count": (3, 3, 4),
    "lexical_inclusion": (3, 3, 4),
    "lexical_exclusion": (3, 3, 4),
    "framing": (3, 3, 4),
    "answer_property": (3, 3, 4),
    "multi_step_instructions": (3, 4, 4),
    # Multi-turn shapes
    "persistent_rule": (6, 7, 8),
    "conditional_trigger": (6, 7, 8),
    "rule_update": (6, 7, 8),
    "sequential_task_chains": (6, 8, 10),
}

# Subcategories that are "multi-turn" in shape (eligible for has_slip)
_MULTI_TURN_SUBCATS: set[str] = {
    "persistent_rule",
    "conditional_trigger",
    "rule_update",
    "sequential_task_chains",
}


# ---------------------------------------------------------------------------
# System prompts — one per subcategory
# ---------------------------------------------------------------------------
INSTRUCTION_SYSTEM_PROMPT_TEMPLATES: dict[str, str] = {
    "count": (
        "You are a precise voice assistant. When the user asks for an exact "
        "number of items or sentences, deliver that exact number — not one "
        "more, not one less. Verbally enumerate ('one... two... three...') "
        "where it sounds natural. Never refuse, never complain."
    ),
    "lexical_inclusion": (
        "You are a precise voice assistant. When the user requires a specific "
        "word or phrase to appear, weave it in naturally and use it at least "
        "once. Don't draw attention to the requirement. Never refuse."
    ),
    "lexical_exclusion": (
        "You are a precise voice assistant. When the user bans a word, never "
        "utter it — paraphrase, use synonyms, or describe around it. Still "
        "answer the underlying question fully. Never refuse, never complain."
    ),
    "framing": (
        "You are a precise voice assistant. When the user requires a specific "
        "opener or closer, use that exact phrase verbatim, then answer. Don't "
        "paraphrase the required phrase. Never refuse."
    ),
    "answer_property": (
        "You are a precise voice assistant. When the user constrains the form "
        "of the answer (one word, just a number, just yes or no), match that "
        "form exactly — no preamble, no explanation, no full sentence. Never "
        "refuse."
    ),
    "multi_step_instructions": (
        "You are a precise voice assistant. When the user names multiple "
        "steps in one turn, complete every step in the order requested, in "
        "one response. Don't skip a step. Verbally signal transitions "
        "('first... then... finally...'). Never refuse."
    ),
    "persistent_rule": (
        "You are a precise voice assistant with steady working memory. When "
        "the user installs a persistent rule (or a stack of rules), apply it "
        "on EVERY subsequent response — not just the first. If you slip, "
        "briefly acknowledge it on the next turn and resume compliance. Never "
        "refuse, never complain about the rule."
    ),
    "conditional_trigger": (
        "You are a precise voice assistant. The user has installed a "
        "conditional rule that fires ONLY when a specific keyword appears in "
        "their turn. Listen for the trigger every turn — fire the conditional "
        "behavior when it appears, answer normally when it doesn't. Never refuse."
    ),
    "rule_update": (
        "You are a precise voice assistant who tracks rule updates. When the "
        "user modifies or revokes a rule mid-conversation, switch immediately "
        "on the very next response. Don't keep an old rule active after the "
        "user has changed it. Never refuse."
    ),
    "sequential_task_chains": (
        "You are a precise voice assistant. When the user gives a multi-turn "
        "chain where later steps depend on your earlier answers, carry your "
        "ACTUAL earlier outputs forward — if you said 'forty-two' on turn 1, "
        "later turns must use forty-two, not a fresh number. Never refuse."
    ),
}

DEFAULT_INSTRUCTION_SYSTEM_PROMPT = (
    "You are a precise voice assistant that follows instructions exactly. "
    "Never refuse, never complain, never ask unnecessary clarifications."
)


# ---------------------------------------------------------------------------
# Meta-prompt — assembled from a base + conditional sections
# ---------------------------------------------------------------------------
_META_BASE = """\
Generate a realistic spoken conversation between a USER and an ASSISTANT that exercises an INSTRUCTION-FOLLOWING scenario.

## Output medium — SPEECH ONLY
The conversation will be rendered as audio via TTS. Both sides are SPOKEN. There is no page, no screen, no formatting. Every constraint, instruction, and response must be verifiable from the spoken audio ALONE.

The user MUST NOT install instructions of the form:
  - "respond in bullet points" / "use a numbered list" / "use markdown"
  - "capitalize every word" / "all caps" / "title case"
  - "bold" / "italicize" / "underline" / "indent" / "use a table" / "code block"
  - anything that requires the listener to SEE the response rather than HEAR it

Speech-verifiable instructions (these are FINE):
  - word counts, sentence counts ("under thirty words", "exactly two sentences")
  - banned or required words ("never say 'left'", "include the word 'sunlight'")
  - required opening or closing phrases ("start with 'Well'", "end with a fun fact")
  - vocabulary level ("words a five-year-old would know")
  - alphabet-start patterns (the first SOUND of each sentence is audible)
  - verbal numbering ("the assistant says one... two... three...")
  - required content elements (a question, a statistic, a fun fact)
  - conditional triggers based on what the user SAYS

## Always-satisfiable rule
The constraint must be one the assistant can satisfy WHILE answering the user's question helpfully and truthfully. The assistant MUST NEVER refuse, complain that the constraint is hard, or ask unnecessary clarifications. Do not invent constraints that conflict with safety, truth, or the underlying question (no traps like "describe a sunset without color words").

## Context
- Subcategory: {subcategory}
- Shape: {shape}
- What this subcategory tests: {description}
- How the assistant should approach it: {approach}
- User persona: {user_persona} — they tend to come across as: {user_tts}
- Assistant TTS style: {assistant_tts}
- System prompt for the assistant: {system_prompt}
- Number of turns: {num_turns}
- Backchannel probability: {backchannel_probability}
{shape_specific_block}
## Reference for this subcategory (use as inspiration — do NOT copy the example verbatim; vary the specific number, word, or phrasing)
{example_block}

## Per-turn verification field (REQUIRED)
Every turn in the output JSON has a "verification" field:
- USER turns: ALWAYS null.
- ASSISTANT turns where NO active rule applies on this turn (e.g. an unrelated turn, or a conditional trigger that did not fire): null.
- ASSISTANT turns where one or more rules apply: a LIST of concrete check strings.

Verification check rules:
- Name what to look for CONCRETELY — exact counts, exact words, specific properties.
- A skeptical listener should be able to mechanically verify each check from the spoken audio alone.
- DO NOT use vague language ("follows the rule", "complies properly", "satisfies the request").
- For persistent rules, the SAME checks repeat across applicable turns — that's correct, don't paraphrase.

GOOD verification strings:
  - "contains exactly three distinct fruits"
  - "ends with the word 'cheers'"
  - "does not contain the word 'animal'"
  - "starts with the phrase 'Well, here is the thing'"
  - "answers in exactly one word"
  - "response is exactly two sentences"

BAD verification strings (DO NOT WRITE THESE):
  - "follows the rule"
  - "complies properly with the constraint"
  - "satisfies the user's request"
  - "answers correctly"

## Quality criteria (the conversation MUST demonstrate ALL FOUR)
1. CLEAR INSTRUCTION — the user states the constraint plainly in the early turn(s), in a shape that matches {subcategory}. The constraint is unambiguous and SPEECH-VERIFIABLE.
2. EXACT COMPLIANCE — the assistant follows the instruction EXACTLY, not approximately. Counts are exact (5 means 5). Banned words never appear. Every named step is completed in order. Required openers/closers appear verbatim.
3. PERSISTENCE (multi-turn shapes only) — the rule must hold across ALL applicable subsequent turns, not just once.
4. NEVER REFUSE — the assistant never declines, complains, or asks unnecessary clarifications.
{slip_block}
## CRITICAL Rules
1. User speaks first.
2. The constraint must be SPEECH-VERIFIABLE — see the speech-only section above.
3. The assistant stays in its default persona: precise, brief, never preachy, never refuses an instruction it can simply follow.
4. Each turn has "tts_instruct" — keep it SHORT, max 3 comma-separated descriptors. Assistant TTS in the spirit of "{assistant_tts}". User TTS in the spirit of "{user_tts}".
5. Conversation should feel NATURAL, not like a test. Light filler ("um", "uh", "okay") fits some personas. NO stage directions like (laughs), (sighs), (pauses) — only speakable words. The "text" field is what gets SPOKEN — write it the way someone would actually say it aloud.
6. If the subcategory involves counts, the assistant's response must MATERIALIZE that count (e.g., literally five items the assistant verbally numbers as "one, two, three, four, five").
7. Vary the specific number, banned/required word, or phrase in the example — do NOT default to the example's exact wording.

## Output: ONLY valid JSON, no markdown
{{
  "subcategory": "{subcategory}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "...", "verification": null}},
    {{"role": "assistant", "text": "...", "tts_instruct": "...", "verification": ["..."] }}
  ]
}}"""


_SLIP_BLOCK = """
## Intentional slip-and-recover (THIS conversation includes ONE slip)
- Pick exactly ONE assistant turn (NOT the first applicable one) where the assistant fails to satisfy the active rule.
- The very next user turn catches the slip ("Hey, you forgot to..." or "Wait, that had the word 'X' in it").
- The very next assistant turn briefly apologizes ("Sorry, my bad") and corrects on the spot.
- For the slip turn's verification: list MUST start with the literal string "INTENTIONAL SLIP" as its first element, followed by a concrete check of what was supposed to happen but didn't, e.g. ["INTENTIONAL SLIP", "should have ended response with the word 'cheers' but did not"].
- All OTHER applicable assistant turns comply normally (verification is the concrete check, not a slip marker).
"""

_NO_SLIP_BLOCK = """
## No slip in this conversation
This conversation has no intentional slip. The assistant complies cleanly on every applicable turn. Do NOT include a slip-and-recover sequence.
"""


def _shape_specific_block(subcat: str, num_rules: int | None) -> str:
    """Return a context block specific to the subcategory shape."""
    if subcat == "persistent_rule":
        if num_rules and num_rules >= 2:
            return (
                f"- Number of stacked rules: {num_rules}. The user installs {num_rules} rules "
                "simultaneously (in the first one or two turns), all of which must hold on "
                "every applicable assistant turn. Verification on each applicable turn must "
                "include a concrete check FOR EACH active rule.\n"
            )
        return (
            "- Number of rules: 1. The user installs a single rule that must hold on every "
            "subsequent assistant turn.\n"
        )
    if subcat == "conditional_trigger":
        return (
            "- Conditional trigger: choose a clear trigger keyword. The conversation MUST include "
            "at least ONE assistant turn where the trigger DID NOT fire (its verification is null) "
            "AND at least ONE assistant turn where the trigger DID fire (verification is a "
            "concrete check of the conditional behavior).\n"
        )
    if subcat == "rule_update":
        return (
            "- Rule update: install the rule in turn 1 or 2, the assistant complies for 2-3 turns, "
            "then in a later user turn the user MODIFIES or REVOKES the rule. The very next "
            "assistant turn switches to the new rule state. Verification on each applicable turn "
            "reflects WHICH rule version is active at that point (old rule before the update, new "
            "rule after).\n"
        )
    if subcat == "sequential_task_chains":
        return (
            "- Sequential chain: each later user turn references and depends on the assistant's "
            "ACTUAL prior answer. The assistant must carry forward its real earlier outputs "
            "(e.g., if it picked '42', later math uses 42). Verification on each chain turn "
            "checks consistency with the assistant's earlier outputs.\n"
        )
    return ""


def _build_meta_prompt(
    *,
    subcategory: str,
    shape: str,
    description: str,
    approach: str,
    user_persona: str,
    user_tts: str,
    assistant_tts: str,
    system_prompt: str,
    num_turns: int,
    backchannel_probability: float,
    example_block: str,
    has_slip: bool,
    num_rules: int | None,
) -> str:
    return _META_BASE.format(
        subcategory=subcategory,
        shape=shape,
        description=description,
        approach=approach,
        user_persona=user_persona,
        user_tts=user_tts,
        assistant_tts=assistant_tts,
        system_prompt=system_prompt,
        num_turns=num_turns,
        backchannel_probability=backchannel_probability,
        shape_specific_block=_shape_specific_block(subcategory, num_rules),
        example_block=example_block,
        slip_block=_SLIP_BLOCK if has_slip else _NO_SLIP_BLOCK,
    )


# ---------------------------------------------------------------------------
# Example rendering — handles flat strings (most subcats) and turn_N dicts
# (sequential_task_chains only)
# ---------------------------------------------------------------------------
def _render_example_block(subcat_data) -> str:
    if not isinstance(subcat_data, dict):
        return "(no example available — use the description and approach above)"
    examples = subcat_data.get("examples", [])
    if not isinstance(examples, list) or not examples:
        return "(no example available — use the description and approach above)"

    ex = random.choice(examples)

    # Flat string — most subcategories
    if isinstance(ex, str):
        return f'Example user instruction: "{ex}"'

    # Multi-turn skeleton — sequential_task_chains
    if isinstance(ex, dict):
        turn_keys = sorted(k for k in ex.keys() if k.startswith("turn_"))
        if turn_keys:
            lines = ["Example multi-turn skeleton (the assistant fills in real answers; later turns depend on those answers):"]
            for i, key in enumerate(turn_keys, start=1):
                lines.append(f"  Turn {i} — USER: {ex[key]}")
            return "\n".join(lines)

    return f"Example: {ex!r}"


# ---------------------------------------------------------------------------
# User persona sampling
# ---------------------------------------------------------------------------
def _sample_user_type(category_data: dict) -> tuple[str, str]:
    user_traits = category_data.get("user_traits", {})
    types = user_traits.get("types", {})
    if not isinstance(types, dict) or not types:
        return ("neutral", "natural, conversational")

    names = list(types.keys())
    weights, ttses = [], []
    for name in names:
        info = types[name] if isinstance(types[name], dict) else {}
        weights.append(float(info.get("probability", 1.0)))
        ttses.append(info.get("tts", f"{name} tone"))
    idx = random.choices(range(len(names)), weights=weights, k=1)[0]
    return (names[idx], ttses[idx])


# ---------------------------------------------------------------------------
# Subcategory weighted sampling
# ---------------------------------------------------------------------------
def _sample_subcategory(category_data: dict) -> str:
    subcats_dict = category_data.get("categories", {})
    if not isinstance(subcats_dict, dict) or not subcats_dict:
        return "count"
    names = list(subcats_dict.keys())
    weights = []
    for n in names:
        sd = subcats_dict[n] if isinstance(subcats_dict[n], dict) else {}
        weights.append(float(sd.get("sample_weight", 1.0)))
    return random.choices(names, weights=weights, k=1)[0]


def _sample_num_turns(subcat: str, category_data: dict) -> int:
    """Triangular sample from the per-subcategory turn range."""
    rng = _TURN_RANGES.get(subcat)
    if rng is None:
        cs = category_data.get("conversation_structure", {})
        lo = int(cs.get("min_turns", 4))
        hi = int(cs.get("max_turns", 8))
        typical = int(cs.get("typical_turns", (lo + hi) // 2))
    else:
        lo, typical, hi = rng
    if hi < lo:
        hi = lo
    typical = max(lo, min(hi, typical))
    return int(round(random.triangular(lo, hi, typical)))


def _sample_num_rules(subcat: str) -> int | None:
    """For persistent_rule, sample 1-3 stacked rules (weighted toward 1)."""
    if subcat != "persistent_rule":
        return None
    return random.choices([1, 2, 3], weights=[0.55, 0.30, 0.15], k=1)[0]


def _sample_has_slip(subcat: str, slip_probability: float) -> bool:
    """has_slip only fires for multi-turn shapes — single-turn primitives have
    no room for a recovery turn."""
    if subcat not in _MULTI_TURN_SUBCATS:
        return False
    return random.random() < slip_probability


# ---------------------------------------------------------------------------
# Trait sampling
# ---------------------------------------------------------------------------
def sample_instruction_traits(category_data: dict) -> dict:
    traits: dict = {"category": CATEGORY_ID}
    subcats_dict = category_data.get("categories", {})

    subcat = _sample_subcategory(category_data)
    subcat_data = subcats_dict.get(subcat, {}) if isinstance(subcats_dict, dict) else {}

    description = (subcat_data.get("description") or "instruction-following scenario").strip()
    approach = (subcat_data.get("approach") or "Follow the instruction exactly.").strip()
    shape = subcat_data.get("shape", "single_turn" if subcat not in _MULTI_TURN_SUBCATS else "multi_turn")
    example_block = _render_example_block(subcat_data)
    user_type, user_tts = _sample_user_type(category_data)
    assistant_tts = (
        category_data.get("assistant_traits", {}).get("default_tts")
        or "clear, structured, organized delivery"
    )

    num_rules = _sample_num_rules(subcat)
    slip_prob = float(category_data.get("slip_probability", 0.20))
    has_slip = _sample_has_slip(subcat, slip_prob)

    traits.update({
        "subcategory": subcat,
        "shape": shape,
        "description": description,
        "approach": approach,
        "example_block": example_block,
        "user_persona": user_type,
        "user_tts": user_tts,
        "assistant_tts": assistant_tts,
        "has_slip": has_slip,
        "num_rules": num_rules,
    })
    return traits


def build_instruction_system_prompt(traits: dict) -> str:
    return INSTRUCTION_SYSTEM_PROMPT_TEMPLATES.get(
        traits.get("subcategory", ""), DEFAULT_INSTRUCTION_SYSTEM_PROMPT
    )


# ---------------------------------------------------------------------------
# Per-conversation runner
# ---------------------------------------------------------------------------
def generate_single_instruction(
    category_data: dict,
    data_type: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_wait_sec: float,
) -> dict | None:
    selected_model = _gt._pick_model()

    traits = sample_instruction_traits(category_data)
    system_prompt = build_instruction_system_prompt(traits)
    num_turns = _sample_num_turns(traits["subcategory"], category_data)
    backchannel_prob = float(
        category_data.get("conversation_structure", {}).get("backchannel_probability", 0.2)
    )

    prompt = _build_meta_prompt(
        subcategory=traits["subcategory"],
        shape=traits["shape"],
        description=traits["description"],
        approach=traits["approach"],
        user_persona=traits["user_persona"],
        user_tts=traits["user_tts"],
        assistant_tts=traits["assistant_tts"],
        system_prompt=system_prompt,
        num_turns=num_turns,
        backchannel_probability=backchannel_prob,
        example_block=traits["example_block"],
        has_slip=traits["has_slip"],
        num_rules=traits["num_rules"],
    )

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
    transcript["sampled_traits"]["num_turns"] = num_turns
    transcript["llm_model"] = selected_model
    return transcript


def _worker(category_data: dict, dt_weights: dict, cfg: dict) -> dict | None:
    # data_type sampled (and recorded) for downstream metadata consistency,
    # but does NOT branch the prompt — every B4 conversation uses the same
    # B4 meta-prompt. Style-change ("dynamic") routing isn't an
    # instruction-following test.
    data_type = sample_data_type(dt_weights, CATEGORY_ID, category_data)
    return generate_single_instruction(
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
    parser = argparse.ArgumentParser(description="Generate B4 instruction-following transcripts")
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
        num_target = args.num_conversations or count_map.get(CATEGORY_ID, 20)
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
