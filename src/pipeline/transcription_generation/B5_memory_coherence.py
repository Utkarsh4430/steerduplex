"""Phase 1 (B5 memory & coherence): Generate memory/coherence transcripts.

Architecture:
    BASE META-PROMPT  (universal memory + audible-mirror behavior)
            |
    CATEGORY SYSTEM-PROMPT MODIFIER  (appended to base; narrows behavior)
            |
    FEW-SHOT EXAMPLE(S)  (1-2 full conversations injected as exemplars)
            |
    Generated transcript

Six categories: reference_resolution, incidental_memory, self_coherence,
numerical_state_tracking, progressive_synthesis, contradiction_detection.
The connective tissue across all six is the AUDIBLE MIRROR pattern.

Conversations are short and spoken-natural (3-6 turns; numerical_state_tracking
gets 6-10). Subcategory sampling is weighted by the yaml's `num_conversations`.

Usage:
    cd src && python -m pipeline.transcription_generation.B5_memory_coherence \\
        --config configs/generation.yaml --scale pilot
"""

import argparse
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import pipeline.generate_transcripts as _gt
from pipeline.distributed import is_done, release_claim, try_claim
from pipeline.generate_transcripts import (
    call_llm,
    parse_llm_response,
    sample_data_type,
    setup_llm_client,
)
from pipeline.utils import ensure_dir, load_yaml, save_json, set_seed

CATEGORY_ID = "B5_memory_coherence"


# ---------------------------------------------------------------------------
# Universal assistant system prompt — the base every B5 conversation runs with
# ---------------------------------------------------------------------------
UNIVERSAL_BASE_SYSTEM_PROMPT = (
    "You are a voice assistant with excellent conversational memory. You track "
    "details the user shares, resolve indirect references, and proactively "
    "connect earlier information to later requests. The audible-mirror pattern "
    "is RESERVED for moments where memory is actively being APPLIED — resolving "
    "an indirect reference, applying an earlier-dropped detail to a current "
    "request, correcting a misattribution, recapping for a synthesis, or "
    "updating a running total. For ordinary information-gathering turns, do "
    "NOT echo the user's words back; a brief acknowledgment, a clarifying "
    "follow-up, or just the next question is enough. Once you've applied a "
    "remembered detail, move on — don't keep restating it on every subsequent "
    "turn. You speak in short, natural spoken sentences."
)


# ---------------------------------------------------------------------------
# Per-category system prompt modifiers — appended to the base
# ---------------------------------------------------------------------------
CATEGORY_SYSTEM_PROMPT_MODIFIERS: dict[str, str] = {
    "reference_resolution": (
        "When the user refers to something indirectly — by attribute (\"the older one,\" "
        "\"the climber\") or vaguely (\"that thing you mentioned,\" \"the enzyme one\") — "
        "resolve the reference explicitly and mirror it back. State the name and the "
        "distinguishing attribute, or restate the specific topic, before continuing."
    ),
    "incidental_memory": (
        "When the user drops a personal detail in passing — an allergy, a living "
        "situation, a family member, a preference — note it. If a later request is "
        "shaped by that detail, proactively connect them: mention the detail "
        "explicitly when you apply it (\"given your shellfish allergy, I'd lean "
        "toward...\"). The user should NOT have to remind you."
    ),
    "self_coherence": (
        "If the user restates something you said earlier and gets it wrong — whether "
        "explicitly (\"you said X\") or implicitly (acting on a flipped version of your "
        "words) — correct the misattribution by pointing back to what you actually "
        "said. Do NOT cave to the user's confidence. Be polite and brief, not "
        "defensive. If the user provides genuinely new evidence, you can update; if "
        "they just confidently misremember, hold your ground."
    ),
    "numerical_state_tracking": (
        "When the user builds a list, budget, or running total across turns, maintain "
        "the cumulative state. After each addition, removal, or substitution, briefly "
        "mirror the change and report the updated total (\"eggs and milk added — "
        "you're at $17 now\"). Handle substitutions as a NET change: removing $4 and "
        "adding $7 is a net +$3, not +$7. When asked for a total, sub-total, or "
        "derived metric (like a ratio or remaining budget), answer precisely."
    ),
    "progressive_synthesis": (
        "When the user is building up requirements or preferences across turns, "
        "collect them efficiently — brief acknowledgments (\"got it\", \"noted\"), "
        "a quick clarifying follow-up, or just the next question. Do NOT echo every "
        "constraint back as it arrives; that gets robotic. Save the FULL mirror for "
        "the final turn: when you deliver the recommendation, recap ALL the "
        "constraints they mentioned and make sure each feature in your "
        "recommendation visibly ties back to a constraint."
    ),
    "contradiction_detection": (
        "If the user says something that contradicts a fact they shared earlier in "
        "the same conversation, gently flag the inconsistency as a question — not "
        "an accusation. Frame it as your own possible confusion (\"I had it in my "
        "head that...\") and offer the user an easy way to clarify (\"did I catch "
        "that wrong, or has something shifted?\"). Be warm and curious, never "
        "preachy or 'gotcha.' The user might have a perfectly good explanation."
    ),
}


# ---------------------------------------------------------------------------
# Few-shot example dialogues — injected verbatim into the meta-prompt
# Plain-text User:/Assistant: format. The LLM uses these as shape exemplars,
# NOT to copy verbatim.
# ---------------------------------------------------------------------------
_FS_REFERENCE_RESOLUTION = """\
Example A — attribute-based entity disambiguation:
User: Help me think through gifts for my brothers? I've got two and they're total opposites.
Assistant: Sure, tell me about them.
User: Marcus is 34, super into woodworking — has a whole shop in his garage. Theo's 31, big rock climber, every weekend at the crag.
Assistant: Fun contrast. What's your budget?
User: Around a hundred each. Let's start with the climber, he's pickier.
Assistant: Okay, so for Theo, the climber — a hangboard's a solid call at that budget, something like a Tension Block. Lets him train at home during the week when he's not outside.
User: Love it. And for the older one?
Assistant: For Marcus, the woodworker — since he's already got the shop dialed in, I'd skip tools and go with a gift card to a specialty hardwood supplier. Lets him splurge on figured walnut or curly maple for a fun project.
(Notice: "the climber" → "Theo, the climber"; "the older one" → "Marcus, the woodworker.")

Example B — vague topic callback:
User: Can you explain how photosynthesis works at a high level?
Assistant: Sure. Plants take in CO2 and water, and use sunlight to convert them into glucose for energy and oxygen as a byproduct. The whole thing happens in chloroplasts, and the key player is an enzyme called Rubisco that grabs the CO2 and kicks off the reaction.
User: Interesting. Is that the same process in algae?
Assistant: Yep, algae use the same machinery — chloroplasts, Rubisco, the works. Land plants actually inherited it from algae evolutionarily.
User: Okay, going back to that enzyme thing you mentioned — why is it so important?
Assistant: Right, Rubisco — it's the gatekeeper for almost all carbon entering the biosphere. Pretty much every carbon atom in your body passed through a Rubisco molecule at some point. It's also notoriously slow, which is why plants make so much of it — thought to be the most abundant protein on Earth.
(Notice: "that enzyme thing" → "Right, Rubisco" — vague reference resolved by naming it.)
"""

_FS_INCIDENTAL_MEMORY = """\
Example — incidental detail proactively applied later:
User: Hey, can you help me plan something special for my anniversary next weekend?
Assistant: Of course. What are you thinking — dinner, a getaway, something at home?
User: Probably dinner out. We've been wanting to try somewhere new. Oh, and just a heads up, I'm allergic to shellfish, so nothing seafood-heavy.
Assistant: Good to know. What kind of cuisine does your partner enjoy?
User: She loves Italian, especially handmade pasta. And she's really into wine, so a place with a good list would be a plus.
Assistant: Got it. Any neighborhood preference, or are you flexible?
User: Flexible. Somewhere a bit romantic if possible — dim lighting, quieter vibe.
Assistant: Then I'd lean toward a smaller Northern Italian spot rather than a trattoria — handmade pasta, strong wine program, more intimate atmosphere. And given the shellfish allergy, Northern Italian works in your favor since the menus skew more toward meat, mushroom, and cream-based pastas rather than the seafood-heavy Southern coastal style.
(Notice: shellfish allergy dropped as side comment in turn 2. Three turns of unrelated planning. Final turn proactively connects the allergy to the recommendation without being asked.)
"""

_FS_SELF_COHERENCE = """\
Example — implicit misattribution caught and corrected:
User: I'm trying to read more this year. Any sci-fi recommendations?
Assistant: Depends what you're after. For big-idea, philosophical sci-fi, Ursula Le Guin's The Left Hand of Darkness is a great pick. For something more plot-driven and modern, Andy Weir's Project Hail Mary — hard to put down.
User: Ooh, the philosophical one sounds great. I'll grab the Andy Weir for my vacation next week.
Assistant: Small mix-up there — the philosophical one is actually The Left Hand of Darkness, the Le Guin. Project Hail Mary is the plot-driven, hard-to-put-down one. Both are great, just want to make sure you grab the right one for what you're in the mood for.
(Notice: the user never says "you said." They just act on a flipped premise as if it's settled. The assistant catches the swap and re-anchors without being preachy.)
"""

_FS_NUMERICAL_STATE_TRACKING = """\
Example — running grocery total with substitution and derived query:
User: Help me put together a grocery list and keep a running total? I'm trying to stay under $50 this week.
Assistant: Sure, I'll keep track. What's first?
User: Two pounds of chicken thighs, around $8.
Assistant: Got it — running total $8.
User: A dozen eggs, $4. And a gallon of milk, $5.
Assistant: Eggs and milk added — you're at $17 now.
User: Pack of pasta, $2, and a jar of marinara, $4.
Assistant: Up to $23.
User: Actually, scratch the marinara — I've got some at home. But add a block of parmesan, $7.
Assistant: Removing the marinara and adding the parmesan — net change is plus $3, so you're at $26.
User: Two pounds of apples, $5, and a bag of spinach, $3.
Assistant: That brings you to $34.
User: Okay, what's left in my budget? And how much have I spent on dairy specifically?
Assistant: You've got $16 left out of your $50. On dairy, you're at $12 — that's the milk at $5 and the parmesan at $7.
(Notice: every quantity change is mirrored with the updated total. Substitution handled as net change. Final query has two parts — remaining budget AND a category sub-total requiring selective recall plus computation.)
"""

_FS_PROGRESSIVE_SYNTHESIS = """\
Example — drip-fed constraints, brief acks mid-conversation, FULL mirror only in the final synthesis:
User: I'm thinking about getting a new bike. Can you help me figure out what to get?
Assistant: Yeah, happy to. What kind of riding are you planning to do?
User: Mostly commuting to work, about 6 miles each way. But I'd also want to do longer weekend rides sometimes, like 30, 40 miles.
Assistant: Got it. What's the terrain like, pavement or trails?
User: All pavement. City streets during the week, paved bike paths on weekends.
Assistant: And what's your budget?
User: Around $1200, maybe up to $1500 if it's really worth it.
Assistant: Any storage situation I should know about?
User: Apartment, fourth floor walk-up. So I'd be carrying it up the stairs.
Assistant: Ah — that's worth knowing. One more thing: any preference on riding posture, upright versus leaned over?
User: I'd rather not be hunched all the way down. Comfort matters, especially for the longer rides.
Assistant: Okay, putting it all together — a pavement bike for a 12-mile daily commute plus 40-mile weekend rides, $1200 to $1500, light enough to carry up four flights, and with relaxed geometry for comfort. That points to an endurance road or fitness hybrid — something like a Specialized Sirrus, a Trek FX Sport, or a Cannondale Quick. All aluminum, around 22-24 pounds for the stairs, and geared for both the commute and the longer paved rides.
(Notice: assistant collects most constraints with bare acknowledgments or just the next question — no per-turn echo. Only the FINAL turn does the comprehensive mirror, where every constraint visibly ties to a feature of the recommendation. This is the pattern to follow.)
"""

_FS_CONTRADICTION_DETECTION = """\
Example — user self-contradiction gently flagged. Mid-conversation acks are BRIEF (no per-turn echo); the only mirror is the contradiction flag at the end:
User: Hey, can you help me find a good place for dinner this weekend? It's our anniversary.
Assistant: Of course. What kind of food are you both in the mood for?
User: Something nice, sit-down. My wife's been wanting to try Italian, and honestly I could go for a good pasta too. Also, just so you know, we're both vegetarian, so somewhere with solid meat-free options would be ideal.
Assistant: Got it. Any neighborhood preference?
User: Somewhere downtown if possible. And ideally a place with good wine.
Assistant: And what's your budget range?
User: Up to around $150 for the two of us. Oh, and one more thing — make sure they have a good steak option, my wife loves a ribeye.
Assistant: Happy to factor that in — though just to double-check, I had it in my head that you both were vegetarian from earlier. Did I catch that wrong, or has something shifted?
(Notice: vegetarian dropped early as side comment. Mid-conversation, the assistant uses bare acks ("Got it. Any neighborhood preference?", "And what's your budget range?") — NO per-turn echo of "Italian", "downtown", "wine list", etc. The ONLY mirror in the whole conversation is the final flag where the earlier vegetarian claim is brought back to surface the contradiction. That's the right density.)
"""

FEW_SHOT_EXAMPLES: dict[str, str] = {
    "reference_resolution": _FS_REFERENCE_RESOLUTION,
    "incidental_memory": _FS_INCIDENTAL_MEMORY,
    "self_coherence": _FS_SELF_COHERENCE,
    "numerical_state_tracking": _FS_NUMERICAL_STATE_TRACKING,
    "progressive_synthesis": _FS_PROGRESSIVE_SYNTHESIS,
    "contradiction_detection": _FS_CONTRADICTION_DETECTION,
}


# ---------------------------------------------------------------------------
# Layered meta-prompt template
# ---------------------------------------------------------------------------
META_PROMPT_MEMORY = """\
Generate a natural spoken conversation ({num_turns} turns) testing MEMORY AND COHERENCE.

## Context
- Memory test type: {memory_type}
- System prompt: {system_prompt}
- Number of turns: {num_turns}

## Universal behavior being trained
The assistant has excellent conversational memory. It tracks details the user
provides, resolves indirect references, and connects earlier information to
later requests — proactively, without being asked to "remember."

## CRITICAL: Audible mirror — when to fire it (and when NOT to)
The audible mirror is the connective tissue of memory in voice — it confirms
correct recall and re-grounds the entity for the listener. But mirroring on
every turn gets robotic and unnatural.

FIRE a mirror when memory is being APPLIED:
- Resolving an indirect reference: "Okay, so for Theo, the climber..." (after user said "the climber")
- Applying an earlier-dropped detail: "Given the shellfish allergy, I'd lean toward..."
- Correcting a misattribution: "Small mix-up — actually the philosophical one is the Le Guin..."
- The final synthesis recap (one comprehensive mirror at the end, not many along the way)
- Updating a running numerical total — every change gets the updated total

DO NOT mirror on ordinary information-gathering turns. When the user is just
revealing the next piece of context ("my budget is around $1200", "all
pavement"), a brief acknowledgment ("got it"), a quick clarifying follow-up,
or simply the next question is enough. If you catch yourself starting every
assistant turn with "Okay, so [echo of what user just said]...", you are
over-mirroring — vary it.

Two category-specific notes:
- numerical_state_tracking is the EXCEPTION: running totals genuinely need an
  updated number mirrored back after each change.
- progressive_synthesis is the HIGHEST RISK for over-mirroring — constraints
  arrive across many turns, but the FULL recap belongs in the final turn.
  Mid-conversation, use brief and VARIED acknowledgments (sometimes "got it",
  sometimes just the next question, occasionally a one-detail mirror).

## CRITICAL: Spoken-natural style
- Utterances are SHORT — typically 1-2 sentences. NOT paragraphs.
- People interrupt, drop side comments, change topics — sound like real speech.
- NO stage directions. NO bullet lists in dialogue.
- tts_instruct: SHORT, max 3 descriptors.

## Category-specific behavior ({memory_type})
{category_modifier}

## Few-shot example(s) of the target behavior
{few_shot_examples}

## Output: ONLY valid JSON, no markdown
{{
  "memory_type": "{memory_type}",
  "turns": [
    {{"role": "user", "text": "...", "tts_instruct": "..."}},
    {{"role": "assistant", "text": "...", "tts_instruct": "..."}}
  ]
}}"""


# ---------------------------------------------------------------------------
# Per-category turn ranges. Defaults to (3, 6); numerical needs more room.
# ---------------------------------------------------------------------------
_DEFAULT_TURN_RANGE: tuple[int, int] = (3, 6)
_TURN_RANGE_OVERRIDES: dict[str, tuple[int, int]] = {
    "numerical_state_tracking": (6, 10),
}


def _resolve_turn_range(memory_type: str, category_data: dict) -> tuple[int, int]:
    """Per-category override > yaml turn_range_overrides > yaml conversation_structure > script default."""
    if memory_type in _TURN_RANGE_OVERRIDES:
        return _TURN_RANGE_OVERRIDES[memory_type]
    cs = category_data.get("conversation_structure", {}) or {}
    yaml_overrides = cs.get("turn_range_overrides", {}) or {}
    if memory_type in yaml_overrides:
        ov = yaml_overrides[memory_type]
        return (int(ov.get("min_turns", _DEFAULT_TURN_RANGE[0])),
                int(ov.get("max_turns", _DEFAULT_TURN_RANGE[1])))
    return (int(cs.get("min_turns", _DEFAULT_TURN_RANGE[0])),
            int(cs.get("max_turns", _DEFAULT_TURN_RANGE[1])))


def _sample_num_turns(memory_type: str, category_data: dict) -> int:
    lo, hi = _resolve_turn_range(memory_type, category_data)
    if hi < lo:
        hi = lo
    return random.randint(lo, hi)


# ---------------------------------------------------------------------------
# Subcategory sampling — weighted by yaml's `num_conversations` per category
# ---------------------------------------------------------------------------
def _sample_memory_type(category_data: dict) -> str:
    subcats_dict = category_data.get("categories", {}) or {}
    if not isinstance(subcats_dict, dict) or not subcats_dict:
        print(f"  [WARN] {CATEGORY_ID}: yaml has no categories block, falling back to 'reference_resolution'")
        return "reference_resolution"
    names = list(subcats_dict.keys())
    weights = []
    for n in names:
        sd = subcats_dict[n] if isinstance(subcats_dict[n], dict) else {}
        weights.append(float(sd.get("num_conversations", 1.0)))
    return random.choices(names, weights=weights, k=1)[0]


# ---------------------------------------------------------------------------
# Trait sampling
# ---------------------------------------------------------------------------
def sample_memory_traits(category_data: dict) -> dict:
    traits: dict = {"category": CATEGORY_ID}
    memory_type = _sample_memory_type(category_data)
    if memory_type not in CATEGORY_SYSTEM_PROMPT_MODIFIERS:
        print(f"  [WARN] {CATEGORY_ID}: sampled memory_type '{memory_type}' has no modifier defined, using base prompt only")
    if memory_type not in FEW_SHOT_EXAMPLES:
        print(f"  [WARN] {CATEGORY_ID}: sampled memory_type '{memory_type}' has no few-shot example defined")
    traits["memory_type"] = memory_type
    return traits


# ---------------------------------------------------------------------------
# Prompt assembly — base + modifier (system); base + modifier + few-shot (meta)
# ---------------------------------------------------------------------------
def build_memory_system_prompt(traits: dict) -> str:
    memory_type = traits.get("memory_type", "")
    modifier = CATEGORY_SYSTEM_PROMPT_MODIFIERS.get(memory_type, "")
    if modifier:
        return UNIVERSAL_BASE_SYSTEM_PROMPT + " " + modifier
    return UNIVERSAL_BASE_SYSTEM_PROMPT


def build_memory_meta_prompt(traits: dict, system_prompt: str, num_turns: int) -> str:
    memory_type = traits.get("memory_type", "reference_resolution")
    modifier = CATEGORY_SYSTEM_PROMPT_MODIFIERS.get(
        memory_type, "(no category-specific modifier)"
    )
    examples = FEW_SHOT_EXAMPLES.get(
        memory_type, "(no few-shot example available — use the description above)"
    )
    return META_PROMPT_MEMORY.format(
        memory_type=memory_type,
        system_prompt=system_prompt,
        num_turns=num_turns,
        category_modifier=modifier,
        few_shot_examples=examples,
    )


# ---------------------------------------------------------------------------
# Validation hook — lightweight per-category sanity checks. Logs warnings
# but does NOT reject the transcript. Failures should be audited offline.
# ---------------------------------------------------------------------------
_NUM_RX = re.compile(r"\d")
_STOPWORDS = {
    "the", "and", "a", "an", "to", "of", "for", "in", "on", "at", "by", "with",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "i", "you", "he", "she", "it", "we", "they", "my",
    "your", "his", "her", "their", "our", "this", "that", "these", "those",
    "but", "or", "so", "if", "as", "from", "about", "what", "when", "where",
    "how", "why", "would", "could", "should", "will", "can", "just", "yeah",
    "okay", "sure", "got", "really", "into", "some", "any", "all", "more",
    "than", "then", "there", "here", "well", "right", "actually", "good",
    "great", "want", "wanted", "going", "make", "let", "lets", "let's",
}


def _content_words(text: str, min_len: int = 5) -> set[str]:
    """Lowercase content words from text, stripped of punctuation, longer than min_len, not stopwords."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return {w for w in cleaned.split() if len(w) > min_len and w not in _STOPWORDS}


def validate_transcript(transcript: dict, memory_type: str) -> list[str]:
    """Return a list of warning strings. Empty list = passed.

    Failures are logged in the worker; transcripts are always saved. The
    purpose is auditability of generation quality, not gating.
    """
    warnings: list[str] = []
    turns = transcript.get("turns", [])
    if not turns:
        return ["transcript has no turns"]

    asst_turns = [t for t in turns if t.get("role") == "assistant"]
    user_turns = [t for t in turns if t.get("role") == "user"]
    if not asst_turns:
        return ["transcript has no assistant turns"]

    last_asst = asst_turns[-1].get("text", "")
    last_asst_lower = last_asst.lower()
    all_asst_lower = " ".join(t.get("text", "").lower() for t in asst_turns)

    if memory_type == "self_coherence":
        corrective = ["actually", "mix-up", "mixup", "i think you mean", "i said", "what i said", "to be precise", "small mix"]
        if not any(p in all_asst_lower for p in corrective):
            warnings.append("self_coherence: no corrective phrase found in assistant turns")

    elif memory_type == "contradiction_detection":
        soft_flags = [
            "did i catch that wrong", "has something shifted", "did something change",
            "i thought you mentioned", "i had it in my head", "did i misunderstand",
            "to double-check", "you mentioned earlier", "you said earlier you",
        ]
        if not any(p in last_asst_lower for p in soft_flags):
            warnings.append("contradiction_detection: no soft-flag phrase in final assistant turn")

    elif memory_type == "numerical_state_tracking":
        if not _NUM_RX.search(all_asst_lower):
            warnings.append("numerical_state_tracking: no numeric digits in assistant turns")

    elif memory_type == "progressive_synthesis":
        early_user_text = " ".join(t.get("text", "") for t in user_turns[:-1])
        early_words = _content_words(early_user_text)
        last_words = _content_words(last_asst, min_len=4)
        recapped = early_words & last_words
        if len(recapped) < 3:
            warnings.append(
                f"progressive_synthesis: only {len(recapped)} early-user content words echoed "
                f"in final synthesis (expected >=3)"
            )

    elif memory_type == "incidental_memory":
        early_text = " ".join(t.get("text", "") for t in user_turns[:2])
        early_content = _content_words(early_text)
        if early_content and not (early_content & _content_words(last_asst, min_len=4)):
            warnings.append("incidental_memory: no early-user content words appear in final assistant turn")

    elif memory_type == "reference_resolution":
        # Lightweight: check for at least one capitalized non-sentence-start
        # token in the assistant turns (proxy for entity mirroring like "Theo, the climber").
        # Skip the check if there isn't a clear plant in early user turns.
        asst_text = " ".join(t.get("text", "") for t in asst_turns)
        sentences = re.split(r"[.!?]\s+", asst_text)
        has_proper = any(
            re.search(r"\b[A-Z][a-z]{2,}", s[1:] if s else "")  # skip first char (sentence-initial cap)
            for s in sentences
        )
        if not has_proper and len(asst_turns) >= 2:
            warnings.append("reference_resolution: no proper-noun-shaped token in assistant turns (weak mirror?)")

    return warnings


# ---------------------------------------------------------------------------
# Per-conversation runner
# ---------------------------------------------------------------------------
def generate_single_memory(
    category_data: dict,
    data_type: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    retry_wait_sec: float,
) -> tuple[dict | None, list[str]]:
    """Returns (transcript_or_none, warnings). Caller logs warnings."""
    selected_model = _gt._pick_model()

    traits = sample_memory_traits(category_data)
    system_prompt = build_memory_system_prompt(traits)
    requested_turns = _sample_num_turns(traits["memory_type"], category_data)
    prompt = build_memory_meta_prompt(traits, system_prompt, requested_turns)

    content = call_llm(
        selected_model, prompt, temperature, max_tokens, max_retries, retry_wait_sec,
        validate_fn=parse_llm_response,
    )
    transcript = parse_llm_response(content) if content else None
    if transcript is None:
        return None, ["llm call or parse failed"]

    actual_turns = len(transcript.get("turns", []))
    transcript["category"] = CATEGORY_ID
    transcript["data_type"] = data_type
    transcript["system_prompt"] = f"<system> {system_prompt} <system>"
    transcript["sampled_traits"] = {
        k: v for k, v in traits.items() if isinstance(v, (str, int, float, bool, list))
    }
    transcript["sampled_traits"]["num_turns_requested"] = requested_turns
    transcript["sampled_traits"]["num_turns_actual"] = actual_turns
    transcript["llm_model"] = selected_model

    warnings: list[str] = []
    if actual_turns != requested_turns:
        warnings.append(f"turn count drift: requested {requested_turns}, got {actual_turns}")
    warnings.extend(validate_transcript(transcript, traits["memory_type"]))

    return transcript, warnings


def _worker(category_data: dict, dt_weights: dict, cfg: dict) -> tuple[dict | None, list[str]]:
    data_type = sample_data_type(dt_weights, CATEGORY_ID, category_data)
    return generate_single_memory(
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
    parser = argparse.ArgumentParser(description="Generate B5 memory/coherence transcripts")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--num_conversations", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--scale", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["transcript"]
    set_seed(args.seed)

    setup_llm_client(cfg)

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
    warned = 0

    def _claim_and_generate(slot_idx: int):
        claim_path = claims_dir / f"{slot_idx:05d}.claim"
        out_path = cat_dir / f"{slot_idx:05d}.json"
        if is_done(out_path):
            return slot_idx, "skip", []
        if not try_claim(claim_path):
            return slot_idx, "skip", []
        try:
            transcript, warnings = _worker(category_data, dt_weights, cfg)
            if transcript is not None:
                transcript["id"] = f"{CATEGORY_ID}_{slot_idx:05d}"
                save_json(transcript, out_path)
                release_claim(claim_path)
                return slot_idx, transcript, warnings
            release_claim(claim_path)
            return slot_idx, None, warnings
        except Exception as e:
            release_claim(claim_path)
            print(f"  [ERROR] slot {slot_idx}: {e}")
            return slot_idx, None, [f"exception: {e}"]

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_claim_and_generate, s): s for s in slots}
        with tqdm(total=len(slots), desc=CATEGORY_ID, unit="conv") as pbar:
            for fut in as_completed(futures):
                try:
                    slot_idx, result, warnings = fut.result()
                except Exception as e:
                    pbar.write(f"  [ERROR] Worker exception: {e}")
                    failed += 1
                else:
                    if result == "skip":
                        skipped += 1
                    elif result is not None:
                        generated += 1
                        if warnings:
                            warned += 1
                            for w in warnings:
                                pbar.write(f"  [WARN slot {slot_idx:05d}] {w}")
                    else:
                        failed += 1
                        for w in warnings:
                            pbar.write(f"  [FAIL slot {slot_idx:05d}] {w}")
                pbar.set_postfix(gen=generated, fail=failed, skip=skipped, warn=warned)
                pbar.update(1)

    print(f"  {CATEGORY_ID}: generated {generated}, failed {failed}, skipped {skipped}, warned {warned}")


if __name__ == "__main__":
    main()
