"""Rephrase system prompts for training data diversity.

Reads transcript JSONs, rephrases each system_prompt using an LLM, and saves
the result in a new field `system_prompt_rephrased` (original is preserved).

Downstream code (format_dataset.py) can then randomly pick between the original
and rephrased version, giving the model exposure to varied phrasings of the
same instructions.

Usage:
    python -m pipeline.rephrase_system_prompts --config configs/generation.yaml
    python -m pipeline.rephrase_system_prompts --config configs/generation.yaml --category B1_speech_reasoning
    python -m pipeline.rephrase_system_prompts --config configs/generation.yaml --num_workers 32
"""

import argparse
import json
import logging
import random
import time
from pathlib import Path

import openai
from openai import OpenAI
from tqdm import tqdm

from pipeline.utils import load_json, load_yaml, save_json

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rephrasing prompt
# ---------------------------------------------------------------------------
REPHRASE_SYSTEM = (
    "You rephrase system prompts for a voice assistant. "
    "Keep the same meaning and constraints. Don't add new instructions or remove existing ones. "
    "Don't make it longer or fancier — just say the same thing differently. "
    "Use natural, plain language. Output ONLY the rephrased prompt, nothing else."
)

REPHRASE_USER = (
    "Rephrase this system prompt. Same meaning, different wording. "
    "Keep it roughly the same length. No markdown, no quotes.\n\n{prompt}"
)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def _make_client(cfg: dict) -> OpenAI:
    return OpenAI(
        api_key=cfg.get("llm_api_key", "sk-WtJy90ltT2hpJSFKQ1u_TA"),
        base_url=cfg.get("llm_base_url", "https://litellm-proxy.ml-serving-internal.scale.com/v1"),
    )


def _pick_model(cfg: dict) -> str:
    """Randomly pick a model from the configured model list."""
    models = cfg.get("llm_models", [])
    if not models:
        return cfg.get("llm_model", "bedrock/qwen.qwen3-32b-v1:0")
    weights = [m["weight"] for m in models]
    chosen = random.choices(models, weights=weights, k=1)[0]
    return chosen["model"]


def rephrase_prompt(
    client: OpenAI,
    model: str,
    original_prompt: str,
    max_retries: int = 3,
) -> str | None:
    """Call LLM to rephrase a system prompt. Returns rephrased text or None."""
    # Strip <system> tags for the LLM — we re-add them later
    clean = original_prompt.strip()
    if clean.startswith("<system>"):
        clean = clean[len("<system>"):].strip()
    if clean.endswith("<system>"):
        clean = clean[:-len("<system>")].strip()

    if not clean:
        return None

    # GPT-5.x models only support temperature=1
    temp = 1.0 if "gpt-5" in model else 0.9

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": REPHRASE_SYSTEM},
                    {"role": "user", "content": REPHRASE_USER.format(prompt=clean)},
                ],
                temperature=temp,
                max_tokens=256,
            )
            text = resp.choices[0].message.content.strip()
            # Clean up any quotes the LLM might add
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            if text.startswith("'") and text.endswith("'"):
                text = text[1:-1]
            # Don't accept if it's way too different in length (>2x or <0.3x)
            if len(text) > len(clean) * 2.5 or len(text) < len(clean) * 0.3:
                logger.debug("Rejected rephrase (length %d vs %d): %s", len(text), len(clean), text[:80])
                continue
            return text
        except openai.RateLimitError:
            wait = 5 * (2 ** attempt)
            logger.warning("Rate limited, waiting %ds...", wait)
            time.sleep(wait)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning("Retry %d: %s", attempt + 1, e)
                time.sleep(2)
            else:
                logger.error("Failed: %s", e)
                return None
    return None


# ---------------------------------------------------------------------------
# Worker for parallel processing
# ---------------------------------------------------------------------------
def _process_file(args_tuple):
    """Process a single transcript file (for use with ThreadPoolExecutor)."""
    path, cfg = args_tuple
    client = _make_client(cfg)

    transcript = load_json(path)
    if "system_prompt_rephrased" in transcript:
        return 0  # already done

    original = transcript.get("system_prompt", "")
    if not original:
        return 0

    model = _pick_model(cfg)
    rephrased = rephrase_prompt(client, model, original)
    if rephrased:
        transcript["system_prompt_rephrased"] = f"<system> {rephrased} <system>"
        save_json(transcript, path)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Rephrase system prompts in transcript JSONs for training diversity",
    )
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--category", default=None, help="Only process this category")
    parser.add_argument("--num_workers", type=int, default=16, help="Parallel workers")
    parser.add_argument("--dry_run", action="store_true", help="Print one example and exit")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_yaml(args.config)["transcript"]
    transcript_dir = Path(cfg["output_dir"])

    if not transcript_dir.exists():
        logger.error("Transcript dir not found: %s", transcript_dir)
        return

    # Collect all transcript files
    all_files = []
    for cat_dir in sorted(transcript_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.category and cat_dir.name != args.category:
            continue
        files = sorted(cat_dir.glob("*.json"))
        all_files.extend(files)

    logger.info("Found %d transcript files", len(all_files))

    # Dry run: grab first file, rephrase, and exit (skip full scan)
    if args.dry_run:
        client = _make_client(cfg)
        sample = load_json(all_files[0])
        original = sample.get("system_prompt", "")
        model = _pick_model(cfg)
        logger.info("Model: %s", model)
        logger.info("Original: %s", original)
        rephrased = rephrase_prompt(client, model, original)
        logger.info("Rephrased: %s", rephrased)
        return

    # Filter out already-rephrased
    to_process = []
    for f in tqdm(all_files, desc="Scanning"):
        d = load_json(f)
        if "system_prompt_rephrased" not in d and d.get("system_prompt"):
            to_process.append(f)

    logger.info("%d need rephrasing (%d already done)", len(to_process), len(all_files) - len(to_process))

    if not to_process:
        logger.info("Nothing to do.")
        return

    # Process in parallel using threads (IO-bound LLM calls)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    work_items = [(f, cfg) for f in to_process]
    done = 0

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(_process_file, item): item for item in work_items}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Rephrasing"):
            try:
                done += future.result()
            except Exception as e:
                logger.error("Worker error: %s", e)

    logger.info("Rephrased %d / %d transcripts", done, len(to_process))


if __name__ == "__main__":
    main()
