"""Transcribe BigBench Audio input questions and add a `question` field to output.json.

Loads the cached BigBench Audio MP3 files, transcribes them via the OpenAI
Whisper API, and writes a `question` key back into the output.json in-place.
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from openai import OpenAI
from tqdm import tqdm

load_dotenv("/fs/gamma-projects/audio/raman/steerd/steerduplex/src/eval/fdb_v2/.env")

CACHE_DIR = "/fs/gamma-projects/audio/raman/steerd/cache"
HF_REPO = "ArtificialAnalysis/big_bench_audio"

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe BigBench Audio question MP3s via OpenAI Whisper API and update output.json."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to output.json to update. Defaults to latest moshi_results_*/output.json.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel API calls (default: 16).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N examples (for debugging).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-transcribe even if `question` field already exists.",
    )
    return parser.parse_args()


def find_default_input() -> Path:
    src_dir = Path(__file__).parent.parent.parent
    candidates = sorted(
        src_dir.glob("moshi_results_*/output.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No moshi_results_*/output.json found. Pass --input explicitly."
        )
    return candidates[0]


def load_results(path: Path) -> List[Dict[str, Any]]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list in {path}")
    return data


def save_results(results: List[Dict[str, Any]], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, path)


def load_snapshot_index(snapshot_dir: Path) -> Dict[str, Path]:
    """Build a mapping of unique_id -> MP3 path from the HF snapshot metadata."""
    metadata_path = snapshot_dir / "metadata.jsonl"
    index: Dict[str, Path] = {}
    with metadata_path.open() as f:
        for line in f:
            record = json.loads(line)
            unique_id = f"{record['category']}_{record['id']}"
            index[unique_id] = snapshot_dir / record["file_name"]
    return index


def resolve_snapshot_dir(token: Optional[str]) -> Path:
    logger.info("Resolving BigBench Audio snapshot from HuggingFace cache...")
    snapshot_dir = snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        cache_dir=CACHE_DIR,
        token=token,
        allow_patterns=["metadata.jsonl", "data/*.mp3"],
    )
    return Path(snapshot_dir)


def transcribe_one(client: OpenAI, mp3_path: Path) -> str:
    with mp3_path.open("rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    return response.strip() if isinstance(response, str) else response.text.strip()


def main() -> None:
    setup_logging()
    args = parse_args()

    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR

    input_path = Path(args.input) if args.input else find_default_input()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    logger.info("Input: %s", input_path)

    results = load_results(input_path)
    if args.limit is not None:
        results = results[: args.limit]
        logger.info("Limited to first %d examples.", len(results))

    pending = [r for r in results if args.force or r.get("question") is None]
    logger.info("%d / %d entries need question transcription.", len(pending), len(results))
    if not pending:
        logger.info("Nothing to do. Use --force to re-transcribe.")
        return

    snapshot_dir = resolve_snapshot_dir(token=os.environ.get("HF_TOKEN"))
    index = load_snapshot_index(snapshot_dir)
    logger.info("Snapshot index loaded: %d entries.", len(index))

    pairs: List[Tuple[Dict[str, Any], Path]] = []
    for entry in pending:
        uid = entry["unique_id"]
        mp3_path = index.get(uid)
        if mp3_path is None or not mp3_path.exists():
            logger.warning("No audio found for %s, skipping.", uid)
            entry["question"] = None
        else:
            pairs.append((entry, mp3_path))

    if not pairs:
        logger.info("No valid audio files to transcribe.")
        save_results(results, input_path)
        return

    logger.info("Transcribing %d files via OpenAI Whisper API with %d workers...", len(pairs), args.workers)
    client = OpenAI(base_url="https://us.api.openai.com/v1")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_entry = {
            executor.submit(transcribe_one, client, mp3_path): entry
            for entry, mp3_path in pairs
        }
        with tqdm(total=len(pairs), desc="Whisper API", unit="file", dynamic_ncols=True) as pbar:
            for future in as_completed(future_to_entry):
                entry = future_to_entry[future]
                try:
                    entry["question"] = future.result()
                except Exception:
                    logger.exception("Transcription failed for %s.", entry["unique_id"])
                    entry["question"] = None
                pbar.update(1)
                pbar.set_postfix({"last": entry["unique_id"]})

    save_results(results, input_path)
    logger.info("Done. Updated %d entries in %s", len(pairs), input_path)


if __name__ == "__main__":
    main()
