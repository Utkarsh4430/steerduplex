"""Add system prompt regions to external datasets (Fisher, Annutacon).

External datasets have raw stereo conversations with no system prompt region.
This script:
  1. Generates a natural system prompt per conversation via LLM (based on
     the first few words of the conversation)
  2. Prepends a prompt region to the stereo WAV:
     - Assistant channel (ch0): voice reference (from conversation) + sine + silence
     - User channel (ch1): 440 Hz sine (PersonaPlex pattern)
  3. Shifts all alignment timestamps forward by prompt_end_sec
  4. Injects <system> text into alignments
  5. Writes new WAVs and alignment JSONs (originals untouched via new filenames)

The result is a new manifest that can replace the original in full_training.yaml.

Usage:
    python -m pipeline.add_prompts_external --dataset annutacon
    python -m pipeline.add_prompts_external --dataset fisher
    python -m pipeline.add_prompts_external --dataset annutacon --dry_run
    python -m pipeline.add_prompts_external --dataset annutacon --num_workers 16
"""

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from pipeline.utils import ensure_dir, load_yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
VOICE_PROMPT_DURATION = 5.0    # seconds of voice reference
SINE_FREQ = 440.0              # Hz
SINE_MARKER_DURATION = 0.2     # seconds (assistant channel marker)
SILENCE_AFTER_MARKER = 0.5     # seconds
SAMPLE_RATE = 24000

# Pool of generic system prompts for natural conversations.
# These describe a natural conversationalist — appropriate for Fisher/Annutacon
# which are real human conversations, not task-oriented.
NATURAL_PROMPTS = [
    "You are a friendly, natural conversationalist. Speak naturally and engage with what the other person says.",
    "You are having a casual conversation. Be yourself, be natural, and respond to what you hear.",
    "You are a warm and attentive conversation partner. Listen actively and respond naturally.",
    "You are chatting naturally with someone. Keep the conversation flowing and be genuine.",
    "You are an easygoing conversationalist. Respond naturally, be yourself, and keep things friendly.",
    "You enjoy having a good conversation. Listen well, respond thoughtfully, and be natural.",
    "You are having an everyday conversation. Be relaxed, genuine, and engaged.",
    "You are a good listener who responds naturally. Keep the conversation comfortable and flowing.",
]


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def generate_sine(freq_hz: float, duration_sec: float, sr: int) -> np.ndarray:
    t = np.arange(int(duration_sec * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def extract_voice_reference(audio_ch0: np.ndarray, sr: int, duration: float) -> np.ndarray:
    """Extract a voice reference from the assistant channel.

    Takes the first segment with speech (RMS above threshold) up to `duration` seconds.
    Falls back to first N seconds if no speech found.
    """
    target_samples = int(duration * sr)

    # Find first segment with reasonable energy (skip leading silence)
    frame_size = sr // 4  # 250ms frames
    start_idx = 0
    for i in range(0, len(audio_ch0) - frame_size, frame_size):
        rms = np.sqrt(np.mean(audio_ch0[i:i + frame_size] ** 2))
        if rms > 0.01:
            start_idx = max(0, i - frame_size)  # include a bit before
            break

    ref = audio_ch0[start_idx:start_idx + target_samples]
    if len(ref) < target_samples:
        ref = np.pad(ref, (0, target_samples - len(ref)))
    return ref


def build_prompt_region(voice_ref: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Build prompt region for both channels. Returns (assistant, user, prompt_end_sec)."""
    # Assistant channel: [voice_ref][sine_marker][silence]
    sine_marker = generate_sine(SINE_FREQ, SINE_MARKER_DURATION, sr)
    silence = np.zeros(int(SILENCE_AFTER_MARKER * sr), dtype=np.float32)
    assistant = np.concatenate([voice_ref, sine_marker, silence])

    # User channel: 440 Hz sine for entire prompt region (PersonaPlex pattern)
    user = generate_sine(SINE_FREQ, len(assistant) / sr, sr)

    prompt_end_sec = len(assistant) / sr
    return assistant, user, prompt_end_sec


# ---------------------------------------------------------------------------
# LLM-based prompt generation (for diverse, context-aware prompts)
# ---------------------------------------------------------------------------
def _make_client(cfg: dict):
    from openai import OpenAI
    return OpenAI(
        api_key=cfg.get("llm_api_key", "sk-WtJy90ltT2hpJSFKQ1u_TA"),
        base_url=cfg.get("llm_base_url", "https://litellm-proxy.ml-serving-internal.scale.com/v1"),
    )


def _pick_model(cfg: dict) -> str:
    models = cfg.get("llm_models", [])
    if not models:
        return cfg.get("llm_model", "bedrock/qwen.qwen3-32b-v1:0")
    weights = [m["weight"] for m in models]
    return random.choices(models, weights=weights, k=1)[0]["model"]


def generate_system_prompt_llm(
    client, model: str, conversation_snippet: str,
) -> str | None:
    """Generate a natural system prompt based on conversation content."""
    import openai as oai

    prompt = (
        "Based on this conversation snippet, write a short system prompt (1-2 sentences) "
        "for the AI participant. The prompt should describe the AI's role/personality in "
        "this specific conversation. Keep it natural and simple — no jargon, no bullet points. "
        "Output ONLY the system prompt text.\n\n"
        f"Conversation:\n{conversation_snippet}"
    )
    temp = 1.0 if "gpt-5" in model else 0.9

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=100,
        )
        text = resp.choices[0].message.content.strip()
        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1]
        return text
    except oai.RateLimitError:
        time.sleep(5)
        return None
    except Exception as e:
        logger.debug("LLM prompt gen failed: %s", e)
        return None


def get_conversation_snippet(align_path: Path, max_words: int = 50) -> str:
    """Extract first N words from alignments for LLM context."""
    if not align_path.exists():
        return ""
    try:
        data = json.load(open(align_path))
        words = [a[0] for a in data.get("alignments", [])[:max_words]]
        return " ".join(words)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------
def process_file(
    src_wav: Path,
    src_json: Path,
    dst_wav: Path,
    dst_json: Path,
    cfg: dict | None = None,
    use_llm: bool = False,
) -> dict | None:
    """Process one external audio file: prepend prompt region, update alignments."""
    if dst_wav.exists() and dst_json.exists():
        # Already processed
        info = sf.info(str(dst_wav))
        return {"path": str(dst_wav.name), "duration": round(info.duration, 2)}

    # Read source audio
    try:
        audio, sr = sf.read(str(src_wav), dtype="float32")
    except Exception as e:
        logger.warning("Failed to read %s: %s", src_wav, e)
        return None

    if audio.ndim == 1:
        # Mono — duplicate to stereo
        audio = np.stack([audio, audio], axis=-1)

    if sr != SAMPLE_RATE:
        import librosa
        ch0 = librosa.resample(audio[:, 0], orig_sr=sr, target_sr=SAMPLE_RATE)
        ch1 = librosa.resample(audio[:, 1], orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = np.stack([ch0, ch1], axis=-1)
        sr = SAMPLE_RATE

    # Extract voice reference from assistant channel
    voice_ref = extract_voice_reference(audio[:, 0], sr, VOICE_PROMPT_DURATION)

    # Build prompt region
    prompt_asst, prompt_user, prompt_end_sec = build_prompt_region(voice_ref, sr)
    prompt_samples = len(prompt_asst)

    # Prepend prompt region to audio
    new_ch0 = np.concatenate([prompt_asst, audio[:, 0]])
    new_ch1 = np.concatenate([prompt_user, audio[:, 1]])
    new_audio = np.stack([new_ch0, new_ch1], axis=-1)

    # Normalize
    for ch in range(2):
        peak = np.abs(new_audio[:, ch]).max()
        if peak > 0:
            new_audio[:, ch] *= 0.95 / peak

    # Write new WAV
    sf.write(str(dst_wav), new_audio, sr)

    # Update alignments: shift timestamps and inject system prompt
    system_prompt = random.choice(NATURAL_PROMPTS)

    # Try LLM-generated prompt if configured
    if use_llm and cfg:
        snippet = get_conversation_snippet(src_json)
        if snippet:
            client = _make_client(cfg)
            model = _pick_model(cfg)
            llm_prompt = generate_system_prompt_llm(client, model, snippet)
            if llm_prompt:
                system_prompt = llm_prompt

    tagged_prompt = f"<system> {system_prompt} <system>"

    # Build new alignments
    new_alignments = []

    # System prompt text in prompt region
    words = tagged_prompt.split()
    if words and prompt_end_sec > 0:
        dur_per_word = prompt_end_sec / len(words)
        for i, word in enumerate(words):
            start = round(i * dur_per_word, 3)
            end = round((i + 1) * dur_per_word, 3)
            new_alignments.append([word, [start, end], "SPEAKER_MAIN"])

    # Shift existing alignments forward by prompt duration
    if src_json.exists():
        try:
            src_data = json.load(open(src_json))
            for align in src_data.get("alignments", []):
                word, (start, end), speaker = align
                new_alignments.append([
                    word,
                    [round(start + prompt_end_sec, 3), round(end + prompt_end_sec, 3)],
                    speaker,
                ])
        except Exception as e:
            logger.warning("Failed to read alignments %s: %s", src_json, e)

    # Write alignment JSON in moshi-finetune format (matching format_dataset.py)
    # text_conditions is read by the training data loader for loss masking
    align_data = {
        "alignments": new_alignments,
        "text_conditions": {
            "prompt_end_sec": str(round(prompt_end_sec, 3)),
            "system_prompt": tagged_prompt,
        },
    }
    with open(dst_json, "w") as f:
        json.dump(align_data, f, indent=2)

    return {"path": str(dst_wav.name), "duration": round(len(new_audio) / sr, 2)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Add system prompt regions to external datasets",
    )
    parser.add_argument("--dataset", required=True,
                        help="Dataset name (annutacon, fisher, or any name in data/external/). Use 'all' for all.")
    parser.add_argument("--config", default="configs/generation.yaml")
    parser.add_argument("--use_llm", action="store_true", default=True,
                        help="Use LLM for context-aware prompts (default: on)")
    parser.add_argument("--no_llm", action="store_true",
                        help="Disable LLM prompts — use pool of generic prompts instead")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    random.seed(args.seed)
    cfg = load_yaml(args.config).get("transcript", {})
    use_llm = args.use_llm and not args.no_llm
    if args.dataset == "all":
        # Find all datasets in data/external/ that have audio/ subdirectories
        ext_root = Path("data/external")
        datasets = [
            d.name for d in sorted(ext_root.iterdir())
            if d.is_dir() and (d / "audio").exists() and not d.name.endswith("_prompted")
        ]
        logger.info("Found datasets: %s", datasets)
    else:
        datasets = [args.dataset]

    for dataset_name in datasets:
        base_dir = Path(f"data/external/{dataset_name}")
        audio_dir = base_dir / "audio"

        if not audio_dir.exists():
            logger.error("Audio dir not found: %s", audio_dir)
            continue

        # Output dir: data/external/{name}_prompted/audio/
        out_dir = ensure_dir(base_dir.parent / f"{dataset_name}_prompted")
        out_audio_dir = ensure_dir(out_dir / "audio")

        # Find all source WAVs
        src_wavs = sorted(audio_dir.glob("*.wav"))
        logger.info("%s: %d WAV files", dataset_name, len(src_wavs))

        if not src_wavs:
            logger.warning("%s: no WAV files found in %s — run import_external first", dataset_name, audio_dir)
            continue

        if args.dry_run:
            # Process just one
            src = src_wavs[0]
            src_json = src.with_suffix(".json")
            dst_wav = out_audio_dir / src.name
            dst_json = out_audio_dir / src_json.name
            result = process_file(src, src_json, dst_wav, dst_json, cfg, use_llm)
            logger.info("Dry run result: %s", result)

            # Show the alignment
            if dst_json.exists():
                data = json.load(open(dst_json))
                tc = data.get("text_conditions", {})
                prompt_end = float(tc.get("prompt_end_sec", 0))
                logger.info("Prompt end: %.3fs", prompt_end)
                logger.info("System prompt: %s", tc.get("system_prompt", ""))
                logger.info("First 5 alignments: %s", data["alignments"][:5])
                logger.info("Alignment at prompt boundary: %s",
                            [a for a in data["alignments"] if a[1][0] >= prompt_end - 0.5][:3])
            return

        # Distributed coordination: use atomic file claims so multiple nodes
        # can process the same dataset concurrently without duplicating work.
        from pipeline.distributed import try_claim, release_claim

        claims_dir = ensure_dir(out_dir / ".claims")

        # Filter to only unfinished files
        to_process = []
        already_done = 0
        for src_wav in src_wavs:
            dst_wav = out_audio_dir / src_wav.name
            dst_json = out_audio_dir / src_wav.with_suffix(".json").name
            if dst_wav.exists() and dst_json.exists():
                already_done += 1
            else:
                to_process.append(src_wav)

        logger.info("%s: %d already done, %d to process", dataset_name, already_done, len(to_process))

        if not to_process:
            logger.info("All files already processed — rebuilding manifest only")
        else:
            def _worker(src_wav):
                stem = src_wav.stem
                claim_path = claims_dir / f"{stem}.claim"
                if not try_claim(claim_path):
                    return None  # another node claimed it

                src_json = src_wav.with_suffix(".json")
                dst_wav = out_audio_dir / src_wav.name
                dst_json = out_audio_dir / src_json.name
                try:
                    result = process_file(src_wav, src_json, dst_wav, dst_json, cfg, use_llm)
                    return result
                except Exception as e:
                    logger.error("Failed %s: %s", src_wav.name, e)
                    release_claim(claim_path)
                    return None

            with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
                futures = {pool.submit(_worker, w): w for w in to_process}
                for future in tqdm(as_completed(futures), total=len(futures), desc=dataset_name):
                    try:
                        future.result()
                    except Exception as e:
                        logger.error("Worker error: %s", e)

        # Rebuild manifest from all completed files (idempotent)
        manifest_entries = []
        for wav_file in sorted(out_audio_dir.glob("*.wav")):
            json_file = wav_file.with_suffix(".json")
            if json_file.exists():
                try:
                    info = sf.info(str(wav_file))
                    manifest_entries.append({
                        "path": f"audio/{wav_file.name}",
                        "duration": round(info.duration, 2),
                    })
                except Exception:
                    pass

        manifest_path = out_dir / "manifest.jsonl"
        with open(manifest_path, "w") as f:
            for entry in manifest_entries:
                f.write(json.dumps(entry) + "\n")

        total_hours = sum(e["duration"] for e in manifest_entries) / 3600
        logger.info("%s: %d files, %.0fh → %s", dataset_name, len(manifest_entries), total_hours, manifest_path)


if __name__ == "__main__":
    main()
