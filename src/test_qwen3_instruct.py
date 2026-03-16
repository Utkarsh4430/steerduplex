"""Quick interactive test for Qwen3-TTS instruct tuning.

Edit SAMPLES below, run, listen, tweak, repeat.

Usage:
    python test_qwen3_instruct.py
    python test_qwen3_instruct.py --device cuda:1
    python test_qwen3_instruct.py --speaker Vivian
"""

import argparse
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

OUT_DIR = Path("data/tts_comparison/qwen3_test")

# ┌─────────────────────────────────────────────────────────────────────┐
# │  EDIT THESE — change instruct, re-run, listen                      │
# │  Set instruct to None to hear the raw voice with no style control  │
# └─────────────────────────────────────────────────────────────────────┘
SAMPLES = [
    {"id": "neutral",     "text": "Hello! Welcome to our customer service line. How can I help you today?",            "instruct": None},
    {"id": "cheerful",    "text": "Oh hey, great to see you again! What are we working on today?",                     "instruct": "cheerful"},
    {"id": "warm",        "text": "I'm really sorry to hear about that. Let me see what I can do to help.",            "instruct": "warm"},
    {"id": "calm",        "text": "Take a deep breath. Let's slow everything down and work through this together.",    "instruct": "calm"},
    {"id": "excited",     "text": "Oh wow, that's absolutely incredible! I can't believe you managed to pull that off!", "instruct": "enthusiastic"},
    {"id": "stern",       "text": "I need to be very clear about this. You must not share that information.",          "instruct": "stern"},
    {"id": "fast",        "text": "The capital of France is Paris, the largest ocean is the Pacific.",                 "instruct": "fast"},
    {"id": "slow",        "text": "Now close your eyes. Breathe in deeply. Hold it. And slowly let it go.",           "instruct": "slow"},
    {"id": "sad",         "text": "I know this isn't easy. Sometimes things just don't work out the way we hoped.",    "instruct": "subdued"},
    {"id": "playful",     "text": "Alright, pop quiz time! Let's see if you've been paying attention.",                "instruct": "playful"},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speaker", default="Ryan", help="CustomVoice preset: Ryan, Vivian, Dylan, Serena")
    args = parser.parse_args()

    from qwen_tts import Qwen3TTSModel

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading Qwen3-TTS CustomVoice on {args.device}...")
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=args.device, dtype=torch.bfloat16,
    )
    print(f"Speaker: {args.speaker}\n")

    for s in SAMPLES:
        out_path = OUT_DIR / f"{args.speaker.lower()}_{s['id']}.wav"
        kwargs = dict(text=s["text"], language="English", speaker=args.speaker)
        if s["instruct"]:
            kwargs["instruct"] = s["instruct"]

        tag = s["instruct"] or "NONE"
        try:
            wavs, sr = model.generate_custom_voice(**kwargs)
            audio = wavs[0] if isinstance(wavs, list) else wavs
            if isinstance(audio, torch.Tensor):
                audio = audio.cpu().numpy()
            audio = audio.astype(np.float32)
            peak = np.abs(audio).max()
            if peak > 0:
                audio = audio * (0.95 / peak)
            sf.write(str(out_path), audio, sr)
            print(f"  [ok] {out_path.name:30s}  instruct={tag:20s}  ({len(audio)/sr:.1f}s)")
        except Exception as e:
            print(f"  [FAIL] {out_path.name}: {e}")

    print(f"\nOutputs in: {OUT_DIR}/")
    print("Edit SAMPLES in this file, delete the wavs, and re-run to iterate.")


if __name__ == "__main__":
    main()
