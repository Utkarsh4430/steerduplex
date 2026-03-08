"""Quality filtering for synthesized audio (lightweight for scale).

Checks:
1. WER via Whisper ASR — is the speech intelligible?
2. Duration — is the turn too short or too long?
3. Silence ratio — is it mostly silence?

Deliberately omits expensive checks (UTMOS, speaker similarity) for scalability.
Those can be run on a sample for spot-checking.

Usage:
    python -m pipeline.quality_filter --config configs/generation.yaml
    python -m pipeline.quality_filter --config configs/generation.yaml --category A3_tone_controlled
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from pipeline.utils import load_json, load_yaml, save_json


class QualityChecker:
    def __init__(self, wer_threshold: float = 0.20, min_duration: float = 0.5, max_silence_ratio: float = 0.5, whisper_model: str = "medium"):
        self.wer_threshold = wer_threshold
        self.min_duration = min_duration
        self.max_silence_ratio = max_silence_ratio
        self.whisper_model_name = whisper_model
        self._whisper = None

    @property
    def whisper(self):
        if self._whisper is None:
            import whisper
            self._whisper = whisper.load_model(self.whisper_model_name)
        return self._whisper

    def check_wer(self, audio_path: str, expected_text: str) -> tuple[float, bool]:
        from jiwer import wer
        result = self.whisper.transcribe(audio_path, language="en")
        transcribed = result["text"].strip().lower()
        expected = expected_text.strip().lower()
        for marker in ["(laughs)", "(sighs)", "(pauses)", "(clears throat)", "(whispers)"]:
            expected = expected.replace(marker, "").strip()
        if not expected:
            return 0.0, True
        error_rate = wer(expected, transcribed)
        return round(error_rate, 3), error_rate <= self.wer_threshold

    def check_duration(self, audio_path: str) -> tuple[float, bool]:
        info = sf.info(audio_path)
        return round(info.duration, 2), info.duration >= self.min_duration

    def check_silence(self, audio_path: str) -> tuple[float, bool]:
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
        energy = np.abs(audio)
        silence_frames = (energy < 0.01).sum()
        ratio = silence_frames / max(len(audio), 1)
        return round(float(ratio), 3), ratio <= self.max_silence_ratio

    def check_turn(self, audio_path: str, expected_text: str) -> dict:
        results = {"passed": True, "checks": {}}

        dur, dur_ok = self.check_duration(audio_path)
        results["checks"]["duration"] = {"value": dur, "passed": dur_ok}
        if not dur_ok:
            results["passed"] = False
            return results  # skip expensive WER if too short

        sil, sil_ok = self.check_silence(audio_path)
        results["checks"]["silence_ratio"] = {"value": sil, "passed": sil_ok}
        if not sil_ok:
            results["passed"] = False

        wer_score, wer_ok = self.check_wer(audio_path, expected_text)
        results["checks"]["wer"] = {"value": wer_score, "passed": wer_ok}
        if not wer_ok:
            results["passed"] = False

        return results


def filter_conversation(checker: QualityChecker, transcript: dict) -> dict:
    all_passed = True
    for turn in transcript["turns"]:
        audio_path = turn.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            turn["quality_check"] = {"passed": False, "reason": "missing_audio"}
            all_passed = False
            continue

        check = checker.check_turn(audio_path, turn["text"])
        turn["quality_check"] = check
        if not check["passed"]:
            all_passed = False

    transcript["quality_passed"] = all_passed
    return transcript


def main():
    parser = argparse.ArgumentParser(description="Quality filter synthesized audio")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--synth_dir", type=str, default=None)
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["quality"]
    checker = QualityChecker(
        wer_threshold=cfg.get("wer_threshold", 0.20),
        min_duration=cfg.get("min_duration_sec", 0.5),
        max_silence_ratio=cfg.get("max_silence_ratio", 0.5),
        whisper_model=cfg.get("whisper_model", "medium"),
    )

    synth_dir = Path(args.synth_dir or load_yaml(args.config)["tts"]["output_dir"])
    passed = 0
    total = 0

    for cat_dir in sorted(synth_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.category and cat_dir.name != args.category:
            continue

        for synth_path in tqdm(sorted(cat_dir.glob("*_synth.json")), desc=cat_dir.name):
            transcript = load_json(synth_path)
            if "quality_passed" in transcript:
                total += 1
                if transcript["quality_passed"]:
                    passed += 1
                continue  # already checked
            transcript = filter_conversation(checker, transcript)
            save_json(transcript, synth_path)
            total += 1
            if transcript["quality_passed"]:
                passed += 1

    print(f"\nQuality: {passed}/{total} passed ({passed/max(total,1)*100:.1f}%)")


if __name__ == "__main__":
    main()
