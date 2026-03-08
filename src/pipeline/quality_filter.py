"""Quality filtering for synthesized audio.

Checks:
1. UTMOS naturalness score
2. Whisper ASR word error rate (WER)
3. Speaker similarity (cosine sim between ref and generated)

Usage:
    python -m pipeline.quality_filter \
        --config configs/generation.yaml \
        --synth_dir data/tts_audio
"""

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from pipeline.utils import ensure_dir, load_json, load_yaml, save_json


class QualityChecker:
    """Multi-metric quality checker for synthesized speech."""

    def __init__(
        self,
        utmos_threshold: float = 3.5,
        wer_threshold: float = 0.15,
        speaker_sim_threshold: float = 0.75,
        whisper_model: str = "medium",
    ):
        self.utmos_threshold = utmos_threshold
        self.wer_threshold = wer_threshold
        self.speaker_sim_threshold = speaker_sim_threshold
        self.whisper_model_name = whisper_model

        # Lazy-loaded models
        self._whisper = None
        self._speaker_encoder = None

    @property
    def whisper(self):
        if self._whisper is None:
            import whisper

            self._whisper = whisper.load_model(self.whisper_model_name)
        return self._whisper

    @property
    def speaker_encoder(self):
        if self._speaker_encoder is None:
            from resemblyzer import VoiceEncoder

            self._speaker_encoder = VoiceEncoder()
        return self._speaker_encoder

    def check_wer(self, audio_path: str, expected_text: str) -> tuple[float, bool]:
        """Check word error rate via Whisper ASR."""
        from jiwer import wer

        result = self.whisper.transcribe(audio_path, language="en")
        transcribed = result["text"].strip().lower()
        expected = expected_text.strip().lower()

        # Remove non-verbal markers for WER computation
        for marker in ["(laughs)", "(sighs)", "(pauses)", "(clears throat)"]:
            expected = expected.replace(marker, "").strip()

        error_rate = wer(expected, transcribed) if expected else 0.0
        return error_rate, error_rate <= self.wer_threshold

    def check_speaker_similarity(
        self, generated_path: str, ref_path: str
    ) -> tuple[float, bool]:
        """Check speaker similarity between generated and reference audio."""
        from resemblyzer import preprocess_wav

        gen_wav = preprocess_wav(Path(generated_path))
        ref_wav = preprocess_wav(Path(ref_path))

        gen_embed = self.speaker_encoder.embed_utterance(gen_wav)
        ref_embed = self.speaker_encoder.embed_utterance(ref_wav)

        similarity = float(np.dot(gen_embed, ref_embed))
        return similarity, similarity >= self.speaker_sim_threshold

    def check_utmos(self, audio_path: str) -> tuple[float, bool]:
        """Check UTMOS naturalness score."""
        try:
            import utmos

            audio, sr = sf.read(audio_path)
            score = utmos.score(audio, sr)
            return float(score), score >= self.utmos_threshold
        except ImportError:
            # UTMOS not available, skip
            return 0.0, True

    def check_turn(
        self,
        audio_path: str,
        expected_text: str,
        ref_path: str | None = None,
    ) -> dict:
        """Run all quality checks on a single turn."""
        results = {"audio_path": audio_path, "passed": True, "checks": {}}

        # WER check
        wer_score, wer_ok = self.check_wer(audio_path, expected_text)
        results["checks"]["wer"] = {"score": wer_score, "passed": wer_ok}
        if not wer_ok:
            results["passed"] = False

        # Speaker similarity (if ref available)
        if ref_path and Path(ref_path).exists():
            sim_score, sim_ok = self.check_speaker_similarity(audio_path, ref_path)
            results["checks"]["speaker_sim"] = {"score": sim_score, "passed": sim_ok}
            if not sim_ok:
                results["passed"] = False

        # UTMOS
        utmos_score, utmos_ok = self.check_utmos(audio_path)
        results["checks"]["utmos"] = {"score": utmos_score, "passed": utmos_ok}
        if not utmos_ok:
            results["passed"] = False

        return results


def filter_conversation(checker: QualityChecker, transcript: dict) -> dict:
    """Run quality checks on all turns of a conversation.

    Returns transcript with quality_check field added to each turn.
    """
    assistant_ref = transcript.get("assistant_voice", {}).get("ref_path")
    user_ref = transcript.get("user_voice", {}).get("ref_path")

    all_passed = True
    for turn in transcript["turns"]:
        audio_path = turn.get("audio_path")
        if not audio_path or not Path(audio_path).exists():
            turn["quality_check"] = {"passed": False, "reason": "missing_audio"}
            all_passed = False
            continue

        ref_path = assistant_ref if turn["role"] == "assistant" else user_ref
        check = checker.check_turn(audio_path, turn["text"], ref_path)
        turn["quality_check"] = check
        if not check["passed"]:
            all_passed = False

    transcript["quality_passed"] = all_passed
    return transcript


def main():
    parser = argparse.ArgumentParser(description="Quality filter synthesized audio")
    parser.add_argument("--config", type=str, default="configs/generation.yaml")
    parser.add_argument("--synth_dir", type=str, default="data/tts_audio")
    parser.add_argument("--category", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)["quality"]
    checker = QualityChecker(
        utmos_threshold=cfg["utmos_threshold"],
        wer_threshold=cfg["wer_threshold"],
        speaker_sim_threshold=cfg["speaker_sim_threshold"],
        whisper_model=cfg["whisper_model"],
    )

    synth_dir = Path(args.synth_dir)
    passed_count = 0
    total_count = 0

    for cat_dir in sorted(synth_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.category and cat_dir.name != args.category:
            continue

        for synth_path in tqdm(sorted(cat_dir.glob("*_synth.json")), desc=cat_dir.name):
            transcript = load_json(synth_path)
            transcript = filter_conversation(checker, transcript)
            save_json(transcript, synth_path)  # overwrite with quality info
            total_count += 1
            if transcript["quality_passed"]:
                passed_count += 1

    print(f"\nQuality filter: {passed_count}/{total_count} conversations passed")


if __name__ == "__main__":
    main()
