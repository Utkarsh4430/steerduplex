"""ASR driver for Full Duplex Bench v1.0 / v1.5.

Two backends:
  - parakeet (default): NVIDIA Parakeet-TDT-0.6B-v2 via NeMo. Mirrors the
    official Full-Duplex-Bench `get_transcript/asr.py`. Runs in the
    raman_fdb_v1 env (which already has nemo_toolkit[asr]).
  - whisperx: WhisperX large-v3 + wav2vec2 alignment, runs in the
    raman_whisperx env. Word boundaries differ from Parakeet, which
    biases all timestamp-driven metrics — keep as a fallback, not the
    default.

Both backends emit the same JSON shape:

    {"text": "...", "chunks": [{"text": "word", "timestamp": [start, end]}, ...]}

For v1.0:
  - task `default`:           output.wav -> output.json
  - task `user_interruption`: output.wav cropped after interrupt.json end-time
                              then transcribed (offsets adjusted).

For v1.5 paired tasks:
  - output.wav       -> output.json
  - clean_output.wav -> clean_output.json
  - input.json / clean_input.json ship with the dataset and are carried into
    the mirror tree by dataset_utils, so we never re-transcribe the user
    inputs (matches the official pipeline, which only transcribes
    output.wav).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple

from tqdm import tqdm

from eval.fdb_v1.dataset_utils import (
    TASKS_V1,
    TASKS_V15,
    discover_sample_dirs,
    mirror_subdir_name,
)

logger = logging.getLogger(__name__)


# ---- Backend state (lazy) ----------------------------------------------------

# WhisperX
_WHISPERX_ASR = None
_WHISPERX_ALIGN = None
_WHISPERX_ALIGN_META = None

# Parakeet
_PARAKEET_MODEL = None


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=4))
    os.replace(tmp, path)


# ---- Parakeet backend --------------------------------------------------------

def _load_parakeet(device: str):
    """Lazy-load NVIDIA Parakeet-TDT-0.6B-v2.

    Mirrors `Full-Duplex-Bench/v1_v1.5/get_transcript/asr.py:18-20`.
    """
    global _PARAKEET_MODEL
    if _PARAKEET_MODEL is not None:
        return _PARAKEET_MODEL
    import nemo.collections.asr as nemo_asr  # noqa: WPS433
    logger.info("Loading Parakeet (nvidia/parakeet-tdt-0.6b-v2) on %s", device)
    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    )
    # Move to the requested device. Prefer model.to(); .cuda() works only for
    # cuda:0 and ignores explicit device strings.
    try:
        model = model.to(device)
    except Exception:
        # Older NeMo versions sometimes need .cuda() and rely on
        # CUDA_VISIBLE_DEVICES for indexing.
        model = model.cuda()
    _PARAKEET_MODEL = model
    return _PARAKEET_MODEL


def _parakeet_transcribe(wav_path: Path, json_path: Path, device: str,
                         offset: float = 0.0, crop_start_sec: float = 0.0) -> None:
    """Transcribe `wav_path` with Parakeet and write FDB-shape JSON.

    Faithfully ports the official asr.py logic:
      - sf.read(audio_path)  (no whisperx-style 16 kHz mono coerce)
      - mean across channels if multichannel
      - crop = waveform[int(crop_start_sec * sr):]
      - tempfile WAV write + nemo .transcribe([tmp.name], timestamps=True)
      - emit chunks with start + offset, end + offset
    """
    import soundfile as sf  # noqa: WPS433

    model = _load_parakeet(device)

    waveform, sr = sf.read(str(wav_path))
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if crop_start_sec > 0:
        start_idx = int(crop_start_sec * sr)
        if start_idx >= waveform.shape[0]:
            _atomic_write_json(json_path, {"text": "", "chunks": []})
            return
        waveform = waveform[start_idx:]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, waveform, sr)
        asr_outputs = model.transcribe([tmp_path], timestamps=True)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not asr_outputs:
        _atomic_write_json(json_path, {"text": "", "chunks": []})
        return

    result = asr_outputs[0]
    word_timestamps = result.timestamp.get("word", []) if hasattr(result, "timestamp") else []

    chunks: List[dict] = []
    text_parts: List[str] = []
    for w in word_timestamps:
        start_time = float(w["start"]) + offset
        end_time = float(w["end"]) + offset
        word = w["word"]
        chunks.append({"text": word, "timestamp": [start_time, end_time]})
        text_parts.append(word)
    payload = {"text": " ".join(text_parts).strip(), "chunks": chunks}
    _atomic_write_json(json_path, payload)


# ---- WhisperX backend (fallback) --------------------------------------------

def _load_whisperx(asr_model: str, device: str, compute_type: str, language: str):
    global _WHISPERX_ASR, _WHISPERX_ALIGN, _WHISPERX_ALIGN_META
    import whisperx  # noqa: WPS433

    if _WHISPERX_ASR is None:
        logger.info("Loading WhisperX ASR model %s on %s (%s)", asr_model, device, compute_type)
        _WHISPERX_ASR = whisperx.load_model(asr_model, device, compute_type=compute_type, language=language)
    if _WHISPERX_ALIGN is None:
        logger.info("Loading wav2vec2 alignment model for language=%s", language)
        _WHISPERX_ALIGN, _WHISPERX_ALIGN_META = whisperx.load_align_model(language_code=language, device=device)
    return _WHISPERX_ASR, _WHISPERX_ALIGN, _WHISPERX_ALIGN_META


def _whisperx_segments_to_fdb_json(aligned, offset: float = 0.0) -> dict:
    chunks: List[dict] = []
    text_parts: List[str] = []
    for seg in aligned.get("segments", []):
        for w in seg.get("words", []) or []:
            if "start" not in w or "end" not in w:
                continue  # unaligned words (rare but possible with noisy audio)
            word_text = w.get("word", "")
            chunks.append({
                "text": word_text,
                "timestamp": [float(w["start"]) + offset, float(w["end"]) + offset],
            })
            text_parts.append(word_text)
    return {"text": " ".join(t.strip() for t in text_parts).strip(), "chunks": chunks}


def _whisperx_transcribe(wav_path: Path, json_path: Path, device: str,
                         batch_size: int, language: str,
                         offset: float = 0.0, crop_start_sec: float = 0.0) -> None:
    import whisperx  # noqa: WPS433
    audio = whisperx.load_audio(str(wav_path))  # → 16 kHz mono float32
    if crop_start_sec > 0:
        sr = 16000
        start_idx = int(crop_start_sec * sr)
        if start_idx >= audio.shape[-1]:
            _atomic_write_json(json_path, {"text": "", "chunks": []})
            return
        audio = audio[start_idx:]

    asr, align_model, align_meta = _WHISPERX_ASR, _WHISPERX_ALIGN, _WHISPERX_ALIGN_META
    result = asr.transcribe(audio, batch_size=batch_size, language=language)
    segments = result.get("segments", [])
    if not segments:
        _atomic_write_json(json_path, {"text": "", "chunks": []})
        return
    aligned = whisperx.align(
        segments, align_model, align_meta, audio, device,
        return_char_alignments=False,
    )
    payload = _whisperx_segments_to_fdb_json(aligned, offset=offset)
    _atomic_write_json(json_path, payload)


# ---- Backend dispatch --------------------------------------------------------

class _Backend:
    def __init__(self, name: str, device: str, *, asr_model: str = "",
                 compute_type: str = "", batch_size: int = 16, language: str = "en"):
        self.name = name
        self.device = device
        self.asr_model = asr_model
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.language = language

    def load(self) -> None:
        if self.name == "parakeet":
            _load_parakeet(self.device)
        elif self.name == "whisperx":
            _load_whisperx(self.asr_model, self.device, self.compute_type, self.language)
        else:
            raise ValueError(f"Unknown ASR backend: {self.name}")

    def transcribe(self, wav_path: Path, json_path: Path,
                   offset: float = 0.0, crop_start_sec: float = 0.0) -> None:
        if self.name == "parakeet":
            _parakeet_transcribe(wav_path, json_path, self.device,
                                 offset=offset, crop_start_sec=crop_start_sec)
        else:
            _whisperx_transcribe(wav_path, json_path, self.device,
                                 self.batch_size, self.language,
                                 offset=offset, crop_start_sec=crop_start_sec)


# ---- Per-sample dispatchers ---------------------------------------------------

def _run_v1_sample(backend: _Backend, sample_dir: Path, asr_task: str,
                   overwrite: bool) -> Tuple[bool, Optional[str]]:
    out_wav = sample_dir / "output.wav"
    out_json = sample_dir / "output.json"
    if not out_wav.exists():
        return False, "missing output.wav"
    if out_json.exists() and not overwrite:
        return True, None
    try:
        if asr_task == "user_interruption":
            interrupt_path = sample_dir / "interrupt.json"
            if not interrupt_path.exists():
                return False, "missing interrupt.json"
            with interrupt_path.open() as fh:
                interrupt_meta = json.load(fh)
            _, end_interrupt = interrupt_meta[0]["timestamp"]
            backend.transcribe(
                out_wav, out_json,
                offset=float(end_interrupt), crop_start_sec=float(end_interrupt),
            )
        else:
            backend.transcribe(out_wav, out_json)
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _run_v15_sample(backend: _Backend, sample_dir: Path,
                    overwrite: bool) -> Tuple[bool, Optional[str]]:
    """Transcribe model outputs only.

    v1.5 zips ship `input.json` / `clean_input.json` aligned with Parakeet;
    we carry those through the mirror as optional annotations and do NOT
    re-transcribe the user inputs (matches the official pipeline, which
    only globs `*/output.wav`).
    """
    pairs: List[Tuple[Path, Path]] = []
    for wav_name, json_name in [
        ("output.wav", "output.json"),
        ("clean_output.wav", "clean_output.json"),
    ]:
        wav = sample_dir / wav_name
        jsn = sample_dir / json_name
        if not wav.exists():
            continue
        if jsn.exists() and not overwrite:
            continue
        pairs.append((wav, jsn))

    errs: List[str] = []
    for wav, jsn in pairs:
        try:
            backend.transcribe(wav, jsn)
        except Exception as exc:  # noqa: BLE001
            errs.append(f"{wav.name}: {type(exc).__name__}: {exc}")
    if errs:
        return False, "; ".join(errs)
    return True, None


# ---- Orchestration -----------------------------------------------------------

def _iter_task_dirs(output_root: Path, version: str,
                    tasks_filter: Optional[List[str]]) -> List[Tuple[str, str, Path, dict]]:
    """Yield (version_prefix, task, task_subdir, task_meta)."""
    out: List[Tuple[str, str, Path, dict]] = []
    if version in ("1.0", "both"):
        for task, meta in TASKS_V1.items():
            if tasks_filter and task not in tasks_filter:
                continue
            subdir = mirror_subdir_name("v1", task)
            out.append(("v1", task, output_root / subdir, meta))
    if version in ("1.5", "both"):
        for task, meta in TASKS_V15.items():
            if tasks_filter and task not in tasks_filter:
                continue
            subdir = mirror_subdir_name("v15", task)
            out.append(("v15", task, output_root / subdir, meta))
    return out


def run_asr(output_root: Path, version: str, tasks_filter: Optional[List[str]],
            backend: _Backend, overwrite: bool) -> dict:
    load_start = time.monotonic()
    print(f"\nLoading ASR backend: {backend.name} on {backend.device}")
    sys.stdout.flush()
    backend.load()
    print(f"  models loaded in {_fmt_duration(time.monotonic() - load_start)}")

    plan: List[Tuple[str, str, Path, dict, List[Path]]] = []
    for version_prefix, task, task_subdir, meta in _iter_task_dirs(output_root, version, tasks_filter):
        sample_dirs = discover_sample_dirs(task_subdir) if task_subdir.is_dir() else []
        plan.append((version_prefix, task, task_subdir, meta, sample_dirs))

    total_samples = sum(len(s) for _, _, _, _, s in plan)
    print("")
    print(f"{'Task':30s}  {'Samples':>9s}  {'Status':>10s}")
    print("-" * 55)
    for version_prefix, task, _task_subdir, _meta, sample_dirs in plan:
        label = f"{version_prefix}/{task}"
        status = "ready" if sample_dirs else "empty/missing"
        print(f"  {label:28s}  {len(sample_dirs):>9d}  {status:>10s}")
    print("-" * 55)
    print(f"  {'TOTAL':28s}  {total_samples:>9d}")
    sys.stdout.flush()

    stats = {"ok": 0, "err": 0, "skipped": 0}
    per_task: dict = defaultdict(lambda: {"ok": 0, "err": 0, "elapsed": 0.0})
    wall_start = time.monotonic()

    for version_prefix, task, _task_subdir, meta, sample_dirs in plan:
        label = f"{version_prefix}/{task}"
        if not sample_dirs:
            stats["skipped"] += 1
            continue

        task_start = time.monotonic()
        pbar = tqdm(total=len(sample_dirs), desc=f"ASR {label}", unit="sample",
                    dynamic_ncols=True)
        for sample_dir in sample_dirs:
            t0 = time.monotonic()
            if version_prefix == "v1":
                ok, err = _run_v1_sample(
                    backend, sample_dir,
                    asr_task=meta.get("asr_task", "default"),
                    overwrite=overwrite,
                )
            else:
                ok, err = _run_v15_sample(backend, sample_dir, overwrite=overwrite)
            elapsed = time.monotonic() - t0
            if ok:
                stats["ok"] += 1
                per_task[label]["ok"] += 1
            else:
                stats["err"] += 1
                per_task[label]["err"] += 1
                tqdm.write(f"ASR ERROR [{label}] {sample_dir.name}: {err}")
            pbar.update(1)
            pbar.set_postfix({"last": f"{elapsed:.1f}s", "ok": stats["ok"], "err": stats["err"]})
        pbar.close()
        per_task[label]["elapsed"] = time.monotonic() - task_start

    wall = time.monotonic() - wall_start
    print("")
    print(f"ASR done in {_fmt_duration(wall)} "
          f"({stats['ok']}/{total_samples} ok, {stats['err']} err, {stats['skipped']} tasks skipped)")
    if per_task:
        print("Per-task breakdown:")
        for label in sorted(per_task):
            c = per_task[label]
            print(f"  {label:30s}  ok={c['ok']:4d}  err={c['err']:4d}  wall={_fmt_duration(c['elapsed'])}")
    sys.stdout.flush()
    return stats


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FDB v1/v1.5 ASR driver")
    parser.add_argument("--output_root", required=True, type=str)
    parser.add_argument("--version", default="both", choices=["1.0", "1.5", "both"])
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument("--asr_backend", default="parakeet", choices=["parakeet", "whisperx"])
    parser.add_argument("--device", default="cuda:0")
    # WhisperX-only knobs (ignored for parakeet)
    parser.add_argument("--asr_model", default="large-v3",
                        help="WhisperX model name (whisperx backend only)")
    parser.add_argument("--compute_type", default="float16",
                        help="faster-whisper compute type (whisperx backend only)")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="WhisperX batch size (whisperx backend only)")
    parser.add_argument("--language", default="en",
                        help="ASR language (whisperx backend only; Parakeet is English-only)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    setup_logging()
    args = parse_args(argv)
    output_root = Path(args.output_root).resolve()
    if not output_root.is_dir():
        raise SystemExit(f"output_root does not exist: {output_root}")

    # Both backends accept "cuda" or "cuda:N". For whisperx specifically we
    # used to remap to CUDA_VISIBLE_DEVICES + "cuda"; do the same for whisperx
    # only, since Parakeet/NeMo handles "cuda:N" directly via .to(device).
    asr_device = args.device
    if args.asr_backend == "whisperx" and asr_device.startswith("cuda:"):
        idx = asr_device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = idx
        asr_device = "cuda"

    backend = _Backend(
        name=args.asr_backend,
        device=asr_device,
        asr_model=args.asr_model,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        language=args.language,
    )

    stats = run_asr(
        output_root=output_root,
        version=args.version,
        tasks_filter=args.tasks,
        backend=backend,
        overwrite=args.overwrite,
    )
    logger.info("ASR done. %s", stats)


if __name__ == "__main__":
    main()
