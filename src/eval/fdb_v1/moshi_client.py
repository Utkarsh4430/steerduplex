"""Moshi WebSocket file client.

Ports `Full-Duplex-Bench/v1_v1.5/model_inference/moshi/inference.py` (the
upstream reference implementation used to produce the published Moshi
baseline numbers) into a reusable module that talks to a Moshi server
spawned by `launch_moshi_servers.py`.

Wire protocol (matching `moshi_server.py`):
  - Each client message is `b"\\x01" + opus_packet_bytes` (audio frame).
  - The server emits messages of the form `b"\\x01" + opus_pcm_bytes`
    (audio response) and `b"\\x02" + utf8_text` (token text, ignored
    here for offline file inference).
  - Sample rate: 24_000 Hz; frame size: 1920 samples (= 80 ms).

Parity invariants vs. the official client:
  - SKIP_FRAMES=1 → the first 1920 output samples are zero-padded so that
    the model's first emitted frame lands at t = FRAME_SEC (~80 ms) in
    the output, matching what the published numbers expect.
  - Output is hard-clipped to `len(input_audio_at_24kHz)` samples so
    timing-derived metrics (latency, JSD denominators, post-split
    windows) align with the input timeline exactly.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import List

import numpy as np
import soundfile as sf
import sphn
import torch
import torchaudio.functional as AF
import websockets
import websockets.exceptions as wsex


SEND_SR = 24_000
FRAME_SMP = 1_920
SKIP_FRAMES = 1
FRAME_SEC = FRAME_SMP / SEND_SR


# Modern sphn (≥ 0.1.x) API — verified in raman_steerduplex:
#   OpusStreamWriter.append_pcm(pcm: np.ndarray[float32])  -> bytes   (encoded opus frame)
#   OpusStreamReader.append_bytes(data: bytes)             -> np.ndarray[float32]  (decoded PCM)
# The upstream Full-Duplex-Bench reference was written against an older two-
# phase API (append_* then separate read_*), which no longer exists. Do NOT
# reintroduce that two-phase pattern or any _patch_sphn() fallback: if the
# drain method is missing the fallback returns empty, and every decoded
# frame gets discarded — output.wav ends up all zeros and every metric
# collapses (TOR=0 everywhere, ZeroDivisionError in eval_smooth_turn_taking).


def _mono(x: np.ndarray) -> np.ndarray:
    return x if x.ndim == 1 else x.mean(axis=1)


def _resample(x: np.ndarray, sr: int, tgt: int) -> np.ndarray:
    if sr == tgt:
        return x
    y = torch.from_numpy(x.astype(np.float32) / 32768).unsqueeze(0)
    y = AF.resample(y, sr, tgt)[0].numpy()
    return (y * 32768).astype(np.int16)


def _chunk(sig: np.ndarray) -> List[np.ndarray]:
    pad = (-len(sig)) % FRAME_SMP
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, sig.dtype)])
    return [sig[i: i + FRAME_SMP] for i in range(0, len(sig), FRAME_SMP)]


class MoshiFileClient:
    """One-shot file ↔ websocket inference for a single Moshi server."""

    def __init__(self, ws_url: str, inp: Path, out: Path):
        self.url = ws_url
        self.inp = inp
        self.out = out

        # Read natively as float32 — some FDB v1 WAVs are FLOAT subtype
        # (e.g. candor_turn_taking/*/input.wav). sf.read(dtype="int16") does
        # NOT rescale floats in [-1, 1] to int16 range; it truncates to the
        # int16 cast, yielding near-silent (peak=1) input. Read float and
        # scale explicitly so both PCM_16 and FLOAT sources work.
        wav_f, sr = sf.read(str(inp), dtype="float32", always_2d=False)
        wav_f = _mono(wav_f)
        sig16 = np.clip(wav_f * 32768.0, -32768.0, 32767.0).astype(np.int16)
        self.sig24 = _resample(sig16, sr, SEND_SR)
        self.max_samples = len(self.sig24)  # output target length

        self.writer = sphn.OpusStreamWriter(SEND_SR)
        self.reader = sphn.OpusStreamReader(SEND_SR)

    # ---------------- sender ----------------
    async def _send(self, ws):
        # append_pcm returns the encoded opus bytes directly in modern sphn.
        # No separate drain call is needed.
        for frame in _chunk(self.sig24):
            enc = self.writer.append_pcm(frame.astype(np.float32) / 32768.0)
            if enc:
                await ws.send(b"\x01" + enc)
            await asyncio.sleep(FRAME_SEC)
        # Brief drain window before closing — matches upstream timing.
        await asyncio.sleep(0.5)
        await ws.close()

    # ---------------- receiver ----------------
    async def _recv(self, ws):
        samples_written = 0
        first_pcm_seen = False

        # PCM_16 mono at 24 kHz, atomic via .tmp + rename in caller.
        # `format="WAV"` is explicit because the caller writes to *.wav.tmp,
        # which soundfile would otherwise fail to auto-detect.
        with sf.SoundFile(
            str(self.out), "w", samplerate=SEND_SR, channels=1,
            subtype="PCM_16", format="WAV",
        ) as fout:
            try:
                async for msg in ws:
                    if not msg or msg[0] not in (1, 2):
                        continue
                    kind, payload = msg[0], msg[1:]

                    if kind == 1:  # audio bytes
                        # Modern sphn: append_bytes returns the decoded
                        # float32 PCM ndarray directly.
                        pcm = self.reader.append_bytes(payload)
                        if pcm is None or pcm.size == 0:
                            continue

                        if not first_pcm_seen:
                            # Zero-pad SKIP_FRAMES * FRAME_SMP up front so
                            # the model's first emitted frame lands at
                            # t = FRAME_SEC. Matches the published baseline.
                            pad = min(SKIP_FRAMES * FRAME_SMP, self.max_samples)
                            fout.write(np.zeros(pad, dtype=np.int16))
                            samples_written += pad
                            first_pcm_seen = True

                        remain = self.max_samples - samples_written
                        if remain <= 0:
                            continue
                        n_write = min(pcm.size, remain)
                        fout.write(
                            np.clip(pcm[:n_write] * 32768.0, -32768.0, 32767.0).astype(np.int16)
                        )
                        samples_written += n_write

                    # kind == 2: text token. Ignored for offline file inference.
            except wsex.ConnectionClosedError:
                # Server-side close mid-stream is normal once max_samples is
                # exhausted; the file we already wrote is what we want.
                pass

            # Pad output to exactly max_samples if the model finished early.
            if samples_written < self.max_samples:
                fout.write(np.zeros(self.max_samples - samples_written, dtype=np.int16))

    # ---------------- entrypoint ----------------
    async def run(self) -> None:
        async with websockets.connect(self.url, max_size=None) as ws:
            # Server greets with a 1-byte handshake (b"\x00") that we drain.
            try:
                first = await asyncio.wait_for(ws.recv(), timeout=30.0)
                if isinstance(first, (bytes, bytearray)) and first and first[0] == 0:
                    pass
                # Anything else: log via exception path is fine; the server
                # may still accept audio.
            except Exception:
                pass

            await asyncio.gather(self._send(ws), self._recv(ws))


async def transcribe_one(ws_url: str, inp: Path, out: Path) -> None:
    """Convenience wrapper for `await MoshiFileClient(...).run()`."""
    await MoshiFileClient(ws_url, inp, out).run()
