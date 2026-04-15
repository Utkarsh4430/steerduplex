#!/usr/bin/env python3
"""
evaluate.py — unified post-recording evaluation for FDB v2 Revamped.

Automatically detects which stages are complete and runs only what's needed:

  Stage 1.5  Preparation  — write prompt.json per task, trim WAVs to target length
  Stage 2    ASR          — transcribe A.wav / B.wav → A.json, B.json, transcripts.json
  Stage 3    LLM Judge    — score each transcript via LiteLLM → {task_id}.json
  Stage 4    Parse        — normalize raw LLM output → {task_id}_processed.json
  Stage 5    Score        — aggregate + print task-specific score table, write scores.json

Usage:
  python evaluate.py
  python evaluate.py --config configs/gemini_live.yaml
  python evaluate.py --config configs/gemini_live.yaml --stages 3,4,5
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS = ["Daily", "Correction", "EntityTracking", "Safety"]


def load_config(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def normalize_litellm_base(base_url: str) -> str:
    """Return base URL with /v1 appended exactly once."""
    b = base_url.rstrip("/")
    if b.endswith("/v1"):
        b = b[:-3]
    return b + "/v1"


# ---------------------------------------------------------------------------
# Task discovery
# ---------------------------------------------------------------------------

def find_task_dirs(output_dir: Path, splits: List[str]) -> List[Dict[str, Any]]:
    """Return tasks that have both A.wav and B.wav recorded."""
    tasks = []
    for split in splits:
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for item_dir in sorted(split_dir.iterdir()):
            if not item_dir.is_dir():
                continue
            if (item_dir / "A.wav").exists() and (item_dir / "B.wav").exists():
                tasks.append({
                    "split": split,
                    "task_id": item_dir.name,
                    "task_dir": item_dir,
                })
    return tasks


# ---------------------------------------------------------------------------
# Stage detection
# ---------------------------------------------------------------------------

def needs_prep(task: Dict) -> bool:
    return not (task["task_dir"] / "prompt.json").exists()


def needs_asr(task: Dict) -> bool:
    d = task["task_dir"]
    return not (d / "A.json").exists() or not (d / "B.json").exists()


def needs_llm_eval(task: Dict, eval_dir: Path) -> bool:
    return not (eval_dir / task["split"] / f"{task['task_id']}.json").exists()


def needs_parse(task: Dict, eval_dir: Path) -> bool:
    return not (eval_dir / task["split"] / f"{task['task_id']}_processed.json").exists()


# ---------------------------------------------------------------------------
# Stage 1.5 — Preparation (write prompt.json, trim WAVs)
# ---------------------------------------------------------------------------

def load_all_prompts(prompts_file: Path) -> Dict[str, Dict]:
    """Index prompts_staged_200.json by task id."""
    data = json.loads(prompts_file.read_text())
    index: Dict[str, Dict] = {}

    def walk(obj: Any) -> None:
        if isinstance(obj, dict):
            if ("id" in obj and "examiner_system_prompt" in obj
                    and "examiner_task_prompt" in obj and "staged_reveal" in obj):
                index[obj["id"]] = obj
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)
    return index


def write_prompt_json(task: Dict, prompts_index: Dict) -> bool:
    item = prompts_index.get(task["task_id"])
    if not item:
        return False
    out = {
        "system_prompt": item.get("examiner_system_prompt"),
        "task_prompt": item.get("examiner_task_prompt"),
        "staged_reveal": item.get("staged_reveal"),
    }
    (task["task_dir"] / "prompt.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False)
    )
    return True


def trim_wav_inplace(path: Path, seconds: int) -> str:
    """Trim WAV to `seconds` in-place. Returns 'trimmed', 'skipped', or 'failed: ...'."""
    duration: Optional[float] = None
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True,
        )
        t = result.stdout.strip()
        if t:
            duration = float(t)
    except Exception:
        pass

    if duration is None:
        try:
            with closing(wave.open(str(path), "rb")) as w:
                duration = w.getnframes() / float(w.getframerate())
        except Exception:
            pass

    if duration is not None and duration <= seconds + 1e-6:
        return "skipped"

    fd, tmp = tempfile.mkstemp(prefix=".trim_", suffix=".wav", dir=str(path.parent))
    os.close(fd)
    try:
        if shutil.which("ffmpeg"):
            subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-i", str(path), "-t", str(seconds), "-c", "copy", tmp],
                check=True,
            )
        else:
            with closing(wave.open(str(path), "rb")) as in_w:
                params = in_w.getparams()
                target_frames = min(params.nframes, int(seconds * params.framerate))
                with closing(wave.open(tmp, "wb")) as out_w:
                    out_w.setparams(params)
                    remaining = target_frames
                    while remaining > 0:
                        chunk = in_w.readframes(min(16384, remaining))
                        if not chunk:
                            break
                        out_w.writeframes(chunk)
                        remaining -= len(chunk) // (params.sampwidth * params.nchannels)
        os.replace(tmp, str(path))
        return "trimmed"
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        return f"failed: {e}"


def run_prep(tasks: List[Dict], prompts_file: Path, trim_seconds: int) -> None:
    n = len(tasks)
    print(f"\n[Stage 1.5] Preparation — {n} task(s)")
    prompts_index = load_all_prompts(prompts_file)

    written = already = missing = 0
    wav_counts: Dict[str, int] = {"trimmed": 0, "skipped": 0, "failed": 0}

    for i, task in enumerate(tasks, 1):
        label = f"  [{i:>{len(str(n))}}/{n}] {task['split']}/{task['task_id']}"
        prompt_path = task["task_dir"] / "prompt.json"

        # --- prompt.json ---
        if prompt_path.exists():
            prompt_status = "prompt.json found, skipping"
            already += 1
        elif write_prompt_json(task, prompts_index):
            prompt_status = "prompt.json written"
            written += 1
        else:
            prompt_status = "WARNING: task ID not in prompts file"
            missing += 1

        # --- WAV trimming ---
        trim_parts: List[str] = []
        for name in ("A.wav", "B.wav", "combined.wav"):
            p = task["task_dir"] / name
            if not p.exists():
                continue
            status = trim_wav_inplace(p, trim_seconds)
            if status == "trimmed":
                wav_counts["trimmed"] += 1
                trim_parts.append(f"{name} trimmed")
            elif status == "skipped":
                wav_counts["skipped"] += 1
            else:
                wav_counts["failed"] += 1
                trim_parts.append(f"{name} FAILED")

        trim_str = " | " + ", ".join(trim_parts) if trim_parts else ""
        print(f"{label} | {prompt_status}{trim_str}")

    print(
        f"\n  Summary: {written} written, {already} already present, {missing} not found in prompts"
        f" | WAVs trimmed={wav_counts['trimmed']} skipped={wav_counts['skipped']} failed={wav_counts['failed']}"
    )


# ---------------------------------------------------------------------------
# Stage 2 — ASR (WhisperX)
# ---------------------------------------------------------------------------

def _segment_sentences(chunks: List[Dict], gap_threshold: float = 1.2) -> List[Dict]:
    if not chunks:
        return []
    sentences, current = [], []
    for i, chunk in enumerate(chunks):
        current.append(chunk)
        is_last = i == len(chunks) - 1
        gap = (0 if is_last
               else chunks[i + 1]["timestamp"][0] - chunk["timestamp"][1])
        if is_last or gap > gap_threshold:
            sentences.append(current)
            current = []
    return [
        {"text": " ".join(c["text"] for c in s),
         "start": s[0]["timestamp"][0],
         "end": s[-1]["timestamp"][1]}
        for s in sentences
    ]


def _combine_transcripts(a_json: Path, b_json: Path, max_time: float = 120.0) -> List[Dict]:
    a_data = json.loads(a_json.read_text())
    b_data = json.loads(b_json.read_text())
    a_chunks = [c for c in a_data["chunks"] if c["timestamp"][1] <= max_time]
    b_chunks = [c for c in b_data["chunks"] if c["timestamp"][1] <= max_time]
    a_sents = [{"speaker": "Examiner", **s} for s in _segment_sentences(a_chunks)]
    b_sents = [{"speaker": "Evaluatee", **s} for s in _segment_sentences(b_chunks)]
    merged = sorted(a_sents + b_sents, key=lambda x: x["start"])
    return [{"speaker": s["speaker"], "text": s["text"]} for s in merged]


def _transcribe_pair(task: Dict, model: Any, align_model: Any, align_metadata: Any,
                     lang: str, batch_size: int, device: str) -> None:
    import whisperx  # imported here so the rest of the script works without it

    task_dir = task["task_dir"]
    for wav_name, json_name in [("A.wav", "A.json"), ("B.wav", "B.json")]:
        json_path = task_dir / json_name
        if json_path.exists():
            continue
        audio = whisperx.load_audio(str(task_dir / wav_name))
        result = model.transcribe(audio, batch_size=batch_size, language=lang)
        result = whisperx.align(
            result["segments"], align_model, align_metadata, audio, device,
            return_char_alignments=False,
        )
        chunks, words = [], []
        for seg in result["segments"]:
            for word in seg.get("words", []):
                s, e = word.get("start"), word.get("end")
                if s is None or e is None:
                    continue
                words.append(word["word"])
                chunks.append({"text": word["word"], "timestamp": [s, e]})
        json_path.write_text(json.dumps({"text": " ".join(words), "chunks": chunks}, indent=2))

    transcripts_path = task_dir / "transcripts.json"
    if not transcripts_path.exists():
        combined = _combine_transcripts(task_dir / "A.json", task_dir / "B.json")
        transcripts_path.write_text(json.dumps(combined, indent=2))


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as H:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def run_asr(tasks: List[Dict], whisper_model: str = "large-v3",
            lang: str = "en", batch_size: int = 16) -> None:
    n = len(tasks)
    print(f"\n[Stage 2] ASR transcription — {n} task(s), model={whisper_model}")
    try:
        import whisperx
    except ImportError:
        print("  ERROR: whisperx not installed. Run: pip install whisperx")
        sys.exit(1)

    device = "cuda"
    print(f"  Loading WhisperX '{whisper_model}' ...", flush=True)
    model = whisperx.load_model(whisper_model, device, compute_type="float16", language=lang)
    align_model, align_metadata = whisperx.load_align_model(language_code=lang, device=device)
    print("  Models loaded. Starting transcription ...\n")

    ok = failed = 0
    width = len(str(n))
    task_start = time.time()
    task_times: List[float] = []

    for i, task in enumerate(tasks, 1):
        label = f"  [{i:>{width}}/{n}] {task['split']}/{task['task_id']}"
        print(f"{label} ...", end="", flush=True)
        t0 = time.time()
        error_msg: Optional[str] = None
        try:
            _transcribe_pair(task, model, align_model, align_metadata, lang, batch_size, device)
            ok += 1
        except Exception as e:
            error_msg = str(e)
            failed += 1

        elapsed_task = time.time() - t0
        task_times.append(elapsed_task)
        elapsed_total = time.time() - task_start
        avg = sum(task_times) / len(task_times)
        eta = avg * (n - i)
        fail_str = f"  [{failed} failed]" if failed else ""

        if error_msg:
            print(
                f" ✗  ({elapsed_task:.1f}s)"
                f"  elapsed {_fmt_duration(elapsed_total)}"
                f"  ETA ~{_fmt_duration(eta)}"
                f"{fail_str}"
                f"\n     ERROR: {error_msg}"
            )
        else:
            print(
                f" ✓  ({elapsed_task:.1f}s)"
                f"  elapsed {_fmt_duration(elapsed_total)}"
                f"  ETA ~{_fmt_duration(eta)}"
                f"{fail_str}"
            )

    total_elapsed = time.time() - task_start
    print(f"\n  Done in {_fmt_duration(total_elapsed)}: {ok} ok, {failed} failed")


# ---------------------------------------------------------------------------
# Stage 3 — LLM Judge (via LiteLLM proxy, OpenAI-compatible)
# ---------------------------------------------------------------------------

def _load_eval_prompts(path: Path) -> Dict:
    data = json.loads(path.read_text())
    for key in ("system_prompt", "turn_taking_fluency_prompt",
                "multi_turn_instruction_following_prompt",
                "task_specific_prompts", "output_format_prompt"):
        if key not in data:
            raise KeyError(f"Missing '{key}' in {path}")
    return data


def _normalize_subset(split: str) -> str:
    s = split.strip().lower().replace(" ", "").replace("_", "")
    if s == "daily":
        return "daily"
    if s in ("correction", "corrections"):
        return "correction"
    if s == "entitytracking":
        return "entity_tracking"
    if s == "safety":
        return "safety"
    raise ValueError(f"Unknown split: {split}")


def _task_specific_prompt(eval_prompts: Dict, subset_norm: str) -> Optional[str]:
    if subset_norm == "daily":
        return None
    for item in eval_prompts.get("task_specific_prompts", []):
        if item.get("id") == subset_norm:
            return item.get("prompt", "").strip() or None
    raise KeyError(f"No task_specific_prompts entry for id='{subset_norm}'")


def _load_staged_reveal(task_dir: Path, task_id: str) -> Dict[str, str]:
    prompt_path = task_dir / "prompt.json"
    if prompt_path.exists():
        data = json.loads(prompt_path.read_text())
        sr = data.get("staged_reveal") or {}
        if all(k in sr for k in ("T1", "T2", "T3", "T4")):
            return {k: sr[k] for k in ("T1", "T2", "T3", "T4")}
    raise FileNotFoundError(
        f"prompt.json missing or incomplete for {task_id}. Run Stage 1.5 first."
    )


def _format_chunks(chunks: List[Dict]) -> str:
    lines = []
    for ch in chunks:
        ts, text = ch.get("timestamp"), ch.get("text", "")
        if isinstance(ts, list) and len(ts) == 2:
            lines.append(f"[{ts[0]:.2f}, {ts[1]:.2f}]: {text}")
        elif text:
            lines.append(text)
    return "\n".join(lines)


def _build_judge_prompt(eval_prompts: Dict, subset_norm: str,
                        task_specific: Optional[str], staged_reveal: Dict,
                        a_data: Dict, b_data: Dict) -> str:
    sections = [
        "SYSTEM INSTRUCTION:\n" + eval_prompts["system_prompt"].strip(),
        "TURN-TAKING FLUENCY RUBRIC:\n" + eval_prompts["turn_taking_fluency_prompt"].strip(),
        "MULTI-TURN INSTRUCTION-FOLLOWING RUBRIC:\n"
        + eval_prompts["multi_turn_instruction_following_prompt"].strip(),
    ]
    if task_specific:
        sections.append("TASK-SPECIFIC RUBRIC:\n" + task_specific)
    sections.append("OUTPUT FORMAT:\n" + eval_prompts["output_format_prompt"].strip())
    sections.append(
        "Dataset staged goals (T1–T4):\n"
        f"T1: {staged_reveal.get('T1', '')}\n"
        f"T2: {staged_reveal.get('T2', '')}\n"
        f"T3: {staged_reveal.get('T3', '')}\n"
        f"T4: {staged_reveal.get('T4', '')}"
    )
    sections.append(
        "Transcript summary:\n"
        f"Channel A (Examiner) summary:\n{a_data.get('text', '')}\n\n"
        f"Channel B (Examinee) summary:\n{b_data.get('text', '')}\n\n"
        f"Channel B (Examinee) segments with timestamps:\n"
        f"{_format_chunks(b_data.get('chunks', []))}"
    )
    sections.append(
        "Constraints:\n"
        "- Keep each evaluation output about ~100 words.\n"
        "- Follow the JSON shape exactly as in OUTPUT FORMAT.\n"
        "- Use time ranges from the provided Channel B timestamps.\n"
    )
    return "\n\n".join(s for s in sections if s)


def _call_litellm(base_url: str, api_key: str, model: str, prompt: str,
                  max_tokens: int = 20000, temperature: float = 0.2,
                  max_retries: int = 5) -> str:
    import requests

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
        except Exception as e:
            if attempt >= max_retries:
                raise RuntimeError(f"HTTP request failed after {attempt} attempts: {e}")
            time.sleep(min(60, 2 ** attempt + random.random()))
            continue

        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]

        if resp.status_code in (429, 500, 502, 503, 504):
            if attempt >= max_retries:
                raise RuntimeError(
                    f"LiteLLM {resp.status_code} after {attempt} attempts: {resp.text[:300]}"
                )
            retry_after = resp.headers.get("Retry-After")
            try:
                sleep_s: float = float(retry_after) if retry_after else 0.0
            except ValueError:
                sleep_s = 0.0
            time.sleep(sleep_s or min(60, 2 ** attempt + random.random()))
            continue

        raise RuntimeError(f"LiteLLM returned {resp.status_code}: {resp.text[:300]}")

    raise RuntimeError("Exceeded max retries")


def _extract_json(text: str) -> Optional[Dict]:
    """Strip markdown fences and extract the first balanced JSON object."""
    t = text.strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl != -1:
            t = t[nl + 1:]
        if t.endswith("```"):
            t = t[:-3]
    t = t.strip()
    start = t.find("{")
    if start == -1:
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start: i + 1])
                    except Exception:
                        return None
    return None


def _eval_single_task(task: Dict, eval_prompts: Dict, litellm_base: str,
                      litellm_key: str, judge_model: str,
                      eval_dir: Path) -> Optional[str]:
    """Run LLM judge for one task. Returns error string or None on success."""
    split, task_id, task_dir = task["split"], task["task_id"], task["task_dir"]
    out_path = eval_dir / split / f"{task_id}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subset_norm = _normalize_subset(split)
        task_specific = _task_specific_prompt(eval_prompts, subset_norm)
        staged_reveal = _load_staged_reveal(task_dir, task_id)
        a_data = json.loads((task_dir / "A.json").read_text())
        b_data = json.loads((task_dir / "B.json").read_text())
        prompt = _build_judge_prompt(
            eval_prompts, subset_norm, task_specific, staged_reveal, a_data, b_data
        )
        response_text = _call_litellm(litellm_base, litellm_key, judge_model, prompt)
        parsed = _extract_json(response_text)
        if parsed is not None:
            out_path.write_text(json.dumps(parsed, indent=2, ensure_ascii=False))
        else:
            out_path.write_text(response_text)
        return None
    except Exception as e:
        return str(e)


def run_llm_eval(tasks: List[Dict], eval_prompts: Dict, litellm_base: str,
                 litellm_key: str, judge_model: str, eval_dir: Path,
                 max_workers: int = 10) -> None:
    n = len(tasks)
    width = len(str(n))
    print(f"\n[Stage 3] LLM eval — {n} task(s) | model={judge_model} | workers={max_workers}")
    done = ok = failed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _eval_single_task, task, eval_prompts, litellm_base,
                litellm_key, judge_model, eval_dir
            ): task
            for task in tasks
        }
        for fut in as_completed(futures):
            task = futures[fut]
            err = fut.result()
            done += 1
            label = f"  [{done:>{width}}/{n}] {task['split']}/{task['task_id']}"
            fail_str = f"  [{failed} failed]" if failed else ""
            if err:
                failed += 1
                fail_str = f"  [{failed} failed]"
                print(f"{label} ✗{fail_str}  — {err}")
            else:
                ok += 1
                print(f"{label} ✓{fail_str}")
    print(f"\n  Done: {ok} ok, {failed} failed")


# ---------------------------------------------------------------------------
# Stage 4 — Parse / normalize  (ported from fdb_v2/scoring/parse.py)
# ---------------------------------------------------------------------------

_NUM_PAT = r"[+-]?(?:\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
_RE_SE = re.compile(
    rf"""\{{\s*['"](?:start_time|start|startTime)['"]\s*:\s*({_NUM_PAT}|"[^"]+")\s*,\s*
         ['"](?:end_time|end|endTime)['"]\s*:\s*({_NUM_PAT}|"[^"]+")\s*\}}""",
    re.VERBOSE,
)
_RE_ES = re.compile(
    rf"""\{{\s*['"](?:end_time|end|endTime)['"]\s*:\s*({_NUM_PAT}|"[^"]+")\s*,\s*
         ['"](?:start_time|start|startTime)['"]\s*:\s*({_NUM_PAT}|"[^"]+")\s*\}}""",
    re.VERBOSE,
)
_RE_BRACE = re.compile(r"\{([^{}]+?)\}", re.DOTALL)
_RE_FENCED = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def _parse_time_tok(tok: Any) -> float:
    if isinstance(tok, (int, float)):
        return float(tok)
    s = str(tok).strip().strip('"').strip("'")
    if ":" in s:
        total, mul = 0.0, 1.0
        for p in reversed(s.split(":")):
            total += float(p) * mul
            mul *= 60.0
        return total
    return float(s)


def _coerce_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(float(str(x).strip().strip('"').strip("'")))
    except Exception:
        return None


def _norm_intervals(text: str) -> str:
    text = _RE_SE.sub(lambda m: f"[{m.group(1)}, {m.group(2)}]", text)
    text = _RE_ES.sub(lambda m: f"[{m.group(2)}, {m.group(1)}]", text)

    def _brace(m: re.Match) -> str:
        parts = [p.strip() for p in m.group(1).split(",")]
        if len(parts) == 2:
            try:
                float(parts[0].strip().strip('"').strip("'"))
                float(parts[1].strip().strip('"').strip("'"))
                return f"[{parts[0]}, {parts[1]}]"
            except Exception:
                pass
        return m.group(0)

    return _RE_BRACE.sub(_brace, text)


def _find_matching_bracket(text: str, start: int, open_c: str, close_c: str) -> int:
    depth, i, n, in_str = 0, start, len(text), False
    while i < n:
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == open_c:
                depth += 1
            elif ch == close_c:
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return -1


def _flatten_tt(data: Dict) -> Dict:
    key = "Turn-taking event and score"
    items = data.get(key, [])
    if not isinstance(items, list):
        data[key] = []
        return data
    new_items: List = []
    for item in items:
        if not isinstance(item, list) or len(item) < 3:
            continue
        s2_raw, s1_raw, intervals_part = item[-1], item[-2], item[:-2]

        def _collect(o: Any) -> List:
            if (isinstance(o, list) and len(o) == 2
                    and all(isinstance(x, (int, float, str)) for x in o)):
                try:
                    return [[_parse_time_tok(o[0]), _parse_time_tok(o[1])]]
                except Exception:
                    return []
            out: List = []
            if isinstance(o, list):
                for e in o:
                    out.extend(_collect(e))
            return out

        for iv in _collect(intervals_part):
            new_items.append([iv, _coerce_int(s1_raw), _coerce_int(s2_raw)])
    data[key] = new_items
    return data


def _try_json_block(block: str) -> Optional[Dict]:
    txt = _norm_intervals(block)
    try:
        d = json.loads(txt)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    if "Turn-taking event and score" not in d and "Task-specific score" not in d:
        return None
    return _flatten_tt(d)


def _skip_wc(text: str, i: int) -> int:
    while i < len(text) and text[i] in " \t\r\n,":
        i += 1
    return i


def _parse_score_tok(text: str, i: int) -> Tuple[Optional[str], int]:
    n = len(text)
    i = _skip_wc(text, i)
    if i >= n:
        return None, i
    if text[i] == '"':
        j = i + 1
        while j < n and text[j] != '"':
            if text[j] == "\\":
                j += 1
            j += 1
        if j < n:
            return text[i + 1:j], j + 1
        return None, i
    j = i
    while j < n and text[j] not in ",[]{}:\n\r\t ":
        j += 1
    tok = text[i:j].strip()
    return tok if tok else None, j


def _coerce_score(tok: Optional[str]) -> Optional[int]:
    if not tok:
        return None
    t = tok.strip().strip('"').strip("'")
    if t.lower() == "null":
        return None
    try:
        return int(float(t))
    except Exception:
        return None


def _parse_turn_array(arr_text: str) -> List:
    i, n, out = 0, len(arr_text), []
    while True:
        i = _skip_wc(arr_text, i)
        if i >= n:
            break
        if arr_text[i] != "[":
            nx = arr_text.find("[", i)
            if nx == -1:
                break
            i = nx
        rb = _find_matching_bracket(arr_text, i, "[", "]")
        if rb == -1:
            break
        interval_expr = arr_text[i:rb + 1]
        i = rb + 1
        j = _skip_wc(arr_text, i)
        if j >= n or arr_text[j] != ":":
            i = j
            continue
        j += 1
        s1_tok, j = _parse_score_tok(arr_text, j)
        s2_tok, j = _parse_score_tok(arr_text, j)

        def _collect_iv(o: Any) -> List:
            if (isinstance(o, list) and len(o) == 2
                    and all(isinstance(x, (int, float, str)) for x in o)):
                try:
                    return [[_parse_time_tok(o[0]), _parse_time_tok(o[1])]]
                except Exception:
                    return []
            res: List = []
            if isinstance(o, list):
                for e in o:
                    res.extend(_collect_iv(e))
            return res

        try:
            expr = _norm_intervals(interval_expr)
            ivs_obj = json.loads(re.sub(r",\s*]", "]", expr))
            for iv in _collect_iv(ivs_obj):
                out.append([iv, _coerce_score(s1_tok), _coerce_score(s2_tok)])
        except Exception:
            pass
        i = j
    return out


def parse_raw_eval(text: str) -> Dict:
    blocks = [m.group(1) for m in _RE_FENCED.finditer(text)
              if "Turn-taking event and score" in m.group(1)] or [text]

    merged_events: List = []
    task_scores: List = []
    for block in blocks:
        parsed = _try_json_block(block)
        if parsed is not None:
            merged_events.extend(parsed.get("Turn-taking event and score", []))
            ts = parsed.get("Task-specific score")
            if ts is not None:
                task_scores.append(ts)
            continue
        # fallback regex scan
        for key_pat in ('"Turn-taking event and score"', "'Turn-taking event and score'"):
            k = block.find(key_pat)
            if k == -1:
                continue
            rest = block[k + len(key_pat):]
            lb_rel = rest.find("[")
            if lb_rel == -1:
                continue
            lb = k + len(key_pat) + lb_rel
            rb = _find_matching_bracket(block, lb, "[", "]")
            if rb == -1:
                continue
            merged_events.extend(_parse_turn_array(block[lb + 1:rb]))
            m = re.search(r'"Task-specific score"\s*:\s*("[^"]+"|[^\s,}\]]+)', block[rb:])
            if m:
                try:
                    task_scores.append(int(float(m.group(1).strip().strip('"'))))
                except Exception:
                    pass
            break

    out: Dict = {"Turn-taking event and score": merged_events}
    if task_scores:
        out["Task-specific score"] = task_scores[-1]
    return out


def run_parse(tasks: List[Dict], eval_dir: Path) -> None:
    n = len(tasks)
    width = len(str(n))
    print(f"\n[Stage 4] Parsing/normalizing — {n} task(s)")
    ok = failed = skipped = 0
    for i, task in enumerate(tasks, 1):
        split, task_id = task["split"], task["task_id"]
        raw_path = eval_dir / split / f"{task_id}.json"
        out_path = eval_dir / split / f"{task_id}_processed.json"
        label = f"  [{i:>{width}}/{n}] {split}/{task_id}"

        if not raw_path.exists():
            skipped += 1
            print(f"{label} — skipped (no raw eval JSON)")
            continue

        fail_str = f"  [{failed} failed]" if failed else ""
        try:
            data = parse_raw_eval(raw_path.read_text(encoding="utf-8"))
            out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            ok += 1
            print(f"{label} ✓{fail_str}")
        except Exception as e:
            failed += 1
            fail_str = f"  [{failed} failed]"
            print(f"{label} ✗{fail_str}  — {e}")

    print(f"\n  Done: {ok} ok, {failed} failed, {skipped} skipped")


# ---------------------------------------------------------------------------
# Stage 5 — Score aggregation + print
# ---------------------------------------------------------------------------

TASK_SPLITS = {
    "Correction": "Correction",
    "EntityTracking": "Entity",
    "Safety": "Safety",
}


def run_score(eval_dir: Path) -> None:
    print(f"\n[Stage 5] Scoring")
    stats: Dict[str, List] = defaultdict(lambda: [0.0, 0])
    per_task: Dict[str, float] = {}

    for split, col in TASK_SPLITS.items():
        split_dir = eval_dir / split
        if not split_dir.exists():
            continue
        for json_file in sorted(split_dir.glob("*_processed.json")):
            try:
                data = json.loads(json_file.read_text())
                score = data.get("Task-specific score")
                if score is not None:
                    stats[col][0] += float(score)
                    stats[col][1] += 1
                    task_key = json_file.stem.replace("_processed", "")
                    per_task[f"{split}/{task_key}"] = float(score)
            except Exception:
                continue

    # Print table
    print(f"\n{'System':<15} | {'Correction':<10} | {'Entity':<10} | {'Safety':<10}")
    print("-" * 55)
    row = ["Result"]
    for col in ("Correction", "Entity", "Safety"):
        total, count = stats[col]
        row.append(f"{total / count:.2f}" if count > 0 else "N/A")
    print(f"{row[0]:<15} | {row[1]:<10} | {row[2]:<10} | {row[3]:<10}")
    print("-" * 55)

    # Write scores JSON
    summary = {
        "averages": {
            col: (stats[col][0] / stats[col][1] if stats[col][1] > 0 else None)
            for col in ("Correction", "Entity", "Safety")
        },
        "counts": {col: stats[col][1] for col in ("Correction", "Entity", "Safety")},
        "per_task": per_task,
    }
    scores_path = eval_dir / "scores.json"
    scores_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Scores written → {scores_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="FDB v2 Revamped — evaluate recorded conversations end-to-end"
    )
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--stages", default=None,
                        help="Comma-separated stages to force-run regardless of existing outputs "
                             "(e.g. '3,4,5'). Default: auto-detect from missing artifacts.")
    args = parser.parse_args(argv)

    load_dotenv(SCRIPT_DIR / ".env")
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    eval_dir = output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    splits: List[str] = cfg.get("splits", SPLITS)
    eval_cfg: Dict = cfg.get("eval", {})
    judge_model: str  = eval_cfg.get("judge_model", "gemini/gemini-2.5-pro")
    max_workers: int  = int(eval_cfg.get("max_workers", 10))
    trim_seconds: int = int(eval_cfg.get("trim_seconds", 120))
    whisper_model: str = eval_cfg.get("whisper_model", "large-v3")

    litellm_base = normalize_litellm_base(cfg["litellm"]["base_url"])
    litellm_key = os.environ.get("LITELLM_API_KEY", "")
    if not litellm_key:
        print("ERROR: LITELLM_API_KEY is not set. Add it to .env or export it.", file=sys.stderr)
        return 1

    prompts_file = SCRIPT_DIR / "prompts_staged_200.json"
    eval_prompts = _load_eval_prompts(SCRIPT_DIR / "eval" / "eval_prompts.json")

    force: set = set(args.stages.split(",")) if args.stages else set()

    tasks = find_task_dirs(output_dir, splits)
    if not tasks:
        print(f"No recorded tasks found in {output_dir}. Run run_sessions.py first.")
        return 1
    print(f"Found {len(tasks)} recorded tasks in {output_dir}")
    print(f"Eval results → {eval_dir}")

    # Stage 1.5 — Prep (always runs per-task; skips internally if already done)
    run_prep(tasks if "1.5" in force else tasks, prompts_file, trim_seconds)

    # Stage 2 — ASR
    asr_tasks = tasks if "2" in force else [t for t in tasks if needs_asr(t)]
    if asr_tasks:
        run_asr(asr_tasks, whisper_model=whisper_model)
    else:
        print("\n[Stage 2] ASR — all transcripts present, skipping")

    # Stage 3 — LLM eval
    eval_tasks = tasks if "3" in force else [t for t in tasks if needs_llm_eval(t, eval_dir)]
    if eval_tasks:
        run_llm_eval(eval_tasks, eval_prompts, litellm_base, litellm_key,
                     judge_model, eval_dir, max_workers=max_workers)
    else:
        print("\n[Stage 3] LLM eval — all scores present, skipping")

    # Stage 4 — Parse
    parse_tasks = tasks if "4" in force else [t for t in tasks if needs_parse(t, eval_dir)]
    if parse_tasks:
        run_parse(parse_tasks, eval_dir)
    else:
        print("\n[Stage 4] Parse — all processed JSONs present, skipping")

    # Stage 5 — always run to print latest numbers
    run_score(eval_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
