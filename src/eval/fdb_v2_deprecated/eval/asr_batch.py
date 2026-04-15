"""
FDB v2 ASR evaluation script using WhisperX.

Drop-in replacement for dep_asr_batch.py (NeMo/Parakeet) with identical
input/output interface but using WhisperX for transcription.
"""
import os
import json
import argparse
from pathlib import Path

import whisperx
from tqdm import tqdm


def transcribe_audio_file(audio_path, model, align_model, align_metadata, lang, batch_size, device):
    """
    Transcribe a single audio file and return the result dict.
    Returns a dict with 'text' and 'chunks' keys.
    """
    print(f"Processing: {audio_path}")

    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=batch_size, language=lang)
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    chunks = []
    text = ""
    for segment in result["segments"]:
        if "words" not in segment:
            continue
        for word in segment["words"]:
            start = word.get("start")
            end = word.get("end")
            if start is None or end is None:
                continue
            text += word["word"] + " "
            chunks.append({
                "text": word["word"],
                "timestamp": [start, end],
            })

    return {
        "text": text.strip(),
        "chunks": chunks,
    }


def find_audio_pairs(root_dir):
    """
    Find all A.wav and B.wav pairs in the directory hierarchy.
    Returns a list of dicts: key, sample_path, a_wav, b_wav, task, setup, model, sample.

    Handles simplified structure: root/Task/Task.sample_id/A.wav
    """
    audio_pairs = []

    task_folders = ["Daily", "Correction", "EntityTracking", "Safety"]

    for task in task_folders:
        task_path = os.path.join(root_dir, task)
        if not os.path.exists(task_path):
            task_lower = task.lower()
            task_path = os.path.join(root_dir, task_lower)
            if not os.path.exists(task_path):
                continue
            task = task_lower

        for sample_folder in os.listdir(task_path):
            sample_path = os.path.join(task_path, sample_folder)
            if not os.path.isdir(sample_path):
                continue

            a_wav = find_file_recursive(sample_path, "A.wav")
            b_wav = find_file_recursive(sample_path, "B.wav")

            if a_wav and b_wav:
                key = f"{task}.{sample_folder}"
                audio_pairs.append({
                    "key": key,
                    "sample_path": sample_path,
                    "a_wav": a_wav,
                    "b_wav": b_wav,
                    "task": task,
                    "setup": "default",
                    "model": "default",
                    "sample": sample_folder,
                })

    return audio_pairs


def find_file_recursive(directory, filename):
    """
    Recursively find a file in a directory.
    Returns the full path if found, None otherwise.
    """
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None


def segment_into_sentences(chunks, time_threshold=1.2):
    """
    Segment word chunks into sentences based on time gaps.

    Args:
        chunks: List of dicts with 'text' and 'timestamp' keys
        time_threshold: Time gap (seconds) that indicates sentence boundary

    Returns:
        List of sentence dicts with 'text', 'start', 'end' keys
    """
    if not chunks:
        return []

    sentences = []
    current_sentence = []

    for i, chunk in enumerate(chunks):
        current_sentence.append(chunk)

        if i == len(chunks) - 1:
            sentences.append(current_sentence)
        else:
            current_end = chunk["timestamp"][1]
            next_start = chunks[i + 1]["timestamp"][0]
            gap = next_start - current_end

            if gap > time_threshold:
                sentences.append(current_sentence)
                current_sentence = []

    result = []
    for sentence_chunks in sentences:
        text = " ".join([c["text"] for c in sentence_chunks])
        start_time = sentence_chunks[0]["timestamp"][0]
        end_time = sentence_chunks[-1]["timestamp"][1]
        result.append({
            "text": text,
            "start": start_time,
            "end": end_time,
        })

    return result


def combine_transcripts(a_json_path, b_json_path, max_time=60.0, time_threshold=1.2):
    """
    Combine A.json (Examiner) and B.json (Evaluatee) into conversation format.

    Args:
        a_json_path: Path to A.json file
        b_json_path: Path to B.json file
        max_time: Maximum timestamp to include (seconds)
        time_threshold: Time gap for sentence segmentation (seconds)

    Returns:
        List of conversation turns with speaker and text
    """
    with open(a_json_path, 'r') as f:
        a_data = json.load(f)
    with open(b_json_path, 'r') as f:
        b_data = json.load(f)

    a_chunks = [c for c in a_data["chunks"] if c["timestamp"][1] <= max_time]
    b_chunks = [c for c in b_data["chunks"] if c["timestamp"][1] <= max_time]

    a_sentences = segment_into_sentences(a_chunks, time_threshold)
    b_sentences = segment_into_sentences(b_chunks, time_threshold)

    a_tagged = [{"speaker": "Examiner", "text": s["text"], "start": s["start"]}
                for s in a_sentences]
    b_tagged = [{"speaker": "Evaluatee", "text": s["text"], "start": s["start"]}
                for s in b_sentences]

    all_sentences = a_tagged + b_tagged
    all_sentences.sort(key=lambda x: x["start"])

    return [{"speaker": s["speaker"], "text": s["text"]} for s in all_sentences]


def main():
    parser = argparse.ArgumentParser(
        description="Batch transcribe audio files using WhisperX (FDB v2 eval)"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing task folders (Daily, Correction, EntityTracking, Safety)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="transcripts.json",
        help="Output JSON file path (default: transcripts.json)",
    )
    parser.add_argument(
        "--max_time",
        type=float,
        default=60.0,
        help="Maximum timestamp to include in transcripts (default: 60.0 seconds)",
    )
    parser.add_argument(
        "--time_threshold",
        type=float,
        default=1.2,
        help="Time gap threshold for sentence segmentation (default: 1.2 seconds)",
    )
    parser.add_argument(
        "--skip_asr",
        action="store_true",
        help="Skip ASR inference and only combine existing A.json/B.json files",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for WhisperX (default: en)",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="large-v3",
        help="WhisperX model size (default: large-v3)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for WhisperX transcription (default: 16). Lower if OOM.",
    )

    args = parser.parse_args()

    print("Scanning directory structure...")
    audio_pairs = find_audio_pairs(args.root_dir)
    print(f"Found {len(audio_pairs)} audio pairs to process")

    if not audio_pairs:
        print("No audio pairs found. Please check the directory structure.")
        return

    model = None
    align_model = None
    align_metadata = None
    device = "cuda"

    if not args.skip_asr:
        print(f"Loading WhisperX model '{args.whisper_model}'...")
        model = whisperx.load_model(
            args.whisper_model, device, compute_type="float16", language=args.lang
        )
        print(f"Loading alignment model for '{args.lang}'...")
        align_model, align_metadata = whisperx.load_align_model(
            language_code=args.lang, device=device
        )
        print("Models loaded.")

    all_transcripts = {}

    for pair in tqdm(audio_pairs, desc="Processing audio pairs"):
        key = pair["key"]
        sample_path = pair["sample_path"]
        a_wav = pair["a_wav"]
        b_wav = pair["b_wav"]

        a_json = os.path.join(sample_path, "A.json")
        b_json = os.path.join(sample_path, "B.json")

        if not args.skip_asr:
            if not os.path.exists(a_json):
                a_result = transcribe_audio_file(
                    a_wav, model, align_model, align_metadata,
                    args.lang, args.batch_size, device,
                )
                with open(a_json, 'w') as f:
                    json.dump(a_result, f, indent=4)

            if not os.path.exists(b_json):
                b_result = transcribe_audio_file(
                    b_wav, model, align_model, align_metadata,
                    args.lang, args.batch_size, device,
                )
                with open(b_json, 'w') as f:
                    json.dump(b_result, f, indent=4)

        combined_transcript = combine_transcripts(
            a_json, b_json,
            max_time=args.max_time,
            time_threshold=args.time_threshold,
        )

        transcript_out = os.path.join(sample_path, "transcripts.json")
        with open(transcript_out, 'w') as f:
            json.dump(combined_transcript, f, indent=2)

        all_transcripts[key] = combined_transcript

    print(f"Writing transcripts to {args.output}...")
    with open(args.output, 'w') as f:
        json.dump(all_transcripts, f, indent=2)

    print(f"Done! Processed {len(all_transcripts)} conversations.")
    print(f"Output saved to: {args.output}")


if __name__ == "__main__":
    main()
