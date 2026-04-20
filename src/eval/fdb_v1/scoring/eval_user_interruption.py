# MODIFIED FROM UPSTREAM (Full-Duplex-Bench/v1_v1.5/evaluation/eval_user_interruption.py):
# Per-sample loop wrapped in ThreadPoolExecutor so the gpt-4-turbo calls run
# concurrently. Default 32 workers; override via MAX_LLM_WORKERS env var.
# Behavior of the metric calculation is unchanged — same TOR / latency /
# rating logic, same prompt, same seed=0, same retry-on-exception. Sample
# results are still keyed by sample dir (file writes are per-sample so no
# race), and the upstream "score appended twice" quirk is preserved verbatim
# so the reported "Average rating" matches what `Full-Duplex-Bench` produces.

import json
import os
import re
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

turn_duration_threshold = 1
turn_num_words_threshold = 3

DEFAULT_MAX_WORKERS = int(os.environ.get("MAX_LLM_WORKERS", "32"))


def parse_output(data):
    # Regular expression to match each example

    example_pattern = re.compile(
        r"Analysis:\s*(.*?)\nI would rate the AI's response as (\d+)", re.DOTALL
    )

    example = {}
    # Parse the examples
    for match in example_pattern.finditer(data):
        analysis = match.group(1).strip()
        rating = match.group(2).strip()

        # Append the parsed example
        example = {"analysis": analysis, "rating": int(rating)}

    return example


_SYSTEM_MSG = """
   The scenario is that the user and AI are talking in the spoken conversation.
   The user first speaks, then the AI responds. But when AI is speaking, the user interrupts the AI's turn.
   Your task is to rate the quality of AI's response after the user interrupt the turn.


   Below is the rating guideline (from 0 to 5, 0 is the worst and 5 is the best):
   - 0: The AI's response is totally unrelated to the user's interrupting turn.
   - 1: The AI's response is not related to the user's interrupting turn.
   - 2: The AI's response is slightly related to the user's interrupting turn.
   - 3: The AI's response is related to the user's interrupting turn.
   - 4: The AI's response is highly related to the user's interrupting turn.
   - 5: The AI's response is perfectly related to the user's interrupting turn.


   Firstly, briefly analyze the user's interrupting turn and the AI's response
   Then, you must return the overall output as the following format:
   Analysis: [Your analysis].
   I would rate the AI's response as [Rating].
   """


def _process_one_sample(file_dir, client, model_name, seed):
    """Compute TOR / latency / rating for one sample directory.

    Returns dict with TOR, latency, scores (list of 0/1/2 entries to mirror
    the upstream double-append quirk; sum/len invariant), or None for the
    rating fields when TOR != 1.
    """
    out_after_interrupt_path = os.path.join(file_dir, "output.json")
    if not os.path.exists(out_after_interrupt_path):
        raise FileNotFoundError("Required file 'output.json' not found.")
    with open(out_after_interrupt_path, "r") as f:
        out_after_interrupt = json.load(f)

    metadata_path = os.path.join(file_dir, "interrupt.json")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError("Required file 'interrupt.json' not found.")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    in_interrupt_text = metadata[0]["interrupt"]
    in_before_interrupt_text = metadata[0]["context"]
    input_end_time = metadata[0]["timestamp"][1]
    out_after_interrupt_text = out_after_interrupt["text"]

    # TOR and latency (cheap, no LLM call)
    TOR = None
    latency = None
    segments_cw = out_after_interrupt["chunks"]
    if len(segments_cw) == 0:
        TOR = 0
    else:
        output_start_time = segments_cw[0]["timestamp"][0]
        duration = (
            segments_cw[-1]["timestamp"][-1] - segments_cw[0]["timestamp"][0]
        )
        if duration < turn_duration_threshold:
            if len(segments_cw) <= turn_num_words_threshold:
                TOR = 0
            else:
                TOR = 1
                latency = output_start_time - input_end_time
        else:
            TOR = 1
            latency = output_start_time - input_end_time

    if TOR != 1:
        return {"file_dir": file_dir, "TOR": TOR, "latency": None, "scores": []}

    user_msg = f"""
                - Contextual user turn: {in_before_interrupt_text}
                - User interrupting turn: {in_interrupt_text}
                - AI's response: {out_after_interrupt_text}
                """
    messages = [
        {"role": "system", "content": _SYSTEM_MSG},
        {"role": "user", "content": user_msg},
    ]

    # Retry until the LLM produces a parseable rating (matches upstream's
    # `while True ... continue if "rating" not in parsed_output ... break`).
    while True:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            seed=seed,
        )
        prediction = response.choices[0].message.content
        parsed_output = parse_output(prediction + "\n")
        if "rating" in parsed_output:
            break

    # save rating.json (per-sample dir, no race across workers)
    with open(os.path.join(file_dir, "rating.json"), "w") as f:
        json.dump(parsed_output, f)

    score = parsed_output["rating"]
    # Preserve upstream double-append quirk verbatim — sum/len invariant so the
    # reported "Average rating" matches Full-Duplex-Bench output exactly.
    return {"file_dir": file_dir, "TOR": TOR, "latency": latency, "scores": [score, score]}


def eval_user_interruption(root_dir, client, max_workers=None):

    MODEL_NAME = "gpt-4-turbo"
    seed = 0

    file_dirs = []
    for root, dirs, files in os.walk(root_dir):
        for dir in dirs:
            file_dirs.append(os.path.join(root, dir))

    file_dirs = sorted(file_dirs)
    if max_workers is None:
        max_workers = DEFAULT_MAX_WORKERS

    score_list = []
    take_turn_list = []
    latency_list = []

    print(f"Running gpt-4-turbo rating on {len(file_dirs)} samples "
          f"with {max_workers} concurrent workers")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_process_one_sample, fd, client, MODEL_NAME, seed): fd
            for fd in file_dirs
        }
        for fut in tqdm(as_completed(futures), total=len(futures)):
            fd = futures[fut]
            try:
                result = fut.result()
            except Exception as exc:
                print(f"[ERROR] sample {fd}: {type(exc).__name__}: {exc}")
                continue
            take_turn_list.append(result["TOR"])
            score_list.extend(result["scores"])
            if result["latency"] is not None:
                if result["latency"] < 0:
                    latency_list.append(0)
                else:
                    latency_list.append(result["latency"])

    print("---------------------------------------------------")
    print("[Result]")
    if score_list:
        print("Average rating: ", sum(score_list) / len(score_list))
    else:
        print("Average rating:  N/A (no TOR=1 samples)")
    print("Average take turn: ", sum(take_turn_list) / len(take_turn_list))
    if latency_list:
        print("Average latency: ", sum(latency_list) / len(latency_list))
    else:
        print("Average latency:  N/A (no TOR=1 samples)")
    print("---------------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser")
    parser.add_argument("--root_dir", type=str)
    args = parser.parse_args()

    eval_user_interruption(args.root_dir)
