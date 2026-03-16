"""Compare Qwen3-TTS vs TADA vs Fish Speech S2 Pro side by side.

Generates samples from all engines across diverse styles:
- Voice cloning quality (same ref audio, same text)
- Emotion/tone control (Qwen3-TTS instruct, Fish S2 Pro inline tags, TADA neutral)
- Speed variation (fast/slow instructs)
- Preset voices with instruct (Qwen3-TTS CustomVoice only)

This mirrors the style diversity in our data categories (A3, A6, A7, A9).

Usage:
    pip install hume-tada qwen-tts fish-speech
    python src/compare_tts.py
    python src/compare_tts.py --device cuda:1
    python src/compare_tts.py --only qwen3
    python src/compare_tts.py --only tada
    python src/compare_tts.py --only fish
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

# Suppress noisy warnings
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"


# ── Config ──────────────────────────────────────────────────────────────
VOICEPOOL = Path("/mnt/efs/utkarshtyagi/personal_projects/steerduplex/src/data/voicepool/val")
OUTPUT_DIR = Path("/mnt/efs/utkarshtyagi/personal_projects/steerduplex/src/data/tts_comparison")

# Pick diverse speakers: male, female, female
SPEAKER_IDS = ["1272", "1462", "1673"]

# ── Diverse test cases: (id, text, tts_instruct, category) ─────────────
# Each case pairs a sentence with a style instruct from our data categories.
# For TADA (no instruct support), only the text is used — so we can compare
# whether TADA infers appropriate prosody from text alone vs Qwen3's explicit control.
TEST_CASES = [
    # --- Neutral baseline ---
    {
        "id": "neutral_greeting",
        "text": "Hello! Welcome to our customer service line. How can I help you today?",
        "instruct": None,
        "fish_tag": None,
        "category": "baseline",
    },
    {
        "id": "neutral_factual",
        "text": "The quarterly revenue increased by fifteen percent, driven primarily by strong international growth.",
        "instruct": None,
        "fish_tag": None,
        "category": "baseline",
    },
    # --- Tone: positive emotions (A3) ---
    {
        "id": "cheerful_greeting",
        "text": "Oh hey, great to see you again! What are we working on today?",
        "instruct": "cheerful, bright tone, smiling voice, upbeat pace",
        "fish_tag": "[cheerful bright tone]",
        "category": "A3_tone",
    },
    {
        "id": "warm_empathy",
        "text": "I'm really sorry to hear about that. Let me see what I can do to help you through this.",
        "instruct": "warm, gentle tone, soft delivery, caring inflection",
        "fish_tag": "[warm gentle tone]",
        "category": "A3_tone",
    },
    {
        "id": "enthusiastic_discovery",
        "text": "Oh wow, that's absolutely incredible! I can't believe you managed to pull that off!",
        "instruct": "enthusiastic, energetic, varied pitch, faster pace, genuine excitement",
        "fish_tag": "[excited enthusiastic]",
        "category": "A3_tone",
    },
    {
        "id": "encouraging_coach",
        "text": "You've got this. Just take it one step at a time, and don't be afraid to make mistakes along the way.",
        "instruct": "encouraging, uplifting tone, confident and warm, building energy",
        "fish_tag": "[encouraging uplifting tone]",
        "category": "A3_tone",
    },
    {
        "id": "calm_soothing",
        "text": "Take a deep breath. Let's slow everything down and work through this together, nice and easy.",
        "instruct": "calm, serene, slow pace, soft and steady, soothing delivery",
        "fish_tag": "[calm soothing soft voice]",
        "category": "A3_tone",
    },
    # --- Tone: negative emotions (A3) ---
    {
        "id": "sad_consolation",
        "text": "I know this isn't easy. Sometimes things just don't work out the way we hoped they would.",
        "instruct": "sad, subdued tone, slower pace, softer voice, occasional sighs",
        "fish_tag": "[sad subdued tone]",
        "category": "A3_tone",
    },
    {
        "id": "frustrated_troubleshoot",
        "text": "Look, I've already explained this three times. Let me try once more from the beginning.",
        "instruct": "frustrated, exasperated sighs, strained patience, uneven pace",
        "fish_tag": "[frustrated exasperated]",
        "category": "A3_tone",
    },
    {
        "id": "sarcastic_deadpan",
        "text": "Oh sure, because that's exactly what I wanted to hear right now. Absolutely perfect timing.",
        "instruct": "dry sarcasm, deadpan delivery, exaggerated flat tone, subtle emphasis",
        "fish_tag": "[sarcastic deadpan flat tone]",
        "category": "A3_tone",
    },
    {
        "id": "stern_warning",
        "text": "I need to be very clear about this. You must not share that information with anyone outside this team.",
        "instruct": "stern, firm delivery, lower pitch, deliberate and commanding",
        "fish_tag": "[stern firm commanding tone]",
        "category": "A3_tone",
    },
    {
        "id": "anxious_uncertain",
        "text": "I'm not entirely sure about this, but I think we should probably check again before moving forward.",
        "instruct": "anxious, slightly trembling voice, hesitant, faster pace with pauses",
        "fish_tag": "[anxious hesitant nervous]",
        "category": "A3_tone",
    },
    # --- Speed control (A6) ---
    {
        "id": "very_fast_trivia",
        "text": "The capital of France is Paris, the largest ocean is the Pacific, and the speed of light is roughly three hundred thousand kilometers per second.",
        "instruct": "speak very fast, rapid pace, energetic, quick delivery",
        "fish_tag": "[speak fast rapid pace]",
        "category": "A6_speed",
    },
    {
        "id": "very_slow_meditation",
        "text": "Now close your eyes. Breathe in deeply through your nose. Hold it. And slowly let it go.",
        "instruct": "speak very slowly, meditative pace, long pauses between phrases",
        "fish_tag": "[speak slowly with long pauses]",
        "category": "A6_speed",
    },
    {
        "id": "fast_brief_qa",
        "text": "Yes, that's correct. Next question.",
        "instruct": "fast, clipped, efficient, rapid-fire",
        "fish_tag": "[fast clipped efficient]",
        "category": "A6_speed",
    },
    {
        "id": "slow_detailed_teacher",
        "text": "So the way this works is, first the client sends a request to the server, then the server processes it and sends back a response.",
        "instruct": "slow, patient, thorough, pausing for comprehension",
        "fish_tag": "[slow patient pace with pauses]",
        "category": "A6_speed",
    },
    # --- Emotional empathy (A7) ---
    {
        "id": "empathy_grief",
        "text": "I'm so sorry for your loss. There are no words that can make this better, but I'm here for you.",
        "instruct": "gentle, empathetic, soft delivery, giving space, warm comfort",
        "fish_tag": "[gentle empathetic soft voice]",
        "category": "A7_emotion",
    },
    {
        "id": "empathy_panic_calm",
        "text": "Hey, it's okay. I know it feels urgent right now, but let's take this one thing at a time. We'll figure it out.",
        "instruct": "very calm, grounding, steady pace, reassuring, anchor-like stability",
        "fish_tag": "[calm reassuring steady]",
        "category": "A7_emotion",
    },
    {
        "id": "match_excitement",
        "text": "No way, that's amazing! Tell me everything, I want to hear all the details!",
        "instruct": "enthusiastic, matching energy, engaged, genuinely interested",
        "fish_tag": "[excited enthusiastic energetic]",
        "category": "A7_emotion",
    },
    # --- Dynamic steering (A9) — same text, different styles ---
    {
        "id": "steer_formal",
        "text": "Based on my analysis, I would recommend proceeding with option B as it offers the best balance of risk and return.",
        "instruct": "confident, steady pace, clear articulation, assured delivery",
        "fish_tag": "[confident professional tone]",
        "category": "A9_steering",
    },
    {
        "id": "steer_casual",
        "text": "So yeah, I'd go with option B honestly. It's just the better deal all around, you know?",
        "instruct": "friendly, casual warmth, relaxed pace, approachable and easy",
        "fish_tag": "[casual friendly relaxed]",
        "category": "A9_steering",
    },
    {
        "id": "steer_wise",
        "text": "In my experience, the path that seems hardest at first often turns out to be the most rewarding in the end.",
        "instruct": "wise, measured pace, deeper tone, thoughtful pauses, gravitas",
        "fish_tag": "[wise measured deep tone]",
        "category": "A9_steering",
    },
    {
        "id": "steer_playful",
        "text": "Alright, pop quiz time! Let's see if you've been paying attention. Ready? Here we go!",
        "instruct": "playful, light and bouncy, teasing delivery, fun energy",
        "fish_tag": "[playful teasing fun energy]",
        "category": "A9_steering",
    },
    {
        "id": "steer_humorous",
        "text": "Well, technically speaking, that's one way to do it. Not the best way, mind you, but certainly the most entertaining.",
        "instruct": "humorous, comedic timing, varied pace, playful emphasis, occasional chuckle",
        "fish_tag": "[humorous with comedic timing]",
        "category": "A9_steering",
    },
]


def load_voicepool(pool_dir: Path) -> dict:
    """Load voice pool metadata, keyed by speaker ID."""
    pool_file = pool_dir / "libri_val_pool.json"
    with open(pool_file) as f:
        entries = json.load(f)
    return {e["id"]: e for e in entries}


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio * (0.95 / peak)
    return audio


def to_numpy(audio) -> np.ndarray:
    if isinstance(audio, list):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()
    return audio.astype(np.float32)


# ── Qwen3-TTS ──────────────────────────────────────────────────────────
def generate_qwen3_samples(pool: dict, device: str):
    """Generate samples using Qwen3-TTS."""
    from qwen_tts import Qwen3TTSModel

    # --- Voice cloning (Base model, no instruct) ---
    print("\n=== Qwen3-TTS: Base model (voice cloning, no instruct) ===")
    base_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map=device, dtype=torch.bfloat16,
    )

    clone_dir = OUTPUT_DIR / "qwen3_clone"
    clone_dir.mkdir(parents=True, exist_ok=True)

    for spk_id in SPEAKER_IDS:
        entry = pool[spk_id]
        ref_path = str(VOICEPOOL / entry["ref_path"])
        ref_text = entry["ref_text"]
        print(f"\n  Speaker {spk_id} ({entry['gender']})")

        clone_prompt = base_model.create_voice_clone_prompt(
            ref_audio=ref_path, ref_text=ref_text,
        )

        for case in TEST_CASES:
            out_path = clone_dir / f"spk{spk_id}_{case['id']}.wav"
            if out_path.exists():
                print(f"    [skip] {out_path.name}")
                continue
            try:
                wavs, sr = base_model.generate_voice_clone(
                    text=case["text"], language="English",
                    voice_clone_prompt=clone_prompt,
                )
                audio = normalize_audio(to_numpy(wavs))
                sf.write(str(out_path), audio, sr)
                print(f"    [ok] {out_path.name} ({len(audio)/sr:.1f}s)")
            except Exception as e:
                print(f"    [FAIL] {out_path.name}: {e}")

    del base_model
    torch.cuda.empty_cache()

    # --- CustomVoice preset (with instruct for style control) ---
    print("\n=== Qwen3-TTS: CustomVoice model (preset + instruct) ===")
    cv_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device, dtype=torch.bfloat16,
    )

    preset_dir = OUTPUT_DIR / "qwen3_instruct"
    preset_dir.mkdir(parents=True, exist_ok=True)

    for speaker in ["Ryan", "Vivian"]:
        print(f"\n  Preset speaker: {speaker}")
        for case in TEST_CASES:
            out_path = preset_dir / f"{speaker.lower()}_{case['id']}.wav"
            if out_path.exists():
                print(f"    [skip] {out_path.name}")
                continue
            try:
                kwargs = dict(text=case["text"], language="English", speaker=speaker)
                if case["instruct"]:
                    kwargs["instruct"] = case["instruct"]
                wavs, sr = cv_model.generate_custom_voice(**kwargs)
                audio = normalize_audio(to_numpy(wavs))
                sf.write(str(out_path), audio, sr)
                tag = f" [{case['instruct'][:40]}...]" if case["instruct"] else ""
                print(f"    [ok] {out_path.name} ({len(audio)/sr:.1f}s){tag}")
            except Exception as e:
                print(f"    [FAIL] {out_path.name}: {e}")

    del cv_model
    torch.cuda.empty_cache()


# ── TADA ────────────────────────────────────────────────────────────────
def generate_tada_samples(pool: dict, device: str):
    """Generate samples using TADA (voice cloning, no explicit style control)."""
    from tada.modules.encoder import Encoder
    from tada.modules.tada import TadaForCausalLM

    print("\n=== TADA 3B: voice cloning (no instruct — style from text only) ===")
    encoder = Encoder.from_pretrained(
        "HumeAI/tada-codec", subfolder="encoder",
    ).to(device)
    model = TadaForCausalLM.from_pretrained("HumeAI/tada-3b-ml").to(device)

    out_dir = OUTPUT_DIR / "tada"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spk_id in SPEAKER_IDS:
        entry = pool[spk_id]
        ref_path = str(VOICEPOOL / entry["ref_path"])
        ref_text = entry["ref_text"]
        print(f"\n  Speaker {spk_id} ({entry['gender']})")

        audio_ref, sr_ref = torchaudio.load(ref_path)
        audio_ref = audio_ref.to(device)
        prompt = encoder(audio_ref, text=[ref_text], sample_rate=sr_ref)

        for case in TEST_CASES:
            out_path = out_dir / f"spk{spk_id}_{case['id']}.wav"
            if out_path.exists():
                print(f"    [skip] {out_path.name}")
                continue
            try:
                output = model.generate(prompt=prompt, text=case["text"])
                # output is GenerationOutput; .audio is a list of tensors
                wav = output.audio[0]
                if wav is None:
                    print(f"    [FAIL] {out_path.name}: decoder returned None")
                    continue
                gen_audio = wav.squeeze().cpu().numpy().astype(np.float32)
                gen_audio = normalize_audio(gen_audio)
                gen_sr = 24000  # TADA decodes at 24kHz
                sf.write(str(out_path), gen_audio, gen_sr)
                print(f"    [ok] {out_path.name} ({len(gen_audio)/gen_sr:.1f}s)")
            except Exception as e:
                print(f"    [FAIL] {out_path.name}: {e}")

    del model, encoder
    torch.cuda.empty_cache()


# ── Fish Speech S2 Pro ─────────────────────────────────────────────────
def generate_fish_samples(pool: dict, device: str):
    """Generate samples using Fish Speech S2 Pro (voice cloning + inline prosody tags).

    Requires the fish-speech conda env and repo cloned separately:
        conda activate fish-speech
        cd /mnt/efs/utkarshtyagi/fish-speech
        pip install -e .[cu129]
        huggingface-cli download fishaudio/s2-pro --local-dir checkpoints/s2-pro

    Then run from inside the fish-speech repo (for hydra config paths):
        cd /mnt/efs/utkarshtyagi/fish-speech
        python /mnt/efs/utkarshtyagi/personal_projects/steerduplex/src/compare_tts.py --only fish
    """
    from fish_speech.models.text2semantic.inference import (
        decode_to_audio,
        encode_audio,
        generate_long,
        init_model,
        load_codec_model,
    )

    FISH_CKPT = Path("checkpoints/s2-pro")
    CODEC_CKPT = FISH_CKPT / "codec.pth"

    precision = torch.bfloat16

    print("\n=== Fish Speech S2 Pro: voice cloning + inline prosody tags ===")

    # Load LLM (DualAR 5B)
    print("  Loading text2semantic model...")
    llm, decode_one_token = init_model(
        checkpoint_path=FISH_CKPT,
        device=device,
        precision=precision,
        compile=False,
    )
    with torch.device(device):
        llm.setup_caches(
            max_batch_size=1,
            max_seq_len=llm.config.max_seq_len,
            dtype=precision,
        )

    # Load codec (DAC-based)
    print("  Loading codec model...")
    codec = load_codec_model(CODEC_CKPT, device, precision)

    out_dir = OUTPUT_DIR / "fish_s2pro"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spk_id in SPEAKER_IDS:
        entry = pool[spk_id]
        ref_path = str(VOICEPOOL / entry["ref_path"])
        ref_text = entry["ref_text"]
        print(f"\n  Speaker {spk_id} ({entry['gender']})")

        # Encode reference audio to VQ tokens (handles resampling internally)
        prompt_tokens = encode_audio(ref_path, codec, device)

        for case in TEST_CASES:
            out_path = out_dir / f"spk{spk_id}_{case['id']}.wav"
            if out_path.exists():
                print(f"    [skip] {out_path.name}")
                continue
            try:
                # Prepend inline prosody tag if available
                text = case["text"]
                if case.get("fish_tag"):
                    text = f"{case['fish_tag']} {text}"

                # generate_long is a generator — collect all code chunks
                all_codes = []
                for response in generate_long(
                    model=llm,
                    device=device,
                    decode_one_token=decode_one_token,
                    text=text,
                    prompt_text=[ref_text],
                    prompt_tokens=[prompt_tokens.cpu()],
                    num_samples=1,
                    max_new_tokens=0,
                    top_p=0.9,
                    top_k=30,
                    temperature=1.0,
                    iterative_prompt=True,
                    chunk_length=300,
                ):
                    if response.action == "sample" and response.codes is not None:
                        all_codes.append(response.codes)

                if not all_codes:
                    print(f"    [FAIL] {out_path.name}: no codes generated")
                    continue

                # Merge code chunks and decode to waveform
                merged_codes = torch.cat(all_codes, dim=1).to(device)
                wav = decode_to_audio(merged_codes, codec)
                gen_audio = wav.cpu().float().numpy()
                gen_audio = normalize_audio(gen_audio)
                gen_sr = codec.sample_rate
                sf.write(str(out_path), gen_audio, gen_sr)
                print(f"    [ok] {out_path.name} ({len(gen_audio)/gen_sr:.1f}s)")
            except Exception as e:
                print(f"    [FAIL] {out_path.name}: {e}")

    del llm, codec
    torch.cuda.empty_cache()


# ── Experiment: Qwen3-TTS toned down (minimal/no instruct) ─────────────
# Strategy: for neutral/baseline → no instruct at all
#           for styled cases → single-word instruct only (the core emotion)
QWEN3_MINIMAL_INSTRUCT = {
    "neutral_greeting": None,
    "neutral_factual": None,
    "cheerful_greeting": "cheerful",
    "warm_empathy": "warm",
    "enthusiastic_discovery": "enthusiastic",
    "encouraging_coach": "encouraging",
    "calm_soothing": "calm",
    "sad_consolation": "subdued",
    "frustrated_troubleshoot": "frustrated",
    "sarcastic_deadpan": "deadpan",
    "stern_warning": "stern",
    "anxious_uncertain": "hesitant",
    "very_fast_trivia": "fast",
    "very_slow_meditation": "slow",
    "fast_brief_qa": "fast",
    "slow_detailed_teacher": "slow",
    "empathy_grief": "gentle",
    "empathy_panic_calm": "calm",
    "match_excitement": "enthusiastic",
    "steer_formal": "confident",
    "steer_casual": "casual",
    "steer_wise": "measured",
    "steer_playful": "playful",
    "steer_humorous": "humorous",
}


def generate_qwen3_minimal_samples(pool: dict, device: str):
    """Qwen3-TTS CustomVoice with minimal single-word instructs (less expressive)."""
    from qwen_tts import Qwen3TTSModel

    print("\n=== Experiment: Qwen3-TTS CustomVoice MINIMAL instruct ===")
    cv_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map=device, dtype=torch.bfloat16,
    )

    out_dir = OUTPUT_DIR / "qwen3_minimal"
    out_dir.mkdir(parents=True, exist_ok=True)

    for speaker in ["Ryan", "Vivian"]:
        print(f"\n  Preset speaker: {speaker}")
        for case in TEST_CASES:
            out_path = out_dir / f"{speaker.lower()}_{case['id']}.wav"
            if out_path.exists():
                print(f"    [skip] {out_path.name}")
                continue
            try:
                instruct = QWEN3_MINIMAL_INSTRUCT.get(case["id"])
                kwargs = dict(text=case["text"], language="English", speaker=speaker)
                if instruct:
                    kwargs["instruct"] = instruct
                wavs, sr = cv_model.generate_custom_voice(**kwargs)
                audio = normalize_audio(to_numpy(wavs))
                sf.write(str(out_path), audio, sr)
                tag = f" [instruct={instruct}]" if instruct else " [no instruct]"
                print(f"    [ok] {out_path.name} ({len(audio)/sr:.1f}s){tag}")
            except Exception as e:
                print(f"    [FAIL] {out_path.name}: {e}")

    del cv_model
    torch.cuda.empty_cache()


# ── Experiment: Fish S2 Pro more expressive (richer inline tags) ───────
# Strategy: longer, more descriptive tags with mid-sentence placement
FISH_RICH_TAGS = {
    "neutral_greeting": "[friendly professional voice] ",
    "neutral_factual": "[calm clear newsreader tone] ",
    "cheerful_greeting": "[very cheerful and bright, smiling while speaking] ",
    "warm_empathy": "[soft warm empathetic voice, speaking gently with care] ",
    "enthusiastic_discovery": "[extremely excited and amazed, voice rising with genuine enthusiasm] ",
    "encouraging_coach": "[warm encouraging coaching voice, building confidence] ",
    "calm_soothing": "[very calm serene soothing voice, speaking slowly like a meditation guide] ",
    "sad_consolation": "[sad subdued voice, speaking softly with gentle sadness] ",
    "frustrated_troubleshoot": "[frustrated and exasperated, strained patience] ",
    "sarcastic_deadpan": "[sarcastic flat deadpan delivery, dry humor] ",
    "stern_warning": "[stern serious commanding voice, firm and authoritative] ",
    "anxious_uncertain": "[anxious nervous hesitant voice, slightly trembling] ",
    "very_fast_trivia": "[speaking very fast, rapid energetic pace] ",
    "very_slow_meditation": "[speaking very slowly, meditative pace with long pauses] ",
    "fast_brief_qa": "[quick clipped efficient delivery] ",
    "slow_detailed_teacher": "[slow patient teacher voice, pausing for understanding] ",
    "empathy_grief": "[very gentle empathetic soft voice, giving emotional space] ",
    "empathy_panic_calm": "[calm grounding reassuring voice, steady anchor-like stability] ",
    "match_excitement": "[excited enthusiastic energetic, matching high energy] ",
    "steer_formal": "[confident professional assured tone] ",
    "steer_casual": "[casual relaxed friendly warmth] ",
    "steer_wise": "[wise measured deep voice, thoughtful pauses with gravitas] ",
    "steer_playful": "[playful bouncy teasing voice, fun lighthearted energy] ",
    "steer_humorous": "[humorous witty delivery, comedic timing with playful emphasis] ",
}


def generate_fish_rich_samples(pool: dict, device: str):
    """Fish S2 Pro with richer, more descriptive inline tags (more expressive)."""
    from fish_speech.models.text2semantic.inference import (
        decode_to_audio,
        encode_audio,
        generate_long,
        init_model,
        load_codec_model,
    )

    FISH_CKPT = Path("checkpoints/s2-pro")
    CODEC_CKPT = FISH_CKPT / "codec.pth"
    precision = torch.bfloat16

    print("\n=== Experiment: Fish S2 Pro RICH inline tags (more expressive) ===")

    print("  Loading text2semantic model...")
    llm, decode_one_token = init_model(
        checkpoint_path=FISH_CKPT,
        device=device,
        precision=precision,
        compile=False,
    )
    with torch.device(device):
        llm.setup_caches(
            max_batch_size=1,
            max_seq_len=llm.config.max_seq_len,
            dtype=precision,
        )

    print("  Loading codec model...")
    codec = load_codec_model(CODEC_CKPT, device, precision)

    out_dir = OUTPUT_DIR / "fish_rich"
    out_dir.mkdir(parents=True, exist_ok=True)

    for spk_id in SPEAKER_IDS:
        entry = pool[spk_id]
        ref_path = str(VOICEPOOL / entry["ref_path"])
        ref_text = entry["ref_text"]
        print(f"\n  Speaker {spk_id} ({entry['gender']})")

        prompt_tokens = encode_audio(ref_path, codec, device)

        for case in TEST_CASES:
            out_path = out_dir / f"spk{spk_id}_{case['id']}.wav"
            if out_path.exists():
                print(f"    [skip] {out_path.name}")
                continue
            try:
                tag = FISH_RICH_TAGS.get(case["id"], "")
                text = f"{tag}{case['text']}"

                all_codes = []
                for response in generate_long(
                    model=llm,
                    device=device,
                    decode_one_token=decode_one_token,
                    text=text,
                    prompt_text=[ref_text],
                    prompt_tokens=[prompt_tokens.cpu()],
                    num_samples=1,
                    max_new_tokens=0,
                    top_p=0.9,
                    top_k=30,
                    temperature=1.0,
                    iterative_prompt=True,
                    chunk_length=300,
                ):
                    if response.action == "sample" and response.codes is not None:
                        all_codes.append(response.codes)

                if not all_codes:
                    print(f"    [FAIL] {out_path.name}: no codes generated")
                    continue

                merged_codes = torch.cat(all_codes, dim=1).to(device)
                wav = decode_to_audio(merged_codes, codec)
                gen_audio = wav.cpu().float().numpy()
                gen_audio = normalize_audio(gen_audio)
                gen_sr = codec.sample_rate
                sf.write(str(out_path), gen_audio, gen_sr)
                print(f"    [ok] {out_path.name} ({len(gen_audio)/gen_sr:.1f}s) {tag.strip()}")
            except Exception as e:
                print(f"    [FAIL] {out_path.name}: {e}")

    del llm, codec
    torch.cuda.empty_cache()


# ── Main ────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Qwen3-TTS vs TADA")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--only", choices=["qwen3", "tada", "fish", "qwen3_minimal", "fish_rich"], default=None,
                        help="Run only one engine/experiment")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pool = load_voicepool(VOICEPOOL)
    print(f"Loaded {len(pool)} speakers from voice pool")
    print(f"Using speakers: {SPEAKER_IDS}")
    print(f"Test cases: {len(TEST_CASES)} ({len(set(c['category'] for c in TEST_CASES))} categories)")
    print(f"Output dir: {OUTPUT_DIR}")

    # Copy reference audios
    ref_dir = OUTPUT_DIR / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for spk_id in SPEAKER_IDS:
        entry = pool[spk_id]
        src = VOICEPOOL / entry["ref_path"]
        dst = ref_dir / f"spk{spk_id}_ref.wav"
        if not dst.exists():
            shutil.copy2(src, dst)

    # Write test case manifest for easy reference
    manifest_path = OUTPUT_DIR / "test_cases.json"
    with open(manifest_path, "w") as f:
        json.dump(TEST_CASES, f, indent=2)

    if args.only is None or args.only == "qwen3":
        generate_qwen3_samples(pool, args.device)

    if args.only is None or args.only == "tada":
        generate_tada_samples(pool, args.device)

    if args.only is None or args.only == "fish":
        generate_fish_samples(pool, args.device)

    if args.only == "qwen3_minimal":
        generate_qwen3_minimal_samples(pool, args.device)

    if args.only == "fish_rich":
        generate_fish_rich_samples(pool, args.device)

    # Print summary
    print(f"\n{'='*70}")
    print(f"Done! Compare samples in: {OUTPUT_DIR}/")
    print(f"  reference/      — original speaker audio (3 speakers)")
    print(f"  qwen3_clone/    — Qwen3 Base voice cloning (no instruct)")
    print(f"  qwen3_instruct/ — Qwen3 CustomVoice + instruct (Ryan, Vivian)")
    print(f"  tada/           — TADA voice cloning (no instruct)")
    print(f"  fish_s2pro/     — Fish S2 Pro voice cloning + inline prosody tags")
    print(f"  test_cases.json — manifest of all {len(TEST_CASES)} test cases")
    print(f"\nKey comparisons:")
    print(f"  1. Speaker similarity:  reference/ vs qwen3_clone/ vs tada/ vs fish_s2pro/")
    print(f"  2. Style control:       qwen3_instruct/ vs fish_s2pro/ (inline tags) vs tada/ (text-only)")
    print(f"  3. Naturalness:         listen across all folders")
    print(f"  4. Emotion range:       compare cheerful/sad/angry/calm samples")
    print(f"  5. Speed control:       compare very_fast_* vs very_slow_* samples")


if __name__ == "__main__":
    main()
