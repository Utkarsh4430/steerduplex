"""Interactive chat UI for SteerDuplex / Moshi models.

Works over SSH via Gradio's share tunnel or SSH port forwarding.

Two modes:
1. Server mode (default): Moshi WebSocket server with built-in web UI
2. Fallback mode: Gradio app with record/upload audio + generate response

Usage:
    # Base Moshi model (creates a public share link for SSH access)
    python -m inference.chat

    # With finetuned checkpoint
    python -m inference.chat --checkpoint runs/full_v1/checkpoints/step_5000

    # Fallback mode (upload audio, get response)
    python -m inference.chat --fallback --checkpoint runs/full_v1/checkpoints/step_5000

    # Without share link (use SSH port forwarding instead)
    python -m inference.chat --no-share --port 8998
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def find_checkpoint_weight(checkpoint_path: str) -> str | None:
    """Find the model weight file in a checkpoint directory."""
    ckpt_dir = Path(checkpoint_path)
    if not ckpt_dir.is_dir():
        if ckpt_dir.exists():
            return str(ckpt_dir)
        return None

    for pattern in ["*.safetensors", "consolidated*.pth", "*.pt"]:
        files = list(ckpt_dir.glob(pattern))
        if files:
            return str(files[0])
    return None


def is_lora_checkpoint(checkpoint_path: str) -> bool:
    """Check if checkpoint is LoRA by reading its config."""
    config_path = Path(checkpoint_path) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        return config.get("lora", False)
    return False


def launch_server_mode(args):
    """Launch moshi server for real-time full-duplex conversation."""
    server_cmd = [
        sys.executable, "-m", "moshi.server",
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--device", args.device,
    ]

    if args.hf_repo:
        server_cmd.extend(["--hf-repo", args.hf_repo])

    # Gradio tunnel for SSH access
    if args.share:
        server_cmd.append("--gradio-tunnel")

    if args.checkpoint:
        weight_path = find_checkpoint_weight(args.checkpoint)
        if weight_path is None:
            print(f"ERROR: No weight file found in {args.checkpoint}")
            sys.exit(1)

        is_lora = args.lora or is_lora_checkpoint(args.checkpoint)
        if is_lora:
            server_cmd.extend(["--lora-weight", weight_path])
            print(f"Loading LoRA checkpoint: {weight_path}")
        else:
            server_cmd.extend(["--moshi-weight", weight_path])
            print(f"Loading full checkpoint: {weight_path}")

        config_path = Path(args.checkpoint) / "config.json"
        if config_path.exists():
            server_cmd.extend(["--config-path", str(config_path)])

    print(f"\n{'=' * 60}")
    print(f"  SteerDuplex Chat")
    print(f"{'=' * 60}")
    if args.checkpoint:
        print(f"  Checkpoint:  {args.checkpoint}")
    else:
        print(f"  Model:       {args.hf_repo}")
    print(f"  Device:      {args.device}")
    print(f"  Port:        {args.port}")
    if args.share:
        print(f"  Access:      Gradio tunnel (public URL printed below)")
    else:
        print(f"  Access:      SSH port forward with:")
        print(f"    ssh -L {args.port}:localhost:{args.port} <your-ssh-host>")
        print(f"    Then open: http://localhost:{args.port}")
    print(f"{'=' * 60}\n")

    print(f"Command: {' '.join(server_cmd)}\n")
    print("Starting server (this takes ~30s to load the model)...")

    try:
        proc = subprocess.Popen(server_cmd)
        proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        proc.terminate()
        proc.wait(timeout=10)


def launch_fallback_mode(args):
    """Launch Gradio app with record/upload audio — works over SSH with --share."""
    import gradio as gr
    import numpy as np

    from inference.generate import MoshiInference

    print(f"\n{'=' * 60}")
    print(f"  SteerDuplex Chat (Fallback Mode)")
    print(f"{'=' * 60}")
    if args.checkpoint:
        print(f"  Checkpoint:  {args.checkpoint}")
    else:
        print(f"  Model:       {args.hf_repo}")
    print(f"  Device:      {args.device}")
    if args.share:
        print(f"  Access:      Gradio share link (printed after model loads)")
    else:
        port = args.port + 1
        print(f"  Access:      SSH port forward with:")
        print(f"    ssh -L {port}:localhost:{port} <your-ssh-host>")
        print(f"    Then open: http://localhost:{port}")
    print(f"{'=' * 60}\n")

    print("Loading model...")
    model = MoshiInference(
        hf_repo_id=args.hf_repo,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )
    print("Model loaded!\n")

    def respond(
        audio_input: tuple[int, np.ndarray] | str | None,
        system_prompt: str,
        voice_prompt: str | None,
        temperature: float,
        top_k: int,
        max_duration: float,
    ) -> tuple[int, np.ndarray] | None:
        if audio_input is None:
            return None

        if isinstance(audio_input, str):
            audio_path = audio_input
        elif isinstance(audio_input, tuple):
            sr, audio_array = audio_input
            import soundfile as sf
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            if audio_array.dtype == np.int16:
                audio_array = audio_array.astype(np.float32) / 32768.0
            if audio_array.ndim > 1:
                audio_array = audio_array[:, 0]
            sf.write(tmp.name, audio_array, sr)
            audio_path = tmp.name
        else:
            return None

        audio_out, sr_out = model.generate(
            user_audio_path=audio_path,
            system_prompt=system_prompt,
            voice_prompt_path=voice_prompt if voice_prompt else None,
            max_duration_sec=max_duration,
            temperature=temperature,
            top_k=top_k,
        )

        return (sr_out, audio_out)

    with gr.Blocks(title="SteerDuplex Chat") as demo:
        gr.Markdown("# SteerDuplex Chat")
        if args.checkpoint:
            gr.Markdown(f"**Checkpoint:** `{args.checkpoint}`")
        else:
            gr.Markdown(f"**Model:** `{args.hf_repo}`")

        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="Record or upload user audio",
                    sources=["microphone", "upload"],
                    type="numpy",
                )
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    value=args.system_prompt,
                    placeholder="You are a helpful assistant.",
                )
                voice_prompt = gr.File(
                    label="Voice Prompt (optional, 3-10s WAV)",
                    file_types=[".wav"],
                    type="filepath",
                )

            with gr.Column(scale=2):
                audio_output = gr.Audio(
                    label="Model Response",
                    type="numpy",
                    autoplay=True,
                )

        with gr.Row():
            with gr.Column():
                temperature = gr.Slider(0.1, 1.5, value=0.8, step=0.1, label="Temperature")
                top_k = gr.Slider(1, 500, value=250, step=10, label="Top-K")
                max_duration = gr.Slider(5, 120, value=30, step=5, label="Max Duration (s)")

        submit_btn = gr.Button("Generate Response", variant="primary")
        submit_btn.click(
            fn=respond,
            inputs=[audio_input, system_prompt, voice_prompt, temperature, top_k, max_duration],
            outputs=[audio_output],
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port + 1,
        share=args.share,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SteerDuplex interactive chat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to checkpoint directory (full or LoRA)",
    )
    parser.add_argument(
        "--lora", action="store_true",
        help="Force treating checkpoint as LoRA (auto-detected if not set)",
    )
    parser.add_argument(
        "--hf-repo", type=str, default="kyutai/moshiko-pytorch-bf16",
        help="HuggingFace model repo",
    )
    parser.add_argument(
        "--system-prompt", type=str, default="",
        help="Default system prompt",
    )
    parser.add_argument("--port", type=int, default=8998)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--fallback", action="store_true",
        help="Use file-upload Gradio UI instead of real-time server",
    )
    parser.add_argument(
        "--share", action="store_true", default=True,
        help="Create a public Gradio/tunnel share link (default: on for SSH)",
    )
    parser.add_argument(
        "--no-share", action="store_true",
        help="Disable share link (use SSH port forwarding instead)",
    )

    args = parser.parse_args()
    if args.no_share:
        args.share = False

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.fallback:
        launch_fallback_mode(args)
    else:
        launch_server_mode(args)


if __name__ == "__main__":
    main()
