# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import asyncio
from dataclasses import dataclass
import inspect
import json
import random
import os
from pathlib import Path
import tarfile
import time
import secrets
import sys
from typing import Any
import aiohttp
from aiohttp import web
from huggingface_hub import hf_hub_download
import numpy as np
import sentencepiece
from safetensors.torch import load_file
import sphn
import torch
from moshi.models import LMGen, loaders
from moshi.run_inference import get_condition_tensors

try:
    from .client_utils import log
except ImportError:
    from client_utils import log

try:
    from .personaplex_loader import is_personaplex, load_personaplex_lm
except ImportError:
    from personaplex_loader import is_personaplex, load_personaplex_lm

# mimi.set_num_codebooks(8) is always called at load time; decode uses only
# the first 8 audio codebooks regardless of the model's dep_q value.
N_MIMI_CODEBOOKS = 8


def seed_all(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU setups
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


def load_checkpoint_weights(model: Any, checkpoint_path: str, device: str | torch.device) -> None:
    ckpt_path = Path(checkpoint_path)
    if ckpt_path.is_file():
        checkpoint_files = [ckpt_path]
    else:
        checkpoint_files = sorted(ckpt_path.glob("*.safetensors")) + sorted(ckpt_path.glob("*.pt"))

    if not checkpoint_files:
        log("warning", f"No checkpoint files found in {ckpt_path}")
        return

    checkpoint_file = checkpoint_files[0]
    log("info", f"Loading checkpoint from {checkpoint_file}")
    if checkpoint_file.suffix == ".safetensors":
        state_dict = load_file(str(checkpoint_file), device=str(device))
    else:
        state_dict = torch.load(str(checkpoint_file), map_location=device)

    missing, _unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        log("info", f"Missing keys while loading checkpoint: {len(missing)}")


def load_moshi_lm(
    checkpoint_info: Any,
    checkpoint_path: str | None,
    device: str,
    dtype: torch.dtype,
    fuse_lora: bool,
    hf_repo: str = "",
) -> Any:
    if is_personaplex(hf_repo):
        # PersonaPlex config.json only has {"model_type": ..., "version": ...}
        # and cannot be used as LMModel kwargs. Use dedicated loader with
        # correct architecture constants (n_q=16, dep_q=16) and weight patching.
        model = load_personaplex_lm(
            hf_repo=hf_repo,
            device=device,
            dtype=dtype,
        )
        if checkpoint_path:
            load_checkpoint_weights(model, checkpoint_path, device)
        return model

    lm_config = dict(
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    if checkpoint_path:
        ckpt_path = Path(checkpoint_path)
        config_path = ckpt_path / "config.json" if ckpt_path.is_dir() else None
        if config_path is not None and config_path.exists():
            with open(config_path) as f:
                ckpt_config = json.load(f)
            if ckpt_config.get("lora"):
                lm_config["lora"] = True
                lm_config["lora_rank"] = ckpt_config.get("lora_rank", 128)
                lm_config["lora_scaling"] = ckpt_config.get("lora_scaling", 2.0)

    get_moshi_kwargs = {"device": device}
    get_moshi_sig = inspect.signature(checkpoint_info.get_moshi)
    if "dtype" in get_moshi_sig.parameters:
        get_moshi_kwargs["dtype"] = dtype
    if "fuse_lora" in get_moshi_sig.parameters:
        get_moshi_kwargs["fuse_lora"] = fuse_lora
    if "lm_kwargs_overrides" in get_moshi_sig.parameters:
        get_moshi_kwargs["lm_kwargs_overrides"] = lm_config

    model = checkpoint_info.get_moshi(**get_moshi_kwargs)

    if checkpoint_path:
        load_checkpoint_weights(model, checkpoint_path, device)

    return model


@dataclass
class ServerState:
    model_type: str
    mimi: Any
    text_tokenizer: sentencepiece.SentencePieceProcessor
    lm_gen: LMGen
    lock: asyncio.Lock

    def __init__(self, model_type: str, mimi: Any, text_tokenizer: sentencepiece.SentencePieceProcessor,
                 lm: Any, cfg_coef: float, device: str | torch.device, **kwargs):
        self.model_type = model_type
        self.mimi = mimi
        self.text_tokenizer = text_tokenizer
        condition_tensors = get_condition_tensors(model_type, lm, batch_size=1, cfg_coef=cfg_coef)
        self.lm_gen = LMGen(lm, cfg_coef=cfg_coef, condition_tensors=condition_tensors, **kwargs)

        self.device = device
        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lock = asyncio.Lock()

        self.mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

    def warmup(self):
        for chunk in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:1 + N_MIMI_CODEBOOKS])

        torch.cuda.synchronize()

    async def decode_and_send(
        self,
        tokens: torch.Tensor,
        ws: web.WebSocketResponse,
        opus_writer: sphn.OpusStreamWriter
    ):
        assert tokens.shape[1] == self.lm_gen.lm_model.dep_q + 1
        main_pcm = self.mimi.decode(tokens[:, 1:1 + N_MIMI_CODEBOOKS])
        main_pcm = main_pcm.cpu()
        opus_bytes = opus_writer.append_pcm(main_pcm[0, 0].numpy())
        if len(opus_bytes) > 0:
            await ws.send_bytes(b"\x01" + opus_bytes)
        text_token = tokens[0, 0, 0].item()
        if text_token not in (0, 3):
            _text = self.text_tokenizer.id_to_piece(text_token)  # type: ignore
            _text = _text.replace("▁", " ")
            msg = b"\x02" + bytes(_text, encoding="utf8")
            log("info", f"text token '{_text}'")
            await ws.send_bytes(msg)

    async def recv_loop(
        self,
        ws: web.WebSocketResponse,
        opus_reader: sphn.OpusStreamReader,
        opus_writer: sphn.OpusStreamWriter
    ):
        all_pcm_data = None
        skip_frames = 1
        try:
            async for message in ws:
                if message.type == aiohttp.WSMsgType.ERROR:
                    log("error", f"{ws.exception()}")
                    break
                elif message.type == aiohttp.WSMsgType.CLOSED:
                    break
                elif message.type != aiohttp.WSMsgType.BINARY:
                    log("error", f"unexpected message type {message.type}")
                    continue
                message = message.data
                if not isinstance(message, bytes):
                    log("error", f"unsupported message type {type(message)}")
                    continue
                if len(message) == 0:
                    log("warning", "empty message")
                    continue
                kind = message[0]
                if kind == 1:  # audio
                    payload = message[1:]
                    pcm = opus_reader.append_bytes(payload)
                    if pcm.shape[-1] == 0:
                        continue
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        be = time.time()
                        chunk = all_pcm_data[: self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        chunk = torch.from_numpy(chunk)
                        chunk = chunk.to(device=self.device)[None, None]
                        codes = self.mimi.encode(chunk)
                        if skip_frames:
                            # The first input audio frame is ignored, as from the point of
                            # view of the model it is in the past. We still `mimi.encode` for simplicity,
                            # however as the first encoded frame has a specific structure (due to the left padding),
                            # we reset the streaming state of the encoder to reapply the padding on the next call.
                            self.mimi.reset_streaming()
                            skip_frames -= 1
                        for c in range(codes.shape[-1]):
                            tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                            if tokens is None:
                                continue
                            await self.decode_and_send(tokens, ws, opus_writer)
                        log("info", f"frame handled in {1000 * (time.time() - be):.1f}ms")
                else:
                    log("warning", f"unknown message kind {kind}")
        finally:
            log("info", "connection closed")

    async def handle_chat(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        log("info", "accepted connection")

        async with self.lock:
            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            # Send the handshake.
            await ws.send_bytes(b"\x00")
            await self.recv_loop(ws, opus_reader, opus_writer)
        log("info", "done with connection")
        return ws


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost", type=str)
    parser.add_argument("--port", default=8998, type=int)
    parser.add_argument("--static", type=str)
    parser.add_argument("--gradio-tunnel", action='store_true', help='Activate a gradio tunnel.')
    parser.add_argument("--gradio-tunnel-token",
                        help='Provide a custom (secret) token here to keep getting the same URL.')

    parser.add_argument("--tokenizer", type=str, help="Path to a local tokenizer file.")
    parser.add_argument("--moshi-weight", type=str, help="Path to a local checkpoint file for Moshi.")
    parser.add_argument("--mimi-weight", type=str, help="Path to a local checkpoint file for Mimi.")
    parser.add_argument("--hf-repo", type=str, default=loaders.DEFAULT_REPO,
                        help="HF repo to look into, defaults Moshiko. "
                             "Use this to select a different pre-trained model.")
    parser.add_argument("--lora-weight", type=str, help="Path to a local checkpoint file for LoRA.", default=None)
    parser.add_argument("--config-path", type=str, help="Path to a local config file.", default=None)
    parser.add_argument("--cfg-coef", type=float, default=1., help="CFG coefficient.")
    parser.add_argument("--device", type=str, default="cuda", help="Device on which to run, defaults to 'cuda'.")
    parser.add_argument("--no_fuse_lora", action="store_false", dest="fuse_lora", default=True,
                        help="Do not fuse LoRA layers intot Linear layers.")
    parser.add_argument("--half", action="store_const", const=torch.float16, default=torch.bfloat16,
                        dest="dtype", help="Run inference with float16, not bfloat16, better for old GPUs.")
    parser.add_argument(
        "--ssl",
        type=str,
        help=(
            "use https instead of http, this flag should point to a directory "
            "that contains valid key.pem and cert.pem files"
        )
    )

    args = parser.parse_args()
    seed_all(42424242)

    setup_tunnel = None
    tunnel_token = ''
    if args.gradio_tunnel:
        try:
            from gradio import networking  # type: ignore
        except ImportError:
            log("error", "Cannot find gradio which is required to activate a tunnel. "
                         "Please install with `pip install gradio`.")
            sys.exit(1)
        setup_tunnel = networking.setup_tunnel
        if args.gradio_tunnel_token is None:
            tunnel_token = secrets.token_urlsafe(32)
        else:
            tunnel_token = args.gradio_tunnel_token

    log("info", "retrieving checkpoint")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(hf_repo=args.hf_repo)
    log("info", "loading mimi")
    # checkpoint_info.get_mimi() reads dep_q/n_q from lm_config, which is absent
    # in PersonaPlex's minimal config.json. Use loaders.get_mimi() directly instead
    # (defaults to num_codebooks=8, correct for both models).
    if is_personaplex(args.hf_repo):
        mimi = loaders.get_mimi(checkpoint_info.mimi_weights, device=args.device)
    else:
        mimi = checkpoint_info.get_mimi(device=args.device)
    log("info", "mimi loaded")

    text_tokenizer = checkpoint_info.get_text_tokenizer()

    log("info", "loading moshi")
    lm = load_moshi_lm(
        checkpoint_info,
        args.moshi_weight,
        args.device,
        args.dtype,
        args.fuse_lora,
        hf_repo=args.hf_repo,
    )
    log("info", "moshi loaded")

    state = ServerState(checkpoint_info.model_type, mimi, text_tokenizer, lm, args.cfg_coef, args.device,
                        **checkpoint_info.lm_gen_config)
    log("info", "warming up the model")
    state.warmup()
    app = web.Application()
    app.router.add_get("/api/chat", state.handle_chat)
    static_path: None | str = None
    if args.static is None:
        log("info", "retrieving the static content")
        dist_tgz = hf_hub_download("kyutai/moshi-artifacts", "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        static_path = str(dist)
    elif args.static != "none":
        # When set to the "none" string, we don't serve any static content.
        static_path = args.static
    if static_path is not None:
        async def handle_root(_):
            return web.FileResponse(os.path.join(static_path, "index.html"))

        log("info", f"serving static content from {static_path}")
        app.router.add_get("/", handle_root)
        app.router.add_static(
            "/", path=static_path, follow_symlinks=True, name="static"
        )
    protocol = "http"
    ssl_context = None
    if args.ssl is not None:
        import ssl

        ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
        cert_file = os.path.join(args.ssl, "cert.pem")
        key_file = os.path.join(args.ssl, "key.pem")
        ssl_context.load_cert_chain(certfile=cert_file, keyfile=key_file)
        protocol = "https"

    log("info", f"Access the Web UI directly at {protocol}://{args.host}:{args.port}")
    if setup_tunnel is not None:
        tunnel_kwargs = {}
        if "share_server_tls_certificate" in inspect.signature(setup_tunnel).parameters:
            tunnel_kwargs["share_server_tls_certificate"] = None
        tunnel = setup_tunnel('localhost', args.port, tunnel_token, None, **tunnel_kwargs)  # type: ignore
        log("info", f"Tunnel started, if executing on a remote GPU, you can use {tunnel}.")
        log("info", "Note that this tunnel goes through the US and you might experience high latency in Europe.")
    web.run_app(app, host=args.host , port=args.port, ssl_context=ssl_context)


with torch.no_grad():
    main()
