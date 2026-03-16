"""SteerDuplex training script.

Adapted from moshi-finetune's train.py with PersonaPlex training methodology:
- Full finetuning (no LoRA) following PersonaPlex
- System prompt loss masking: zeros out loss in the voice_prompt + text_prompt region
- Separate learning rates: depth transformer (4e-6) and temporal transformer (2e-6)
- Cosine annealing with linear warmup (PersonaPlex: "Adam with cosine annealing")
- Token weighting follows PersonaPlex:
  - Non-semantic audio tokens downweighted by 0.02 (first_codebook_weight_multiplier=50)
  - Padded text tokens downweighted by 0.3
- Checkpoint resume support

Usage:
    # Via launch.sh (recommended)
    bash training/launch.sh configs/full_training.yaml 8

    # Direct invocation
    torchrun --nproc-per-node 8 -m training.train configs/full_training.yaml

    # Resume from checkpoint (set moshi_paths.moshi_path in config to checkpoint file)
    bash training/launch.sh configs/full_training.yaml 8
"""

import dataclasses
import logging
import math
import os
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch
import torch.cuda
import torch.distributed as dist
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders

from training.loss_masking import create_prompt_mask, seconds_to_frames

logger = logging.getLogger("train")

MIMI_FRAME_RATE = 12.5


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


# ---------------------------------------------------------------------------
# Custom config helpers
# ---------------------------------------------------------------------------

def _load_custom_config(config_path: str) -> dict:
    """Load custom SteerDuplex config fields from the 'steerduplex' section."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("steerduplex", {})


# ---------------------------------------------------------------------------
# System prompt loss masking
# ---------------------------------------------------------------------------

def _get_prompt_end_frames(batch) -> list[int]:
    """Extract prompt_end_sec from batch condition_attributes and convert to frame indices.

    moshi-finetune's InterleavedTokenizer passes text_conditions as a raw dict
    (not a ConditionAttributes object). This function handles both cases.
    """
    prompt_end_frames = []
    batch_size = batch.codes.shape[0]

    if batch.condition_attributes is not None:
        for attr in batch.condition_attributes:
            prompt_end_sec = 0.0
            if isinstance(attr, dict):
                # Raw dict from JSON — moshi-finetune passes text_conditions as dict
                val = attr.get("prompt_end_sec")
                if val is not None:
                    try:
                        prompt_end_sec = float(val)
                    except (ValueError, TypeError):
                        pass
            elif hasattr(attr, 'text') and attr.text:
                # ConditionAttributes object (future-proofing)
                val = attr.text.get("prompt_end_sec")
                if val is not None:
                    try:
                        prompt_end_sec = float(val)
                    except (ValueError, TypeError):
                        pass
            prompt_end_frames.append(seconds_to_frames(prompt_end_sec))
    else:
        prompt_end_frames = [0] * batch_size

    return prompt_end_frames


def _apply_prompt_mask(batch, text_mask, audio_mask):
    """Apply system prompt loss masking to text and audio masks.

    Returns modified (text_mask, audio_mask) with prompt region zeroed out.
    """
    prompt_end_frames = _get_prompt_end_frames(batch)
    seq_len = batch.codes.shape[-1]
    prompt_mask = create_prompt_mask(
        batch_size=batch.codes.shape[0],
        seq_len=seq_len,
        prompt_end_frames=prompt_end_frames,
        device=batch.codes.device,
    )

    # Only apply if there are actual prompt regions to mask
    if not prompt_mask.all():
        if text_mask.dim() == 3:
            pm_text = prompt_mask.unsqueeze(1).expand_as(text_mask)
            text_mask = text_mask & pm_text
        elif text_mask.dim() == 2:
            pm_text = prompt_mask[:, :text_mask.shape[-1]]
            text_mask = text_mask & pm_text

        if audio_mask.dim() == 3:
            pm_audio = prompt_mask.unsqueeze(1).expand_as(audio_mask)
            audio_mask = audio_mask & pm_audio
        elif audio_mask.dim() == 2:
            pm_audio = prompt_mask[:, :audio_mask.shape[-1]]
            audio_mask = audio_mask & pm_audio

    return text_mask, audio_mask


# ---------------------------------------------------------------------------
# Optimizer param groups (PersonaPlex: separate LR for depth/temporal)
# ---------------------------------------------------------------------------

def _build_param_groups(model, temporal_lr: float, depth_lr: float, weight_decay: float):
    """Build optimizer param groups with separate LR for temporal and depth transformers.

    PersonaPlex: depth transformer LR = 4e-6, temporal transformer LR = 2e-6.
    """
    temporal_params = []
    depth_params = []
    other_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "depformer" in name:
            depth_params.append(param)
        elif "transformer" in name:
            temporal_params.append(param)
        else:
            other_params.append(param)

    groups = []
    if temporal_params:
        groups.append({
            "params": temporal_params,
            "lr": temporal_lr,
            "weight_decay": weight_decay,
        })
    if depth_params:
        groups.append({
            "params": depth_params,
            "lr": depth_lr,
            "weight_decay": weight_decay,
        })
    if other_params:
        groups.append({
            "params": other_params,
            "lr": temporal_lr,
            "weight_decay": weight_decay,
        })

    n_temporal = sum(p.numel() for p in temporal_params)
    n_depth = sum(p.numel() for p in depth_params)
    n_other = sum(p.numel() for p in other_params)
    main_logger_info(
        f"Param groups: temporal={n_temporal:,d} @ lr={temporal_lr}, "
        f"depth={n_depth:,d} @ lr={depth_lr}, "
        f"other={n_other:,d} @ lr={temporal_lr}"
    )
    return groups


# ---------------------------------------------------------------------------
# Evaluation with prompt masking
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_prompt_mask(model, data_loader, state, args):
    """Evaluate with system prompt loss masking applied (upstream eval ignores it)."""
    model.eval()

    total_loss = 0.0
    num_samples = 0
    max_samples = 40 // max(get_world_size(), 1)

    for batch in data_loader:
        if num_samples >= max_samples:
            break

        codes = batch.codes
        condition_tensors = None
        if batch.condition_attributes is not None and model.condition_provider is not None:
            condition_tensors = model.condition_provider.prepare(batch.condition_attributes)

        output = model(codes=codes, condition_tensors=condition_tensors)

        text_mask, audio_mask = _apply_prompt_mask(batch, output.text_mask, output.mask)

        text_loss = compute_loss_with_mask(
            output.text_logits,
            codes[:, :model.audio_offset],
            text_mask,
            mode="text",
            text_padding_weight=args.text_padding_weight,
            text_padding_ids={model.text_padding_token_id, model.end_of_text_padding_id},
        )
        audio_loss = compute_loss_with_mask(
            output.logits,
            codes[:, model.audio_offset:model.audio_offset + model.dep_q],
            audio_mask,
            mode="audio",
            first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
        )

        total_loss += (text_loss + audio_loss).item()
        num_samples += 1

    avg_loss = total_loss / max(num_samples, 1)
    avg_loss = avg_aggregate(avg_loss)

    state.this_eval_loss = avg_loss
    state.this_eval_perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")

    model.train()


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

def _find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Find the latest checkpoint directory in run_dir."""
    ckpt_dirs = sorted(
        run_dir.glob("checkpoints/step_*"),
        key=lambda p: int(p.name.split("_")[1]),
    )
    return ckpt_dirs[-1] if ckpt_dirs else None


def _save_resume_state(run_dir: Path, step: int, scheduler):
    """Save training state for resume (rank 0 only)."""
    if get_rank() == 0:
        torch.save(
            {"step": step, "scheduler_state": scheduler.state_dict()},
            run_dir / "resume_state.pt",
        )


def _load_resume_state(run_dir: Path) -> dict | None:
    """Load training state for resume."""
    path = run_dir / "resume_state.pt"
    if path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    return None


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------

def train(config: str):
    custom = _load_custom_config(config)
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=True)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, custom, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, custom: dict, exit_stack: ExitStack):
    set_random_seed(args.seed)
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Disable torch.compile/dynamo — causes triton race conditions in multi-GPU training
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")
        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. "
            "This message should only be displayed when testing."
        )

    # Init run dir with resume detection
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    resume_state = None
    if run_dir.exists() and is_torchrun():
        latest_ckpt = _find_latest_checkpoint(run_dir)
        if latest_ckpt is not None:
            resume_state = _load_resume_state(run_dir)
            if resume_state:
                main_logger_info(
                    f"Resuming from step {resume_state['step']} "
                    f"(checkpoint: {latest_ckpt})"
                )
            else:
                main_logger_info(
                    f"Found checkpoint {latest_ckpt} but no resume_state.pt. "
                    f"Starting fresh — set moshi_paths.moshi_path to checkpoint "
                    f"file for weight resume."
                )
        elif args.overwrite_run_dir:
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)
    elif not run_dir.exists() and is_torchrun():
        pass  # Fresh start

    # Validate finetuning mode
    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning."

    dist.barrier()
    run_dir.mkdir(exist_ok=True, parents=True)

    args_path = run_dir / "args.yaml"
    if not args_path.exists():
        args.save(args_path)

    main_logger_info(f"TrainArgs: {pprint.pformat(dataclasses.asdict(args))}")

    # Loggers
    metrics_logger: MetricsLogger = MetricsLogger(
        run_dir, tag="train", is_master=get_rank() == 0,
        wandb_args=args.wandb, config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(metrics_logger, "metrics_logger"))

    eval_logger: MetricsLogger = MetricsLogger(
        run_dir, tag="eval", is_master=get_rank() == 0,
        wandb_args=args.wandb, config=dataclasses.asdict(args),
    )
    exit_stack.enter_context(logged_closing(eval_logger, "eval_logger"))

    # Load Mimi and Moshi
    main_logger_info("Loading Mimi and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # Load and shard model
    model = get_fsdp_model(args, checkpoint_info)

    spm = checkpoint_info.get_text_tokenizer()

    interleaver = Interleaver(
        spm,
        mimi.frame_rate,
        model.text_padding_token_id,
        model.end_of_text_padding_id,
        model.zero_token_id,
        keep_main_only=True,
    )
    interleaved_tokenizer = InterleavedTokenizer(
        mimi, interleaver, duration_sec=args.duration_sec
    )

    # Data loaders
    data_loader = build_data_loader(
        instruct_tokenizer=interleaved_tokenizer,
        args=args.data,
        batch_size=args.batch_size,
        seed=args.seed,
        rank=get_rank(),
        world_size=get_world_size(),
        is_eval=False,
    )

    if args.do_eval:
        eval_data_loader = build_data_loader(
            instruct_tokenizer=interleaved_tokenizer,
            args=args.data,
            batch_size=args.batch_size,
            seed=None,
            rank=get_rank(),
            world_size=get_world_size(),
            is_eval=True,
        )

    # Mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    # Optimizer with separate LR for depth/temporal (PersonaPlex)
    depth_lr = custom.get("depth_lr", args.optim.lr * 2)
    warmup_steps = custom.get("warmup_steps", max(1, int(args.max_steps * args.optim.pct_start)))

    main_logger_info(
        f"PersonaPlex config: temporal_lr={args.optim.lr}, depth_lr={depth_lr}, "
        f"warmup_steps={warmup_steps}, cosine_annealing to step {args.max_steps}"
    )

    param_groups = _build_param_groups(
        model,
        temporal_lr=args.optim.lr,
        depth_lr=depth_lr,
        weight_decay=args.optim.weight_decay,
    )
    optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-08)

    # Cosine annealing with linear warmup (PersonaPlex)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_steps - warmup_steps,
        eta_min=1e-7,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    state = TrainState(args.max_steps)

    # Checkpointer
    if args.do_ckpt:
        checkpointer = Checkpointer(
            model=model,
            state=state,
            config=lm_config,
            run_dir=run_dir,
            optimizer=optimizer,
            num_ckpt_keep=args.num_ckpt_keep,
            full_finetuning=args.full_finetuning,
        )

    # Resume from checkpoint
    if resume_state is not None:
        start_step = resume_state["step"]
        scheduler.load_state_dict(resume_state["scheduler_state"])
        state.step = start_step
        main_logger_info(
            f"Resumed: step={start_step}, scheduler restored. "
            f"Note: optimizer momentum rebuilt from scratch."
        )

    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # Training loop
    model.train()
    torch.cuda.empty_cache()

    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    main_logger_info(
        f"Total params: {total_params:,d}, "
        f"Trainable: {trainable_params:,d} ({trainable_params / total_params * 100:.1f}%)"
    )

    while state.step < args.max_steps:
        state.start_step()
        is_last_step = state.step == args.max_steps

        optimizer.zero_grad()

        loss = torch.tensor([0.0], device="cuda")
        n_batch_tokens: int = 0
        n_real_tokens: int = 0

        for i in range(args.num_microbatches):
            batch = next(data_loader)
            codes = batch.codes

            condition_tensors = None
            if batch.condition_attributes is not None and model.condition_provider is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            output = model(codes=codes, condition_tensors=condition_tensors)

            # Apply system prompt loss masking (PersonaPlex)
            text_mask, audio_mask = _apply_prompt_mask(
                batch, output.text_mask, output.mask
            )

            text_loss = compute_loss_with_mask(
                output.text_logits,
                codes[:, :model.audio_offset],
                text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                },
            )
            audio_loss = compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset:model.audio_offset + model.dep_q],
                audio_mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )

            mb_loss = text_loss + audio_loss
            mb_loss.backward()

            loss += mb_loss.detach()
            n_batch_tokens += text_mask.numel() + audio_mask.numel()
            n_real_tokens += (
                torch.sum(text_mask).item() + torch.sum(audio_mask).item()
            )

            if i < args.num_microbatches - 1:
                assert args.num_microbatches > 1
                torch.cuda.synchronize()

        if args.num_microbatches > 1:
            loss /= args.num_microbatches
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.div_(args.num_microbatches)

        upcast_mixed_precision(model.parameters(), optim_dtype=optim_dtype)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
        downcast_mixed_precision(model.parameters(), param_dtype=param_dtype)

        last_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        loss_item = loss.item()
        avg_loss = avg_aggregate(loss_item)

        if args.do_eval and (
            (args.eval_freq > 0 and state.step % args.eval_freq == 0) or is_last_step
        ):
            evaluate_with_prompt_mask(model, eval_data_loader, state, args)

            eval_logs = get_eval_logs(
                state.step, avg_loss,
                state.this_eval_perplexity, state.this_eval_loss,
            )
            main_logger_info(eval_log_msg(eval_logs))
            eval_logger.log(eval_logs, step=state.step)

        state.end_step(n_batch_tokens)

        if state.step % args.log_freq == 0:
            train_logs = get_train_logs(
                state, avg_loss, n_real_tokens, last_lr,
                torch.cuda.max_memory_allocated(),
                torch.cuda.memory_allocated(),
                args,
            )
            main_logger_info(train_log_msg(state, logs=train_logs, loss=avg_loss))
            metrics_logger.log(train_logs, step=state.step)

        if args.do_ckpt and (
            (args.ckpt_freq > 0 and state.step % args.ckpt_freq == 0) or is_last_step
        ):
            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )
            _save_resume_state(run_dir, state.step, scheduler)

    main_logger_info("done!")


if __name__ == "__main__":
    fire.Fire(train)
