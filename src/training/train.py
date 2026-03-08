"""SteerDuplex training script.

Adapted from moshi-finetune's train.py with integrated system prompt loss masking.
Requires the `finetune` package (installed via pip from moshi-finetune).

Key differences from upstream moshi-finetune:
- System prompt loss masking: zeros out loss in the voice_prompt + text_prompt region
  (PersonaPlex: "we mask out loss backpropagation to the system prompt")
- Token weighting follows PersonaPlex:
  - Non-semantic audio tokens downweighted by 0.02 (first_codebook_weight_multiplier=50)
  - Padded text tokens downweighted by 0.3

Usage:
    # Via launch.sh (recommended)
    bash training/launch.sh configs/pilot_training.yaml 8

    # Direct invocation
    torchrun --nproc-per-node 8 -m training.train configs/pilot_training.yaml
"""

import dataclasses
import json
import logging
import os
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler

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
from finetune.eval import evaluate
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

# Mimi codec frame rate
MIMI_FRAME_RATE = 12.5


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def _get_prompt_end_frames(batch) -> list[int]:
    """Extract prompt_end_sec from batch condition_attributes and convert to frame indices.

    format_dataset.py stores prompt_end_sec in text_conditions, which flows through
    moshi-finetune's data loader into ConditionAttributes.text. If not available,
    returns zeros (no masking applied).
    """
    prompt_end_frames = []
    batch_size = batch.codes.shape[0]

    if batch.condition_attributes is not None:
        for attr in batch.condition_attributes:
            prompt_end_sec = 0.0
            # text_conditions → ConditionAttributes.text (Dict[str, Optional[str]])
            if hasattr(attr, 'text') and attr.text:
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


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")


def _train(args: TrainArgs, exit_stack: ExitStack):
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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

    # Init run dir
    main_logger_info(f"Run dir: {args.run_dir}")
    run_dir = Path(args.run_dir)

    if is_torchrun():
        if run_dir.exists() and not args.overwrite_run_dir:
            raise RuntimeError(
                f"Run dir {run_dir} already exists. "
                f"Either rename `run_dir` or remove {run_dir}."
            )
        elif run_dir.exists():
            main_logger_info(f"Removing run dir {run_dir}...")
            shutil.rmtree(run_dir)

    if args.full_finetuning:
        assert not args.lora.enable, "LoRA should not be enabled for full finetuning."
    else:
        assert args.lora.enable, "LoRA should be enabled for partial finetuning"

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

    assert args.lora is not None, "`args.lora` should be set to a valid value."

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.optim.lr,
        betas=(0.9, 0.95),
        eps=1e-08,
        weight_decay=args.optim.weight_decay,
    )

    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.optim.lr,
        total_steps=args.max_steps,
        pct_start=args.optim.pct_start,
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

    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # Training loop
    model.train()
    torch.cuda.empty_cache()

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
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            # Build system prompt mask for this batch
            prompt_end_frames = _get_prompt_end_frames(batch)
            seq_len = codes.shape[-1]
            prompt_mask = create_prompt_mask(
                batch_size=codes.shape[0],
                seq_len=seq_len,
                prompt_end_frames=prompt_end_frames,
                device=codes.device,
            )

            output = model(codes=codes, condition_tensors=condition_tensors)

            # Apply prompt mask to both text and audio loss masks
            text_mask = output.text_mask
            audio_mask = output.mask
            if prompt_mask.any():
                # Expand prompt_mask to match mask dimensions
                # text_mask shape: [B, 1, T], audio_mask shape: [B, Q, T]
                pm = prompt_mask
                if text_mask.dim() == 3:
                    pm_text = pm.unsqueeze(1).expand_as(text_mask)
                    text_mask = text_mask & pm_text
                elif text_mask.dim() == 2:
                    pm_text = pm[:, :text_mask.shape[-1]]
                    text_mask = text_mask & pm_text

                if audio_mask.dim() == 3:
                    pm_audio = pm.unsqueeze(1).expand_as(audio_mask)
                    audio_mask = audio_mask & pm_audio
                elif audio_mask.dim() == 2:
                    pm_audio = pm[:, :audio_mask.shape[-1]]
                    audio_mask = audio_mask & pm_audio

            text_loss = compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
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
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
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
                if p.requires_grad:
                    assert p.grad is not None
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
            evaluate(model, eval_data_loader, state, args)

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

    main_logger_info("done!")


if __name__ == "__main__":
    fire.Fire(train)
