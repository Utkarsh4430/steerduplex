"""System prompt loss masking for moshi-finetune.

This module provides a patched loss function that masks out the
system prompt region from the training loss. During the prompt
region (voice prompt + system text), the model shouldn't be
penalized for its predictions since those tokens are conditioning.

FOR FULL TRAINING ONLY — the pilot bakes prompts into audio and
trains without masking, which is simpler but less optimal.

To use: apply the patch in train.py before the training loop:
    from training.loss_masking import patch_loss_for_prompt_masking
    patch_loss_for_prompt_masking(args, data_loader)

The patch works by modifying the loss mask to zero out positions
in the prompt region (0 to prompt_end_sec for each sample).
"""

import torch
import torch.nn.functional as F


def compute_loss_with_prompt_mask(
    logits: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    mode: str,
    first_codebook_weight_multiplier: float = 1.0,
    text_padding_weight: float = 1.0,
    text_padding_ids: set | None = None,
    prompt_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Loss computation with additional prompt masking.

    Same as moshi-finetune's compute_loss_with_mask, but with an
    additional prompt_mask that zeros out the system prompt region.

    Args:
        prompt_mask: Optional [B, T] bool tensor. True = keep, False = mask out.
                     If None, behaves identically to the original loss function.
    """
    target = torch.where(target_mask, target, torch.zeros_like(target))
    weights = target_mask.float()

    if mode == "audio":
        weights[:, 0] *= first_codebook_weight_multiplier
    elif mode == "text":
        if text_padding_ids is not None:
            for pid in text_padding_ids:
                weights[target == pid] *= text_padding_weight

    # Apply prompt mask: zero out loss in the prompt region
    if prompt_mask is not None:
        # prompt_mask shape: [B, T] — expand to match weights shape
        if weights.dim() == 3:  # [B, Q, T]
            prompt_mask = prompt_mask.unsqueeze(1).expand_as(weights)
        weights = weights * prompt_mask.float()

    logits = logits.reshape(-1, logits.size(-1)).float()
    target = target.reshape(-1)
    weights = weights.reshape(-1)

    loss = F.cross_entropy(logits, target, reduction="none")
    loss_weighted = torch.where(weights > 0.0, loss * weights, torch.zeros_like(loss))
    total = loss_weighted.sum() / weights.sum().clamp(min=1.0)

    return total


def create_prompt_mask(
    batch_size: int,
    seq_len: int,
    prompt_end_frames: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Create a boolean mask that is False during the prompt region.

    Args:
        batch_size: Batch size
        seq_len: Sequence length in frames
        prompt_end_frames: Per-sample prompt end frame index
        device: Target device

    Returns:
        Boolean tensor [B, T]: True = compute loss, False = mask out
    """
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    for i, end_frame in enumerate(prompt_end_frames):
        if end_frame > 0:
            mask[i, :end_frame] = False
    return mask


# Frame rate for Mimi codec
MIMI_FRAME_RATE = 12.5


def seconds_to_frames(seconds: float) -> int:
    """Convert seconds to Mimi frame index."""
    return int(seconds * MIMI_FRAME_RATE)
