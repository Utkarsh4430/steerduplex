"""System prompt loss masking for SteerDuplex training.

Masks out the system prompt region (voice prompt + text prompt)
from the training loss. Following PersonaPlex: "we mask out loss
backpropagation to the system prompt."

Used by training/train.py which calls create_prompt_mask() and
seconds_to_frames() each training step to build per-batch masks.

The mask zeros out loss for frames 0..prompt_end_sec, where
prompt_end_sec is stored in each sample's JSON metadata.
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
