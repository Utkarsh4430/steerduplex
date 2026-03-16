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

# Mimi codec frame rate
MIMI_FRAME_RATE = 12.5


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


def seconds_to_frames(seconds: float) -> int:
    """Convert seconds to Mimi frame index."""
    return int(seconds * MIMI_FRAME_RATE)
