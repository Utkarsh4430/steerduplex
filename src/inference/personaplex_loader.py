"""PersonaPlex-specific model loading for steerd.

PersonaPlex (nvidia/personaplex-7b-v1) cannot be loaded via the standard
CheckpointInfo.get_moshi() path because its config.json only contains
{"model_type": "personaplex", "version": "7b-v1"} — not the architecture
params needed to instantiate LMModel.

This module provides a dedicated loader that:
  - Uses the correct architecture constants (n_q=16, dep_q=16, 17-element delays)
  - Applies weight patching to expand dep_q=8 checkpoint weights to dep_q=16
  - Downloads model weights from HF if no local path is given

Architecture constants are sourced from:
  personaplex/moshi/moshi/models/loaders.py
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from moshi.models.lm import LMModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PersonaPlex constants
# ---------------------------------------------------------------------------

PERSONAPLEX_REPO = "nvidia/personaplex-7b-v1"
MOSHI_NAME = "model.safetensors"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
TEXT_TOKENIZER_NAME = "tokenizer_spm_32k_3.model"

# Number of mimi codebooks used for audio decode (same for both moshi and personaplex).
# mimi.set_num_codebooks(8) is called at load time; decode always uses the first 8.
MIMI_CODEBOOKS = 8

# Full architecture kwargs for PersonaPlex 7B-v1.
# dep_q is listed as 8 here (matching the checkpoint), but get_personaplex_lm
# overrides it to 16 at model instantiation time.
_PERSONAPLEX_LM_KWARGS = {
    "dim": 4096,
    "text_card": 32000,
    "existing_text_padding_id": 3,
    "n_q": 16,
    "dep_q": 8,        # checkpoint has 8; overridden to 16 at init
    "card": 2048,
    "num_heads": 32,
    "num_layers": 32,
    "hidden_scale": 4.125,
    "causal": True,
    "layer_scale": None,
    "context": 3000,
    "max_period": 10000,
    "gating": "silu",
    "norm": "rms_norm_f32",
    "positional_embedding": "rope",
    "depformer_dim": 1024,
    "depformer_dim_feedforward": int(4.125 * 1024),
    "depformer_num_heads": 16,
    "depformer_num_layers": 6,
    "depformer_layer_scale": None,
    "depformer_multi_linear": True,
    "depformer_context": 8,
    "depformer_max_period": 10000,
    "depformer_gating": "silu",
    "depformer_pos_emb": "none",
    "depformer_weights_per_step": True,
    "delays": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def is_personaplex(hf_repo_or_model_type: str) -> bool:
    """Return True if the string identifies a PersonaPlex model."""
    return "personaplex" in hf_repo_or_model_type.lower()


# ---------------------------------------------------------------------------
# LM loader
# ---------------------------------------------------------------------------

def load_personaplex_lm(
    hf_repo: str = PERSONAPLEX_REPO,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    local_weights_path: Optional[str | Path] = None,
) -> LMModel:
    """Load the PersonaPlex LM with correct architecture and weight patching.

    The checkpoint was trained with dep_q=8 depformer steps.  At inference time
    PersonaPlex runs with dep_q=16 — the extra steps are initialised by copying
    the existing weights (Patch 2) and expanding the self-attention projection
    (Patch 1).

    Args:
        hf_repo: HuggingFace repo ID.  Used to download weights when
            ``local_weights_path`` is None.
        device: Target device for the loaded model.
        dtype: Model dtype (default bfloat16).
        local_weights_path: If provided, load from this safetensors file
            instead of downloading from HF.

    Returns:
        Loaded, eval-mode LMModel with dep_q=16.
    """
    # Resolve weights path
    if local_weights_path is not None:
        weights_path = str(local_weights_path)
    else:
        logger.info("Downloading PersonaPlex weights from %s", hf_repo)
        weights_path = hf_hub_download(hf_repo, MOSHI_NAME)
    logger.info("Loading PersonaPlex LM from %s", weights_path)

    # Build kwargs with dep_q=16
    lm_kwargs = dict(_PERSONAPLEX_LM_KWARGS)
    lm_kwargs["dep_q"] = 16

    # Instantiate on meta device to avoid wasting memory on random init
    model = LMModel(device="meta", dtype=dtype, **lm_kwargs)

    # Load state_dict
    dev = torch.device(device) if isinstance(device, str) else device
    load_device = dev.type if dev.type != "mps" else "cpu"
    state_dict = load_file(weights_path, device=load_device)

    # ------------------------------------------------------------------
    # Patch 1: expand depformer self_attn projection weights.
    # The checkpoint has dep_q=8 self_attn rows; the model expects dep_q=16.
    # We duplicate the existing rows to fill the expanded dimension.
    # ------------------------------------------------------------------
    model_sd = model.state_dict()
    for name, tensor in list(state_dict.items()):
        if "depformer" in name and "self_attn" in name and name in model_sd:
            if tensor.shape != model_sd[name].shape:
                logger.debug("PersonaPlex Patch1: expanding %s", name)
                state_dict[name] = torch.cat([tensor, tensor], dim=0)

    # ------------------------------------------------------------------
    # Patch 2: fill missing codebook-indexed keys by copying layers 0..7 → 8..15.
    # Affected module groups: gating, linears, depformer_in, depformer_emb.
    # ------------------------------------------------------------------
    to_replace = ["gating", "linears", "depformer_in", "depformer_emb"]
    for name in model_sd.keys():
        if name in state_dict:
            continue
        replaced = False
        for old, new in zip(range(8), range(8, 16)):
            for rep in to_replace:
                needle = f"{rep}.{new}."
                if needle in name:
                    src = name.replace(needle, f"{rep}.{old}.")
                    if src in state_dict:
                        logger.debug("PersonaPlex Patch2: %s <- %s", name, src)
                        state_dict[name] = state_dict[src]
                        replaced = True
                    break
            if replaced:
                break
        if not replaced:
            logger.debug("PersonaPlex: key %s not in checkpoint (random init)", name)

    # Move all tensors to the target device and dtype
    for key in state_dict:
        state_dict[key] = state_dict[key].to(device=dev, dtype=dtype)

    model.load_state_dict(state_dict, strict=False, assign=True)
    model.eval()
    return model.to(device=device, dtype=dtype)
