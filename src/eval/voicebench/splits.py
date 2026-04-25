"""Registry of VoiceBench splits evaluated by this harness.

Only single-turn, non-MCQ subsets are included. sd-qa defaults to the `usa`
region; opt into more regions via --sdqa_regions.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


# All 11 sd-qa regions available on HuggingFace (hlt-lab/voicebench).
SDQA_ALL_REGIONS = (
    "aus", "gbr", "ind_n", "ind_s", "irl", "kenya",
    "nga", "nzl", "phl", "usa", "zaf",
)
SDQA_DEFAULT_REGIONS: Tuple[str, ...] = ("usa",)


@dataclass(frozen=True)
class SplitSpec:
    """One concrete evaluation unit — a subset + HF split combination."""
    name: str           # stable identifier, used for output dir (e.g. "wildvoice", "sd-qa/usa")
    subset: str         # HF subset name (e.g. "wildvoice")
    hf_split: str       # HF split name (e.g. "test", or a region code for sd-qa)
    evaluator: str      # one of: open, qa, ifeval, bbh, harm
    id_field: Optional[str]  # dataset field used to derive unique_id; None → use integer index
    carry_fields: Tuple[str, ...]  # fields to copy from dataset row into the output record


# All supported subsets. sd-qa is expanded to one SplitSpec per region at build time.
_SUBSET_REGISTRY = {
    "alpacaeval_full": dict(evaluator="open",  id_field=None, carry=("prompt",)),
    "commoneval":      dict(evaluator="open",  id_field=None, carry=("prompt",)),
    "wildvoice":       dict(evaluator="open",  id_field="conversation_hash", carry=("prompt", "conversation_hash")),
    "sd-qa":           dict(evaluator="qa",    id_field=None, carry=("prompt", "reference")),
    "ifeval":          dict(evaluator="ifeval", id_field="key", carry=("prompt", "key", "instruction_id_list", "kwargs")),
    "bbh":             dict(evaluator="bbh",   id_field="id", carry=("prompt", "reference", "id")),
    "advbench":        dict(evaluator="harm",  id_field=None, carry=("prompt",)),
}

SUPPORTED_SUBSETS: Tuple[str, ...] = tuple(_SUBSET_REGISTRY.keys())


def build_specs(subsets: List[str], sdqa_regions: List[str]) -> List[SplitSpec]:
    """Expand a list of subset names into concrete SplitSpec entries."""
    specs: List[SplitSpec] = []
    for subset in subsets:
        if subset not in _SUBSET_REGISTRY:
            raise ValueError(
                f"Unknown subset {subset!r}. Supported: {sorted(SUPPORTED_SUBSETS)}"
            )
        cfg = _SUBSET_REGISTRY[subset]
        if subset == "sd-qa":
            for region in sdqa_regions:
                if region not in SDQA_ALL_REGIONS:
                    raise ValueError(
                        f"Unknown sd-qa region {region!r}. "
                        f"Available: {sorted(SDQA_ALL_REGIONS)}"
                    )
                specs.append(SplitSpec(
                    name=f"sd-qa/{region}",
                    subset="sd-qa",
                    hf_split=region,
                    evaluator=cfg["evaluator"],
                    id_field=cfg["id_field"],
                    carry_fields=cfg["carry"],
                ))
        else:
            specs.append(SplitSpec(
                name=subset,
                subset=subset,
                hf_split="test",
                evaluator=cfg["evaluator"],
                id_field=cfg["id_field"],
                carry_fields=cfg["carry"],
            ))
    return specs


def unique_id_for(spec: SplitSpec, index: int, row: dict) -> str:
    """Build a stable unique_id for a dataset row under a given SplitSpec."""
    tag = spec.name.replace("/", "_")
    if spec.id_field is not None and row.get(spec.id_field) is not None:
        return f"{tag}__{row[spec.id_field]}"
    return f"{tag}__{index}"
