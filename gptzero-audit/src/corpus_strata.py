"""Corpus strata taxonomy and helpers for large-scale empirical evaluation."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence


DEFAULT_STRATA: Sequence[str] = (
    "human_pre_llm",
    "human_conversational",
    "ai_instructional",
    "ai_formal",
    "mixed_blended",
    "boundary_near",
)


def infer_stratum(record: Mapping[str, object]) -> str:
    """
    Infer a corpus stratum from dataset record metadata.

    This function uses explicit `stratum` when available and falls back
    to heuristics from sample ID, labels, provenance and style metadata.
    """
    explicit = str(record.get("stratum", "")).strip()
    if explicit:
        return explicit

    sample_id = str(record.get("sample_id", "")).lower()
    label = str(record.get("ground_truth_label", "")).lower()
    provenance = str(record.get("provenance", "")).lower()
    source = str(record.get("source", "")).lower()
    style_variant = str(record.get("style_variant", "")).lower()

    if any(sample_id.startswith(prefix) for prefix in ("bound_", "grad_", "hmm_", "ppl_", "mem_", "mine_")):
        return "boundary_near"

    if label == "mixed" or style_variant == "mixed" or "mixed_" in sample_id:
        return "mixed_blended"

    if label == "human":
        if "archived" in provenance or "2022" in provenance or "pre-chatgpt" in provenance:
            return "human_pre_llm"
        return "human_conversational"

    if label == "ai":
        if any(token in sample_id for token in ("formal", "technical", "brief", "template", "memo")):
            return "ai_formal"
        return "ai_instructional"

    if style_variant == "human":
        return "human_conversational"
    if style_variant == "ai":
        return "ai_formal"

    if source.startswith("ui_reason_guided"):
        if style_variant == "mixed":
            return "mixed_blended"
        if style_variant == "human":
            return "human_conversational"
        if style_variant == "ai":
            return "ai_formal"

    return "ai_instructional"


def annotate_with_stratum(record: Mapping[str, object]) -> Dict[str, object]:
    """Return a shallow copy with `stratum` populated."""
    out = dict(record)
    out["stratum"] = infer_stratum(record)
    return out
