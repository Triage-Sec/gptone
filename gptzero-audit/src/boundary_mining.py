"""Utilities for active boundary mining against GPTZero-style outputs."""

from __future__ import annotations

import re
from typing import List, Sequence


_CONTRACTIONS = (
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bdid not\b", "didn't"),
    (r"\bcan not\b", "can't"),
    (r"\bcannot\b", "can't"),
    (r"\bwill not\b", "won't"),
    (r"\bit is\b", "it's"),
    (r"\bthat is\b", "that's"),
    (r"\bthere is\b", "there's"),
    (r"\bi am\b", "I'm"),
    (r"\bi have\b", "I've"),
    (r"\bwe are\b", "we're"),
    (r"\bthey are\b", "they're"),
)

_DECONTRACTIONS = (
    (r"\bdon't\b", "do not"),
    (r"\bdoesn't\b", "does not"),
    (r"\bdidn't\b", "did not"),
    (r"\bcan't\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bit's\b", "it is"),
    (r"\bthat's\b", "that is"),
    (r"\bthere's\b", "there is"),
    (r"\bi'm\b", "I am"),
    (r"\bi've\b", "I have"),
    (r"\bwe're\b", "we are"),
    (r"\bthey're\b", "they are"),
)

_CASUAL_TO_FORMAL = (
    (r"\ba lot of\b", "a significant amount of"),
    (r"\ba lot\b", "substantially"),
    (r"\bkind of\b", "somewhat"),
    (r"\bsort of\b", "to some extent"),
    (r"\breally\b", "considerably"),
    (r"\bpretty\b", "notably"),
    (r"\bkids\b", "children"),
    (r"\bguy\b", "individual"),
    (r"\bgot\b", "obtained"),
    (r"\bhuge\b", "substantial"),
)

_FORMAL_TO_CASUAL = (
    (r"\bmoreover\b", "also"),
    (r"\bfurthermore\b", "also"),
    (r"\btherefore\b", "so"),
    (r"\bconsequently\b", "so"),
    (r"\bhowever\b", "but"),
    (r"\butilize\b", "use"),
    (r"\bdemonstrate\b", "show"),
    (r"\bsignificant\b", "big"),
    (r"\bapproximately\b", "about"),
    (r"\bindividuals\b", "people"),
)


def normalize_text(text: str) -> str:
    """Normalize spacing while preserving punctuation."""
    return re.sub(r"\s+", " ", text.strip())


def split_sentences(text: str) -> List[str]:
    """Split text on sentence boundaries."""
    text = normalize_text(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def join_sentences(sentences: Sequence[str]) -> str:
    return normalize_text(" ".join(s.strip() for s in sentences if s.strip()))


def _apply_case_like(source_fragment: str, replacement: str) -> str:
    if source_fragment.isupper():
        return replacement.upper()
    if source_fragment[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _sub_wordwise(text: str, pattern: str, replacement: str) -> str:
    return re.sub(
        pattern,
        lambda m: _apply_case_like(m.group(0), replacement),
        text,
        flags=re.IGNORECASE,
    )


def contractionize(text: str) -> str:
    out = text
    for pattern, replacement in _CONTRACTIONS:
        out = _sub_wordwise(out, pattern, replacement)
    return normalize_text(out)


def decontractionize(text: str) -> str:
    out = text
    for pattern, replacement in _DECONTRACTIONS:
        out = _sub_wordwise(out, pattern, replacement)
    return normalize_text(out)


def casualize(text: str) -> str:
    out = text
    for pattern, replacement in _FORMAL_TO_CASUAL:
        out = _sub_wordwise(out, pattern, replacement)
    return normalize_text(out)


def formalize(text: str) -> str:
    out = text
    for pattern, replacement in _CASUAL_TO_FORMAL:
        out = _sub_wordwise(out, pattern, replacement)
    return normalize_text(out)


def _prepend_if_missing(text: str, sentence: str) -> str:
    if sentence.lower() in text.lower():
        return normalize_text(text)
    return normalize_text(f"{sentence} {text}")


def _append_if_missing(text: str, sentence: str) -> str:
    if sentence.lower() in text.lower():
        return normalize_text(text)
    return normalize_text(f"{text} {sentence}")


def humanize_variants(text: str) -> List[str]:
    """Generate text variants likely to shift toward human-like output."""
    base = normalize_text(text)
    variants = [
        contractionize(base),
        casualize(base),
        _prepend_if_missing(base, "From what I saw on my block last week, this matches real day to day behavior."),
        _append_if_missing(base, "Honestly, this is the part people in my neighborhood talk about the most."),
    ]

    sents = split_sentences(base)
    if sents:
        lead = sents[0]
        variants.append(
            join_sentences(
                [f"I used to think this sounded abstract, but {lead[:1].lower() + lead[1:] if len(lead) > 1 else lead}"]
                + sents[1:]
            )
        )
        variants.append(
            join_sentences(
                sents
                + ["You can see the impact in small details, not just in the headline numbers."]
            )
        )

    # Deduplicate and remove unchanged variants.
    deduped = []
    seen = set()
    for v in variants:
        if v == base or not v:
            continue
        if v not in seen:
            deduped.append(v)
            seen.add(v)
    return deduped


def ai_formalize_variants(text: str) -> List[str]:
    """Generate text variants likely to shift toward AI-like output."""
    base = normalize_text(text)
    variants = [
        decontractionize(base),
        formalize(base),
        _prepend_if_missing(base, "It is important to note that a structured implementation framework is required."),
        _append_if_missing(base, "Accordingly, measurable outcomes should be evaluated through a standardized process."),
    ]

    sents = split_sentences(base)
    if sents:
        variants.append(
            join_sentences(
                ["The following analysis presents a methodical overview of the issue."]
                + sents
            )
        )
        variants.append(
            join_sentences(
                sents
                + ["Furthermore, the operational implications should be assessed across relevant stakeholders."]
            )
        )

    deduped = []
    seen = set()
    for v in variants:
        if v == base or not v:
            continue
        if v not in seen:
            deduped.append(v)
            seen.add(v)
    return deduped


def generate_directional_variants(text: str, direction: str) -> List[str]:
    """
    Generate mutation variants for a boundary-mining step.

    direction:
      - "toward_human": reduce AI-ness
      - "toward_ai": increase AI-ness
    """
    if direction == "toward_human":
        return humanize_variants(text)
    if direction == "toward_ai":
        return ai_formalize_variants(text)
    raise ValueError(f"Unknown direction: {direction}")


def boundary_distance(prob_ai: float) -> float:
    """Absolute distance from the decision midpoint."""
    return abs(float(prob_ai) - 0.5)


def in_boundary_band(prob_ai: float, low: float, high: float) -> bool:
    return float(low) <= float(prob_ai) <= float(high)


def adaptive_boundary_band(
    *,
    step_idx: int,
    max_steps: int,
    initial_low: float,
    initial_high: float,
    tight_low: float = 0.40,
    tight_high: float = 0.60,
) -> tuple[float, float]:
    """
    Interpolate from a wider boundary band to a tighter one across steps.

    This keeps early exploration broad while making later selections stricter.
    """
    if max_steps <= 1:
        return float(initial_low), float(initial_high)
    progress = min(max(step_idx, 0), max_steps - 1) / float(max_steps - 1)
    low = float(initial_low) + (float(tight_low) - float(initial_low)) * progress
    high = float(initial_high) + (float(tight_high) - float(initial_high)) * progress
    return low, high
