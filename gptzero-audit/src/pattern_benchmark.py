"""Perturbation-based pattern benchmark utilities."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict
from statistics import mean
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

from src.boundary_mining import generate_directional_variants


def _split_sentences(text: str) -> List[str]:
    chunks = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.strip()) if part.strip()]
    if chunks:
        return chunks
    return [part.strip() for part in text.split("\n") if part.strip()]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _jitter_case(text: str, rng: random.Random) -> str:
    words = text.split()
    if not words:
        return text
    idx = rng.randrange(len(words))
    token = words[idx]
    words[idx] = token.upper() if token.lower() == token else token.lower()
    return " ".join(words)


def _drop_middle_sentence(text: str) -> str:
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return text
    mid = len(sentences) // 2
    kept = [s for i, s in enumerate(sentences) if i != mid]
    return " ".join(kept)


def _shuffle_sentence_order(text: str, rng: random.Random) -> str:
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return text
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    return " ".join(shuffled)


def generate_variants(
    text: str,
    *,
    rng: random.Random,
    max_variants: int = 8,
    branching_factor: int = 4,
) -> List[Dict[str, object]]:
    """
    Generate deterministic perturbation variants used in pattern evaluation.

    Each variant dict includes:
      - variant_id
      - perturbation_kind
      - perturbation_strength (0..1)
      - text
    """
    base_text = _normalize_whitespace(text)
    variants: List[Dict[str, object]] = [
        {
            "variant_id": "base",
            "perturbation_kind": "identity",
            "perturbation_strength": 0.0,
            "text": base_text,
        }
    ]

    directional = [("toward_human", 0.30), ("toward_ai", 0.30)]
    for direction, base_strength in directional:
        cands = generate_directional_variants(base_text, direction=direction)
        if not cands:
            continue
        for rank, pick in enumerate(cands[: max(1, int(branching_factor))]):
            if len(variants) >= max_variants:
                break
            strength = min(0.85, base_strength + 0.10 * float(rank))
            variants.append(
                {
                    "variant_id": f"{direction}_{rank}",
                    "perturbation_kind": direction,
                    "perturbation_strength": float(strength),
                    "text": _normalize_whitespace(pick),
                }
            )
        if len(variants) >= max_variants:
            break

    while len(variants) < max_variants:
        choice = len(variants) % 3
        if choice == 0:
            vt = _shuffle_sentence_order(base_text, rng)
            kind = "shuffle_sentences"
            strength = 0.4
        elif choice == 1:
            vt = _drop_middle_sentence(base_text)
            kind = "drop_sentence"
            strength = 0.5
        else:
            vt = _jitter_case(base_text, rng)
            kind = "case_jitter"
            strength = 0.2
        variants.append(
            {
                "variant_id": f"{kind}_{len(variants)}",
                "perturbation_kind": kind,
                "perturbation_strength": float(strength),
                "text": _normalize_whitespace(vt),
            }
        )

    dedup: List[Dict[str, object]] = []
    seen = set()
    for row in variants:
        payload = str(row.get("text", "")).strip()
        if payload in seen:
            continue
        seen.add(payload)
        dedup.append(row)

    return dedup[:max_variants]


def select_boundary_variants(
    variants: Sequence[Mapping[str, object]],
    predicted_probs: Sequence[float],
    *,
    target_count: int,
    kind_min_counts: Mapping[str, int] | None = None,
) -> List[Dict[str, object]]:
    """
    Select variants closest to the decision boundary with directional diversity.

    The selector keeps the identity variant (when present), then prefers:
      1) one variant below base probability,
      2) one variant above base probability,
      3) optional per-kind minimum quotas,
      4) remaining variants nearest to 0.5 with perturbation-kind diversity.
    """
    if len(variants) != len(predicted_probs):
        raise ValueError("variants and predicted_probs must have identical length.")
    if target_count <= 0:
        return []
    if not variants:
        return []

    rows = [dict(v) for v in variants]
    probs = [float(p) for p in predicted_probs]
    n = len(rows)
    normalized_min_counts: Dict[str, int] = {}
    if isinstance(kind_min_counts, Mapping):
        for raw_kind, raw_count in kind_min_counts.items():
            kind = str(raw_kind).strip()
            if not kind:
                continue
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                normalized_min_counts[kind] = count

    base_idx = 0
    for idx, row in enumerate(rows):
        if str(row.get("variant_id", "")) == "base":
            base_idx = idx
            break
    base_prob = probs[base_idx]

    selected: List[int] = [base_idx]
    remaining = [idx for idx in range(n) if idx != base_idx]

    def boundary_key(idx: int) -> tuple[float, float]:
        p = probs[idx]
        return (abs(p - 0.5), -abs(p - base_prob))

    lower = [idx for idx in remaining if probs[idx] < base_prob - 1e-6]
    upper = [idx for idx in remaining if probs[idx] > base_prob + 1e-6]
    for group in (lower, upper):
        if len(selected) >= target_count or not group:
            continue
        pick = min(group, key=boundary_key)
        selected.append(pick)
        remaining = [idx for idx in remaining if idx != pick]

    used_kinds = {str(rows[idx].get("perturbation_kind", "")) for idx in selected}
    if normalized_min_counts and len(selected) < target_count:
        selected_set = set(selected)
        for kind, min_count in sorted(normalized_min_counts.items(), key=lambda item: (-item[1], item[0])):
            if len(selected) >= target_count:
                break
            already = sum(1 for idx in selected if str(rows[idx].get("perturbation_kind", "")) == kind)
            need = max(0, int(min_count) - already)
            if need <= 0:
                continue
            candidates = [
                idx
                for idx in remaining
                if idx not in selected_set and str(rows[idx].get("perturbation_kind", "")) == kind
            ]
            for idx in sorted(candidates, key=boundary_key):
                if len(selected) >= target_count or need <= 0:
                    break
                selected.append(idx)
                selected_set.add(idx)
                used_kinds.add(kind)
                need -= 1
        remaining = [idx for idx in remaining if idx not in selected_set]

    for prefer_novel in (True, False):
        if len(selected) >= target_count:
            break
        ordered = sorted(remaining, key=boundary_key)
        for idx in ordered:
            if len(selected) >= target_count:
                break
            kind = str(rows[idx].get("perturbation_kind", ""))
            if prefer_novel and kind in used_kinds:
                continue
            selected.append(idx)
            used_kinds.add(kind)
            remaining = [rid for rid in remaining if rid != idx]

    selected = selected[:target_count]
    return [rows[idx] for idx in selected]


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _sign(value: float, eps: float = 1e-6) -> int:
    if abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def _js_divergence(probs_a: Sequence[float], probs_b: Sequence[float], bins: int = 20) -> float:
    if not probs_a or not probs_b:
        return 0.0

    hist_a = [0.0] * bins
    hist_b = [0.0] * bins
    for p in probs_a:
        idx = min(max(int(p * bins), 0), bins - 1)
        hist_a[idx] += 1.0
    for p in probs_b:
        idx = min(max(int(p * bins), 0), bins - 1)
        hist_b[idx] += 1.0

    total_a = sum(hist_a)
    total_b = sum(hist_b)
    if total_a <= 0 or total_b <= 0:
        return 0.0
    pa = [v / total_a for v in hist_a]
    pb = [v / total_b for v in hist_b]
    m = [(a + b) / 2.0 for a, b in zip(pa, pb)]

    def kl(p: Sequence[float], q: Sequence[float]) -> float:
        out = 0.0
        for pi, qi in zip(p, q):
            if pi <= 0 or qi <= 0:
                continue
            out += pi * math.log(pi / qi)
        return out

    return float((kl(pa, m) + kl(pb, m)) / 2.0)


def _transition_agreement(rows: Sequence[Mapping[str, object]]) -> float:
    groups: MutableMapping[str, List[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("base_sample_id", ""))].append(row)

    all_flags: List[float] = []
    for _, group in groups.items():
        ordered = sorted(group, key=lambda x: int(x.get("variant_order", 0)))
        if not ordered:
            continue
        base = ordered[0]
        base_gz = float(base.get("gptzero_prob", 0.0) or 0.0)
        base_sg = float(base.get("surrogate_prob", 0.0) or 0.0)
        for item in ordered[1:]:
            d_gz = float(item.get("gptzero_prob", 0.0) or 0.0) - base_gz
            d_sg = float(item.get("surrogate_prob", 0.0) or 0.0) - base_sg
            all_flags.append(1.0 if _sign(d_gz) == _sign(d_sg) else 0.0)
    return _safe_mean(all_flags)


def summarize_prediction_rows(rows: Iterable[Mapping[str, object]]) -> Dict[str, object]:
    """Summarize per-variant prediction rows into overall and per-stratum metrics."""
    materialized = [dict(r) for r in rows]
    by_stratum: MutableMapping[str, List[Dict[str, object]]] = defaultdict(list)
    for row in materialized:
        by_stratum[str(row.get("stratum", "unknown"))].append(row)

    def summarize_one(chunk: Sequence[Mapping[str, object]]) -> Dict[str, float]:
        matches = [1.0 if bool(r.get("match", False)) else 0.0 for r in chunk]
        abs_err = [
            abs(float(r.get("gptzero_prob", 0.0) or 0.0) - float(r.get("surrogate_prob", 0.0) or 0.0))
            for r in chunk
        ]
        gz_probs = [float(r.get("gptzero_prob", 0.0) or 0.0) for r in chunk]
        sg_probs = [float(r.get("surrogate_prob", 0.0) or 0.0) for r in chunk]
        return {
            "agreement_rate": _safe_mean(matches),
            "probability_mae": _safe_mean(abs_err),
            "transition_agreement": _transition_agreement(chunk),
            "probability_js_divergence": _js_divergence(gz_probs, sg_probs),
            "sample_count": float(len(chunk)),
        }

    overall = summarize_one(materialized)
    strata = {name: summarize_one(chunk) for name, chunk in sorted(by_stratum.items())}
    return {
        "overall": overall,
        "strata": strata,
        "variant_count": len(materialized),
        "base_sample_count": len({str(r.get("base_sample_id", "")) for r in materialized}),
    }


def summarize_prediction_rows_by_stratum_kind(
    rows: Iterable[Mapping[str, object]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Summarize prediction rows by stratum and perturbation kind.

    Transition agreement at this granularity is computed by comparing each
    non-base row against its sample's base row.
    """
    materialized = [dict(r) for r in rows]
    base_by_sample: Dict[str, Mapping[str, object]] = {}
    grouped: MutableMapping[str, MutableMapping[str, List[Dict[str, object]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in materialized:
        sample_id = str(row.get("base_sample_id", ""))
        if int(row.get("variant_order", 0)) == 0 and sample_id and sample_id not in base_by_sample:
            base_by_sample[sample_id] = row
        stratum = str(row.get("stratum", "unknown"))
        kind = str(row.get("perturbation_kind", "unknown"))
        grouped[stratum][kind].append(row)

    def summarize_one(chunk: Sequence[Mapping[str, object]]) -> Dict[str, float]:
        matches = [1.0 if bool(r.get("match", False)) else 0.0 for r in chunk]
        abs_err = [
            abs(float(r.get("gptzero_prob", 0.0) or 0.0) - float(r.get("surrogate_prob", 0.0) or 0.0))
            for r in chunk
        ]
        gz_probs = [float(r.get("gptzero_prob", 0.0) or 0.0) for r in chunk]
        sg_probs = [float(r.get("surrogate_prob", 0.0) or 0.0) for r in chunk]

        transition_flags: List[float] = []
        for row in chunk:
            order = int(row.get("variant_order", 0))
            if order <= 0:
                continue
            sample_id = str(row.get("base_sample_id", ""))
            base = base_by_sample.get(sample_id)
            if base is None:
                continue
            d_gz = float(row.get("gptzero_prob", 0.0) or 0.0) - float(base.get("gptzero_prob", 0.0) or 0.0)
            d_sg = float(row.get("surrogate_prob", 0.0) or 0.0) - float(base.get("surrogate_prob", 0.0) or 0.0)
            transition_flags.append(1.0 if _sign(d_gz) == _sign(d_sg) else 0.0)

        return {
            "agreement_rate": _safe_mean(matches),
            "probability_mae": _safe_mean(abs_err),
            "transition_agreement": _safe_mean(transition_flags),
            "probability_js_divergence": _js_divergence(gz_probs, sg_probs),
            "sample_count": float(len(chunk)),
            "transition_sample_count": float(len(transition_flags)),
        }

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for stratum, by_kind in sorted(grouped.items()):
        out[stratum] = {}
        for kind, chunk in sorted(by_kind.items()):
            out[stratum][kind] = summarize_one(chunk)
    return out
