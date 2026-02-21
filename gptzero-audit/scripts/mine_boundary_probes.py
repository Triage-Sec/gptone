#!/usr/bin/env python3
"""Actively mine near-boundary probes and build a new dataset version."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.boundary_mining import (
    adaptive_boundary_band,
    boundary_distance,
    generate_directional_variants,
    in_boundary_band,
    normalize_text,
)
from src.client import GPTZeroClient
from src.dataset_io import load_jsonl, write_jsonl


PRIORITY_PREFIXES: Sequence[str] = (
    "style_",
    "train_mixed_",
    "train_human_",
    "bound_",
    "grad_",
    "hmm_",
    "mem_",
    "ppl_",
)


@dataclass
class ScoredText:
    text: str
    prob_ai: float
    predicted_class: str
    response: Dict[str, object]
    origin: str


@dataclass
class MinedCandidate:
    base_sample_id: str
    scored: ScoredText
    in_band: bool
    distance: float
    trace: Dict[str, object]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_required_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return load_jsonl(path)


def _coerce_prob_ai(result: Dict[str, object]) -> Optional[float]:
    raw_prob = result.get("average_generated_prob")
    if raw_prob is None:
        class_probs = result.get("class_probabilities")
        if isinstance(class_probs, dict):
            raw_prob = class_probs.get("ai")
    try:
        prob = float(raw_prob)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, prob))


def _predict(client: GPTZeroClient, text: str, delay: float) -> Optional[ScoredText]:
    response = client.predict(text, delay=delay)
    if "error" in response:
        return None
    prob_ai = _coerce_prob_ai(response)
    if prob_ai is None:
        return None
    predicted_class = str(response.get("predicted_class", "unknown"))
    return ScoredText(
        text=normalize_text(text),
        prob_ai=prob_ai,
        predicted_class=predicted_class,
        response=response,
        origin="base",
    )


def _select_base_records(
    records: Sequence[Dict[str, object]],
    *,
    max_count: int,
    seed: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    by_prefix: Dict[str, List[Dict[str, object]]] = {}
    selected: List[Dict[str, object]] = []
    used_ids = set()

    for prefix in PRIORITY_PREFIXES:
        bucket = [
            r for r in records
            if str(r.get("sample_id", "")).startswith(prefix)
            and str(r.get("sample_id", "")) not in used_ids
        ]
        rng.shuffle(bucket)
        by_prefix[prefix] = bucket

    while len(selected) < max_count:
        progressed = False
        for prefix in PRIORITY_PREFIXES:
            bucket = by_prefix[prefix]
            if not bucket or len(selected) >= max_count:
                continue
            row = bucket.pop()
            sid = str(row.get("sample_id", ""))
            if sid in used_ids:
                continue
            selected.append(row)
            used_ids.add(sid)
            progressed = True
        if not progressed:
            break

    if len(selected) < max_count:
        remaining = [r for r in records if str(r.get("sample_id", "")) not in used_ids]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_count - len(selected)])

    return selected[:max_count]


def _build_candidate_rows(
    *,
    source_probes: Sequence[Dict[str, object]],
    train_rows: Sequence[Dict[str, object]],
    include_verified_bases: bool,
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = [
        r for r in source_probes if str(r.get("collection", "")) == "attack_train"
    ]
    if include_verified_bases:
        for row in train_rows:
            candidates.append(
                {
                    "sample_id": str(row["sample_id"]),
                    "text": str(row["text"]),
                    "collection": "verified_train_seed",
                    "source": "verified_seed_pool",
                }
            )
    return candidates


def _class_quota_targets(
    *,
    max_mined_samples: int,
    quota_ai: Optional[int],
    quota_human: Optional[int],
    quota_mixed: Optional[int],
) -> Dict[str, int]:
    explicit = {
        "ai": quota_ai,
        "human": quota_human,
        "mixed": quota_mixed,
    }
    if any(v is not None for v in explicit.values()):
        total = sum(int(v or 0) for v in explicit.values())
        if total <= 0:
            raise ValueError("At least one class quota must be > 0 when explicit quotas are provided.")
        if total > max_mined_samples:
            raise ValueError("Sum of class quotas cannot exceed --max-mined-samples.")
        quotas = {k: int(v or 0) for k, v in explicit.items()}
        remainder = max_mined_samples - total
        if remainder > 0:
            quotas["mixed"] += remainder
        return quotas

    # Default: evenly split target with mixed carrying remainder.
    base = max_mined_samples // 3
    remainder = max_mined_samples - (base * 3)
    return {
        "ai": base,
        "human": base,
        "mixed": base + remainder,
    }


def _safe_sample_id(base_sample_id: str, text: str, existing: set[str]) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]+", "_", base_sample_id).strip("_").lower()
    stem = stem[:24] if stem else "probe"
    suffix = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    candidate = f"mine_{stem}_{suffix}"
    if candidate not in existing:
        return candidate
    for i in range(1, 1000):
        alt = f"{candidate}_{i}"
        if alt not in existing:
            return alt
    raise RuntimeError(f"Could not allocate unique sample_id for base={base_sample_id}")


def _mine_for_base(
    *,
    client: GPTZeroClient,
    base_sample_id: str,
    base_text: str,
    delay: float,
    boundary_low: float,
    boundary_high: float,
    max_steps: int,
    branching_factor: int,
    initial_scored: Optional[ScoredText] = None,
    adaptive_band: bool = True,
    tight_boundary_low: float = 0.40,
    tight_boundary_high: float = 0.60,
) -> Tuple[Optional[MinedCandidate], Dict[str, object]]:
    base_scored = initial_scored or _predict(client, base_text, delay=delay)
    trace: Dict[str, object] = {
        "base_sample_id": base_sample_id,
        "accepted": False,
        "steps": [],
    }
    if base_scored is None:
        trace["error"] = "base_prediction_failed"
        return None, trace

    trace["base_prob_ai"] = base_scored.prob_ai
    trace["base_predicted_class"] = base_scored.predicted_class

    current = base_scored
    best = base_scored
    best_in_band: Optional[ScoredText] = None
    visited = {hashlib.sha256(current.text.encode("utf-8")).hexdigest()}

    for step_idx in range(max_steps):
        if adaptive_band:
            step_low, step_high = adaptive_boundary_band(
                step_idx=step_idx,
                max_steps=max_steps,
                initial_low=boundary_low,
                initial_high=boundary_high,
                tight_low=tight_boundary_low,
                tight_high=tight_boundary_high,
            )
        else:
            step_low, step_high = boundary_low, boundary_high

        primary_direction = "toward_human" if current.prob_ai >= 0.5 else "toward_ai"
        secondary_direction = "toward_ai" if primary_direction == "toward_human" else "toward_human"
        variants: List[Tuple[str, str]] = []
        for direction in (primary_direction, secondary_direction):
            for candidate_text in generate_directional_variants(current.text, direction):
                variants.append((direction, candidate_text))

        candidates: List[ScoredText] = []

        for direction, variant in variants[: (branching_factor * 2)]:
            fingerprint = hashlib.sha256(variant.encode("utf-8")).hexdigest()
            if fingerprint in visited:
                continue
            visited.add(fingerprint)
            scored = _predict(client, variant, delay=delay)
            if scored is None:
                continue
            scored.origin = f"{direction}_step_{step_idx + 1}"
            candidates.append(scored)

        if not candidates:
            trace["steps"].append(
                {
                    "step": step_idx + 1,
                    "direction": primary_direction,
                    "band_low": step_low,
                    "band_high": step_high,
                    "evaluated": 0,
                    "selected_prob_ai": current.prob_ai,
                }
            )
            continue

        candidates.sort(key=lambda c: boundary_distance(c.prob_ai))
        current = candidates[0]
        if boundary_distance(current.prob_ai) < boundary_distance(best.prob_ai):
            best = current

        trace["steps"].append(
            {
                "step": step_idx + 1,
                "direction": primary_direction,
                "band_low": step_low,
                "band_high": step_high,
                "evaluated": len(candidates),
                "selected_prob_ai": current.prob_ai,
                "selected_predicted_class": current.predicted_class,
                "distance_to_boundary": boundary_distance(current.prob_ai),
            }
        )

        if in_boundary_band(current.prob_ai, step_low, step_high):
            if best_in_band is None or boundary_distance(current.prob_ai) < boundary_distance(best_in_band.prob_ai):
                best_in_band = current

    selected = best_in_band or best
    selected_in_band = best_in_band is not None
    trace["accepted"] = selected_in_band
    trace["selected_in_band"] = selected_in_band
    trace["selected_prob_ai"] = selected.prob_ai
    trace["selected_predicted_class"] = selected.predicted_class
    trace["selected_distance_to_boundary"] = boundary_distance(selected.prob_ai)
    trace["best_prob_ai"] = best.prob_ai
    trace["best_predicted_class"] = best.predicted_class
    trace["best_distance_to_boundary"] = boundary_distance(best.prob_ai)

    candidate = MinedCandidate(
        base_sample_id=base_sample_id,
        scored=selected,
        in_band=selected_in_band,
        distance=boundary_distance(selected.prob_ai),
        trace=trace,
    )
    return candidate, trace


def _write_dataset_manifest(
    *,
    target_dir: Path,
    source_version: str,
    target_version: str,
    probes_rows: List[Dict[str, object]],
    train_rows: List[Dict[str, object]],
    test_rows: List[Dict[str, object]],
    mining_report_path: Path,
    config: Dict[str, object],
) -> None:
    probes_path = target_dir / "probes.jsonl"
    train_path = target_dir / "train_verified.jsonl"
    test_path = target_dir / "test_verified.jsonl"
    manifest_path = target_dir / "manifest.json"

    def count_labels(rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
        counts = Counter(str(r["ground_truth_label"]) for r in rows)
        return dict(sorted(counts.items()))

    def count_collections(rows: Iterable[Dict[str, object]]) -> Dict[str, int]:
        counts = Counter(str(r.get("collection", "unknown")) for r in rows)
        return dict(sorted(counts.items()))

    manifest = {
        "dataset_version": target_version,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_dataset_version": source_version,
        "augmentation": {
            "boundary_mined_records_added": config["accepted_count"],
            "boundary_band": [config["boundary_low"], config["boundary_high"]],
            "max_steps": config["max_steps"],
            "branching_factor": config["branching_factor"],
            "max_base_samples": config["max_base_samples"],
            "seed": config["seed"],
            "mining_report": str(mining_report_path),
            "mining_report_sha256": file_sha256(mining_report_path),
        },
        "files": {
            "probes.jsonl": {
                "sha256": file_sha256(probes_path),
                "record_count": len(probes_rows),
            },
            "train_verified.jsonl": {
                "sha256": file_sha256(train_path),
                "record_count": len(train_rows),
            },
            "test_verified.jsonl": {
                "sha256": file_sha256(test_path),
                "record_count": len(test_rows),
            },
        },
        "counts": {
            "probes_by_collection": count_collections(probes_rows),
            "train_verified_by_label": count_labels(train_rows),
            "test_verified_by_label": count_labels(test_rows),
        },
        "provenance_notes": {
            "train_verified": f"Copied from {source_version} verified-provenance records.",
            "test_verified": f"Copied from {source_version} held-out verified-provenance records.",
            "probes": (
                f"{source_version} probes plus active boundary-mined records "
                "guided by GPTZero live probabilities."
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine boundary probes and build a new dataset version")
    parser.add_argument("--source-version", default="v2")
    parser.add_argument("--target-version", default="v3")
    parser.add_argument("--api-key", help="GPTZero API key (optional if env var is set)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--max-base-samples", type=int, default=12)
    parser.add_argument("--base-screen-size", type=int, default=36)
    parser.add_argument("--max-mined-samples", type=int, default=24)
    parser.add_argument("--fallback-top-k", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--branching-factor", type=int, default=4)
    parser.add_argument("--boundary-low", type=float, default=0.35)
    parser.add_argument("--boundary-high", type=float, default=0.65)
    parser.add_argument("--tight-boundary-low", type=float, default=0.40)
    parser.add_argument("--tight-boundary-high", type=float, default=0.60)
    parser.add_argument(
        "--adaptive-band",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate from initial boundary band to tight boundary band across mining steps.",
    )
    parser.add_argument("--quota-ai", type=int, help="Optional class quota for AI-labeled mined probes.")
    parser.add_argument("--quota-human", type=int, help="Optional class quota for human-labeled mined probes.")
    parser.add_argument("--quota-mixed", type=int, help="Optional class quota for mixed-labeled mined probes.")
    parser.add_argument(
        "--include-verified-bases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include verified training texts as candidate base seeds",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.boundary_low >= args.boundary_high:
        raise ValueError("--boundary-low must be less than --boundary-high")
    if args.tight_boundary_low >= args.tight_boundary_high:
        raise ValueError("--tight-boundary-low must be less than --tight-boundary-high")

    api_key = (
        args.api_key
        or os.environ.get("GPTZERO_API_KEY")
        or os.environ.get("GPT_ZERO_API_KEY")
    )
    if not api_key:
        raise ValueError("Set --api-key, GPTZERO_API_KEY, or GPT_ZERO_API_KEY")

    source_dir = PROJECT_ROOT / "datasets" / args.source_version
    target_dir = PROJECT_ROOT / "datasets" / args.target_version
    if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
        raise FileExistsError(
            f"Target directory exists and is non-empty: {target_dir}. Use --force to overwrite."
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    source_probes = _load_required_jsonl(source_dir / "probes.jsonl")
    train_rows = _load_required_jsonl(source_dir / "train_verified.jsonl")
    test_rows = _load_required_jsonl(source_dir / "test_verified.jsonl")
    client = GPTZeroClient(api_key=api_key)

    candidate_rows = _build_candidate_rows(
        source_probes=source_probes,
        train_rows=train_rows,
        include_verified_bases=args.include_verified_bases,
    )
    screen_rows = _select_base_records(
        candidate_rows,
        max_count=min(args.base_screen_size, len(candidate_rows)),
        seed=args.seed,
    )
    screened: List[Tuple[Dict[str, object], ScoredText]] = []
    print(f"Screening {len(screen_rows)} base candidates for boundary closeness...", flush=True)
    for row in screen_rows:
        scored = _predict(client, str(row["text"]), delay=args.delay)
        if scored is None:
            continue
        screened.append((row, scored))
    screened.sort(key=lambda item: boundary_distance(item[1].prob_ai))
    screened = screened[: args.max_base_samples]
    base_rows = [row for row, _ in screened]
    base_scored_lookup = {str(row["sample_id"]): scored for row, scored in screened}
    print(f"Selected {len(base_rows)} base seeds after screening.", flush=True)

    mined_rows: List[Dict[str, object]] = []
    trace_rows: List[Dict[str, object]] = []
    candidate_pool: List[Tuple[MinedCandidate, str]] = []
    existing_ids = {str(r["sample_id"]) for r in source_probes}
    existing_text_hashes = {
        hashlib.sha256(normalize_text(str(r["text"])).encode("utf-8")).hexdigest()
        for r in source_probes
    }
    candidate_text_hashes = set(existing_text_hashes)
    quota_targets = _class_quota_targets(
        max_mined_samples=args.max_mined_samples,
        quota_ai=args.quota_ai,
        quota_human=args.quota_human,
        quota_mixed=args.quota_mixed,
    )
    quota_counts = {k: 0 for k in quota_targets}
    print(f"Class quota targets: {quota_targets}", flush=True)

    for idx, row in enumerate(base_rows, start=1):
        base_id = str(row["sample_id"])
        base_text = str(row["text"])
        print(f"[{idx}/{len(base_rows)}] mining from {base_id} ...", flush=True)
        candidate, trace = _mine_for_base(
            client=client,
            base_sample_id=base_id,
            base_text=base_text,
            delay=args.delay,
            boundary_low=args.boundary_low,
            boundary_high=args.boundary_high,
            max_steps=args.max_steps,
            branching_factor=args.branching_factor,
            initial_scored=base_scored_lookup.get(base_id),
            adaptive_band=bool(args.adaptive_band),
            tight_boundary_low=args.tight_boundary_low,
            tight_boundary_high=args.tight_boundary_high,
        )
        trace_rows.append(trace)

        if candidate is None:
            print("  candidate generation failed", flush=True)
            continue

        text_hash = hashlib.sha256(normalize_text(candidate.scored.text).encode("utf-8")).hexdigest()
        if text_hash in candidate_text_hashes:
            trace["accepted"] = False
            trace["rejected_reason"] = "duplicate_text"
            print("  candidate duplicate text; skipped", flush=True)
            continue
        candidate_text_hashes.add(text_hash)
        candidate_pool.append((candidate, text_hash))
        print(
            f"  candidate prob_ai={candidate.scored.prob_ai:.3f} "
            f"class={candidate.scored.predicted_class} "
            f"in_band={candidate.in_band}",
            flush=True,
        )

    candidate_pool.sort(key=lambda pair: pair[0].distance)
    selected_hashes: set[str] = set()

    def _norm_cls(raw: str) -> str:
        norm = str(raw).strip().lower()
        return norm if norm in quota_targets else "mixed"

    def _materialize_row(candidate: MinedCandidate, text_hash: str, mode: str) -> None:
        sample_id = _safe_sample_id(candidate.base_sample_id, candidate.scored.text, existing_ids)
        existing_ids.add(sample_id)
        existing_text_hashes.add(text_hash)
        selected_hashes.add(text_hash)
        pred_class = _norm_cls(candidate.scored.predicted_class)
        quota_counts[pred_class] = quota_counts.get(pred_class, 0) + 1
        mined_rows.append(
            {
                "sample_id": sample_id,
                "text": candidate.scored.text,
                "collection": "attack_train",
                "source": f"boundary_mining_{args.target_version}",
                "ground_truth_label": None,
                "mined_from_sample_id": candidate.base_sample_id,
                "mined_prob_ai": candidate.scored.prob_ai,
                "mined_predicted_class": candidate.scored.predicted_class,
                "mined_distance_to_boundary": candidate.distance,
                "mined_selection_mode": mode,
                "boundary_band": [args.boundary_low, args.boundary_high],
                "tight_boundary_band": [args.tight_boundary_low, args.tight_boundary_high],
            }
        )
        candidate.trace["accepted"] = True
        candidate.trace["accepted_sample_id"] = sample_id
        candidate.trace["selection_mode"] = mode

    # Pass 1: in-band candidates honoring quotas.
    for candidate, text_hash in candidate_pool:
        if len(mined_rows) >= args.max_mined_samples:
            break
        if not candidate.in_band:
            continue
        cls = _norm_cls(candidate.scored.predicted_class)
        if quota_counts.get(cls, 0) >= quota_targets.get(cls, 0):
            continue
        _materialize_row(candidate, text_hash, mode="in_band")

    # Pass 2: closest fallback candidates when in-band hits are low.
    remaining = [
        (candidate, text_hash)
        for candidate, text_hash in candidate_pool
        if text_hash not in selected_hashes
    ]
    remaining.sort(
        key=lambda pair: (
            quota_counts.get(_norm_cls(pair[0].scored.predicted_class), 0)
            >= quota_targets.get(_norm_cls(pair[0].scored.predicted_class), 0),
            pair[0].distance,
        )
    )
    fallback_budget = max(args.fallback_top_k, args.max_mined_samples - len(mined_rows))
    for candidate, text_hash in remaining[:fallback_budget]:
        if len(mined_rows) >= args.max_mined_samples:
            break
        cls = _norm_cls(candidate.scored.predicted_class)
        if quota_counts.get(cls, 0) < quota_targets.get(cls, 0):
            _materialize_row(candidate, text_hash, mode="fallback_topk")

    # Pass 3: fill remaining slots by nearest distance regardless of class quota.
    if len(mined_rows) < args.max_mined_samples:
        for candidate, text_hash in remaining:
            if len(mined_rows) >= args.max_mined_samples:
                break
            if text_hash in selected_hashes:
                continue
            _materialize_row(candidate, text_hash, mode="fallback_fill")

    probes_target = source_probes + mined_rows
    probes_path = target_dir / "probes.jsonl"
    train_path = target_dir / "train_verified.jsonl"
    test_path = target_dir / "test_verified.jsonl"

    write_jsonl(probes_path, probes_target)
    write_jsonl(train_path, train_rows)
    write_jsonl(test_path, test_rows)

    mining_report = {
        "source_dataset_version": args.source_version,
        "target_dataset_version": args.target_version,
        "started_from_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "seed": args.seed,
            "delay": args.delay,
            "max_base_samples": args.max_base_samples,
            "base_screen_size": args.base_screen_size,
            "max_mined_samples": args.max_mined_samples,
            "fallback_top_k": args.fallback_top_k,
            "max_steps": args.max_steps,
            "branching_factor": args.branching_factor,
            "boundary_low": args.boundary_low,
            "boundary_high": args.boundary_high,
            "tight_boundary_low": args.tight_boundary_low,
            "tight_boundary_high": args.tight_boundary_high,
            "adaptive_band": args.adaptive_band,
            "quota_targets": quota_targets,
            "quota_ai": args.quota_ai,
            "quota_human": args.quota_human,
            "quota_mixed": args.quota_mixed,
            "include_verified_bases": args.include_verified_bases,
        },
        "summary": {
            "base_candidates_screened": len(screen_rows),
            "base_samples_selected": len(base_rows),
            "candidate_pool_size": len(candidate_pool),
            "accepted_boundary_samples": len(mined_rows),
            "accepted_by_predicted_class": quota_counts,
            "api_calls": client.call_count,
            "api_words": client.total_words,
        },
        "trace": trace_rows,
    }
    mining_report_path = target_dir / "boundary_mining_report.json"
    mining_report_path.write_text(json.dumps(mining_report, indent=2), encoding="utf-8")

    _write_dataset_manifest(
        target_dir=target_dir,
        source_version=args.source_version,
        target_version=args.target_version,
        probes_rows=probes_target,
        train_rows=train_rows,
        test_rows=test_rows,
        mining_report_path=mining_report_path,
        config={
            "accepted_count": len(mined_rows),
            "boundary_low": args.boundary_low,
            "boundary_high": args.boundary_high,
            "max_steps": args.max_steps,
            "branching_factor": args.branching_factor,
            "base_screen_size": args.base_screen_size,
            "fallback_top_k": args.fallback_top_k,
            "max_base_samples": args.max_base_samples,
            "seed": args.seed,
        },
    )

    print(f"Dataset written to: {target_dir}")
    print(f"Mined boundary samples: {len(mined_rows)}")
    print(f"API calls: {client.call_count}, words: {client.total_words}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
