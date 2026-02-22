#!/usr/bin/env python3
"""Build dataset v5 with explicit corpus strata and pattern-eval split."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus_strata import DEFAULT_STRATA, annotate_with_stratum
from src.dataset_io import load_jsonl, write_jsonl
from src.pattern_benchmark import generate_variants


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_required(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")
    return load_jsonl(path)


def _count_labels(rows: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    counts = Counter(str(r.get("ground_truth_label", "null")) for r in rows)
    return dict(sorted(counts.items()))


def _count_strata(rows: Iterable[Mapping[str, object]]) -> Dict[str, int]:
    counts = Counter(str(r.get("stratum", "unknown")) for r in rows)
    return dict(sorted(counts.items()))


def _load_external_sources(config_path: Path) -> List[Dict[str, object]]:
    """
    Load optional external corpus records.

    Config schema: list[dict] where each dict can include:
      - path (required)
      - format ("jsonl" or "txt_dir"), default "jsonl"
      - text_field (for jsonl, default "text")
      - id_field (for jsonl, default "sample_id")
      - ground_truth_label (optional)
      - stratum (optional)
      - source (optional)
    """
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("External corpus config must be a JSON list of source definitions.")

    merged: List[Dict[str, object]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"External source #{idx} must be an object.")
        source_path = Path(str(item.get("path", ""))).expanduser().resolve()
        fmt = str(item.get("format", "jsonl")).strip().lower()
        text_field = str(item.get("text_field", "text"))
        id_field = str(item.get("id_field", "sample_id"))
        source_name = str(item.get("source", source_path.name))
        ground_truth = item.get("ground_truth_label")
        declared_stratum = str(item.get("stratum", "")).strip()

        if fmt == "jsonl":
            rows = load_jsonl(source_path)
            for row_i, row in enumerate(rows):
                text = str(row.get(text_field, "")).strip()
                if not text:
                    continue
                sample_id = str(row.get(id_field, f"external_{idx}_{row_i:06d}"))
                merged.append(
                    {
                        "sample_id": sample_id,
                        "text": text,
                        "ground_truth_label": ground_truth if ground_truth is not None else row.get("ground_truth_label"),
                        "source": source_name,
                        "provenance": str(row.get("provenance", source_name)),
                        "stratum": declared_stratum or str(row.get("stratum", "")),
                        "collection": "external_pattern_eval",
                        "split": "pattern_eval",
                    }
                )
            continue

        if fmt == "txt_dir":
            if not source_path.is_dir():
                raise ValueError(f"External txt_dir source is not a directory: {source_path}")
            for row_i, text_path in enumerate(sorted(source_path.glob("*.txt"))):
                text = text_path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                merged.append(
                    {
                        "sample_id": f"external_{idx}_{row_i:06d}",
                        "text": text,
                        "ground_truth_label": ground_truth,
                        "source": source_name,
                        "provenance": str(text_path),
                        "stratum": declared_stratum,
                        "collection": "external_pattern_eval",
                        "split": "pattern_eval",
                    }
                )
            continue

        raise ValueError(f"Unsupported external source format: {fmt}")
    return merged


def _to_pattern_candidates(
    probes: List[Dict[str, object]],
    train_verified: List[Dict[str, object]],
    test_verified: List[Dict[str, object]],
    external: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for item in probes:
        rec = annotate_with_stratum(item)
        rec.setdefault("split", "probes")
        rows.append(rec)
    for item in train_verified:
        rec = annotate_with_stratum(item)
        rec.setdefault("split", "train_verified")
        rows.append(rec)
    for item in test_verified:
        rec = annotate_with_stratum(item)
        rec.setdefault("split", "test_verified")
        rows.append(rec)
    for item in external:
        rec = annotate_with_stratum(item)
        rec.setdefault("split", "pattern_eval")
        rows.append(rec)
    return rows


def _dedupe_text(rows: Iterable[Mapping[str, object]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    seen = set()
    for row in rows:
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        out.append(dict(row))
    return out


def _balanced_pattern_eval(
    *,
    candidates: List[Dict[str, object]],
    target_per_stratum: int,
    seed: int,
    variants_per_augmented_base: int,
) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    by_stratum: MutableMapping[str, List[Dict[str, object]]] = defaultdict(list)
    for row in candidates:
        by_stratum[str(row.get("stratum", "unknown"))].append(row)

    strata = sorted(set(DEFAULT_STRATA) | set(by_stratum.keys()))
    selected: List[Dict[str, object]] = []
    seen_hashes = set()
    next_aug_idx = 0

    for stratum in strata:
        pool = list(by_stratum.get(stratum, []))
        if not pool:
            continue

        rng.shuffle(pool)
        stratum_rows: List[Dict[str, object]] = []
        for row in pool:
            if len(stratum_rows) >= target_per_stratum:
                break
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            copied = dict(row)
            copied.setdefault("split", "pattern_eval")
            copied.setdefault("collection", "pattern_eval")
            stratum_rows.append(copied)

        if not stratum_rows:
            continue

        per_base = max(1, variants_per_augmented_base)
        max_attempts = max(target_per_stratum * 50, 500)
        attempts = 0
        while len(stratum_rows) < target_per_stratum and attempts < max_attempts:
            attempts += 1
            base = rng.choice(stratum_rows)
            variants = generate_variants(
                str(base.get("text", "")),
                rng=rng,
                max_variants=max(per_base + 3, 6),
            )
            non_identity = [v for v in variants if str(v.get("variant_id", "")) != "base"]
            if not non_identity:
                continue
            variant = rng.choice(non_identity)
            variant_text = str(variant["text"]).strip()
            if not variant_text:
                continue
            digest = hashlib.sha256(variant_text.encode("utf-8")).hexdigest()
            if digest in seen_hashes:
                continue
            seen_hashes.add(digest)
            next_aug_idx += 1
            stratum_rows.append(
                {
                    "sample_id": f"v5aug_{stratum}_{next_aug_idx:06d}",
                    "text": variant_text,
                    "ground_truth_label": base.get("ground_truth_label"),
                    "collection": "pattern_eval",
                    "source": "v5_augmentation",
                    "provenance": str(base.get("sample_id", "")),
                    "split": "pattern_eval_augmented",
                    "stratum": stratum,
                    "perturbation_kind": variant["perturbation_kind"],
                    "perturbation_strength": variant["perturbation_strength"],
                    "base_sample_id": base.get("sample_id"),
                }
            )

        selected.extend(stratum_rows)

    return selected


def build_dataset(
    *,
    source_version: str,
    target_version: str,
    force: bool,
    target_per_stratum: int,
    variants_per_augmented_base: int,
    seed: int,
    external_corpus_config: Path | None,
) -> Path:
    source_dir = PROJECT_ROOT / "datasets" / source_version
    target_dir = PROJECT_ROOT / "datasets" / target_version
    if target_dir.exists() and any(target_dir.iterdir()) and not force:
        raise FileExistsError(
            f"Target dataset directory already exists and is not empty: {target_dir}. "
            "Use --force to overwrite."
        )
    target_dir.mkdir(parents=True, exist_ok=True)

    probes_src = _load_required(source_dir / "probes.jsonl")
    train_src = _load_required(source_dir / "train_verified.jsonl")
    test_src = _load_required(source_dir / "test_verified.jsonl")
    external_rows: List[Dict[str, object]] = []
    if external_corpus_config is not None:
        external_rows = _load_external_sources(external_corpus_config)

    probes = [annotate_with_stratum(row) for row in probes_src]
    train_verified = [annotate_with_stratum(row) for row in train_src]
    test_verified = [annotate_with_stratum(row) for row in test_src]

    pattern_candidates = _to_pattern_candidates(
        probes=probes,
        train_verified=train_verified,
        test_verified=test_verified,
        external=external_rows,
    )
    pattern_eval = _balanced_pattern_eval(
        candidates=pattern_candidates,
        target_per_stratum=target_per_stratum,
        seed=seed,
        variants_per_augmented_base=variants_per_augmented_base,
    )

    probes_path = target_dir / "probes.jsonl"
    train_path = target_dir / "train_verified.jsonl"
    test_path = target_dir / "test_verified.jsonl"
    pattern_path = target_dir / "pattern_eval.jsonl"
    manifest_path = target_dir / "manifest.json"

    write_jsonl(probes_path, probes)
    write_jsonl(train_path, train_verified)
    write_jsonl(test_path, test_verified)
    write_jsonl(pattern_path, pattern_eval)

    manifest = {
        "dataset_version": target_version,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_dataset_version": source_version,
        "schema": {
            "strata": list(DEFAULT_STRATA),
            "pattern_eval_fields": [
                "sample_id",
                "text",
                "ground_truth_label",
                "stratum",
                "source",
                "provenance",
                "split",
            ],
        },
        "build_config": {
            "target_per_stratum": int(target_per_stratum),
            "variants_per_augmented_base": int(variants_per_augmented_base),
            "seed": int(seed),
            "external_corpus_config": str(external_corpus_config) if external_corpus_config else None,
            "external_record_count": len(external_rows),
        },
        "files": {
            "probes.jsonl": {"sha256": file_sha256(probes_path), "record_count": len(probes)},
            "train_verified.jsonl": {"sha256": file_sha256(train_path), "record_count": len(train_verified)},
            "test_verified.jsonl": {"sha256": file_sha256(test_path), "record_count": len(test_verified)},
            "pattern_eval.jsonl": {"sha256": file_sha256(pattern_path), "record_count": len(pattern_eval)},
        },
        "counts": {
            "probes_by_stratum": _count_strata(probes),
            "train_verified_by_label": _count_labels(train_verified),
            "test_verified_by_label": _count_labels(test_verified),
            "pattern_eval_by_stratum": _count_strata(pattern_eval),
            "pattern_eval_by_label": _count_labels(pattern_eval),
        },
        "notes": {
            "probes": "Copied from source version and annotated with stratum.",
            "train_verified": "Copied from source version and annotated with stratum.",
            "test_verified": "Copied from source version and annotated with stratum.",
            "pattern_eval": "Balanced corpus matrix with deterministic augmentation for empirical pattern testing.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset v5 for large-scale empirical matrix tests")
    parser.add_argument("--source-version", default="v4")
    parser.add_argument("--target-version", default="v5")
    parser.add_argument("--force", action="store_true", help="Overwrite target dataset directory if it exists")
    parser.add_argument("--target-per-stratum", type=int, default=120)
    parser.add_argument("--variants-per-augmented-base", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--external-corpus-config",
        type=Path,
        default=None,
        help="Optional JSON file describing external corpora sources to include in pattern_eval.",
    )
    args = parser.parse_args()

    out_dir = build_dataset(
        source_version=args.source_version,
        target_version=args.target_version,
        force=args.force,
        target_per_stratum=args.target_per_stratum,
        variants_per_augmented_base=args.variants_per_augmented_base,
        seed=args.seed,
        external_corpus_config=args.external_corpus_config,
    )
    print(f"Dataset written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
