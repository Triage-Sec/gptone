#!/usr/bin/env python3
"""Build dataset v2 by augmenting v1 with UI-reason-guided probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_io import load_jsonl, write_jsonl
from src.style_reason_probes import build_style_reason_probe_records


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required dataset file: {path}")
    return load_jsonl(path)


def _count_labels(rows: List[Dict[str, object]]) -> Dict[str, int]:
    counts = Counter(str(r["ground_truth_label"]) for r in rows)
    return dict(sorted(counts.items()))


def _count_collections(rows: List[Dict[str, object]]) -> Dict[str, int]:
    counts = Counter(str(r.get("collection", "unknown")) for r in rows)
    return dict(sorted(counts.items()))


def build_dataset(*, source_version: str, target_version: str, force: bool) -> Path:
    source_dir = PROJECT_ROOT / "datasets" / source_version
    target_dir = PROJECT_ROOT / "datasets" / target_version

    if target_dir.exists():
        has_files = any(target_dir.iterdir())
        if has_files and not force:
            raise FileExistsError(
                f"Target dataset directory already exists and is not empty: {target_dir}. "
                "Use --force to overwrite."
            )
    target_dir.mkdir(parents=True, exist_ok=True)

    source_probes = _load_rows(source_dir / "probes.jsonl")
    source_train = _load_rows(source_dir / "train_verified.jsonl")
    source_test = _load_rows(source_dir / "test_verified.jsonl")

    extension = build_style_reason_probe_records()
    existing_ids = {str(row["sample_id"]) for row in source_probes}
    duplicate_ids = [str(row["sample_id"]) for row in extension if str(row["sample_id"]) in existing_ids]
    if duplicate_ids:
        raise ValueError(f"Duplicate sample IDs detected in extension set: {duplicate_ids[:5]}")

    probes_v2 = source_probes + extension

    probes_path = target_dir / "probes.jsonl"
    train_path = target_dir / "train_verified.jsonl"
    test_path = target_dir / "test_verified.jsonl"
    manifest_path = target_dir / "manifest.json"

    write_jsonl(probes_path, probes_v2)
    write_jsonl(train_path, source_train)
    write_jsonl(test_path, source_test)

    manifest = {
        "dataset_version": target_version,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_dataset_version": source_version,
        "augmentation": {
            "style_reason_records_added": len(extension),
            "style_reason_generator": "src.style_reason_probes.build_style_reason_probe_records",
            "source": "GPTZero product explanation taxonomy",
        },
        "files": {
            "probes.jsonl": {
                "sha256": file_sha256(probes_path),
                "record_count": len(probes_v2),
            },
            "train_verified.jsonl": {
                "sha256": file_sha256(train_path),
                "record_count": len(source_train),
            },
            "test_verified.jsonl": {
                "sha256": file_sha256(test_path),
                "record_count": len(source_test),
            },
        },
        "counts": {
            "probes_by_collection": _count_collections(probes_v2),
            "train_verified_by_label": _count_labels(source_train),
            "test_verified_by_label": _count_labels(source_test),
        },
        "provenance_notes": {
            "train_verified": "Copied from v1 verified-provenance records.",
            "test_verified": "Copied from v1 held-out verified-provenance records.",
            "probes": "v1 synthetic probes plus UI-reason-guided style-axis augmentation.",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Build dataset v2 with style-reason probe augmentation")
    parser.add_argument("--source-version", default="v1")
    parser.add_argument("--target-version", default="v2")
    parser.add_argument("--force", action="store_true", help="Overwrite target dataset directory if it exists")
    args = parser.parse_args()

    out_dir = build_dataset(
        source_version=args.source_version,
        target_version=args.target_version,
        force=args.force,
    )
    print(f"Dataset written to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
