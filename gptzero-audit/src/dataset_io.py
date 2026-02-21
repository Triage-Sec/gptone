"""Dataset file I/O utilities for versioned benchmark corpora."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def dataset_dir(dataset_version: str) -> Path:
    """Return the directory containing versioned datasets."""
    return PROJECT_ROOT / "datasets" / dataset_version


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    """Load a JSONL file into memory."""
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")

