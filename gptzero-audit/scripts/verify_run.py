#!/usr/bin/env python3
"""CLI wrapper for run-to-run reproducibility verification."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.verification import verify_runs


def resolve_manifest_paths(run_ids: List[str], output_dir: Path) -> List[Path]:
    """Resolve run IDs to manifest paths."""
    return [output_dir / "artifacts" / run_id / "run_manifest.json" for run_id in run_ids]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify reproducibility across run manifests")
    parser.add_argument(
        "--run-ids",
        nargs="+",
        help="Run IDs to compare (resolved from --output-dir/artifacts/<run-id>/run_manifest.json)",
    )
    parser.add_argument(
        "--manifest-paths",
        nargs="+",
        help="Explicit manifest paths; overrides --run-ids when provided",
    )
    parser.add_argument("--output-dir", default=".", help="Base output directory used by main.py")
    parser.add_argument("--clone-score-delta-max", type=float, default=0.03)
    parser.add_argument("--gptzero-accuracy-delta-max", type=float, default=0.05)
    parser.add_argument("--surrogate-accuracy-delta-max", type=float, default=0.05)
    args = parser.parse_args()

    if args.manifest_paths:
        manifests = [Path(p).resolve() for p in args.manifest_paths]
    elif args.run_ids:
        manifests = resolve_manifest_paths(args.run_ids, Path(args.output_dir).resolve())
    else:
        parser.error("Provide --manifest-paths or --run-ids.")

    result = verify_runs(
        manifest_paths=manifests,
        tolerances={
            "clone_score_delta_max": args.clone_score_delta_max,
            "gptzero_accuracy_delta_max": args.gptzero_accuracy_delta_max,
            "surrogate_accuracy_delta_max": args.surrogate_accuracy_delta_max,
        },
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
