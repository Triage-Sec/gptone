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

from src.verification import DEFAULT_PATTERN_TARGETS, DEFAULT_PROMOTION_TARGETS, verify_runs


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
    parser.add_argument(
        "--enforce-promotion-targets",
        action="store_true",
        help="Apply strict promotion gates (agreement/MAE/CI/per-class/stability).",
    )
    parser.add_argument("--agreement-min", type=float, default=DEFAULT_PROMOTION_TARGETS["agreement_min"])
    parser.add_argument("--probability-mae-max", type=float, default=DEFAULT_PROMOTION_TARGETS["probability_mae_max"])
    parser.add_argument("--ci-lower-min", type=float, default=DEFAULT_PROMOTION_TARGETS["ci_lower_min"])
    parser.add_argument("--per-class-min-ai", type=float, default=DEFAULT_PROMOTION_TARGETS["per_class_min"]["ai"])
    parser.add_argument("--per-class-min-human", type=float, default=DEFAULT_PROMOTION_TARGETS["per_class_min"]["human"])
    parser.add_argument("--per-class-min-mixed", type=float, default=DEFAULT_PROMOTION_TARGETS["per_class_min"]["mixed"])
    parser.add_argument("--clone-spread-max", type=float, default=DEFAULT_PROMOTION_TARGETS["clone_score_spread_max"])
    parser.add_argument("--mae-spread-max", type=float, default=DEFAULT_PROMOTION_TARGETS["mae_spread_max"])
    parser.add_argument("--require-stability-runs", type=int, default=DEFAULT_PROMOTION_TARGETS["require_runs"])
    parser.add_argument(
        "--enforce-pattern-targets",
        action="store_true",
        help="Require pattern benchmark overall + per-stratum targets.",
    )
    parser.add_argument(
        "--pattern-overall-agreement-min",
        type=float,
        default=DEFAULT_PATTERN_TARGETS["overall_agreement_min"],
    )
    parser.add_argument(
        "--pattern-overall-probability-mae-max",
        type=float,
        default=DEFAULT_PATTERN_TARGETS["overall_probability_mae_max"],
    )
    parser.add_argument(
        "--pattern-overall-transition-min",
        type=float,
        default=DEFAULT_PATTERN_TARGETS["overall_transition_agreement_min"],
    )
    parser.add_argument(
        "--pattern-stratum-agreement-min",
        type=float,
        default=DEFAULT_PATTERN_TARGETS["stratum_agreement_min"],
    )
    parser.add_argument(
        "--pattern-stratum-probability-mae-max",
        type=float,
        default=DEFAULT_PATTERN_TARGETS["stratum_probability_mae_max"],
    )
    parser.add_argument(
        "--pattern-stratum-transition-min",
        type=float,
        default=DEFAULT_PATTERN_TARGETS["stratum_transition_agreement_min"],
    )
    parser.add_argument(
        "--pattern-stratum-min-samples",
        type=int,
        default=DEFAULT_PATTERN_TARGETS["stratum_min_samples"],
    )
    parser.add_argument(
        "--pattern-required-strata",
        nargs="*",
        default=list(DEFAULT_PATTERN_TARGETS["required_strata"]),
        help="Optional list of strata that must be present in each run's pattern benchmark.",
    )
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
        enforce_promotion=bool(args.enforce_promotion_targets),
        enforce_pattern=bool(args.enforce_pattern_targets),
        promotion_targets={
            "agreement_min": args.agreement_min,
            "probability_mae_max": args.probability_mae_max,
            "ci_lower_min": args.ci_lower_min,
            "per_class_min": {
                "ai": args.per_class_min_ai,
                "human": args.per_class_min_human,
                "mixed": args.per_class_min_mixed,
            },
            "clone_score_spread_max": args.clone_spread_max,
            "mae_spread_max": args.mae_spread_max,
            "require_runs": args.require_stability_runs,
        },
        pattern_targets={
            "overall_agreement_min": args.pattern_overall_agreement_min,
            "overall_probability_mae_max": args.pattern_overall_probability_mae_max,
            "overall_transition_agreement_min": args.pattern_overall_transition_min,
            "stratum_agreement_min": args.pattern_stratum_agreement_min,
            "stratum_probability_mae_max": args.pattern_stratum_probability_mae_max,
            "stratum_transition_agreement_min": args.pattern_stratum_transition_min,
            "stratum_min_samples": args.pattern_stratum_min_samples,
            "required_strata": args.pattern_required_strata,
        },
        require_stability_runs=args.require_stability_runs,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
