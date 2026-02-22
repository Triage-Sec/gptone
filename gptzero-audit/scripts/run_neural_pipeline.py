"""Orchestrate the full neural surrogate pipeline on Modal.

Usage:
  python scripts/run_neural_pipeline.py --stage all
  python scripts/run_neural_pipeline.py --stage train
  python scripts/run_neural_pipeline.py --stage benchmark --seed 42
  python scripts/run_neural_pipeline.py --stage verify
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def stage_corpus(args: argparse.Namespace) -> dict:
    import modal_app

    config = {
        "seed": args.seed,
        "target_per_stratum": args.target_per_stratum,
        "boundary_count": args.boundary_count,
        "mixed_count": args.mixed_count,
        "variants_per_base": args.variants_per_base,
        "delay": args.delay,
        "batch_size": args.batch_size,
        "output_dir": args.output_dir,
    }
    print("=" * 60)
    print("STAGE: Corpus Build + GPTZero Labeling")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")

    result = modal_app.build_and_label_corpus.remote(config)
    print("\n" + json.dumps(result, indent=2))
    return result


def stage_train(args: argparse.Namespace) -> dict:
    import modal_app

    config_path = Path(args.train_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = args.seed

    print("=" * 60)
    print("STAGE: Neural Surrogate Training")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")

    result = modal_app.train_neural_surrogate.remote(config)
    print("\n" + json.dumps(result, indent=2))
    return result


def stage_benchmark(args: argparse.Namespace) -> dict:
    import modal_app

    config_path = Path(args.benchmark_config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["seed"] = args.seed

    print("=" * 60)
    print(f"STAGE: Neural Pattern Benchmark (seed={args.seed})")
    print("=" * 60)
    print(f"Config: {json.dumps(config, indent=2)}")

    result = modal_app.run_neural_pattern_benchmark.remote(config)
    print("\n" + json.dumps(result, indent=2))
    return result


def stage_multi_seed_benchmark(args: argparse.Namespace) -> list:
    import modal_app

    config_path = Path(args.benchmark_config)
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 60)
    print(f"STAGE: Multi-Seed Pattern Benchmark (seeds={seeds})")
    print("=" * 60)

    results = []
    for seed in seeds:
        config = dict(base_config)
        config["seed"] = seed
        config["cache_name"] = f"neural_pattern_cache_s{seed}.json"
        print(f"\n--- Seed {seed} ---")
        result = modal_app.run_neural_pattern_benchmark.remote(config)
        results.append(result)

        overall = result.get("summary", {}).get("overall", {})
        print(f"  Agreement:  {overall.get('agreement_rate', 0):.4f}")
        print(f"  MAE:        {overall.get('probability_mae', 1):.4f}")
        print(f"  Transition: {overall.get('transition_agreement', 0):.4f}")
        passed = result.get("gate_check", {}).get("all_passed", False)
        print(f"  Gates: {'PASSED' if passed else 'FAILED'}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Neural surrogate pipeline orchestrator")
    parser.add_argument("--stage", required=True, choices=["corpus", "train", "benchmark", "multi-benchmark", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default="42,1337,2026")
    parser.add_argument("--output-dir", default="/artifacts/gptzero-audit")
    parser.add_argument("--target-per-stratum", type=int, default=300)
    parser.add_argument("--boundary-count", type=int, default=200)
    parser.add_argument("--mixed-count", type=int, default=200)
    parser.add_argument("--variants-per-base", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--train-config", default="configs/neural_surrogate_train.json")
    parser.add_argument("--benchmark-config", default="configs/neural_pattern_benchmark.json")
    args = parser.parse_args()

    import modal_app

    with modal_app.app.run():
        if args.stage == "corpus":
            stage_corpus(args)
        elif args.stage == "train":
            stage_train(args)
        elif args.stage == "benchmark":
            stage_benchmark(args)
        elif args.stage == "multi-benchmark":
            stage_multi_seed_benchmark(args)
        elif args.stage == "all":
            print("Running full pipeline: corpus -> train -> multi-seed benchmark\n")
            stage_corpus(args)
            stage_train(args)
            stage_multi_seed_benchmark(args)


if __name__ == "__main__":
    main()
