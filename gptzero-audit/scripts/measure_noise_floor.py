"""Measure GPTZero API determinism by querying the same texts multiple times.

Usage:
  python scripts/measure_noise_floor.py \
    --api-key $GPTZERO_API_KEY \
    --output-path noise_floor_results.json \
    --num-texts 50 \
    --repeats 3
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.client import GPTZeroClient


def _load_diverse_texts(
    dataset_version: str, num_texts: int, seed: int
) -> List[Dict[str, str]]:
    eval_path = PROJECT_ROOT / "datasets" / dataset_version / "pattern_eval.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"Pattern eval not found: {eval_path}")

    by_stratum: Dict[str, list] = {}
    with eval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            stratum = rec.get("stratum", "unknown")
            by_stratum.setdefault(stratum, []).append(rec)

    rng = random.Random(seed)
    per_stratum = max(1, num_texts // max(1, len(by_stratum)))
    selected: List[Dict[str, str]] = []
    for stratum in sorted(by_stratum.keys()):
        pool = by_stratum[stratum]
        rng.shuffle(pool)
        for rec in pool[:per_stratum]:
            selected.append(
                {
                    "sample_id": rec["sample_id"],
                    "stratum": stratum,
                    "text": rec["text"],
                }
            )

    rng.shuffle(selected)
    return selected[:num_texts]


def measure_noise_floor(
    *,
    api_key: str,
    num_texts: int = 50,
    repeats: int = 3,
    dataset_version: str = "v5",
    seed: int = 42,
    delay: float = 0.5,
    output_path: str | None = None,
) -> Dict:
    client = GPTZeroClient(api_key=api_key)
    texts = _load_diverse_texts(dataset_version, num_texts, seed)
    print(f"Selected {len(texts)} texts across strata for noise floor measurement")

    results: Dict[str, Dict] = {}
    total_queries = len(texts) * repeats
    query_idx = 0

    for rec in texts:
        sid = rec["sample_id"]
        text = rec["text"]
        probs: List[float] = []
        classes: List[str] = []

        for rep in range(repeats):
            query_idx += 1
            print(
                f"  [{query_idx}/{total_queries}] {sid} rep={rep+1}",
                end=" ",
                flush=True,
            )
            resp = client.predict(text, delay=delay)
            if "error" in resp:
                print(f"ERROR: {resp['error']}")
                continue
            prob = resp.get("average_generated_prob")
            cls = resp.get("predicted_class")
            if prob is not None:
                probs.append(float(prob))
            if cls is not None:
                classes.append(str(cls))
            print(f"p={prob} cls={cls}")

        if len(probs) >= 2:
            mean_p = sum(probs) / len(probs)
            variance = sum((p - mean_p) ** 2 for p in probs) / len(probs)
            spread = max(probs) - min(probs)
        else:
            mean_p = probs[0] if probs else None
            variance = 0.0
            spread = 0.0

        class_stable = len(set(classes)) <= 1 if classes else None
        results[sid] = {
            "stratum": rec["stratum"],
            "probs": probs,
            "classes": classes,
            "mean_prob": mean_p,
            "variance": variance,
            "spread": spread,
            "class_stable": class_stable,
        }

    all_variances = [r["variance"] for r in results.values() if r["variance"] is not None]
    all_spreads = [r["spread"] for r in results.values() if r["spread"] is not None]
    class_flips = sum(
        1 for r in results.values() if r["class_stable"] is False
    )

    import math

    summary = {
        "num_texts": len(texts),
        "repeats": repeats,
        "total_queries": total_queries,
        "mean_variance": (sum(all_variances) / len(all_variances)) if all_variances else None,
        "mean_spread": (sum(all_spreads) / len(all_spreads)) if all_spreads else None,
        "max_spread": max(all_spreads) if all_spreads else None,
        "mean_stddev": (
            math.sqrt(sum(all_variances) / len(all_variances))
            if all_variances
            else None
        ),
        "class_flip_count": class_flips,
        "class_flip_rate": class_flips / len(results) if results else None,
    }

    by_stratum_summary: Dict[str, Dict] = {}
    for sid, r in results.items():
        stratum = r["stratum"]
        by_stratum_summary.setdefault(stratum, {"variances": [], "spreads": [], "flips": 0, "count": 0})
        by_stratum_summary[stratum]["count"] += 1
        if r["variance"] is not None:
            by_stratum_summary[stratum]["variances"].append(r["variance"])
        if r["spread"] is not None:
            by_stratum_summary[stratum]["spreads"].append(r["spread"])
        if r["class_stable"] is False:
            by_stratum_summary[stratum]["flips"] += 1

    strata_report = {}
    for stratum, s in sorted(by_stratum_summary.items()):
        strata_report[stratum] = {
            "count": s["count"],
            "mean_variance": (sum(s["variances"]) / len(s["variances"])) if s["variances"] else None,
            "mean_spread": (sum(s["spreads"]) / len(s["spreads"])) if s["spreads"] else None,
            "max_spread": max(s["spreads"]) if s["spreads"] else None,
            "class_flips": s["flips"],
        }

    output = {
        "summary": summary,
        "strata": strata_report,
        "per_text": results,
    }

    print("\n=== NOISE FLOOR SUMMARY ===")
    print(f"  Texts queried:     {summary['num_texts']}")
    print(f"  Repeats per text:  {summary['repeats']}")
    print(f"  Mean stddev:       {summary['mean_stddev']:.6f}" if summary["mean_stddev"] else "  Mean stddev: N/A")
    print(f"  Mean spread:       {summary['mean_spread']:.6f}" if summary["mean_spread"] else "  Mean spread: N/A")
    print(f"  Max spread:        {summary['max_spread']:.6f}" if summary["max_spread"] else "  Max spread: N/A")
    print(f"  Class flips:       {summary['class_flip_count']}/{len(results)}")
    print()
    for stratum, sr in strata_report.items():
        ms = sr["mean_spread"]
        print(f"  {stratum:25s}  mean_spread={ms:.6f}" if ms is not None else f"  {stratum:25s}  N/A")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"\nResults written to {out}")

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure GPTZero API noise floor")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--num-texts", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dataset-version", default="v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    measure_noise_floor(
        api_key=args.api_key,
        num_texts=args.num_texts,
        repeats=args.repeats,
        dataset_version=args.dataset_version,
        seed=args.seed,
        delay=args.delay,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
