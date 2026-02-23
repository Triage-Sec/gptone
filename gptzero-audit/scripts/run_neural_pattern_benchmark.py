"""Pattern benchmark using the neural surrogate instead of sklearn.

The neural surrogate predicts GPTZero's probability from raw text alone,
without any GPTZero API features. Evaluation compares neural predictions
against GPTZero's actual API responses.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.client import GPTZeroClient
from src.pattern_benchmark import (
    generate_variants,
    select_boundary_variants,
    set_transition_eps,
    summarize_prediction_rows,
    summarize_prediction_rows_by_stratum_kind,
)


def _load_pattern_eval(dataset_version: str) -> List[Dict[str, Any]]:
    eval_path = PROJECT_ROOT / "datasets" / dataset_version / "pattern_eval.jsonl"
    records: List[Dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _load_cache(cache_path: Path) -> Dict[str, Dict]:
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            return raw.get("responses", {})
        except Exception:
            pass
    return {}


def _save_cache(cache_path: Path, cache: Dict[str, Dict]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "responses": cache,
            }
        ),
        encoding="utf-8",
    )


def run(
    api_key: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run pattern benchmark with neural surrogate."""
    model_dir = str(config.get("model_dir", "/artifacts/gptzero-audit/neural_surrogate"))
    output_dir = str(config.get("output_dir", "/artifacts/gptzero-audit"))
    dataset_version = str(config.get("dataset_version", "v5"))
    max_base_per_stratum = int(config.get("max_base_per_stratum", 40))
    variants_per_base = int(config.get("variants_per_base", 6))
    seed = int(config.get("seed", 42))
    delay = float(config.get("delay", 0.25))
    batch_size = int(config.get("batch_size", 64))
    boundary_focus = bool(config.get("boundary_focus", True))
    cache_name = str(config.get("cache_name", "neural_pattern_cache.json"))
    transition_eps = float(config.get("transition_eps", 1e-6))

    set_transition_eps(transition_eps)
    rng = random.Random(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    data_dir = out_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    cache_path = data_dir / cache_name
    cache = _load_cache(cache_path)
    print(f"Loaded {len(cache)} cached GPTZero responses")

    model_type = str(config.get("model_type", "neural"))

    if model_type == "perplexity":
        from src.perplexity_surrogate import PerplexitySurrogate
        print(f"Loading perplexity surrogate from {model_dir}...")
        surrogate = PerplexitySurrogate(model_dir)
    else:
        from src.neural_surrogate import GPTZeroSurrogate
        print(f"Loading neural surrogate from {model_dir}...")
        surrogate = GPTZeroSurrogate(model_dir)

    print(f"Loading pattern eval from {dataset_version}...")
    eval_records = _load_pattern_eval(dataset_version)

    by_stratum: Dict[str, List[Dict]] = {}
    for rec in eval_records:
        stratum = rec.get("stratum", "unknown")
        by_stratum.setdefault(stratum, []).append(rec)

    selected_bases: List[Dict[str, Any]] = []
    for stratum in sorted(by_stratum.keys()):
        pool = by_stratum[stratum]
        rng.shuffle(pool)
        for rec in pool[:max_base_per_stratum]:
            selected_bases.append(rec)

    print(f"Selected {len(selected_bases)} base texts across {len(by_stratum)} strata")

    print("Generating perturbation variants...")
    all_rows: List[Dict[str, Any]] = []
    for rec in selected_bases:
        variants = generate_variants(
            rec["text"], rng=rng, max_variants=variants_per_base + 1
        )
        for v_idx, var in enumerate(variants):
            all_rows.append(
                {
                    "base_sample_id": rec["sample_id"],
                    "stratum": rec.get("stratum", "unknown"),
                    "variant_order": v_idx,
                    "variant_id": var["variant_id"],
                    "perturbation_kind": var["perturbation_kind"],
                    "perturbation_strength": var["perturbation_strength"],
                    "text": var["text"],
                }
            )
    print(f"Total variant rows: {len(all_rows)}")

    if boundary_focus:
        print("Running boundary-focused variant selection...")
        texts_for_scoring = [str(r["text"]) for r in all_rows]
        scoring_probs = surrogate.predict_prob(texts_for_scoring)

        groups: Dict[str, List[int]] = {}
        for i, row in enumerate(all_rows):
            groups.setdefault(row["base_sample_id"], []).append(i)

        focused_rows: List[Dict[str, Any]] = []
        for bid, indices in groups.items():
            group_variants = [all_rows[i] for i in indices]
            group_probs = [float(scoring_probs[i]) for i in indices]
            selected = select_boundary_variants(
                group_variants, group_probs, target_count=variants_per_base
            )
            focused_rows.extend(selected)

        all_rows = focused_rows
        print(f"After boundary selection: {len(all_rows)} rows")

    print("Querying GPTZero API for ground truth...")
    client = GPTZeroClient(api_key=api_key)
    queries_made = 0
    cache_hits = 0

    for i, row in enumerate(all_rows):
        text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if text_hash in cache:
            resp = cache[text_hash]
            cache_hits += 1
        else:
            resp = client.predict(row["text"], delay=delay)
            if "error" not in resp:
                cache[text_hash] = resp
            queries_made += 1

        if "error" not in resp:
            row["gptzero_prob"] = resp.get("average_generated_prob")
            row["gptzero_class"] = resp.get("predicted_class")
        else:
            row["gptzero_prob"] = None
            row["gptzero_class"] = None

        if (i + 1) % batch_size == 0:
            pct = (i + 1) / len(all_rows) * 100
            print(f"  [{i+1}/{len(all_rows)}] ({pct:.0f}%) queries={queries_made} cache={cache_hits}")
            _save_cache(cache_path, cache)

    _save_cache(cache_path, cache)
    print(f"GPTZero labeling: {queries_made} queries, {cache_hits} cache hits")

    valid_rows = [r for r in all_rows if r.get("gptzero_prob") is not None]
    print(f"Valid rows for evaluation: {len(valid_rows)}")

    print("Running neural surrogate inference...")
    texts = [str(r["text"]) for r in valid_rows]
    surrogate_probs = surrogate.predict_prob(texts)

    ai_threshold = float(config.get("ai_threshold", 0.65))
    human_threshold = float(config.get("human_threshold", 0.35))

    pred_classes = None
    if hasattr(surrogate, "predict_class") and hasattr(surrogate, "class_model") and surrogate.class_model is not None:
        print("Using dedicated class predictor for class labels")
        pred_classes = surrogate.predict_class(texts)

    for i, row in enumerate(valid_rows):
        row["surrogate_prob"] = float(surrogate_probs[i])
        gz_cls = row.get("gptzero_class", "human")
        if pred_classes is not None:
            pred_cls = pred_classes[i]
        else:
            sp = float(surrogate_probs[i])
            if sp >= ai_threshold:
                pred_cls = "ai"
            elif sp <= human_threshold:
                pred_cls = "human"
            else:
                pred_cls = "mixed"
        row["surrogate_class"] = pred_cls
        row["match"] = pred_cls == gz_cls
        row["prob_error"] = abs(float(row["gptzero_prob"]) - float(surrogate_probs[i]))

    summary = summarize_prediction_rows(valid_rows)
    stratum_kind = summarize_prediction_rows_by_stratum_kind(valid_rows)

    from src.verification import DEFAULT_PATTERN_TARGETS

    targets = dict(DEFAULT_PATTERN_TARGETS)
    gate_check = _check_gates(summary, targets)

    results = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_type": "neural_surrogate",
        "model_dir": model_dir,
        "dataset_version": dataset_version,
        "seed": seed,
        "config": {
            "max_base_per_stratum": max_base_per_stratum,
            "variants_per_base": variants_per_base,
            "boundary_focus": boundary_focus,
        },
        "summary": summary,
        "stratum_kind_summary": stratum_kind,
        "gate_check": gate_check,
        "row_count": len(valid_rows),
        "api_queries": queries_made,
    }

    results_path = data_dir / "neural_pattern_benchmark_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {results_path}")

    scored_rows_path = data_dir / "neural_pattern_scored_rows.jsonl"
    with scored_rows_path.open("w", encoding="utf-8") as fh:
        for row in valid_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n=== PATTERN BENCHMARK RESULTS (Neural Surrogate) ===")
    overall = summary.get("overall", {})
    print(f"  Agreement:    {overall.get('agreement_rate', 0):.4f}  (target: >= {targets.get('overall_agreement_min', 0.99)})")
    print(f"  MAE:          {overall.get('probability_mae', 1):.4f}  (target: <= {targets.get('overall_probability_mae_max', 0.02)})")
    print(f"  Transition:   {overall.get('transition_agreement', 0):.4f}  (target: >= {targets.get('overall_transition_agreement_min', 0.97)})")
    print(f"\n  Gates passed: {gate_check['all_passed']}")
    if not gate_check["all_passed"]:
        print(f"  Failures: {gate_check['failures']}")

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    return results


def _check_gates(
    summary: Dict[str, Any], targets: Dict[str, Any]
) -> Dict[str, Any]:
    failures: List[str] = []
    overall = summary.get("overall", {})

    if overall.get("agreement_rate", 0) < targets.get("overall_agreement_min", 0.99):
        failures.append(
            f"overall_agreement: {overall.get('agreement_rate', 0):.4f} < {targets['overall_agreement_min']}"
        )
    if overall.get("probability_mae", 1) > targets.get("overall_probability_mae_max", 0.02):
        failures.append(
            f"overall_mae: {overall.get('probability_mae', 1):.4f} > {targets['overall_probability_mae_max']}"
        )
    if overall.get("transition_agreement", 0) < targets.get("overall_transition_agreement_min", 0.97):
        failures.append(
            f"overall_transition: {overall.get('transition_agreement', 0):.4f} < {targets['overall_transition_agreement_min']}"
        )

    strata = summary.get("strata", {})
    min_samples = targets.get("stratum_min_samples", 30)
    for stratum_name, metrics in strata.items():
        if metrics.get("sample_count", 0) < min_samples:
            continue
        if metrics.get("agreement_rate", 0) < targets.get("stratum_agreement_min", 0.97):
            failures.append(f"{stratum_name}_agreement: {metrics['agreement_rate']:.4f}")
        if metrics.get("probability_mae", 1) > targets.get("stratum_probability_mae_max", 0.03):
            failures.append(f"{stratum_name}_mae: {metrics['probability_mae']:.4f}")
        if metrics.get("transition_agreement", 0) < targets.get("stratum_transition_agreement_min", 0.94):
            failures.append(f"{stratum_name}_transition: {metrics['transition_agreement']:.4f}")

    return {"all_passed": len(failures) == 0, "failures": failures}
