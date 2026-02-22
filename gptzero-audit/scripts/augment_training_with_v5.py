"""Augment v6 training data with GPTZero-labeled v5 pattern_eval texts.

Queries GPTZero for v5 pattern_eval texts and their perturbation variants,
then merges with existing v6 labeled data for a combined training set.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def augment(
    api_key: str,
    output_dir: str = "/artifacts/gptzero-audit",
    dataset_version: str = "v5",
    variants_per_base: int = 5,
    delay: float = 0.25,
    seed: int = 42,
) -> Dict[str, Any]:
    from src.client import GPTZeroClient
    from src.pattern_benchmark import generate_variants

    rng = random.Random(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    eval_path = PROJECT_ROOT / "datasets" / dataset_version / "pattern_eval.jsonl"
    v5_records: List[Dict[str, Any]] = []
    with eval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                v5_records.append(json.loads(line))
    print(f"Loaded {len(v5_records)} v5 pattern_eval records")

    all_rows: List[Dict[str, Any]] = []
    for rec in v5_records:
        variants = generate_variants(rec["text"], rng=rng, max_variants=variants_per_base + 1)
        for v_idx, var in enumerate(variants):
            all_rows.append({
                "base_sample_id": rec["sample_id"],
                "stratum": rec.get("stratum", "unknown"),
                "ground_truth_label": rec.get("ground_truth_label"),
                "source": "v5_pattern_eval",
                "variant_order": v_idx,
                "variant_id": var["variant_id"],
                "perturbation_kind": var["perturbation_kind"],
                "perturbation_strength": var["perturbation_strength"],
                "text": var["text"],
            })
    print(f"Generated {len(all_rows)} rows (base + variants)")

    cache_path = out_path / "v5_gptzero_cache.json"
    cache: Dict[str, Dict] = {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            cache = raw.get("responses", {})
            print(f"Loaded {len(cache)} cached responses")
        except Exception:
            pass

    client = GPTZeroClient(api_key=api_key)
    queries = 0
    hits = 0

    for i, row in enumerate(all_rows):
        text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if text_hash in cache:
            resp = cache[text_hash]
            hits += 1
        else:
            resp = client.predict(row["text"], delay=delay)
            if "error" not in resp:
                cache[text_hash] = resp
            queries += 1

        if "error" not in resp:
            row["gptzero_prob"] = resp.get("average_generated_prob")
            row["gptzero_class"] = resp.get("predicted_class")
            row["gptzero_class_probs"] = resp.get("class_probabilities")
        else:
            row["gptzero_prob"] = None
            row["gptzero_class"] = None
            row["has_error"] = True

        if (i + 1) % 96 == 0:
            pct = (i + 1) / len(all_rows) * 100
            print(f"  [{i+1}/{len(all_rows)}] ({pct:.0f}%) queries={queries} cache={hits}")
            cache_path.write_text(json.dumps({
                "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "responses": cache,
            }), encoding="utf-8")

    cache_path.write_text(json.dumps({
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "responses": cache,
    }), encoding="utf-8")

    v5_labeled = [r for r in all_rows if r.get("gptzero_prob") is not None]
    v5_path = out_path / "v5_augment_labeled.jsonl"
    with v5_path.open("w", encoding="utf-8") as fh:
        for r in v5_labeled:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    v6_path = out_path / "corpus_v6_labeled.jsonl"
    combined_path = out_path / "combined_training_data.jsonl"
    combined_count = 0

    with combined_path.open("w", encoding="utf-8") as out_fh:
        if v6_path.exists():
            with v6_path.open("r", encoding="utf-8") as v6_fh:
                for line in v6_fh:
                    out_fh.write(line)
                    combined_count += 1
            print(f"Included {combined_count} v6 records")

        for r in v5_labeled:
            out_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            combined_count += 1

    print(f"\nCombined training data: {combined_count} records")
    print(f"  v6: {combined_count - len(v5_labeled)}")
    print(f"  v5: {len(v5_labeled)}")
    print(f"Saved to {combined_path}")

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    return {
        "v5_records": len(v5_records),
        "v5_labeled": len(v5_labeled),
        "v6_records": combined_count - len(v5_labeled),
        "combined_total": combined_count,
        "api_queries": queries,
        "cache_hits": hits,
        "combined_path": str(combined_path),
    }
