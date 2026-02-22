"""Build a training set from v5 pattern_eval using cached GPTZero responses.

Uses the pattern benchmark cache from previous M-runs (already on Modal volume)
to avoid re-querying the GPTZero API.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def build_v5_training_set(
    *,
    cache_dir: str = "/artifacts/gptzero-audit",
    output_dir: str = "/artifacts/gptzero-audit",
    variants_per_base: int = 5,
    seed: int = 42,
    api_key: Optional[str] = None,
    delay: float = 0.25,
) -> Dict[str, Any]:
    """Build training JSONL from v5 pattern_eval + any available GPTZero caches."""
    from src.pattern_benchmark import generate_variants

    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eval_path = PROJECT_ROOT / "datasets" / "v5" / "pattern_eval.jsonl"
    records: List[Dict] = []
    with eval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} v5 pattern_eval records")

    cache: Dict[str, Dict] = {}
    cache_base = Path(cache_dir)
    cache_files = list(cache_base.rglob("*cache*.json")) + list(cache_base.rglob("*gptzero*cache*.json"))
    for cp in cache_files:
        try:
            raw = json.loads(cp.read_text(encoding="utf-8"))
            responses = raw.get("responses", {})
            if responses:
                cache.update(responses)
                print(f"  Loaded {len(responses)} entries from {cp}")
        except Exception:
            continue
    print(f"Total cached responses: {len(cache)}")

    all_rows: List[Dict[str, Any]] = []
    for rec in records:
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

    labeled = 0
    unlabeled = 0
    for row in all_rows:
        text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if text_hash in cache:
            resp = cache[text_hash]
            row["gptzero_prob"] = resp.get("average_generated_prob")
            row["gptzero_class"] = resp.get("predicted_class")
            row["gptzero_class_probs"] = resp.get("class_probabilities")
            row["has_error"] = False
            labeled += 1
        else:
            row["gptzero_prob"] = None
            row["gptzero_class"] = None
            row["has_error"] = True
            unlabeled += 1

    print(f"Cache hit: {labeled}/{len(all_rows)} ({100*labeled/len(all_rows):.1f}%)")
    print(f"Cache miss: {unlabeled}")

    if unlabeled > 0 and api_key:
        from src.client import GPTZeroClient
        print(f"Querying GPTZero for {unlabeled} uncached texts...")
        client = GPTZeroClient(api_key=api_key)
        queries = 0
        for row in all_rows:
            if row.get("gptzero_prob") is not None:
                continue
            resp = client.predict(row["text"], delay=delay)
            queries += 1
            if "error" not in resp:
                text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                cache[text_hash] = resp
                row["gptzero_prob"] = resp.get("average_generated_prob")
                row["gptzero_class"] = resp.get("predicted_class")
                row["gptzero_class_probs"] = resp.get("class_probabilities")
                row["has_error"] = False
            if queries % 96 == 0:
                print(f"  {queries} queries made...")
        print(f"  Total queries: {queries}")

    v6_path = Path(cache_dir) / "corpus_v6_labeled.jsonl"
    v5_valid = [r for r in all_rows if r.get("gptzero_prob") is not None]

    combined_path = out / "combined_training_data.jsonl"
    v6_count = 0
    with combined_path.open("w", encoding="utf-8") as fh:
        if v6_path.exists():
            with v6_path.open("r", encoding="utf-8") as v6_fh:
                for line in v6_fh:
                    fh.write(line)
                    v6_count += 1
        for r in v5_valid:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = v6_count + len(v5_valid)
    print(f"\nCombined training set: {total} records (v6: {v6_count}, v5: {len(v5_valid)})")
    print(f"Written to {combined_path}")

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    return {
        "v5_records": len(records),
        "v5_labeled": len(v5_valid),
        "v5_unlabeled": unlabeled,
        "v6_records": v6_count,
        "combined_total": total,
    }
