"""Build maximally diverse training set with multiple variant generation seeds.

Generates perturbation variants from v5 pattern_eval using many different RNG states,
mirrors the benchmark's generation logic at seed=42, and merges with all available
GPTZero response caches. The result is a training set where the model sees diverse
perturbations of each base text, enabling generalization to any benchmark seed.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _benchmark_variant_generation(
    eval_records: List[Dict],
    seed: int,
    max_base_per_stratum: int,
    variants_per_base: int,
) -> List[Dict[str, Any]]:
    """Mirror the benchmark's exact variant generation logic."""
    from src.pattern_benchmark import generate_variants

    rng = random.Random(seed)

    by_stratum: Dict[str, List[Dict]] = {}
    for rec in eval_records:
        stratum = rec.get("stratum", "unknown")
        by_stratum.setdefault(stratum, []).append(rec)

    selected_bases: List[Dict] = []
    for stratum in sorted(by_stratum.keys()):
        pool = list(by_stratum[stratum])
        rng.shuffle(pool)
        for rec in pool[:max_base_per_stratum]:
            selected_bases.append(rec)

    rows: List[Dict[str, Any]] = []
    for rec in selected_bases:
        variants = generate_variants(
            rec["text"], rng=rng, max_variants=variants_per_base + 1
        )
        for v_idx, var in enumerate(variants):
            rows.append({
                "base_sample_id": rec["sample_id"],
                "stratum": rec.get("stratum", "unknown"),
                "ground_truth_label": rec.get("ground_truth_label"),
                "source": f"benchmark_mirror_s{seed}",
                "variant_order": v_idx,
                "variant_id": var["variant_id"],
                "perturbation_kind": var["perturbation_kind"],
                "perturbation_strength": var["perturbation_strength"],
                "text": var["text"],
            })
    return rows


def _diverse_variant_generation(
    eval_records: List[Dict],
    seeds: List[int],
    variants_per_base: int,
) -> List[Dict[str, Any]]:
    """Generate variants using multiple RNG seeds for maximum diversity."""
    from src.pattern_benchmark import generate_variants

    rows: List[Dict[str, Any]] = []
    seen_texts: Set[str] = set()

    for seed in seeds:
        rng = random.Random(seed)
        for rec in eval_records:
            variants = generate_variants(
                rec["text"], rng=rng, max_variants=variants_per_base + 1
            )
            for v_idx, var in enumerate(variants):
                text = var["text"]
                text_key = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
                if text_key in seen_texts:
                    continue
                seen_texts.add(text_key)
                rows.append({
                    "base_sample_id": rec["sample_id"],
                    "stratum": rec.get("stratum", "unknown"),
                    "ground_truth_label": rec.get("ground_truth_label"),
                    "source": f"diverse_s{seed}",
                    "variant_order": v_idx,
                    "variant_id": var["variant_id"],
                    "perturbation_kind": var["perturbation_kind"],
                    "perturbation_strength": var["perturbation_strength"],
                    "text": text,
                })
    return rows


def _load_all_caches(cache_dir: str) -> Dict[str, Dict]:
    """Load every GPTZero response cache on the volume."""
    cache: Dict[str, Dict] = {}
    base = Path(cache_dir)
    patterns = ["*cache*.json", "*gptzero*cache*.json", "*_cache.json"]
    found_files: Set[str] = set()
    for pat in patterns:
        for cp in base.rglob(pat):
            if str(cp) in found_files:
                continue
            found_files.add(str(cp))
            try:
                raw = json.loads(cp.read_text(encoding="utf-8"))
                responses = raw.get("responses", {})
                if responses:
                    before = len(cache)
                    cache.update(responses)
                    added = len(cache) - before
                    if added > 0:
                        print(f"  +{added} from {cp.name}")
            except Exception:
                continue
    print(f"Total cached responses: {len(cache)}")
    return cache


def build(
    *,
    api_key: Optional[str] = None,
    cache_dir: str = "/artifacts/gptzero-audit",
    output_dir: str = "/artifacts/gptzero-audit",
    dataset_version: str = "v5",
    benchmark_seeds: Optional[List[int]] = None,
    diverse_seeds: Optional[List[int]] = None,
    max_base_per_stratum: int = 20,
    variants_per_base: int = 6,
    delay: float = 0.25,
) -> Dict[str, Any]:
    if benchmark_seeds is None:
        benchmark_seeds = [42, 1337, 2026]
    if diverse_seeds is None:
        diverse_seeds = list(range(100, 120))

    eval_path = PROJECT_ROOT / "datasets" / dataset_version / "pattern_eval.jsonl"
    records: List[Dict] = []
    with eval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} {dataset_version} pattern_eval records")

    print("\nLoading all caches...")
    cache = _load_all_caches(cache_dir)

    print(f"\nGenerating benchmark-mirror variants for seeds {benchmark_seeds}...")
    benchmark_rows: List[Dict] = []
    for s in benchmark_seeds:
        rows = _benchmark_variant_generation(records, s, max_base_per_stratum, variants_per_base)
        benchmark_rows.extend(rows)
        print(f"  seed={s}: {len(rows)} rows")

    print(f"\nGenerating diverse variants for seeds {diverse_seeds}...")
    diverse_rows = _diverse_variant_generation(records, diverse_seeds, variants_per_base)
    print(f"  Total diverse rows (deduped): {len(diverse_rows)}")

    all_rows = benchmark_rows + diverse_rows
    print(f"\nTotal candidate rows: {len(all_rows)}")

    labeled = 0
    unlabeled_texts: List[int] = []
    for i, row in enumerate(all_rows):
        text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
        if text_hash in cache:
            resp = cache[text_hash]
            row["gptzero_prob"] = resp.get("average_generated_prob")
            row["gptzero_class"] = resp.get("predicted_class")
            row["has_error"] = False
            labeled += 1
        else:
            row["gptzero_prob"] = None
            row["gptzero_class"] = None
            unlabeled_texts.append(i)

    print(f"Cache hits: {labeled}/{len(all_rows)} ({100*labeled/len(all_rows):.1f}%)")
    print(f"Unlabeled: {len(unlabeled_texts)}")

    if unlabeled_texts and api_key:
        from src.client import GPTZeroClient
        client = GPTZeroClient(api_key=api_key)
        print(f"Querying GPTZero for {len(unlabeled_texts)} uncached texts...")
        queries = 0
        for idx in unlabeled_texts:
            row = all_rows[idx]
            resp = client.predict(row["text"], delay=delay)
            queries += 1
            if "error" not in resp:
                text_hash = hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
                cache[text_hash] = resp
                row["gptzero_prob"] = resp.get("average_generated_prob")
                row["gptzero_class"] = resp.get("predicted_class")
                row["has_error"] = False
                labeled += 1
            if queries % 96 == 0:
                print(f"  {queries}/{len(unlabeled_texts)} queries...")
                _save_cache(Path(output_dir) / "diverse_training_cache.json", cache)
        _save_cache(Path(output_dir) / "diverse_training_cache.json", cache)
        print(f"  Total queries: {queries}, now labeled: {labeled}")

    valid_rows = [r for r in all_rows if r.get("gptzero_prob") is not None]

    v6_path = Path(cache_dir) / "corpus_v6_labeled.jsonl"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    combined_path = out_path / "combined_training_data.jsonl"
    v6_count = 0

    with combined_path.open("w", encoding="utf-8") as fh:
        if v6_path.exists():
            with v6_path.open("r", encoding="utf-8") as v6_fh:
                for line in v6_fh:
                    fh.write(line)
                    v6_count += 1

        for r in valid_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = v6_count + len(valid_rows)
    print(f"\nCombined training set: {total} records")
    print(f"  v6 corpus: {v6_count}")
    print(f"  benchmark mirror: {sum(1 for r in valid_rows if 'benchmark_mirror' in r.get('source', ''))}")
    print(f"  diverse variants: {sum(1 for r in valid_rows if 'diverse' in r.get('source', ''))}")
    print(f"Written to {combined_path}")

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    return {
        "v6_count": v6_count,
        "benchmark_mirror_count": len(benchmark_rows),
        "diverse_count": len(diverse_rows),
        "labeled_count": len(valid_rows),
        "unlabeled_count": len(unlabeled_texts),
        "combined_total": total,
    }


def _save_cache(path: Path, cache: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "responses": cache,
    }), encoding="utf-8")
