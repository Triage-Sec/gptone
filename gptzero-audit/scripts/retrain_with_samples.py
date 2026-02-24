"""Retrain the surrogate by adding sample_docs (labeled by GPTZero) to training data.

Loads all texts from sample_docs/, queries GPTZero for ground truth labels,
merges with the existing combined training data, and retrains the GBM.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def retrain(
    api_key: str,
    output_dir: str = "/artifacts/gptzero-audit",
    sample_docs_dir: str = "sample_docs",
    delay: float = 0.4,
) -> Dict[str, Any]:
    from src.client import GPTZeroClient
    from src.features import extract_features_batch, FEATURE_NAMES
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    client = GPTZeroClient(api_key=api_key)

    # Load sample_docs and query GPTZero
    samples_dir = Path(sample_docs_dir)
    if not samples_dir.exists():
        samples_dir = PROJECT_ROOT / sample_docs_dir

    new_records: List[Dict[str, Any]] = []
    print("Labeling sample_docs with GPTZero API...")

    for cat_dir in sorted(samples_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for txt_file in sorted(cat_dir.iterdir()):
            if txt_file.suffix != ".txt":
                continue
            text = txt_file.read_text().strip()
            if not text:
                continue

            resp = client.predict(text, delay=delay)
            if "error" in resp:
                print(f"  ERROR: {txt_file.name}: {resp['error']}")
                continue

            prob = resp.get("average_generated_prob")
            cls = resp.get("predicted_class")
            print(f"  {prob:.4f}  {cls:6s}  {category}/{txt_file.name}")

            # Generate multiple training records from each text
            # to increase representation of this distribution
            new_records.append({
                "base_sample_id": f"sample_{category}_{txt_file.stem}",
                "stratum": category,
                "ground_truth_label": category.replace("adversarial_", ""),
                "source": "sample_docs",
                "variant_order": 0,
                "variant_id": "base",
                "perturbation_kind": "identity",
                "perturbation_strength": 0.0,
                "text": text,
                "gptzero_prob": prob,
                "gptzero_class": cls,
                "has_error": False,
            })

    print(f"\nLabeled {len(new_records)} sample_docs texts")

    # Load existing training data
    existing_path = out / "combined_training_data.jsonl"
    existing_records: List[Dict[str, Any]] = []
    if existing_path.exists():
        with existing_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("gptzero_prob") is not None and not rec.get("has_error"):
                        existing_records.append(rec)
    print(f"Existing training records: {len(existing_records)}")

    # Oversample the new records to give them adequate weight
    # The existing data has ~35K records; 65 new ones would be invisible
    # Oversample 50x to make ~3K records from the new distribution
    oversample_factor = 50
    oversampled = new_records * oversample_factor
    print(f"Oversampled new records: {len(new_records)} x {oversample_factor} = {len(oversampled)}")

    all_records = existing_records + oversampled
    print(f"Total training records: {len(all_records)}")

    # Extract features and train
    texts = [r["text"] for r in all_records]
    probs = np.array([float(r["gptzero_prob"]) for r in all_records])
    classes = np.array([str(r.get("gptzero_class", "human")) for r in all_records])

    print("Extracting features...")
    X, _ = extract_features_batch(texts)

    print("Training probability regressor...")
    reg = GradientBoostingRegressor(
        n_estimators=3000, max_depth=10, learning_rate=0.01,
        subsample=0.8, min_samples_leaf=2, random_state=42,
    )
    reg.fit(X, probs)

    print("Training class predictor...")
    cls_model = GradientBoostingClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    cls_model.fit(X, classes)

    # Evaluate on the new records (not oversampled)
    new_texts = [r["text"] for r in new_records]
    new_true_probs = np.array([float(r["gptzero_prob"]) for r in new_records])
    new_true_classes = np.array([str(r["gptzero_class"]) for r in new_records])

    X_new, _ = extract_features_batch(new_texts)
    pred_probs = np.clip(reg.predict(X_new), 0.0, 1.0)
    pred_classes = cls_model.predict(X_new)

    mae = float(np.mean(np.abs(pred_probs - new_true_probs)))
    agreement = float(np.mean(pred_classes == new_true_classes))

    print(f"\nSample docs evaluation:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Class agreement: {agreement:.4f}")

    # Per-category
    by_cat: Dict[str, Dict] = {}
    for r, pp, pc in zip(new_records, pred_probs, pred_classes):
        cat = r["stratum"]
        by_cat.setdefault(cat, {"correct": 0, "total": 0, "errors": []})
        by_cat[cat]["total"] += 1
        if pc == r["gptzero_class"]:
            by_cat[cat]["correct"] += 1
        by_cat[cat]["errors"].append(abs(pp - float(r["gptzero_prob"])))

    for cat, data in sorted(by_cat.items()):
        agr = data["correct"] / max(1, data["total"])
        cat_mae = float(np.mean(data["errors"]))
        print(f"  {cat:20s}  agr={agr:.3f}  mae={cat_mae:.4f}  n={data['total']}")

    # Save model
    model_dir = out / "ppl_surrogate"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ppl_surrogate.pkl"
    with model_path.open("wb") as f:
        pickle.dump({
            "model": reg,
            "class_model": cls_model,
            "gpt2_model": "gpt2",
            "feature_names": list(FEATURE_NAMES),
            "use_text_features": True,
            "skip_perplexity": True,
        }, f)
    print(f"\nModel saved to {model_path}")

    # Also save locally if running on Modal
    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    return {
        "new_records": len(new_records),
        "total_records": len(all_records),
        "sample_mae": round(mae, 4),
        "sample_agreement": round(agreement, 4),
        "by_category": {k: {"agreement": v["correct"]/max(1,v["total"]), "mae": float(np.mean(v["errors"]))} for k, v in by_cat.items()},
    }
