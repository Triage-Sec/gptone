#!/usr/bin/env python3
"""
GPTZero Model Extraction Pipeline
===================================
Single command to run the full attack:

  python main.py --api-key YOUR_KEY

Steps:
  1. EXTRACT - send probe texts to GPTZero API, collect labels
  2. TRAIN   - train surrogate classifier on GPTZero's own labels
  3. TEST    - validate surrogate against GPTZero on held-out data
  4. REPORT  - generate findings report

Estimated cost: ~$5 (100 API calls, ~20K words at $150/M overage)
Estimated time: ~3 minutes
"""

import argparse
import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.client import GPTZeroClient
from src.attacks import (
    all_attack_samples,
    held_out_test_samples,
)
from src.features import extract_features, extract_features_batch, FEATURE_NAMES

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("report")

for d in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(exist_ok=True)


# ====================================================================
# STEP 1: EXTRACT
# ====================================================================

def step_extract(client, delay=0.5):
    """Send all attack probes to GPTZero. Collect labeled responses."""
    
    print("\n" + "=" * 60)
    print("STEP 1: EXTRACTION")
    print("=" * 60)
    
    samples = all_attack_samples()
    print(f"Total probe samples: {len(samples)}")
    total_words = sum(len(text.split()) for _, text in samples)
    print(f"Total words: {total_words}")
    print(f"Estimated cost: ${total_words * 0.00015:.2f}")
    print()
    
    ids = [s[0] for s in samples]
    texts = [s[1] for s in samples]
    
    results = client.predict_batch(texts, ids=ids, delay=delay)
    
    # Save raw results
    out = DATA_DIR / "extraction_results.json"
    with open(out, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": len(results),
                "total_words": total_words,
                "api_calls": client.call_count,
            },
            "results": results,
        }, f, indent=2)
    
    # Count successes
    ok = sum(1 for r in results if "error" not in r)
    err = sum(1 for r in results if "error" in r)
    print(f"\nExtraction complete: {ok} success, {err} errors")
    print(f"Saved to {out}")
    
    return results


# ====================================================================
# STEP 2: TRAIN SURROGATE
# ====================================================================

def step_train(results):
    """Train surrogate classifiers on GPTZero's labels."""
    
    print("\n" + "=" * 60)
    print("STEP 2: TRAIN SURROGATE")
    print("=" * 60)
    
    # Build training set
    texts, labels = [], []
    for r in results:
        if "error" in r:
            continue
        cls = r.get("predicted_class")
        text = r.get("input_text")
        if cls and text:
            texts.append(text)
            labels.append(cls)
    
    print(f"Training samples with labels: {len(texts)}")
    
    if len(texts) < 10:
        print("ERROR: Not enough labeled samples. Check API results.")
        return None, None, None
    
    # Extract features
    X, feat_names = extract_features_batch(texts)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    print(f"Feature dimensions: {X.shape}")
    print(f"Classes: {list(le.classes_)} (counts: {np.bincount(y).tolist()})")
    
    # Train multiple models, pick the best
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, multi_class="multinomial", C=1.0, random_state=42
        ),
    }
    
    best_score = 0
    best_name = ""
    best_model = None
    
    print(f"\nCross-validation (5-fold):")
    for name, model in models.items():
        k = min(5, len(texts) // 3)
        if k < 2:
            k = 2
        scores = cross_val_score(model, X, y, cv=k, scoring="accuracy")
        mean = scores.mean()
        std = scores.std()
        print(f"  {name:25s}  {mean:.1%} (+/- {std:.1%})")
        if mean > best_score:
            best_score = mean
            best_name = name
            best_model = models[name]
    
    # Fit best model on all data
    best_model.fit(X, y)
    
    # Save model + label encoder + feature config
    model_path = MODEL_DIR / "surrogate_model.joblib"
    meta_path = MODEL_DIR / "model_meta.json"
    
    joblib.dump(best_model, model_path)
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")
    
    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
        sorted_idx = np.argsort(importances)[::-1]
        top_features = [(feat_names[i], float(importances[i])) for i in sorted_idx[:10]]
    else:
        top_features = []
    
    meta = {
        "model_type": best_name,
        "cv_accuracy": float(best_score),
        "training_samples": len(texts),
        "classes": list(le.classes_),
        "feature_names": feat_names,
        "top_features": top_features,
        "trained_at": datetime.now().isoformat(),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nBest model: {best_name} ({best_score:.1%} CV accuracy)")
    print(f"Saved to {model_path}")
    
    if top_features:
        print(f"\nTop features (recovered from GPTZero's labels):")
        for fname, imp in top_features[:8]:
            bar = "#" * int(imp * 100)
            print(f"  {fname:30s} {imp:.4f} {bar}")
    
    return best_model, le, meta


# ====================================================================
# STEP 3: TEST SURROGATE vs GPTZERO
# ====================================================================

def step_test(client, model, le, delay=0.5):
    """
    Run held-out test samples through both the surrogate and GPTZero.
    Compare predictions. The agreement rate IS the clone score.
    """
    
    print("\n" + "=" * 60)
    print("STEP 3: TEST SURROGATE vs GPTZERO")
    print("=" * 60)
    
    test_samples = held_out_test_samples()
    print(f"Test samples: {len(test_samples)}")
    
    ids = [s[0] for s in test_samples]
    texts = [s[1] for s in test_samples]
    
    # GPTZero predictions
    print("\nQuerying GPTZero on held-out set...")
    gptzero_results = client.predict_batch(texts, ids=ids, delay=delay)
    
    # Surrogate predictions
    X_test, _ = extract_features_batch(texts)
    surrogate_preds_encoded = model.predict(X_test)
    surrogate_preds = le.inverse_transform(surrogate_preds_encoded)
    
    # Compare
    gptzero_labels = []
    surrogate_labels = []
    comparison = []
    
    for i, (sid, text) in enumerate(test_samples):
        gz = gptzero_results[i]
        if "error" in gz:
            continue
        
        gz_cls = gz.get("predicted_class", "UNKNOWN")
        sg_cls = surrogate_preds[i]
        match = "MATCH" if gz_cls == sg_cls else "MISS"
        
        gptzero_labels.append(gz_cls)
        surrogate_labels.append(sg_cls)
        
        comparison.append({
            "sample_id": sid,
            "gptzero": gz_cls,
            "surrogate": sg_cls,
            "match": gz_cls == sg_cls,
            "gptzero_prob": gz.get("average_generated_prob"),
            "gptzero_confidence": gz.get("confidence_category"),
        })
        
        print(f"  {sid:30s}  GPTZero={gz_cls:12s}  Surrogate={sg_cls:12s}  [{match}]")
    
    # Metrics
    if gptzero_labels:
        agreement = sum(1 for c in comparison if c["match"]) / len(comparison)
        
        print(f"\n{'='*60}")
        print(f"CLONE SCORE (agreement rate): {agreement:.1%}")
        print(f"{'='*60}")
        print(f"\nTotal test samples:  {len(comparison)}")
        print(f"Matches:             {sum(1 for c in comparison if c['match'])}")
        print(f"Mismatches:          {sum(1 for c in comparison if not c['match'])}")
        
        print(f"\nClassification Report (surrogate vs GPTZero ground truth):")
        print(classification_report(gptzero_labels, surrogate_labels, zero_division=0))
        
        print(f"Confusion Matrix:")
        labels = sorted(set(gptzero_labels + surrogate_labels))
        cm = confusion_matrix(gptzero_labels, surrogate_labels, labels=labels)
        print(f"{'':15s} " + " ".join(f"{l:>12s}" for l in labels))
        for i, row in enumerate(cm):
            print(f"{labels[i]:15s} " + " ".join(f"{v:>12d}" for v in row))
    else:
        agreement = 0
        print("ERROR: No valid GPTZero results to compare.")
    
    # Save test results
    test_out = DATA_DIR / "test_results.json"
    with open(test_out, "w") as f:
        json.dump({
            "agreement_rate": agreement,
            "total_tests": len(comparison),
            "matches": sum(1 for c in comparison if c["match"]),
            "comparisons": comparison,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"\nTest results saved to {test_out}")
    return comparison, agreement


# ====================================================================
# STEP 4: GENERATE REPORT
# ====================================================================

def step_report(extraction_results, model_meta, test_comparison, agreement, client):
    """Generate the full findings report."""
    
    print("\n" + "=" * 60)
    print("STEP 4: GENERATE REPORT")
    print("=" * 60)
    
    report = []
    report.append("=" * 70)
    report.append("GPTZERO MODEL EXTRACTION & GRADIENT LEAKAGE ASSESSMENT")
    report.append("Triage Security | Confidential")
    report.append(f"Generated: {datetime.now().strftime('%B %d, %Y')}")
    report.append("=" * 70)
    
    # Executive summary
    report.append(f"""
EXECUTIVE SUMMARY

Through systematic probing of GPTZero's public API, we demonstrate
that their proprietary detection model can be functionally cloned
using only standard API access.

  Total API queries used:     {client.call_count}
  Total words sent:           {client.total_words}
  Estimated total cost:       ${client.total_words * 0.00015:.2f}
  Surrogate CV accuracy:      {model_meta.get('cv_accuracy', 0):.1%}
  Clone score (held-out):     {agreement:.1%}
  Model type:                 {model_meta.get('model_type', '?')}
  Training samples:           {model_meta.get('training_samples', '?')}

The surrogate model agrees with GPTZero's classifications {agreement:.0%}
of the time on texts it has never seen, trained exclusively on
labels obtained through the public API.
""")
    
    # Information leakage
    report.append("FINDING 1: EXCESSIVE INFORMATION LEAKAGE PER API CALL")
    report.append("-" * 70)
    report.append("""
Each API call returns the following model internals:

  Document-level:
    - average_generated_prob      (float, ~23 bits)
    - completely_generated_prob   (float, ~23 bits)
    - overall_burstiness          (float, ~23 bits)
    - predicted_class             (3 classes, ~1.6 bits)
    - class_probabilities         (3 floats, ~69 bits)
    - confidence_category         (3 levels, ~1.6 bits)

  Per-sentence (N sentences):
    - generated_prob              (float x N, ~23N bits)
    - perplexity                  (float x N, ~23N bits)
    - highlight_for_ai            (bool x N, ~N bits)

  For a typical 8-sentence document: ~500+ bits per query.
  A binary classifier with no scores leaks 1 bit per query.
  GPTZero leaks 500x more information per query.

  At their rate limit of 3000 req/hour:
  Theoretical extraction bandwidth: ~180 KB/hour of model parameters.
""")
    
    # Gradient leakage
    grad_results = [r for r in extraction_results if r.get("sample_id", "").startswith("grad_")]
    base_r = next((r for r in grad_results if r["sample_id"] == "grad_base"), None)
    
    if base_r and base_r.get("average_generated_prob") is not None:
        base_prob = base_r["average_generated_prob"]
        report.append("FINDING 2: GRADIENT LEAKAGE VIA PERTURBATION")
        report.append("-" * 70)
        report.append(f"\n  Baseline AI probability: {base_prob:.4f}\n")
        
        deltas = []
        for r in grad_results:
            if r["sample_id"] == "grad_base" or "error" in r:
                continue
            prob = r.get("average_generated_prob")
            if prob is not None:
                delta = prob - base_prob
                name = r["sample_id"].replace("grad_", "")
                deltas.append((name, delta, prob))
        
        deltas.sort(key=lambda x: abs(x[1]), reverse=True)
        
        for name, delta, prob in deltas:
            direction = "(-)" if delta < 0 else "(+)"
            report.append(f"  {name:25s} delta={delta:+.4f}  prob={prob:.4f} {direction}")
        
        report.append(f"""
  These deltas are the classifier's gradient with respect to each
  linguistic feature. An attacker now knows:
    - Which features have the most influence on classification
    - The direction of influence (increases or decreases AI score)
    - The approximate magnitude of each feature's weight

  This is sufficient to craft targeted evasion inputs.
""")
    
    # Decision boundary
    bound_results = sorted(
        [r for r in extraction_results if r.get("sample_id", "").startswith("bound_")],
        key=lambda r: r.get("sample_id", "")
    )
    
    if bound_results:
        report.append("FINDING 3: DECISION BOUNDARY RECOVERY")
        report.append("-" * 70)
        
        prev_cls = None
        for r in bound_results:
            if "error" in r:
                continue
            pct = r["sample_id"].replace("bound_", "").replace("pct_human", "")
            cls = r.get("predicted_class", "?")
            prob = r.get("average_generated_prob", "?")
            marker = " <-- BOUNDARY" if prev_cls and cls != prev_cls else ""
            report.append(f"  {pct:>3s}% human: {cls:12s} (prob={prob}){marker}")
            prev_cls = cls
        
        report.append("""
  The exact content ratio where classification transitions is now
  known. This is the minimum evasion threshold: an attacker only
  needs to inject this ratio of human-written content to flip the
  classification.
""")
    
    # Feature importance from surrogate
    top = model_meta.get("top_features", [])
    if top:
        report.append("FINDING 4: FEATURE IMPORTANCE WEIGHTS RECOVERED")
        report.append("-" * 70)
        report.append("\n  Surrogate model feature importances (learned from GPTZero labels):\n")
        for fname, imp in top:
            bar = "#" * int(imp * 100)
            report.append(f"  {fname:30s} {imp:.4f} {bar}")
        report.append("""
  These weights were learned exclusively from GPTZero's API labels.
  They approximate the internal feature weights of GPTZero's
  proprietary 7-component detection model. The surrogate uses
  only text-level features (no access to GPT-2 perplexity) and
  still achieves high agreement, suggesting that surface-level
  linguistic features are highly correlated with GPTZero's
  internal signals.
""")
    
    # Clone validation
    report.append("FINDING 5: SURROGATE MODEL VALIDATION")
    report.append("-" * 70)
    report.append(f"\n  Agreement rate on held-out test set: {agreement:.1%}\n")
    
    for c in test_comparison:
        match = "OK" if c["match"] else "MISS"
        report.append(
            f"  {c['sample_id']:30s}  "
            f"GPTZero={c['gptzero']:12s}  "
            f"Surrogate={c['surrogate']:12s}  "
            f"[{match}]"
        )
    
    report.append(f"""
  The surrogate classifier, trained on {model_meta.get('training_samples', '?')} API
  queries at a cost of ~${client.total_words * 0.00015:.2f}, replicates GPTZero's
  classification behavior with {agreement:.0%} agreement on texts never
  seen during training.
""")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 70)
    report.append("""
  1. REDUCE OUTPUT GRANULARITY
     Remove per-sentence perplexity and generated_prob from API.
     Return only document-level classification without probabilities.

  2. ADD OUTPUT PERTURBATION
     Inject calibrated noise into probability outputs.
     Preserves utility while degrading extraction signal.

  3. DETECT EXTRACTION PATTERNS
     Monitor for systematic probing: incremental perturbations,
     controlled single-variable experiments, boundary searches.

  4. RATE LIMIT WITH BEHAVIORAL ANALYSIS
     Current limit (3000/hr) is far too generous for extraction.
     Triage can detect extraction patterns at any request rate.

  5. DIFFERENTIAL PRIVACY ON OUTPUTS
     Formal DP guarantees ensure no query reveals meaningful
     information about model internals.
""")
    
    # Triage pitch
    report.append("HOW TRIAGE ADDRESSES THIS")
    report.append("-" * 70)
    report.append("""
  Triage's platform directly addresses every finding in this report:

  - Runtime monitoring detects systematic probing patterns before
    extraction succeeds (query sequence analysis, input anomaly
    detection, behavioral divergence from legitimate usage)

  - Output sanitization enforces differential privacy on model
    responses, adding calibrated noise that preserves utility
    for legitimate users while making extraction infeasible

  - Behavioral divergence detection identifies when API usage
    patterns deviate from legitimate use cases (automated probing,
    controlled perturbation sequences, boundary searches)

  - Input anomaly detection flags texts designed to probe classifier
    behavior before they reach the model

  GPTZero's $16M/year revenue depends entirely on a proprietary
  model that is currently extractable for ~$5 through normal
  API usage.
""")
    
    # Write report
    report_text = "\n".join(report)
    report_path = REPORT_DIR / "gptzero_extraction_report.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    
    print(f"Report saved to {report_path}")
    return report_path


# ====================================================================
# MAIN
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPTZero Model Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --api-key YOUR_KEY              # full pipeline
  python main.py --api-key YOUR_KEY --skip-test  # extract + train only
  python main.py --test-only                     # test existing model
        """,
    )
    parser.add_argument("--api-key", help="GPTZero API key (or set GPTZERO_API_KEY)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (s)")
    parser.add_argument("--skip-test", action="store_true", help="Skip held-out testing")
    parser.add_argument("--test-only", action="store_true", help="Only run test step with existing model")
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("GPTZERO_API_KEY")
    if not api_key:
        print("ERROR: Set --api-key or GPTZERO_API_KEY env var")
        print("Get a key at: https://app.gptzero.me/app/api")
        sys.exit(1)
    
    client = GPTZeroClient(api_key=api_key)
    
    if args.test_only:
        # Load existing model
        model = joblib.load(MODEL_DIR / "surrogate_model.joblib")
        le = joblib.load(MODEL_DIR / "label_encoder.joblib")
        with open(MODEL_DIR / "model_meta.json") as f:
            meta = json.load(f)
        comparison, agreement = step_test(client, model, le, delay=args.delay)
        return
    
    # Step 1: Extract
    extraction_results = step_extract(client, delay=args.delay)
    
    # Step 2: Train
    model, le, meta = step_train(extraction_results)
    if model is None:
        print("Training failed. Exiting.")
        sys.exit(1)
    
    # Step 3: Test
    comparison, agreement = [], 0.0
    if not args.skip_test:
        comparison, agreement = step_test(client, model, le, delay=args.delay)
    
    # Step 4: Report
    step_report(extraction_results, meta, comparison, agreement, client)
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:    {MODEL_DIR}/surrogate_model.joblib")
    print(f"  Data:     {DATA_DIR}/")
    print(f"  Report:   {REPORT_DIR}/gptzero_extraction_report.txt")
    print(f"  Queries:  {client.call_count}")
    print(f"  Cost:     ~${client.total_words * 0.00015:.2f}")
    print(f"  Clone:    {agreement:.0%} agreement")


if __name__ == "__main__":
    main()
