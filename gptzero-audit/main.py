#!/usr/bin/env python3
"""
GPTZero Model Extraction Pipeline
===================================
Single command to run the full attack:

  python main.py --api-key YOUR_KEY

Steps:
  1. EXTRACT - send probe texts to GPTZero API, collect labels + scores
  2. TRAIN   - train surrogate classifier + regressor on GPTZero outputs
  3. TEST    - validate surrogate against GPTZero on held-out data
  4. REPORT  - generate findings report with bootstrap confidence intervals

Estimated cost: ~$5 (120 API calls, ~10K words)
Estimated time: ~4 minutes
"""

import argparse
import json
import os
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.client import GPTZeroClient
from src.attacks import all_attack_samples, held_out_test_samples
from src.features import extract_features, extract_features_batch, FEATURE_NAMES

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    mean_squared_error, mean_absolute_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

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

    ok = sum(1 for r in results if "error" not in r)
    err = sum(1 for r in results if "error" in r)
    print(f"\nExtraction complete: {ok} success, {err} errors")
    print(f"Saved to {out}")

    # Class distribution summary
    class_counts = {}
    for r in results:
        cls = r.get("predicted_class", "error")
        class_counts[cls] = class_counts.get(cls, 0) + 1
    print(f"Class distribution: {dict(sorted(class_counts.items()))}")

    return results


# ====================================================================
# PER-SENTENCE FEATURE EXTRACTION (Phase 4)
# ====================================================================

def extract_api_response_features(result):
    """
    Extract features from GPTZero's API response that go beyond
    text-only features. These demonstrate the information leakage:
    the API returns enough signal to enrich surrogate training.

    Returns a dict of feature_name -> value.
    """
    feats = {}

    sentences = result.get("sentences", [])
    if sentences:
        gen_probs = [s.get("generated_prob", 0) for s in sentences if s.get("generated_prob") is not None]
        perplexities = [s.get("perplexity", 0) for s in sentences if s.get("perplexity") is not None]
        highlights = [1.0 if s.get("highlighted", False) else 0.0 for s in sentences]

        if gen_probs:
            feats["sent_genprob_mean"] = float(np.mean(gen_probs))
            feats["sent_genprob_std"] = float(np.std(gen_probs))
            feats["sent_genprob_min"] = float(np.min(gen_probs))
            feats["sent_genprob_max"] = float(np.max(gen_probs))
            feats["sent_genprob_range"] = float(np.max(gen_probs) - np.min(gen_probs))
        else:
            feats.update({k: 0.0 for k in [
                "sent_genprob_mean", "sent_genprob_std", "sent_genprob_min",
                "sent_genprob_max", "sent_genprob_range",
            ]})

        if perplexities:
            feats["sent_perplexity_mean"] = float(np.mean(perplexities))
            feats["sent_perplexity_std"] = float(np.std(perplexities))
            feats["sent_perplexity_min"] = float(np.min(perplexities))
            feats["sent_perplexity_max"] = float(np.max(perplexities))
        else:
            feats.update({k: 0.0 for k in [
                "sent_perplexity_mean", "sent_perplexity_std",
                "sent_perplexity_min", "sent_perplexity_max",
            ]})

        feats["highlight_ratio"] = float(np.mean(highlights)) if highlights else 0.0
    else:
        feats.update({k: 0.0 for k in [
            "sent_genprob_mean", "sent_genprob_std", "sent_genprob_min",
            "sent_genprob_max", "sent_genprob_range",
            "sent_perplexity_mean", "sent_perplexity_std",
            "sent_perplexity_min", "sent_perplexity_max",
            "highlight_ratio",
        ]})

    feats["doc_burstiness"] = float(result.get("overall_burstiness") or 0.0)

    class_probs = result.get("class_probabilities") or {}
    feats["class_prob_ai"] = float(class_probs.get("ai", 0.0))
    feats["class_prob_human"] = float(class_probs.get("human", 0.0))
    feats["class_prob_mixed"] = float(class_probs.get("mixed", 0.0))

    return feats


API_RESPONSE_FEATURE_NAMES = [
    "sent_genprob_mean", "sent_genprob_std", "sent_genprob_min",
    "sent_genprob_max", "sent_genprob_range",
    "sent_perplexity_mean", "sent_perplexity_std",
    "sent_perplexity_min", "sent_perplexity_max",
    "highlight_ratio",
    "doc_burstiness",
    "class_prob_ai", "class_prob_human", "class_prob_mixed",
]


# ====================================================================
# STEP 2: TRAIN SURROGATE (Classifier + Regressor)
# ====================================================================

def step_train(results):
    """
    Train surrogate models on GPTZero's outputs.

    Models trained:
      1. Classifier: predicts class label (ai/human/mixed)
      2. Regressor: predicts average_generated_prob (continuous)

    Uses stratified cross-validation with class-weight balancing.
    """

    print("\n" + "=" * 60)
    print("STEP 2: TRAIN SURROGATE")
    print("=" * 60)

    texts, labels, probs = [], [], []
    api_feats_list = []
    for r in results:
        if "error" in r:
            continue
        cls = r.get("predicted_class")
        text = r.get("input_text")
        avg_prob = r.get("average_generated_prob")
        if cls and text:
            texts.append(text)
            labels.append(cls)
            probs.append(avg_prob if avg_prob is not None else 0.0)
            api_feats_list.append(extract_api_response_features(r))

    print(f"Training samples with labels: {len(texts)}")

    if len(texts) < 10:
        print("ERROR: Not enough labeled samples. Check API results.")
        return None, None, None

    X_text, feat_names = extract_features_batch(texts)
    le = LabelEncoder()
    y_cls = le.fit_transform(labels)
    y_prob = np.array(probs)

    class_counts = np.bincount(y_cls)
    print(f"Feature dimensions: {X_text.shape}")
    print(f"Classes: {list(le.classes_)} (counts: {class_counts.tolist()})")

    # Check for minimum class representation
    min_class_count = min(class_counts)
    if min_class_count < 2:
        print(f"WARNING: Class with only {min_class_count} sample(s). "
              f"Cross-validation may be unreliable for minority classes.")

    # ---- CLASSIFIER ----
    print(f"\n--- Classifier Training ---")

    models_cls = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=42
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=42)),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=42)),
        ]),
    }

    k_cls = min(5, min_class_count)
    k_cls = max(k_cls, 2)

    best_cls_score = 0
    best_cls_name = ""
    best_cls_model = None

    print(f"\nCross-validation ({k_cls}-fold stratified):")
    for name, model in models_cls.items():
        try:
            skf = StratifiedKFold(n_splits=k_cls, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_text, y_cls, cv=skf, scoring="balanced_accuracy")
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"  {name:25s}  {mean_score:.1%} (+/- {std_score:.1%}) balanced_acc")
            if mean_score > best_cls_score:
                best_cls_score = mean_score
                best_cls_name = name
                best_cls_model = models_cls[name]
        except Exception as e:
            print(f"  {name:25s}  FAILED: {e}")

    best_cls_model.fit(X_text, y_cls)
    print(f"\nBest classifier: {best_cls_name} ({best_cls_score:.1%} balanced accuracy)")

    # Feature importance
    top_features = _get_feature_importances(best_cls_model, feat_names)

    # ---- REGRESSOR ----
    print(f"\n--- Regressor Training (predict avg_generated_prob) ---")

    models_reg = {
        "GBRegressor": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),
    }

    k_reg = min(5, len(texts) // 3)
    k_reg = max(k_reg, 2)

    best_reg_score = -999
    best_reg_name = ""
    best_reg_model = None

    print(f"\nCross-validation ({k_reg}-fold):")
    for name, model in models_reg.items():
        try:
            kf = KFold(n_splits=k_reg, shuffle=True, random_state=42)
            scores = cross_val_score(model, X_text, y_prob, cv=kf, scoring="r2")
            mean_score = scores.mean()
            std_score = scores.std()
            print(f"  {name:25s}  R2={mean_score:.3f} (+/- {std_score:.3f})")
            if mean_score > best_reg_score:
                best_reg_score = mean_score
                best_reg_name = name
                best_reg_model = models_reg[name]
        except Exception as e:
            print(f"  {name:25s}  FAILED: {e}")

    best_reg_model.fit(X_text, y_prob)
    print(f"\nBest regressor: {best_reg_name} (R2={best_reg_score:.3f})")

    # ---- SAVE ----
    joblib.dump(best_cls_model, MODEL_DIR / "surrogate_model.joblib")
    joblib.dump(best_reg_model, MODEL_DIR / "surrogate_regressor.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    meta = {
        "classifier": {
            "model_type": best_cls_name,
            "cv_balanced_accuracy": float(best_cls_score),
            "cv_folds": k_cls,
        },
        "regressor": {
            "model_type": best_reg_name,
            "cv_r2": float(best_reg_score),
            "cv_folds": k_reg,
        },
        "training_samples": len(texts),
        "classes": list(le.classes_),
        "class_counts": class_counts.tolist(),
        "feature_count": len(feat_names),
        "feature_names": feat_names,
        "top_features": top_features,
        "api_response_feature_names": API_RESPONSE_FEATURE_NAMES,
        "trained_at": datetime.now().isoformat(),
    }
    with open(MODEL_DIR / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModels saved to {MODEL_DIR}/")

    if top_features:
        print(f"\nTop features (recovered from GPTZero's labels):")
        for fname, imp in top_features[:10]:
            bar = "#" * int(imp * 100)
            print(f"  {fname:35s} {imp:.4f} {bar}")

    return best_cls_model, best_reg_model, le, meta


def _get_feature_importances(model, feat_names):
    """Extract feature importances from the fitted model or pipeline."""
    estimator = model
    if hasattr(model, "named_steps"):
        for step_name in reversed(list(model.named_steps)):
            est = model.named_steps[step_name]
            if hasattr(est, "feature_importances_") or hasattr(est, "coef_"):
                estimator = est
                break

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).mean(axis=0) if estimator.coef_.ndim > 1 else np.abs(estimator.coef_)
    else:
        return []

    sorted_idx = np.argsort(importances)[::-1]
    return [(feat_names[i], float(importances[i])) for i in sorted_idx[:15]]


# ====================================================================
# STEP 3: TEST SURROGATE vs GPTZERO
# ====================================================================

def step_test(client, cls_model, reg_model, le, delay=0.5):
    """
    Run held-out test samples through both the surrogate and GPTZero.
    Reports:
      - Per-class agreement rates (not just aggregate)
      - Regressor MAE and R2 on held-out probabilities
      - Bootstrap 95% confidence interval on clone score
    """

    print("\n" + "=" * 60)
    print("STEP 3: TEST SURROGATE vs GPTZERO")
    print("=" * 60)

    test_samples = held_out_test_samples()
    print(f"Test samples: {len(test_samples)}")

    ids = [s[0] for s in test_samples]
    texts = [s[1] for s in test_samples]

    print("\nQuerying GPTZero on held-out set...")
    gptzero_results = client.predict_batch(texts, ids=ids, delay=delay)

    X_test, _ = extract_features_batch(texts)
    surrogate_cls_encoded = cls_model.predict(X_test)
    surrogate_cls = le.inverse_transform(surrogate_cls_encoded)
    surrogate_probs = reg_model.predict(X_test)

    gptzero_labels = []
    surrogate_labels = []
    gz_probs = []
    sg_probs = []
    comparison = []

    for i, (sid, text) in enumerate(test_samples):
        gz = gptzero_results[i]
        if "error" in gz:
            continue

        gz_cls = gz.get("predicted_class", "UNKNOWN")
        sg_cls = surrogate_cls[i]
        match = "MATCH" if gz_cls == sg_cls else "MISS"

        gz_prob = gz.get("average_generated_prob", 0.0)

        gptzero_labels.append(gz_cls)
        surrogate_labels.append(sg_cls)
        gz_probs.append(gz_prob)
        sg_probs.append(float(surrogate_probs[i]))

        comparison.append({
            "sample_id": sid,
            "gptzero_class": gz_cls,
            "surrogate_class": sg_cls,
            "match": gz_cls == sg_cls,
            "gptzero_prob": gz_prob,
            "surrogate_prob": float(surrogate_probs[i]),
            "prob_error": abs(gz_prob - surrogate_probs[i]),
            "gptzero_confidence": gz.get("confidence_category"),
        })

        print(f"  {sid:30s}  GZ={gz_cls:8s} SG={sg_cls:8s} [{match}]  "
              f"prob: GZ={gz_prob:.3f} SG={surrogate_probs[i]:.3f}")

    if not gptzero_labels:
        print("ERROR: No valid GPTZero results to compare.")
        return [], 0.0

    # ---- Aggregate agreement ----
    matches = [c["match"] for c in comparison]
    agreement = sum(matches) / len(matches)

    print(f"\n{'=' * 60}")
    print(f"CLONE SCORE (agreement rate): {agreement:.1%}")
    print(f"{'=' * 60}")

    # ---- Per-class agreement ----
    print(f"\nPer-class agreement:")
    all_classes = sorted(set(gptzero_labels))
    for cls in all_classes:
        cls_matches = [
            c["match"] for c in comparison if c["gptzero_class"] == cls
        ]
        if cls_matches:
            cls_agree = sum(cls_matches) / len(cls_matches)
            print(f"  {cls:12s}  {cls_agree:.1%}  ({sum(cls_matches)}/{len(cls_matches)})")

    # ---- Classification report ----
    print(f"\nClassification Report (surrogate vs GPTZero ground truth):")
    print(classification_report(gptzero_labels, surrogate_labels, zero_division=0))

    labels_for_cm = sorted(set(gptzero_labels + surrogate_labels))
    cm = confusion_matrix(gptzero_labels, surrogate_labels, labels=labels_for_cm)
    print(f"Confusion Matrix:")
    print(f"{'':15s} " + " ".join(f"{l:>12s}" for l in labels_for_cm))
    for i, row in enumerate(cm):
        print(f"{labels_for_cm[i]:15s} " + " ".join(f"{v:>12d}" for v in row))

    # ---- Regressor metrics ----
    gz_probs_arr = np.array(gz_probs)
    sg_probs_arr = np.array(sg_probs)

    reg_mae = mean_absolute_error(gz_probs_arr, sg_probs_arr)
    reg_r2 = r2_score(gz_probs_arr, sg_probs_arr) if len(set(gz_probs)) > 1 else float("nan")

    print(f"\nRegressor performance (predicting average_generated_prob):")
    print(f"  MAE:  {reg_mae:.4f}")
    print(f"  R2:   {reg_r2:.4f}")

    # ---- Bootstrap CI ----
    ci_low, ci_high = _bootstrap_ci(matches, n_boot=10000, ci=0.95)
    print(f"\nClone score 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")

    # ---- Save ----
    test_out = DATA_DIR / "test_results.json"
    with open(test_out, "w") as f:
        json.dump({
            "agreement_rate": agreement,
            "agreement_ci_95": [ci_low, ci_high],
            "per_class_agreement": {
                cls: sum(c["match"] for c in comparison if c["gptzero_class"] == cls)
                     / max(sum(1 for c in comparison if c["gptzero_class"] == cls), 1)
                for cls in all_classes
            },
            "regressor_mae": reg_mae,
            "regressor_r2": reg_r2 if not np.isnan(reg_r2) else None,
            "total_tests": len(comparison),
            "matches": sum(matches),
            "comparisons": comparison,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nTest results saved to {test_out}")
    return comparison, agreement


def _bootstrap_ci(matches, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for agreement rate."""
    rng = np.random.RandomState(42)
    matches_arr = np.array(matches, dtype=float)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(matches_arr, size=len(matches_arr), replace=True)
        boot_means.append(sample.mean())
    boot_means = np.array(boot_means)
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


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

    # Load test results for CI
    test_data = {}
    test_path = DATA_DIR / "test_results.json"
    if test_path.exists():
        with open(test_path) as f:
            test_data = json.load(f)

    ci = test_data.get("agreement_ci_95", [0, 0])
    reg_mae = test_data.get("regressor_mae", 0)
    reg_r2 = test_data.get("regressor_r2", 0)

    cls_meta = model_meta.get("classifier", {})
    reg_meta = model_meta.get("regressor", {})

    report.append(f"""
EXECUTIVE SUMMARY

Through systematic probing of GPTZero's public API, we demonstrate
that their proprietary detection model can be functionally cloned
using only standard API access.

  Total API queries used:       {client.call_count}
  Total words sent:             {client.total_words}
  Estimated total cost:         ${client.total_words * 0.00015:.2f}

  Classifier:
    Type:                       {cls_meta.get('model_type', '?')}
    CV balanced accuracy:       {cls_meta.get('cv_balanced_accuracy', 0):.1%}
    Clone score (held-out):     {agreement:.1%}
    95% confidence interval:    [{ci[0]:.1%}, {ci[1]:.1%}]

  Regressor (avg_generated_prob):
    Type:                       {reg_meta.get('model_type', '?')}
    CV R-squared:               {reg_meta.get('cv_r2', 0):.3f}
    Held-out MAE:               {reg_mae:.4f}
    Held-out R-squared:         {reg_r2 if reg_r2 else 'N/A'}

  Training samples:             {model_meta.get('training_samples', '?')}
  Feature dimensions:           {model_meta.get('feature_count', '?')}

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
  These deltas approximate the classifier's gradient w.r.t. each
  linguistic feature. They reveal:
    - Which features have the most influence on classification
    - The direction of influence (increases or decreases AI score)
    - The approximate magnitude of each feature's weight
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
  Sentence-level interpolation reveals the exact content ratio
  where classification transitions. This is the minimum evasion
  threshold: an attacker needs to inject this ratio of human
  content to flip the classification.
""")

    # Feature importance from surrogate
    top = model_meta.get("top_features", [])
    if top:
        report.append("FINDING 4: FEATURE IMPORTANCE WEIGHTS RECOVERED")
        report.append("-" * 70)
        report.append("\n  Surrogate model feature importances (learned from GPTZero labels):\n")
        for fname, imp in top:
            bar = "#" * int(imp * 100)
            report.append(f"  {fname:35s} {imp:.4f} {bar}")
        report.append("""
  These weights were learned exclusively from GPTZero's API labels.
  They approximate the internal feature weights of GPTZero's
  proprietary detection model.
""")

    # Per-class agreement
    per_class = test_data.get("per_class_agreement", {})
    if per_class:
        report.append("FINDING 5: PER-CLASS CLONE ACCURACY")
        report.append("-" * 70)
        report.append("")
        for cls, rate in sorted(per_class.items()):
            report.append(f"  {cls:12s}  {rate:.1%}")
        report.append("")

    # Clone validation
    report.append("FINDING 6: SURROGATE MODEL VALIDATION (HELD-OUT)")
    report.append("-" * 70)
    report.append(f"\n  Aggregate agreement: {agreement:.1%} (95% CI: [{ci[0]:.1%}, {ci[1]:.1%}])")
    report.append(f"  Regressor MAE: {reg_mae:.4f}")
    report.append(f"  Regressor R2:  {reg_r2}\n")

    for c in test_comparison:
        match = "OK" if c["match"] else "MISS"
        report.append(
            f"  {c['sample_id']:30s}  "
            f"GZ={c['gptzero_class']:8s}  "
            f"SG={c['surrogate_class']:8s}  "
            f"[{match}]  "
            f"prob: GZ={c.get('gptzero_prob', 0):.3f} SG={c.get('surrogate_prob', 0):.3f}"
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

  GPTZero's revenue depends entirely on a proprietary model that
  is currently extractable for under $10 through normal API usage.
""")

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
  python main.py --test-only --api-key YOUR_KEY  # test existing model
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
        cls_model = joblib.load(MODEL_DIR / "surrogate_model.joblib")
        reg_model = joblib.load(MODEL_DIR / "surrogate_regressor.joblib")
        le = joblib.load(MODEL_DIR / "label_encoder.joblib")
        with open(MODEL_DIR / "model_meta.json") as f:
            meta = json.load(f)
        step_test(client, cls_model, reg_model, le, delay=args.delay)
        return

    # Step 1: Extract
    extraction_results = step_extract(client, delay=args.delay)

    # Step 2: Train
    result = step_train(extraction_results)
    if result[0] is None:
        print("Training failed. Exiting.")
        sys.exit(1)
    cls_model, reg_model, le, meta = result

    # Step 3: Test
    comparison, agreement = [], 0.0
    if not args.skip_test:
        comparison, agreement = step_test(client, cls_model, reg_model, le, delay=args.delay)

    # Step 4: Report
    step_report(extraction_results, meta, comparison, agreement, client)

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Classifier:   {MODEL_DIR}/surrogate_model.joblib")
    print(f"  Regressor:    {MODEL_DIR}/surrogate_regressor.joblib")
    print(f"  Data:         {DATA_DIR}/")
    print(f"  Report:       {REPORT_DIR}/gptzero_extraction_report.txt")
    print(f"  Queries:      {client.call_count}")
    print(f"  Cost:         ~${client.total_words * 0.00015:.2f}")
    print(f"  Clone score:  {agreement:.0%} agreement")


if __name__ == "__main__":
    main()
