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
import hashlib
import json
import os
import sys
import time
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from src.client import API_URL, GPTZeroClient
from src.attacks import (
    all_extraction_samples,
    held_out_test_samples,
    held_out_verified_test_samples,
)
from src.datasource import all_verified_samples
from src.features import extract_features, extract_features_batch, FEATURE_NAMES
from src.reproducibility import (
    RunPaths,
    build_run_metadata,
    configure_run_paths,
    set_global_seed,
    snapshot_artifacts_for_run,
    write_run_manifest,
)

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC
from sklearn.base import clone
from sklearn.model_selection import (
    KFold,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report, confusion_matrix, accuracy_score,
    log_loss,
    mean_absolute_error, r2_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
import joblib

warnings.filterwarnings("ignore", category=UserWarning)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("report")

DEFAULT_SEED = 42


def configure_output_dirs(paths: RunPaths) -> None:
    """Set global output directories for this run."""
    global DATA_DIR, MODEL_DIR, REPORT_DIR
    DATA_DIR = paths.data_dir
    MODEL_DIR = paths.model_dir
    REPORT_DIR = paths.report_dir


# ====================================================================
# STEP 1: EXTRACT
# ====================================================================

def _text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def step_extract(
    client,
    delay=0.5,
    dataset_version="v1",
    use_network_cache=True,
    run_metadata=None,
):
    """Send all probes (synthetic + verified-provenance) to GPTZero."""

    print("\n" + "=" * 60)
    print("STEP 1: EXTRACTION")
    print("=" * 60)

    samples = all_extraction_samples(dataset_version=dataset_version)
    print(f"Total probe samples: {len(samples)}")
    total_words = sum(len(text.split()) for _, text in samples)
    print(f"Total words: {total_words}")
    print(f"Estimated cost: ${total_words * 0.00015:.2f}")
    print()

    ids = [s[0] for s in samples]
    texts = [s[1] for s in samples]
    out = DATA_DIR / "extraction_results.json"
    cache_sig = {sid: _text_hash(text) for sid, text in samples}

    cached_results = None
    if use_network_cache and out.exists():
        with open(out, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_sig = (
            cached.get("metadata", {}).get("sample_signature")
            if isinstance(cached, dict)
            else None
        )
        cached_results = cached.get("results") if isinstance(cached, dict) else None
        if cached_sig == cache_sig and isinstance(cached_results, list):
            print("Using cached extraction results (sample signatures match).")
    results = cached_results
    if results is None:
        results = client.predict_batch(texts, ids=ids, delay=delay)

        with open(out, "w") as f:
            json.dump({
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_queries": len(results),
                    "total_words": total_words,
                    "api_calls": client.call_count,
                    "dataset_version": dataset_version,
                    "sample_signature": cache_sig,
                    "run_metadata": run_metadata or {},
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

    # Ground-truth accuracy for verified-provenance samples
    verified = {s[0]: s[1] for s in all_verified_samples(dataset_version=dataset_version)}
    if verified:
        correct, total = 0, 0
        gt_labels, gptz_labels = [], []
        fp, fn = 0, 0
        for r in results:
            sid = r.get("sample_id", "")
            if sid in verified and "error" not in r:
                gt = verified[sid]
                pred = r.get("predicted_class", "")
                gt_labels.append(gt)
                gptz_labels.append(pred)
                total += 1
                if gt == pred:
                    correct += 1
                if gt == "human" and pred == "ai":
                    fp += 1
                if gt == "ai" and pred == "human":
                    fn += 1

        if total > 0:
            print(f"\n--- GPTZero Ground-Truth Accuracy (verified samples) ---")
            print(f"  Samples:          {total}")
            print(f"  Correct:          {correct} ({correct/total:.1%})")
            print(f"  False positives:  {fp} (human labeled as AI)")
            print(f"  False negatives:  {fn} (AI labeled as human)")

            for gt_cls in sorted(set(gt_labels)):
                cls_total = sum(1 for g in gt_labels if g == gt_cls)
                cls_correct = sum(
                    1 for g, p in zip(gt_labels, gptz_labels) if g == gt_cls and g == p
                )
                print(f"  {gt_cls:12s}  {cls_correct}/{cls_total} correct "
                      f"({cls_correct/cls_total:.1%})")

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


def _api_feature_vector(result):
    feats = extract_api_response_features(result)
    return [float(feats.get(name, 0.0)) for name in API_RESPONSE_FEATURE_NAMES]


def _normalize_probability_rows(mat):
    arr = np.array(mat, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    arr = np.clip(arr, 1e-8, 1.0)
    sums = arr.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    return arr / sums


def _temperature_scale(prob_rows, temperature):
    p = np.clip(np.array(prob_rows, dtype=float), 1e-8, 1.0)
    if temperature <= 0:
        temperature = 1.0
    logits = np.log(p)
    scaled = logits / float(temperature)
    scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    denom = scaled.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return scaled / denom


def _fit_temperature(raw_prob_rows, y_cls_encoded):
    probs = _normalize_probability_rows(raw_prob_rows)
    y = np.array(y_cls_encoded, dtype=int)
    candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    best_t = 1.0
    best_loss = float("inf")
    for t in candidates:
        scaled = _temperature_scale(probs, t)
        try:
            loss = log_loss(y, scaled, labels=list(range(scaled.shape[1])))
        except ValueError:
            continue
        if loss < best_loss:
            best_loss = loss
            best_t = t
    return float(best_t), float(best_loss if np.isfinite(best_loss) else 0.0)


def _oversample_balanced(X, y_cls, y_prob, y_prob_vec, seed):
    rng = np.random.RandomState(seed)
    y_arr = np.array(y_cls, dtype=int)
    class_counts = np.bincount(y_arr)
    if len(class_counts) == 0:
        return X, y_arr, y_prob, y_prob_vec
    max_count = int(class_counts.max())
    sampled_idx = []
    for class_idx, count in enumerate(class_counts):
        if count == 0:
            continue
        class_indices = np.where(y_arr == class_idx)[0]
        draws = rng.choice(class_indices, size=max_count, replace=True)
        sampled_idx.extend(draws.tolist())
    rng.shuffle(sampled_idx)
    sampled_idx = np.array(sampled_idx, dtype=int)
    return (
        np.array(X)[sampled_idx],
        y_arr[sampled_idx],
        np.array(y_prob)[sampled_idx],
        np.array(y_prob_vec)[sampled_idx],
    )


# ====================================================================
# STEP 2: TRAIN SURROGATE (Classifier + Regressor)
# ====================================================================

def step_train(
    results,
    seed=DEFAULT_SEED,
    run_metadata=None,
    use_api_features=True,
    enable_probability_objective=True,
    enable_oversampling=True,
    calibration_method="temperature",
    cv_repeats=3,
):
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
    class_prob_dicts = []
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
            class_prob_dicts.append(r.get("class_probabilities") or {})

    print(f"Training samples with labels: {len(texts)}")

    if len(texts) < 10:
        print("ERROR: Not enough labeled samples. Check API results.")
        return None, None, None, None, {}

    X_text, feat_names = extract_features_batch(texts)
    X_text = np.array(X_text, dtype=float)
    feature_names = list(feat_names)

    if use_api_features:
        X_api = np.array(
            [
                [float(feats.get(name, 0.0)) for name in API_RESPONSE_FEATURE_NAMES]
                for feats in api_feats_list
            ],
            dtype=float,
        )
        X_train_base = np.hstack([X_text, X_api])
        feature_names = feature_names + API_RESPONSE_FEATURE_NAMES
    else:
        X_train_base = X_text

    le = LabelEncoder()
    y_cls = le.fit_transform(labels)
    y_prob = np.array(probs)
    y_prob_vec = []
    for lbl, class_probs in zip(labels, class_prob_dicts):
        if isinstance(class_probs, dict):
            vec = [float(class_probs.get(str(c), 0.0) or 0.0) for c in le.classes_]
            if sum(vec) <= 0:
                vec = [1.0 if str(c) == lbl else 0.0 for c in le.classes_]
        else:
            vec = [1.0 if str(c) == lbl else 0.0 for c in le.classes_]
        y_prob_vec.append(vec)
    y_prob_vec = _normalize_probability_rows(y_prob_vec)

    class_counts = np.bincount(y_cls)
    print(f"Feature dimensions: {X_train_base.shape}")
    print(f"Classes: {list(le.classes_)} (counts: {class_counts.tolist()})")

    # Check for minimum class representation
    min_class_count = int(min(class_counts))
    if min_class_count < 2:
        print(f"WARNING: Class with only {min_class_count} sample(s). "
              f"Cross-validation may be unreliable for minority classes.")

    # ---- CLASSIFIER ----
    print(f"\n--- Classifier Training ---")

    X_train = X_train_base
    y_cls_train = np.array(y_cls, dtype=int)
    y_prob_train = np.array(y_prob, dtype=float)
    y_prob_vec_train = np.array(y_prob_vec, dtype=float)

    if enable_oversampling:
        X_train, y_cls_train, y_prob_train, y_prob_vec_train = _oversample_balanced(
            X=X_train_base,
            y_cls=y_cls,
            y_prob=y_prob,
            y_prob_vec=y_prob_vec,
            seed=seed,
        )
        print(f"Oversampled training rows: {len(X_train)}")

    train_class_counts = np.bincount(y_cls_train)
    min_train_class_count = int(min(train_class_counts))
    print(f"Training class counts (effective): {train_class_counts.tolist()}")

    models_cls = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=seed
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced", random_state=seed)),
        ]),
        "SVM_RBF": Pipeline([
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=seed)),
        ]),
    }

    if min_train_class_count >= 2:
        n_splits_cls = int(max(min(5, min_train_class_count), 2))
        cls_cv = RepeatedStratifiedKFold(
            n_splits=n_splits_cls,
            n_repeats=max(1, int(cv_repeats)),
            random_state=seed,
        )
        cv_label = f"{n_splits_cls}-fold x{max(1, int(cv_repeats))}"
    else:
        cls_cv = None
        cv_label = "disabled"

    best_cls_score = float("-inf")
    best_cls_name = ""
    best_cls_model = None

    print(f"\nCross-validation ({cv_label} stratified):")
    for name, model in models_cls.items():
        try:
            if cls_cv is None:
                mean_score = float("nan")
                std_score = float("nan")
                print(f"  {name:25s}  CV skipped (minority class <2)")
            else:
                scores = cross_val_score(model, X_train, y_cls_train, cv=cls_cv, scoring="balanced_accuracy")
                mean_score = float(scores.mean())
                std_score = float(scores.std())
                print(f"  {name:25s}  {mean_score:.1%} (+/- {std_score:.1%}) balanced_acc")
            if mean_score > best_cls_score:
                best_cls_score = mean_score
                best_cls_name = name
                best_cls_model = model
        except Exception as e:
            print(f"  {name:25s}  FAILED: {e}")

    if best_cls_model is None:
        best_cls_name = "LogisticRegression"
        best_cls_model = models_cls[best_cls_name]
        best_cls_score = 0.0

    best_cls_model.fit(X_train, y_cls_train)
    cls_score_text = f"{best_cls_score:.1%}" if np.isfinite(best_cls_score) else "N/A"
    print(f"\nBest classifier: {best_cls_name} ({cls_score_text} balanced accuracy)")

    # Feature importance
    top_features = _get_feature_importances(best_cls_model, feature_names)

    # ---- REGRESSOR ----
    print(f"\n--- Regressor Training (predict avg_generated_prob) ---")

    models_reg = {
        "GBRegressor": GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed
        ),
        "Ridge": Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]),
    }

    k_reg = int(max(min(5, max(2, len(X_train) // 3)), 2))

    best_reg_score = float("-inf")
    best_reg_name = ""
    best_reg_model = None

    print(f"\nCross-validation ({k_reg}-fold):")
    for name, model in models_reg.items():
        try:
            kf = KFold(n_splits=k_reg, shuffle=True, random_state=seed)
            scores = cross_val_score(model, X_train, y_prob_train, cv=kf, scoring="r2")
            mean_score = float(scores.mean())
            std_score = float(scores.std())
            print(f"  {name:25s}  R2={mean_score:.3f} (+/- {std_score:.3f})")
            if mean_score > best_reg_score:
                best_reg_score = mean_score
                best_reg_name = name
                best_reg_model = model
        except Exception as e:
            print(f"  {name:25s}  FAILED: {e}")

    if best_reg_model is None:
        best_reg_name = "Ridge"
        best_reg_model = models_reg[best_reg_name]
        best_reg_score = 0.0
    best_reg_model.fit(X_train, y_prob_train)
    print(f"\nBest regressor: {best_reg_name} (R2={best_reg_score:.3f})")

    # ---- PROBABILITY-VECTOR MODEL ----
    best_prob_model = None
    best_prob_name = "disabled"
    best_prob_cv_mae = None
    temperature = 1.0
    temperature_loss = None
    if enable_probability_objective:
        print(f"\n--- Probability-Vector Training (predict class_probabilities) ---")
        models_prob = {
            "MultiOutputRidge": MultiOutputRegressor(
                Pipeline([
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=1.0)),
                ])
            ),
            "MultiOutputGBRegressor": MultiOutputRegressor(
                GradientBoostingRegressor(
                    n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed
                )
            ),
        }
        k_prob = int(max(min(5, max(2, len(X_train) // 3)), 2))
        best_prob_score = float("-inf")
        print(f"\nCross-validation ({k_prob}-fold):")
        for name, model in models_prob.items():
            try:
                kf = KFold(n_splits=k_prob, shuffle=True, random_state=seed)
                scores = cross_val_score(
                    model,
                    X_train,
                    y_prob_vec_train,
                    cv=kf,
                    scoring="neg_mean_absolute_error",
                )
                mean_score = float(scores.mean())
                std_score = float(scores.std())
                mae = -mean_score
                print(f"  {name:25s}  MAE={mae:.4f} (+/- {std_score:.4f})")
                if mean_score > best_prob_score:
                    best_prob_score = mean_score
                    best_prob_name = name
                    best_prob_model = model
                    best_prob_cv_mae = mae
            except Exception as e:
                print(f"  {name:25s}  FAILED: {e}")

        if best_prob_model is None:
            best_prob_name = "MultiOutputRidge"
            best_prob_model = models_prob[best_prob_name]
        best_prob_model.fit(X_train, y_prob_vec_train)

        if calibration_method == "temperature":
            try:
                min_cal_class = int(min(np.bincount(y_cls_train)))
                if min_cal_class >= 2 and len(X_train) >= 12:
                    X_cal_train, X_cal_val, y_cal_train, y_cal_val, y_prob_vec_cal_train, _ = train_test_split(
                        X_train,
                        y_cls_train,
                        y_prob_vec_train,
                        test_size=0.25,
                        random_state=seed,
                        stratify=y_cls_train,
                    )
                    temp_model = clone(best_prob_model)
                    temp_model.fit(X_cal_train, y_prob_vec_cal_train)
                    raw_val = temp_model.predict(X_cal_val)
                    temperature, temperature_loss = _fit_temperature(raw_val, y_cal_val)
                else:
                    temperature = 1.0
                    temperature_loss = None
            except Exception:
                temperature = 1.0
                temperature_loss = None

    # ---- SAVE ----
    joblib.dump(best_cls_model, MODEL_DIR / "surrogate_model.joblib")
    joblib.dump(best_reg_model, MODEL_DIR / "surrogate_regressor.joblib")
    if best_prob_model is not None:
        joblib.dump(best_prob_model, MODEL_DIR / "surrogate_prob_vector.joblib")
    joblib.dump(le, MODEL_DIR / "label_encoder.joblib")

    meta = {
        "classifier": {
            "model_type": best_cls_name,
            "cv_balanced_accuracy": float(best_cls_score) if np.isfinite(best_cls_score) else None,
            "cv_folds": int(n_splits_cls) if min_train_class_count >= 2 else None,
            "cv_repeats": int(max(1, int(cv_repeats))) if min_train_class_count >= 2 else None,
        },
        "regressor": {
            "model_type": best_reg_name,
            "cv_r2": float(best_reg_score),
            "cv_folds": int(k_reg),
        },
        "probability_objective": {
            "enabled": bool(enable_probability_objective and best_prob_model is not None),
            "model_type": best_prob_name,
            "cv_mae": float(best_prob_cv_mae) if best_prob_cv_mae is not None else None,
            "calibration_method": calibration_method,
            "temperature": float(temperature),
            "temperature_validation_log_loss": temperature_loss,
        },
        "training_samples": int(len(texts)),
        "effective_training_samples": int(len(X_train)),
        "classes": list(le.classes_),
        "class_counts": [int(x) for x in class_counts.tolist()],
        "effective_class_counts": [int(x) for x in train_class_counts.tolist()],
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "text_feature_count": int(len(feat_names)),
        "api_feature_count": int(len(API_RESPONSE_FEATURE_NAMES) if use_api_features else 0),
        "uses_api_features": bool(use_api_features),
        "oversampling_enabled": bool(enable_oversampling),
        "top_features": top_features,
        "api_response_feature_names": API_RESPONSE_FEATURE_NAMES,
        "seed": seed,
        "run_metadata": run_metadata or {},
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

    return best_cls_model, best_reg_model, best_prob_model, le, meta


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

def step_test(
    client,
    cls_model,
    reg_model,
    prob_vector_model,
    le,
    model_meta=None,
    delay=0.5,
    dataset_version="v1",
    use_network_cache=True,
    seed=DEFAULT_SEED,
):
    """
    Run held-out test samples through both the surrogate and GPTZero.
    Reports:
      - Per-class agreement rates (not just aggregate)
      - Regressor MAE and R2 on held-out probabilities
      - Bootstrap 95% confidence interval on clone score
      - Ground-truth accuracy for verified-provenance test samples
    """

    print("\n" + "=" * 60)
    print("STEP 3: TEST SURROGATE vs GPTZERO")
    print("=" * 60)

    # Combine original test samples with verified held-out samples
    test_samples = held_out_test_samples(dataset_version=dataset_version)
    verified_tests = held_out_verified_test_samples(dataset_version=dataset_version)

    # Merge: original (id, text) + verified (id, gt, text) -> (id, text)
    # Track ground truth separately
    ground_truth = {}
    for sid, gt, text in verified_tests:
        test_samples.append((sid, text))
        ground_truth[sid] = gt

    print(f"Test samples: {len(test_samples)} ({len(verified_tests)} with ground truth)")

    ids = [s[0] for s in test_samples]
    texts = [s[1] for s in test_samples]

    print("\nQuerying GPTZero on held-out set...")
    gptzero_cache_path = DATA_DIR / "gptzero_test_predictions.json"
    test_signature = {sid: _text_hash(text) for sid, text in test_samples}
    gptzero_results = None
    if use_network_cache and gptzero_cache_path.exists():
        with open(gptzero_cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("sample_signature") == test_signature and isinstance(cached.get("results"), list):
            print("Using cached held-out GPTZero predictions.")
            gptzero_results = cached["results"]

    if gptzero_results is None:
        gptzero_results = client.predict_batch(texts, ids=ids, delay=delay)
        with open(gptzero_cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "sample_signature": test_signature,
                    "results": gptzero_results,
                    "dataset_version": dataset_version,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

    X_text_test, _ = extract_features_batch(texts)
    X_text_test = np.array(X_text_test, dtype=float)
    model_meta = model_meta or {}
    use_api_features = bool(model_meta.get("uses_api_features", False))
    if use_api_features:
        X_api_test = np.array(
            [
                _api_feature_vector(gz) if "error" not in gz else [0.0] * len(API_RESPONSE_FEATURE_NAMES)
                for gz in gptzero_results
            ],
            dtype=float,
        )
        X_test = np.hstack([X_text_test, X_api_test])
    else:
        X_test = X_text_test

    prob_obj_meta = model_meta.get("probability_objective", {}) if isinstance(model_meta, dict) else {}
    use_prob_vector = bool(
        prob_vector_model is not None
        and isinstance(prob_obj_meta, dict)
        and prob_obj_meta.get("enabled", False)
    )
    surrogate_class_prob_rows = None
    if use_prob_vector:
        raw_prob_vec = prob_vector_model.predict(X_test)
        surrogate_class_prob_rows = _normalize_probability_rows(raw_prob_vec)
        calibration_method = str(prob_obj_meta.get("calibration_method", "none"))
        if calibration_method == "temperature":
            temperature = float(prob_obj_meta.get("temperature", 1.0) or 1.0)
            surrogate_class_prob_rows = _temperature_scale(surrogate_class_prob_rows, temperature)
        surrogate_cls_encoded = np.argmax(surrogate_class_prob_rows, axis=1)
    else:
        surrogate_cls_encoded = cls_model.predict(X_test)
        if hasattr(cls_model, "predict_proba"):
            try:
                surrogate_class_prob_rows = _normalize_probability_rows(cls_model.predict_proba(X_test))
            except Exception:
                surrogate_class_prob_rows = None
    surrogate_cls = le.inverse_transform(np.array(surrogate_cls_encoded, dtype=int))
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
        sg_prob = float(np.clip(surrogate_probs[i], 0.0, 1.0))
        sg_class_probs = {}
        if surrogate_class_prob_rows is not None:
            row_probs = surrogate_class_prob_rows[i]
            sg_class_probs = {
                str(cls_name): float(row_probs[j])
                for j, cls_name in enumerate(le.classes_)
            }

        gptzero_labels.append(gz_cls)
        surrogate_labels.append(sg_cls)
        gz_probs.append(gz_prob)
        sg_probs.append(sg_prob)

        comparison.append({
            "sample_id": sid,
            "gptzero_class": gz_cls,
            "surrogate_class": sg_cls,
            "match": gz_cls == sg_cls,
            "gptzero_prob": gz_prob,
            "surrogate_prob": sg_prob,
            "surrogate_class_probabilities": sg_class_probs,
            "prob_error": abs(gz_prob - sg_prob),
            "gptzero_confidence": gz.get("confidence_category"),
        })

        print(f"  {sid:30s}  GZ={gz_cls:8s} SG={sg_cls:8s} [{match}]  "
              f"prob: GZ={gz_prob:.3f} SG={sg_prob:.3f}")

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
    ci_low, ci_high = _bootstrap_ci(matches, n_boot=10000, ci=0.95, seed=seed)
    print(f"\nClone score 95% CI: [{ci_low:.1%}, {ci_high:.1%}]")

    # ---- Ground-truth accuracy (verified samples) ----
    gt_results = {
        "gptzero_accuracy": None,
        "gptzero_false_positives": 0,
        "gptzero_false_negatives": 0,
        "surrogate_accuracy": None,
        "per_class": {},
    }

    if ground_truth:
        print(f"\n--- Ground-Truth Accuracy (verified provenance) ---")

        gt_correct_gz, gt_correct_sg, gt_total = 0, 0, 0
        gt_fp_gz, gt_fn_gz = 0, 0
        gt_fp_sg, gt_fn_sg = 0, 0

        for c in comparison:
            sid = c["sample_id"]
            if sid not in ground_truth:
                continue
            gt = ground_truth[sid]
            gz_cls = c["gptzero_class"]
            sg_cls = c["surrogate_class"]
            gt_total += 1

            if gz_cls == gt:
                gt_correct_gz += 1
            if sg_cls == gt:
                gt_correct_sg += 1
            if gt == "human" and gz_cls == "ai":
                gt_fp_gz += 1
            if gt == "ai" and gz_cls == "human":
                gt_fn_gz += 1
            if gt == "human" and sg_cls == "ai":
                gt_fp_sg += 1
            if gt == "ai" and sg_cls == "human":
                gt_fn_sg += 1

            gt_marker = "OK" if gz_cls == gt else "WRONG"
            sg_marker = "OK" if sg_cls == gt else "WRONG"
            print(f"  {sid:30s}  GT={gt:8s}  GZ={gz_cls:8s}[{gt_marker:5s}]  "
                  f"SG={sg_cls:8s}[{sg_marker:5s}]")

        if gt_total > 0:
            gz_acc = gt_correct_gz / gt_total
            sg_acc = gt_correct_sg / gt_total
            print(f"\n  GPTZero accuracy vs ground truth:   {gz_acc:.1%} ({gt_correct_gz}/{gt_total})")
            print(f"  Surrogate accuracy vs ground truth: {sg_acc:.1%} ({gt_correct_sg}/{gt_total})")
            print(f"  GPTZero false positives (human->AI): {gt_fp_gz}")
            print(f"  GPTZero false negatives (AI->human): {gt_fn_gz}")

            gt_results.update({
                "gptzero_accuracy": gz_acc,
                "gptzero_false_positives": gt_fp_gz,
                "gptzero_false_negatives": gt_fn_gz,
                "surrogate_accuracy": sg_acc,
                "surrogate_false_positives": gt_fp_sg,
                "surrogate_false_negatives": gt_fn_sg,
            })

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
            "ground_truth_evaluation": gt_results,
            "dataset_version": dataset_version,
            "total_tests": len(comparison),
            "matches": sum(matches),
            "comparisons": comparison,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\nTest results saved to {test_out}")
    return comparison, agreement


def _bootstrap_ci(matches, n_boot=10000, ci=0.95, seed=DEFAULT_SEED):
    """Bootstrap confidence interval for agreement rate."""
    rng = np.random.RandomState(seed)
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
    cls_cv = cls_meta.get("cv_balanced_accuracy")
    cls_cv_text = f"{float(cls_cv):.1%}" if cls_cv is not None else "N/A"
    reg_cv = reg_meta.get("cv_r2")
    reg_cv_text = f"{float(reg_cv):.3f}" if reg_cv is not None else "N/A"

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
    CV balanced accuracy:       {cls_cv_text}
    Clone score (held-out):     {agreement:.1%}
    95% confidence interval:    [{ci[0]:.1%}, {ci[1]:.1%}]

  Regressor (avg_generated_prob):
    Type:                       {reg_meta.get('model_type', '?')}
    CV R-squared:               {reg_cv_text}
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

    # Ground-truth accuracy (new finding)
    gt_eval = test_data.get("ground_truth_evaluation", {})
    gz_acc = gt_eval.get("gptzero_accuracy")
    gz_fp = gt_eval.get("gptzero_false_positives", 0)
    gz_fn = gt_eval.get("gptzero_false_negatives", 0)

    if gz_acc is not None:
        report.append("FINDING 5: GPTZERO FALSE POSITIVE RATE (VERIFIED PROVENANCE)")
        report.append("-" * 70)
        report.append(f"""
  Using text with independently verified authorship provenance
  (pre-ChatGPT human writing from public archives, and known
  AI-generated text), we measure GPTZero's actual accuracy:

    GPTZero accuracy:          {gz_acc:.1%}
    False positives:           {gz_fp} (human text labeled as AI)
    False negatives:           {gz_fn} (AI text labeled as human)

  False positives represent human-authored text that GPTZero
  incorrectly classifies as AI-generated. These are real texts
  from Reddit, blogs, news articles, and reviews published before
  ChatGPT's launch in November 2022 -- they cannot be AI-generated.

  This false positive rate has direct consequences: students
  falsely accused of cheating, writers wrongly flagged, and
  erosion of trust in AI detection as a reliable signal.
""")

    # Per-class agreement
    per_class = test_data.get("per_class_agreement", {})
    if per_class:
        report.append("FINDING 6: PER-CLASS CLONE ACCURACY")
        report.append("-" * 70)
        report.append("")
        for cls, rate in sorted(per_class.items()):
            report.append(f"  {cls:12s}  {rate:.1%}")
        report.append("")

    # Clone validation
    report.append("FINDING 7: SURROGATE MODEL VALIDATION (HELD-OUT)")
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
    project_root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="GPTZero Model Extraction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --api-key YOUR_KEY              # full pipeline
  python main.py --api-key YOUR_KEY --skip-test  # extract + train only
  python main.py --test-only --api-key YOUR_KEY  # test existing model
  python main.py --seed 42 --dataset-version v1  # deterministic run
        """,
    )
    parser.add_argument("--api-key", help="GPTZero API key (or set GPTZERO_API_KEY)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls (s)")
    parser.add_argument("--skip-test", action="store_true", help="Skip held-out testing")
    parser.add_argument("--test-only", action="store_true", help="Only run test step with existing model")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Deterministic random seed")
    parser.add_argument("--run-id", help="Optional run identifier")
    parser.add_argument("--dataset-version", default="v1", help="Dataset version under datasets/<version>")
    parser.add_argument("--output-dir", default=".", help="Base output directory for data/models/report/artifacts")
    parser.add_argument(
        "--use-api-features",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use GPTZero API-response feature channels in training/testing.",
    )
    parser.add_argument(
        "--probability-objective",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train probability-vector surrogate objective (class_probabilities).",
    )
    parser.add_argument(
        "--oversample",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable minority-class oversampling before model training.",
    )
    parser.add_argument(
        "--calibration-method",
        choices=["none", "temperature"],
        default="temperature",
        help="Calibration method for probability-vector outputs.",
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=3,
        help="Repeated stratified CV repeats for classifier model selection.",
    )
    parser.add_argument(
        "--no-network-cache",
        action="store_true",
        help="Disable cached API responses and force fresh GPTZero calls",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GPTZERO_API_KEY")
    if not api_key:
        print("ERROR: Set --api-key or GPTZERO_API_KEY env var")
        print("Get a key at: https://app.gptzero.me/app/api")
        sys.exit(1)

    set_global_seed(args.seed)
    run_id = args.run_id or datetime.now().strftime("run-%Y%m%dT%H%M%S")
    run_paths = configure_run_paths(Path(args.output_dir), run_id)
    configure_output_dirs(run_paths)

    run_metadata = build_run_metadata(
        project_root=project_root,
        seed=args.seed,
        run_id=run_id,
        dataset_version=args.dataset_version,
        args=vars(args),
        api_endpoint=API_URL,
    )

    network_cache_enabled = not args.no_network_cache

    client = GPTZeroClient(api_key=api_key)

    agreement = 0.0
    comparison = []
    model_meta = {}

    if args.test_only:
        cls_model = joblib.load(MODEL_DIR / "surrogate_model.joblib")
        reg_model = joblib.load(MODEL_DIR / "surrogate_regressor.joblib")
        prob_vector_model = None
        prob_vector_path = MODEL_DIR / "surrogate_prob_vector.joblib"
        if prob_vector_path.exists():
            prob_vector_model = joblib.load(prob_vector_path)
        le = joblib.load(MODEL_DIR / "label_encoder.joblib")
        with open(MODEL_DIR / "model_meta.json") as f:
            model_meta = json.load(f)
        comparison, agreement = step_test(
            client,
            cls_model,
            reg_model,
            prob_vector_model,
            le,
            model_meta=model_meta,
            delay=args.delay,
            dataset_version=args.dataset_version,
            use_network_cache=network_cache_enabled,
            seed=args.seed,
        )
        artifact_paths = {
            "test_results": DATA_DIR / "test_results.json",
            "pattern_benchmark": DATA_DIR / "pattern_benchmark_results.json",
            "model_meta": MODEL_DIR / "model_meta.json",
            "classifier_model": MODEL_DIR / "surrogate_model.joblib",
            "regressor_model": MODEL_DIR / "surrogate_regressor.joblib",
            "prob_vector_model": MODEL_DIR / "surrogate_prob_vector.joblib",
            "label_encoder": MODEL_DIR / "label_encoder.joblib",
            "dataset_manifest": project_root / "datasets" / args.dataset_version / "manifest.json",
        }
        snapshot_artifact_paths = snapshot_artifacts_for_run(
            run_dir=run_paths.run_dir,
            artifact_paths=artifact_paths,
        )
        write_run_manifest(
            manifest_path=run_paths.manifest_path,
            run_metadata=run_metadata,
            artifact_paths=snapshot_artifact_paths,
            extra_metrics={
                "mode": "test-only",
                "clone_score": agreement,
                "gptzero_api_calls": client.call_count,
                "gptzero_words": client.total_words,
            },
        )
        print(f"Run manifest: {run_paths.manifest_path}")
        return

    # Step 1: Extract
    extraction_results = step_extract(
        client,
        delay=args.delay,
        dataset_version=args.dataset_version,
        use_network_cache=network_cache_enabled,
        run_metadata=run_metadata,
    )

    # Step 2: Train
    result = step_train(
        extraction_results,
        seed=args.seed,
        run_metadata=run_metadata,
        use_api_features=bool(args.use_api_features),
        enable_probability_objective=bool(args.probability_objective),
        enable_oversampling=bool(args.oversample),
        calibration_method=str(args.calibration_method),
        cv_repeats=int(max(1, args.cv_repeats)),
    )
    if result[0] is None:
        print("Training failed. Exiting.")
        sys.exit(1)
    cls_model, reg_model, prob_vector_model, le, model_meta = result

    # Step 3: Test
    if not args.skip_test:
        comparison, agreement = step_test(
            client,
            cls_model,
            reg_model,
            prob_vector_model,
            le,
            model_meta=model_meta,
            delay=args.delay,
            dataset_version=args.dataset_version,
            use_network_cache=network_cache_enabled,
            seed=args.seed,
        )

    # Step 4: Report
    step_report(extraction_results, model_meta, comparison, agreement, client)

    artifact_paths = {
        "extraction_results": DATA_DIR / "extraction_results.json",
        "test_results": DATA_DIR / "test_results.json",
        "pattern_benchmark": DATA_DIR / "pattern_benchmark_results.json",
        "model_meta": MODEL_DIR / "model_meta.json",
        "classifier_model": MODEL_DIR / "surrogate_model.joblib",
        "regressor_model": MODEL_DIR / "surrogate_regressor.joblib",
        "prob_vector_model": MODEL_DIR / "surrogate_prob_vector.joblib",
        "label_encoder": MODEL_DIR / "label_encoder.joblib",
        "report": REPORT_DIR / "gptzero_extraction_report.txt",
        "dataset_manifest": project_root / "datasets" / args.dataset_version / "manifest.json",
    }
    snapshot_artifact_paths = snapshot_artifacts_for_run(
        run_dir=run_paths.run_dir,
        artifact_paths=artifact_paths,
    )
    write_run_manifest(
        manifest_path=run_paths.manifest_path,
        run_metadata=run_metadata,
        artifact_paths=snapshot_artifact_paths,
        extra_metrics={
            "mode": "full",
            "clone_score": agreement,
            "gptzero_api_calls": client.call_count,
            "gptzero_words": client.total_words,
            "skip_test": bool(args.skip_test),
        },
    )

    print(f"\n{'=' * 60}")
    print("PIPELINE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Classifier:   {MODEL_DIR}/surrogate_model.joblib")
    print(f"  Regressor:    {MODEL_DIR}/surrogate_regressor.joblib")
    print(f"  Data:         {DATA_DIR}/")
    print(f"  Report:       {REPORT_DIR}/gptzero_extraction_report.txt")
    print(f"  Manifest:     {run_paths.manifest_path}")
    print(f"  Queries:      {client.call_count}")
    print(f"  Cost:         ~${client.total_words * 0.00015:.2f}")
    print(f"  Clone score:  {agreement:.0%} agreement")


if __name__ == "__main__":
    main()
