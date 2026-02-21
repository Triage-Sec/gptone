#!/usr/bin/env python3
"""
Surrogate Predictor
====================
Run the trained surrogate model on any text.
Optionally compare against GPTZero live.

Usage:
  python predict.py "Your text here"
  python predict.py --compare --api-key YOUR_KEY "Your text here"
  echo "Some text" | python predict.py --stdin
"""

import argparse
import json
import sys
import os
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.features import extract_features, FEATURE_NAMES
from main import API_RESPONSE_FEATURE_NAMES, extract_api_response_features


MODEL_DIR = Path("models")


def load_models():
    """Load classifier, regressor, label encoder, and metadata."""
    cls_model = joblib.load(MODEL_DIR / "surrogate_model.joblib")
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "model_meta.json") as f:
        meta = json.load(f)

    reg_model = None
    reg_path = MODEL_DIR / "surrogate_regressor.joblib"
    if reg_path.exists():
        reg_model = joblib.load(reg_path)

    prob_vector_model = None
    prob_vector_path = MODEL_DIR / "surrogate_prob_vector.joblib"
    if prob_vector_path.exists():
        prob_vector_model = joblib.load(prob_vector_path)

    return cls_model, reg_model, prob_vector_model, le, meta


def _normalize_prob_rows(arr):
    arr = np.clip(np.array(arr, dtype=float), 1e-8, 1.0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    sums = arr.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    return arr / sums


def _temperature_scale(probs, temperature):
    p = np.clip(np.array(probs, dtype=float), 1e-8, 1.0)
    if temperature <= 0:
        temperature = 1.0
    logits = np.log(p)
    scaled = logits / float(temperature)
    scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    denom = scaled.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return scaled / denom


def _build_feature_vector(text, meta, api_response=None):
    text_vec, _ = extract_features(text)
    vec = text_vec.reshape(1, -1)
    if not bool(meta.get("uses_api_features", False)):
        return vec
    if api_response is not None:
        api_feats = extract_api_response_features(api_response)
        api_vec = np.array(
            [[float(api_feats.get(name, 0.0)) for name in API_RESPONSE_FEATURE_NAMES]],
            dtype=float,
        )
    else:
        api_vec = np.zeros((1, len(API_RESPONSE_FEATURE_NAMES)), dtype=float)
    return np.hstack([vec, api_vec])


def predict(text, cls_model, reg_model, prob_vector_model, le, meta, api_response=None):
    """Run surrogate on text. Returns (class, class_probs, generated_prob)."""
    vec = _build_feature_vector(text, meta, api_response=api_response)

    class_probs = {}
    prob_obj = meta.get("probability_objective", {})
    if prob_vector_model is not None and prob_obj.get("enabled", False):
        p = _normalize_prob_rows(prob_vector_model.predict(vec))
        if prob_obj.get("calibration_method") == "temperature":
            p = _temperature_scale(p, float(prob_obj.get("temperature", 1.0)))
        p = p[0]
        for cls, prob in zip(le.classes_, p):
            class_probs[cls] = float(prob)
        pred_class = str(le.classes_[int(np.argmax(p))])
    else:
        pred_encoded = cls_model.predict(vec)[0]
        pred_class = le.inverse_transform([pred_encoded])[0]
        if hasattr(cls_model, "predict_proba"):
            p = cls_model.predict_proba(vec)[0]
            for cls, prob in zip(le.classes_, p):
                class_probs[cls] = float(prob)
        elif hasattr(cls_model, "named_steps"):
            last_step = list(cls_model.named_steps.values())[-1]
            if hasattr(last_step, "predict_proba"):
                p = cls_model.predict_proba(vec)[0]
                for cls, prob in zip(le.classes_, p):
                    class_probs[cls] = float(prob)

    gen_prob = None
    if reg_model is not None:
        gen_prob = float(np.clip(reg_model.predict(vec)[0], 0.0, 1.0))

    return pred_class, class_probs, gen_prob


def main():
    parser = argparse.ArgumentParser(description="GPTZero Surrogate Predictor")
    parser.add_argument("text", nargs="?", help="Text to classify")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--compare", action="store_true", help="Compare with GPTZero live")
    parser.add_argument("--api-key", help="GPTZero API key (for --compare)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if args.stdin:
        text = sys.stdin.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    cls_model, reg_model, prob_vector_model, le, meta = load_models()
    pred_class, class_probs, gen_prob = predict(
        text,
        cls_model,
        reg_model,
        prob_vector_model,
        le,
        meta,
    )

    cls_meta = meta.get("classifier", meta)
    result = {
        "surrogate_class": pred_class,
        "surrogate_class_probabilities": class_probs,
        "surrogate_generated_prob": gen_prob,
        "model_type": cls_meta.get("model_type"),
        "model_cv_accuracy": cls_meta.get("cv_balanced_accuracy", cls_meta.get("cv_accuracy")),
    }

    if args.compare:
        from src.client import GPTZeroClient
        api_key = args.api_key or os.environ.get("GPTZERO_API_KEY")
        if not api_key:
            print("ERROR: --compare requires --api-key or GPTZERO_API_KEY")
            sys.exit(1)

        client = GPTZeroClient(api_key=api_key)
        gz = client.predict(text, delay=0)
        pred_class, class_probs, gen_prob = predict(
            text,
            cls_model,
            reg_model,
            prob_vector_model,
            le,
            meta,
            api_response=gz if "error" not in gz else None,
        )
        result["surrogate_class"] = pred_class
        result["surrogate_class_probabilities"] = class_probs
        result["surrogate_generated_prob"] = gen_prob

        result["gptzero_class"] = gz.get("predicted_class")
        result["gptzero_probability"] = gz.get("average_generated_prob")
        result["gptzero_confidence"] = gz.get("confidence_category")
        result["class_match"] = pred_class == gz.get("predicted_class")

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nText ({len(text.split())} words):")
        print(f"  {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"\nSurrogate class: {pred_class}")
        if gen_prob is not None:
            print(f"Surrogate prob:  {gen_prob:.4f}")
        if class_probs:
            print(f"  Class probabilities:")
            for cls, p in sorted(class_probs.items(), key=lambda x: -x[1]):
                bar = "#" * int(p * 40)
                print(f"    {cls:12s} {p:.1%} {bar}")

        if args.compare and "gptzero_class" in result:
            print(f"\nGPTZero class:   {result['gptzero_class']}")
            print(f"GPTZero prob:    {result['gptzero_probability']}")
            print(f"GPTZero conf:    {result['gptzero_confidence']}")
            match = "MATCH" if result.get("class_match") else "MISMATCH"
            print(f"\n  [{match}]")


if __name__ == "__main__":
    main()
