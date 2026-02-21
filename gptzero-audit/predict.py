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

    return cls_model, reg_model, le, meta


def predict(text, cls_model, reg_model, le):
    """Run surrogate on text. Returns (class, class_probs, generated_prob)."""
    vec, _ = extract_features(text)
    vec = vec.reshape(1, -1)

    pred_encoded = cls_model.predict(vec)[0]
    pred_class = le.inverse_transform([pred_encoded])[0]

    class_probs = {}
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
        gen_prob = float(reg_model.predict(vec)[0])

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

    cls_model, reg_model, le, meta = load_models()
    pred_class, class_probs, gen_prob = predict(text, cls_model, reg_model, le)

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
