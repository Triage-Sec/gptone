#!/usr/bin/env python3
"""
Surrogate Predictor
====================
Run the trained surrogate model on any text.
Optionally compare against GPTZero live.

Usage:
  # Surrogate only (no API key needed)
  python predict.py "Your text here"
  
  # Compare surrogate vs GPTZero live
  python predict.py --compare --api-key YOUR_KEY "Your text here"

  # From stdin
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


def load_model():
    model = joblib.load(MODEL_DIR / "surrogate_model.joblib")
    le = joblib.load(MODEL_DIR / "label_encoder.joblib")
    with open(MODEL_DIR / "model_meta.json") as f:
        meta = json.load(f)
    return model, le, meta


def predict(text, model, le):
    """Run surrogate on text. Returns (class, probabilities)."""
    vec, _ = extract_features(text)
    vec = vec.reshape(1, -1)
    
    pred_encoded = model.predict(vec)[0]
    pred_class = le.inverse_transform([pred_encoded])[0]
    
    probs = {}
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(vec)[0]
        for cls, prob in zip(le.classes_, p):
            probs[cls] = float(prob)
    
    return pred_class, probs


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
    
    model, le, meta = load_model()
    pred_class, probs = predict(text, model, le)
    
    result = {
        "surrogate_prediction": pred_class,
        "surrogate_probabilities": probs,
        "model_type": meta.get("model_type"),
        "model_cv_accuracy": meta.get("cv_accuracy"),
    }
    
    if args.compare:
        from src.client import GPTZeroClient
        api_key = args.api_key or os.environ.get("GPTZERO_API_KEY")
        if not api_key:
            print("ERROR: --compare requires --api-key or GPTZERO_API_KEY")
            sys.exit(1)
        
        client = GPTZeroClient(api_key=api_key)
        gz = client.predict(text, delay=0)
        
        result["gptzero_prediction"] = gz.get("predicted_class")
        result["gptzero_probability"] = gz.get("average_generated_prob")
        result["gptzero_confidence"] = gz.get("confidence_category")
        result["match"] = pred_class == gz.get("predicted_class")
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"\nText ({len(text.split())} words):")
        print(f"  {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"\nSurrogate prediction: {pred_class}")
        if probs:
            print(f"  Probabilities:")
            for cls, p in sorted(probs.items(), key=lambda x: -x[1]):
                bar = "#" * int(p * 40)
                print(f"    {cls:12s} {p:.1%} {bar}")
        
        if args.compare and "gptzero_prediction" in result:
            print(f"\nGPTZero prediction:   {result['gptzero_prediction']}")
            print(f"  Probability:        {result['gptzero_probability']}")
            print(f"  Confidence:         {result['gptzero_confidence']}")
            match = "MATCH" if result.get("match") else "MISMATCH"
            print(f"\n  [{match}]")


if __name__ == "__main__":
    main()
