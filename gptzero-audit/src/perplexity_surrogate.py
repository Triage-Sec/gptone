"""GPTZero surrogate using GPT-2 perplexity features.

GPTZero's core detection signal is per-sentence perplexity computed by a language model.
AI text has low, uniform perplexity. Human text has high, variable perplexity (burstiness).
This module computes the same features using GPT-2 and trains a regression model on top.

Features computed per text:
  - Per-sentence perplexity (geometric mean of token NLL)
  - Document-level aggregates: mean, std, min, max, range, entropy, skew
  - Burstiness: coefficient of variation of per-sentence perplexity
  - Fraction of "low perplexity" sentences (below threshold)
"""

from __future__ import annotations

import json
import math
import os
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


PERPLEXITY_FEATURE_NAMES = [
    "ppl_mean",
    "ppl_std",
    "ppl_min",
    "ppl_max",
    "ppl_range",
    "ppl_median",
    "ppl_q25",
    "ppl_q75",
    "ppl_iqr",
    "ppl_cv",
    "ppl_skew",
    "ppl_low_frac",
    "ppl_high_frac",
    "sent_count",
    "log_ppl_mean",
    "log_ppl_std",
]


def compute_perplexity_features(
    texts: Sequence[str],
    model_name: str = "gpt2",
    device: str | None = None,
    batch_size: int = 8,
    low_threshold: float = 30.0,
    high_threshold: float = 200.0,
) -> np.ndarray:
    """Compute per-sentence perplexity features for a batch of texts.

    Returns array of shape (len(texts), len(PERPLEXITY_FEATURE_NAMES)).
    """
    import re
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()

    def _split_sentences(text: str) -> List[str]:
        chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
        return chunks if chunks else [text.strip()]

    def _sentence_perplexity(sentence: str) -> float:
        tokens = tokenizer.encode(sentence, return_tensors="pt").to(device)
        if tokens.shape[1] < 2:
            return 100.0
        with torch.no_grad():
            outputs = model(tokens, labels=tokens)
            nll = outputs.loss.item()
        return math.exp(min(nll, 20.0))

    all_features: List[np.ndarray] = []

    for i, text in enumerate(texts):
        sentences = _split_sentences(text)
        ppls = [_sentence_perplexity(s) for s in sentences]

        if not ppls:
            all_features.append(np.zeros(len(PERPLEXITY_FEATURE_NAMES)))
            continue

        arr = np.array(ppls, dtype=float)
        log_arr = np.log1p(arr)
        n = len(arr)

        mean_ppl = float(np.mean(arr))
        std_ppl = float(np.std(arr)) if n > 1 else 0.0
        min_ppl = float(np.min(arr))
        max_ppl = float(np.max(arr))
        range_ppl = max_ppl - min_ppl
        median_ppl = float(np.median(arr))
        q25 = float(np.percentile(arr, 25)) if n >= 4 else min_ppl
        q75 = float(np.percentile(arr, 75)) if n >= 4 else max_ppl
        iqr = q75 - q25
        cv = std_ppl / max(mean_ppl, 1e-6)

        if n > 2 and std_ppl > 1e-6:
            skew = float(np.mean(((arr - mean_ppl) / std_ppl) ** 3))
        else:
            skew = 0.0

        low_frac = float(np.mean(arr < low_threshold))
        high_frac = float(np.mean(arr > high_threshold))

        features = np.array([
            mean_ppl, std_ppl, min_ppl, max_ppl, range_ppl,
            median_ppl, q25, q75, iqr, cv, skew,
            low_frac, high_frac, float(n),
            float(np.mean(log_arr)), float(np.std(log_arr)) if n > 1 else 0.0,
        ], dtype=float)

        all_features.append(features)

        if (i + 1) % 100 == 0:
            print(f"  Perplexity features: {i+1}/{len(texts)}...")

    return np.vstack(all_features)


def train_perplexity_surrogate(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Train a GBM surrogate using GPT-2 perplexity + hand-crafted text features."""
    from sklearn.ensemble import GradientBoostingRegressor
    import random

    data_path = str(config.get("data_path", "/artifacts/gptzero-audit/combined_training_data.jsonl"))
    output_dir = str(config.get("output_dir", "/artifacts/gptzero-audit/ppl_surrogate"))
    gpt2_model = str(config.get("gpt2_model", "gpt2"))
    use_text_features = bool(config.get("use_text_features", True))
    seed = int(config.get("seed", 42))
    val_split = float(config.get("val_split", 0.15))

    random.seed(seed)
    np.random.seed(seed)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading training data from {data_path}...")
    records: List[Dict] = []
    with open(data_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("gptzero_prob") is not None and not rec.get("has_error"):
                records.append(rec)
    print(f"Loaded {len(records)} valid records")

    texts = [r["text"] for r in records]
    probs = np.array([float(r["gptzero_prob"]) for r in records])

    ppl_cache = out_path / "perplexity_features.npy"
    if ppl_cache.exists():
        print("Loading cached perplexity features...")
        X_ppl = np.load(str(ppl_cache))
        if X_ppl.shape[0] != len(texts):
            print(f"Cache size mismatch, recomputing...")
            X_ppl = compute_perplexity_features(texts, model_name=gpt2_model)
            np.save(str(ppl_cache), X_ppl)
    else:
        print(f"Computing GPT-2 perplexity features for {len(texts)} texts...")
        X_ppl = compute_perplexity_features(texts, model_name=gpt2_model)
        np.save(str(ppl_cache), X_ppl)

    feature_names = list(PERPLEXITY_FEATURE_NAMES)

    if use_text_features:
        from src.features import extract_features_batch, FEATURE_NAMES
        print(f"Computing text features for {len(texts)} texts...")
        txt_cache = out_path / "text_features.npy"
        if txt_cache.exists():
            X_txt = np.load(str(txt_cache))
            if X_txt.shape[0] != len(texts):
                X_txt, _ = extract_features_batch(texts)
                np.save(str(txt_cache), X_txt)
        else:
            X_txt, _ = extract_features_batch(texts)
            np.save(str(txt_cache), X_txt)
        X = np.hstack([X_ppl, X_txt])
        feature_names.extend(FEATURE_NAMES)
        print(f"Combined features: {X_ppl.shape[1]} ppl + {X_txt.shape[1]} text = {X.shape[1]}")
    else:
        X = X_ppl

    print(f"Feature matrix: {X.shape}")

    base_ids = sorted(set(r.get("base_sample_id", r.get("sample_id", "")) for r in records))
    rng = random.Random(seed)
    rng.shuffle(base_ids)
    val_count = max(1, int(len(base_ids) * val_split))
    val_ids = set(base_ids[:val_count])

    train_mask = np.array([r.get("base_sample_id", r.get("sample_id", "")) not in val_ids for r in records])
    val_mask = ~train_mask

    X_train, y_train = X[train_mask], probs[train_mask]
    X_val, y_val = X[val_mask], probs[val_mask]
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}")

    print("Training GradientBoosting regressor...")
    gbm = GradientBoostingRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=10,
        random_state=seed,
    )
    gbm.fit(X_train, y_train)

    train_pred = np.clip(gbm.predict(X_train), 0.0, 1.0)
    val_pred = np.clip(gbm.predict(X_val), 0.0, 1.0)

    train_mae = float(np.mean(np.abs(train_pred - y_train)))
    val_mae = float(np.mean(np.abs(val_pred - y_val)))

    train_records_list = [records[i] for i in range(len(records)) if train_mask[i]]
    val_records_list = [records[i] for i in range(len(records)) if val_mask[i]]

    val_trans = _compute_transition_agreement(val_records_list, val_pred, y_val)
    train_trans = _compute_transition_agreement(train_records_list, train_pred, y_train)

    val_cls_pred = np.where(val_pred >= 0.65, "ai", np.where(val_pred <= 0.35, "human", "mixed"))
    val_cls_true = np.array([r.get("gptzero_class", "human") for r in val_records_list])
    val_agreement = float(np.mean(val_cls_pred == val_cls_true))

    print(f"\nResults:")
    print(f"  Train MAE: {train_mae:.4f}, Transition: {train_trans:.4f}")
    print(f"  Val MAE:   {val_mae:.4f}, Transition: {val_trans:.4f}")
    print(f"  Val Agreement: {val_agreement:.4f}")

    model_path = out_path / "ppl_surrogate.pkl"
    with model_path.open("wb") as fh:
        pickle.dump({
            "model": gbm,
            "gpt2_model": gpt2_model,
            "feature_names": feature_names,
            "use_text_features": use_text_features,
        }, fh)

    importances = dict(zip(feature_names, gbm.feature_importances_.tolist()))

    meta = {
        "model_type": "perplexity_surrogate",
        "gpt2_model": gpt2_model,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "val_transition": val_trans,
        "val_agreement": val_agreement,
        "train_size": int(X_train.shape[0]),
        "val_size": int(X_val.shape[0]),
        "feature_importances": importances,
        "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_path / "ppl_surrogate_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    print(f"Model saved to {model_path}")
    return meta


def _compute_transition_agreement(
    records: List[Dict],
    pred_probs: np.ndarray,
    true_probs: np.ndarray,
) -> float:
    groups: Dict[str, List[int]] = {}
    for i, rec in enumerate(records):
        bid = rec.get("base_sample_id", rec.get("sample_id", ""))
        groups.setdefault(bid, []).append(i)

    flags: List[float] = []
    for bid, indices in groups.items():
        ordered = sorted(indices, key=lambda i: int(records[i].get("variant_order", 0)))
        if len(ordered) < 2:
            continue
        base_idx = ordered[0]
        base_pred = float(pred_probs[base_idx])
        base_true = float(true_probs[base_idx])
        for var_idx in ordered[1:]:
            delta_pred = float(pred_probs[var_idx]) - base_pred
            delta_true = float(true_probs[var_idx]) - base_true
            if abs(delta_true) < 0.003:
                continue
            sign_match = (delta_pred > 0) == (delta_true > 0) or (delta_pred < 0) == (delta_true < 0)
            flags.append(1.0 if sign_match else 0.0)

    return float(np.mean(flags)) if flags else 0.0


class PerplexitySurrogate:
    """Inference wrapper for the perplexity + text feature surrogate."""

    def __init__(self, model_dir: str, device: str | None = None):
        self.model_dir = Path(model_dir)
        self.device = device

        model_path = self.model_dir / "ppl_surrogate.pkl"
        with model_path.open("rb") as fh:
            data = pickle.load(fh)
        self.model = data["model"]
        self.gpt2_model = data.get("gpt2_model", "gpt2")
        self.use_text_features = data.get("use_text_features", False)

    def predict_prob(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        X_ppl = compute_perplexity_features(texts, model_name=self.gpt2_model, device=self.device)
        if self.use_text_features:
            from src.features import extract_features_batch
            X_txt, _ = extract_features_batch(list(texts))
            X = np.hstack([X_ppl, X_txt])
        else:
            X = X_ppl
        return np.clip(self.model.predict(X), 0.0, 1.0)
