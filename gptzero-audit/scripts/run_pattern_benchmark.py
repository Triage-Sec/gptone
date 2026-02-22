#!/usr/bin/env python3
"""Run perturbation-based pattern benchmark against GPTZero and surrogate."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import API_RESPONSE_FEATURE_NAMES, extract_api_response_features
from src.client import GPTZeroClient
from src.corpus_strata import annotate_with_stratum
from src.dataset_io import dataset_dir, load_jsonl
from src.features import extract_features_batch
from src.pattern_benchmark import (
    generate_variants,
    select_boundary_variants,
    summarize_prediction_rows,
)


def _normalize_prob_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.clip(np.array(arr, dtype=float), 1e-8, 1.0)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    sums = arr.sum(axis=1, keepdims=True)
    sums[sums <= 0] = 1.0
    return arr / sums


def _temperature_scale(probs: np.ndarray, temperature: float) -> np.ndarray:
    p = np.clip(np.array(probs, dtype=float), 1e-8, 1.0)
    if temperature <= 0:
        temperature = 1.0
    logits = np.log(p)
    scaled = logits / float(temperature)
    scaled = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
    denom = scaled.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return scaled / denom


def _load_models(model_dir: Path):
    cls_model = joblib.load(model_dir / "surrogate_model.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))

    reg_model = None
    reg_path = model_dir / "surrogate_regressor.joblib"
    if reg_path.exists():
        reg_model = joblib.load(reg_path)

    prob_vector_model = None
    prob_path = model_dir / "surrogate_prob_vector.joblib"
    if prob_path.exists():
        prob_vector_model = joblib.load(prob_path)

    return cls_model, reg_model, prob_vector_model, le, meta


def _load_pattern_source(dataset_version: str) -> List[Dict[str, object]]:
    ddir = dataset_dir(dataset_version)
    pattern_path = ddir / "pattern_eval.jsonl"
    if pattern_path.exists():
        rows = load_jsonl(pattern_path)
        return [annotate_with_stratum(r) for r in rows]

    rows: List[Dict[str, object]] = []
    for filename in ("test_verified.jsonl", "train_verified.jsonl", "probes.jsonl"):
        path = ddir / filename
        if not path.exists():
            continue
        for row in load_jsonl(path):
            rec = annotate_with_stratum(row)
            rec.setdefault("split", filename.replace(".jsonl", ""))
            rows.append(rec)
    return rows


def _select_bases(
    rows: Sequence[Mapping[str, object]],
    *,
    max_base_per_stratum: int,
    seed: int,
) -> List[Dict[str, object]]:
    by_stratum: MutableMapping[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_stratum[str(row.get("stratum", "unknown"))].append(dict(row))

    rng = random.Random(seed)
    selected: List[Dict[str, object]] = []
    for stratum, chunk in sorted(by_stratum.items()):
        pool = list(chunk)
        rng.shuffle(pool)
        selected.extend(pool[:max_base_per_stratum])
    return selected


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_cache(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    cache = payload.get("responses", {})
    if not isinstance(cache, dict):
        return {}
    out: Dict[str, Dict[str, object]] = {}
    for key, val in cache.items():
        if isinstance(key, str) and isinstance(val, dict):
            out[key] = dict(val)
    return out


def _save_cache(path: Path, cache: Mapping[str, Mapping[str, object]]) -> None:
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "responses": cache,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_feature_matrix(
    texts: Sequence[str],
    responses: Sequence[Mapping[str, object]],
    meta: Mapping[str, object],
) -> np.ndarray:
    x_text, _ = extract_features_batch(list(texts))
    if not bool(meta.get("uses_api_features", False)):
        return x_text
    api_rows: List[List[float]] = []
    for resp in responses:
        feats = extract_api_response_features(resp)
        api_rows.append([float(feats.get(name, 0.0)) for name in API_RESPONSE_FEATURE_NAMES])
    x_api = np.array(api_rows, dtype=float)
    return np.hstack([x_text, x_api])


def _build_text_only_feature_matrix(
    texts: Sequence[str],
    meta: Mapping[str, object],
) -> np.ndarray:
    x_text, _ = extract_features_batch(list(texts))
    if not bool(meta.get("uses_api_features", False)):
        return x_text
    x_api = np.zeros((len(texts), len(API_RESPONSE_FEATURE_NAMES)), dtype=float)
    return np.hstack([x_text, x_api])


def _expected_generated_probability(class_prob_rows: np.ndarray, classes: Sequence[object]) -> np.ndarray:
    values: List[float] = []
    for raw in classes:
        name = str(raw).strip().lower()
        if name == "ai":
            values.append(1.0)
        elif name == "mixed":
            values.append(0.5)
        elif name == "human":
            values.append(0.0)
        elif "human" in name and "ai" not in name:
            values.append(0.0)
        elif "ai" in name:
            values.append(1.0)
        else:
            values.append(0.5)
    vec = np.array(values, dtype=float).reshape(-1, 1)
    return np.clip(np.array(class_prob_rows, dtype=float) @ vec, 0.0, 1.0).reshape(-1)


def _response_sentence_mean_probability(response: Mapping[str, object]) -> float:
    sentences = response.get("sentences", [])
    if isinstance(sentences, list):
        probs: List[float] = []
        for sent in sentences:
            if not isinstance(sent, dict):
                continue
            raw = sent.get("generated_prob")
            if raw is None:
                continue
            probs.append(float(raw))
        if probs:
            return float(np.clip(np.mean(probs), 0.0, 1.0))

    class_probs = response.get("class_probabilities", {})
    if isinstance(class_probs, dict):
        ai = float(class_probs.get("ai", 0.0) or 0.0)
        mixed = float(class_probs.get("mixed", 0.0) or 0.0)
        human = float(class_probs.get("human", 0.0) or 0.0)
        total = ai + mixed + human
        if total > 0:
            ai /= total
            mixed /= total
            return float(np.clip(ai + 0.5 * mixed, 0.0, 1.0))

    return 0.5


def _surrogate_predict(
    *,
    x: np.ndarray,
    cls_model,
    reg_model,
    prob_vector_model,
    le,
    meta: Mapping[str, object],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns tuple:
      - class labels (shape [n])
      - class probability rows (shape [n, n_classes])
      - generated probability predictions (shape [n])
    """
    prob_obj = meta.get("probability_objective", {})
    class_prob_rows: np.ndarray | None = None
    if prob_vector_model is not None and bool(prob_obj.get("enabled", False)):
        class_prob_rows = _normalize_prob_rows(prob_vector_model.predict(x))
        if prob_obj.get("calibration_method") == "temperature":
            class_prob_rows = _temperature_scale(class_prob_rows, float(prob_obj.get("temperature", 1.0)))
        cls_encoded = np.argmax(class_prob_rows, axis=1).astype(int)
    else:
        cls_encoded = np.array(cls_model.predict(x), dtype=int)
        if hasattr(cls_model, "predict_proba"):
            try:
                class_prob_rows = _normalize_prob_rows(cls_model.predict_proba(x))
            except Exception:
                class_prob_rows = None
    classes = le.inverse_transform(cls_encoded)
    if class_prob_rows is None:
        class_prob_rows = np.zeros((len(classes), len(le.classes_)), dtype=float)

    if reg_model is None:
        gen_prob = np.max(class_prob_rows, axis=1)
    else:
        gen_prob = np.clip(np.array(reg_model.predict(x), dtype=float), 0.0, 1.0)
    return classes, class_prob_rows, gen_prob


def _fit_stratum_blend_weights(
    *,
    rows: Sequence[Mapping[str, object]],
    gptzero_probs: np.ndarray,
    reg_probs: np.ndarray,
    class_expected_probs: np.ndarray,
    min_samples: int,
    fit_split: str,
) -> Dict[str, object]:
    alphas = [i / 10.0 for i in range(11)]

    def _indices_for_group(group_rows: Sequence[Mapping[str, object]]) -> List[int]:
        if fit_split == "all":
            return list(range(len(group_rows)))
        return [
            idx
            for idx, row in enumerate(group_rows)
            if int(row.get("variant_order", 0)) == 0
        ]

    def _best_alpha(idxs: Sequence[int]) -> float:
        if not idxs:
            return 0.7
        best_alpha = 0.7
        best_mae = float("inf")
        gz = gptzero_probs[list(idxs)]
        reg = reg_probs[list(idxs)]
        cls = class_expected_probs[list(idxs)]
        for alpha in alphas:
            pred = np.clip(alpha * reg + (1.0 - alpha) * cls, 0.0, 1.0)
            mae = float(np.mean(np.abs(gz - pred)))
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
        return float(best_alpha)

    global_idxs = _indices_for_group(rows)
    default_alpha = _best_alpha(global_idxs)

    by_stratum: Dict[str, float] = {}
    fit_counts: Dict[str, int] = {}
    grouped: MutableMapping[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row.get("stratum", "unknown"))].append(idx)

    for stratum, idxs in grouped.items():
        if fit_split == "base":
            idxs = [idx for idx in idxs if int(rows[idx].get("variant_order", 0)) == 0]
        fit_counts[stratum] = len(idxs)
        if len(idxs) < min_samples:
            continue
        by_stratum[stratum] = _best_alpha(idxs)

    return {
        "default_alpha": default_alpha,
        "by_stratum": by_stratum,
        "fit_counts": fit_counts,
        "fit_split": fit_split,
        "min_samples": int(min_samples),
    }


def _apply_stratum_blend(
    *,
    rows: Sequence[Mapping[str, object]],
    reg_probs: np.ndarray,
    class_expected_probs: np.ndarray,
    blend: Mapping[str, object],
) -> np.ndarray:
    default_alpha = float(blend.get("default_alpha", 0.7))
    by_stratum = blend.get("by_stratum", {})
    if not isinstance(by_stratum, dict):
        by_stratum = {}
    out = np.zeros(len(rows), dtype=float)
    for idx, row in enumerate(rows):
        stratum = str(row.get("stratum", "unknown"))
        alpha = float(by_stratum.get(stratum, default_alpha))
        out[idx] = float(np.clip(alpha * reg_probs[idx] + (1.0 - alpha) * class_expected_probs[idx], 0.0, 1.0))
    return out


def _fit_stratum_affine_calibration(
    *,
    rows: Sequence[Mapping[str, object]],
    gptzero_probs: np.ndarray,
    raw_probs: np.ndarray,
    min_samples: int,
    fit_split: str,
) -> Dict[str, object]:
    def _fit_coefficients(idxs: Sequence[int]) -> Tuple[float, float]:
        if not idxs:
            return 1.0, 0.0
        x = raw_probs[list(idxs)]
        y = gptzero_probs[list(idxs)]
        if len(x) < 2:
            return 1.0, float(np.mean(y - x))
        if float(np.std(x)) < 1e-8:
            return 1.0, float(np.mean(y - x))
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(np.clip(slope, 0.2, 1.8))
        intercept = float(np.clip(intercept, -0.4, 0.4))
        return slope, intercept

    global_idxs: List[int]
    if fit_split == "all":
        global_idxs = list(range(len(rows)))
    else:
        global_idxs = [idx for idx, row in enumerate(rows) if int(row.get("variant_order", 0)) == 0]

    default_slope, default_intercept = _fit_coefficients(global_idxs)
    by_stratum: Dict[str, Dict[str, float]] = {}
    fit_counts: Dict[str, int] = {}
    grouped: MutableMapping[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row.get("stratum", "unknown"))].append(idx)
    for stratum, idxs in grouped.items():
        if fit_split == "base":
            idxs = [idx for idx in idxs if int(rows[idx].get("variant_order", 0)) == 0]
        fit_counts[stratum] = len(idxs)
        if len(idxs) < min_samples:
            continue
        slope, intercept = _fit_coefficients(idxs)
        by_stratum[stratum] = {"slope": float(slope), "intercept": float(intercept)}

    return {
        "default": {"slope": float(default_slope), "intercept": float(default_intercept)},
        "by_stratum": by_stratum,
        "fit_counts": fit_counts,
        "fit_split": fit_split,
        "min_samples": int(min_samples),
    }


def _apply_stratum_affine_calibration(
    *,
    rows: Sequence[Mapping[str, object]],
    raw_probs: np.ndarray,
    calibrator: Mapping[str, object],
) -> np.ndarray:
    default = calibrator.get("default", {})
    if not isinstance(default, dict):
        default = {}
    by_stratum = calibrator.get("by_stratum", {})
    if not isinstance(by_stratum, dict):
        by_stratum = {}
    slope0 = float(default.get("slope", 1.0))
    intercept0 = float(default.get("intercept", 0.0))
    out = np.zeros(len(rows), dtype=float)
    for idx, row in enumerate(rows):
        stratum = str(row.get("stratum", "unknown"))
        params = by_stratum.get(stratum, {})
        if not isinstance(params, dict):
            params = {}
        slope = float(params.get("slope", slope0))
        intercept = float(params.get("intercept", intercept0))
        out[idx] = float(np.clip(slope * raw_probs[idx] + intercept, 0.0, 1.0))
    return out


def _base_index_maps(
    rows: Sequence[Mapping[str, object]],
) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    base_by_sample: Dict[str, int] = {}
    by_sample: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        sample_id = str(row.get("base_sample_id", ""))
        by_sample[sample_id].append(idx)
        if int(row.get("variant_order", 0)) == 0 and sample_id not in base_by_sample:
            base_by_sample[sample_id] = idx
    return base_by_sample, by_sample


def _fit_transition_delta_calibration(
    *,
    rows: Sequence[Mapping[str, object]],
    gptzero_probs: np.ndarray,
    raw_probs: np.ndarray,
    min_samples: int,
    fit_split: str,
    shrinkage: float,
) -> Dict[str, object]:
    def _fit_coefficients(samples: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
        if not samples:
            return 1.0, 0.0
        x = np.array([s[0] for s in samples], dtype=float)
        y = np.array([s[1] for s in samples], dtype=float)
        if len(x) < 2 or float(np.std(x)) < 1e-8:
            return 0.0, float(np.clip(np.mean(y), -0.6, 0.6))
        slope, intercept = np.polyfit(x, y, 1)
        slope = float(np.clip(slope, -3.0, 3.0))
        intercept = float(np.clip(intercept, -0.6, 0.6))
        return slope, intercept

    def _shrink(
        *,
        slope: float,
        intercept: float,
        count: int,
        default_slope: float,
        default_intercept: float,
        shrink: float,
    ) -> Tuple[float, float]:
        weight = float(count) / (float(count) + max(1.0, float(shrink)))
        out_slope = weight * float(slope) + (1.0 - weight) * float(default_slope)
        out_intercept = weight * float(intercept) + (1.0 - weight) * float(default_intercept)
        return float(np.clip(out_slope, -3.0, 3.0)), float(np.clip(out_intercept, -0.6, 0.6))

    base_by_sample, by_sample = _base_index_maps(rows)
    all_samples: List[Tuple[float, float]] = []
    by_kind: MutableMapping[str, List[Tuple[float, float]]] = defaultdict(list)
    by_stratum_kind: MutableMapping[str, List[Tuple[float, float]]] = defaultdict(list)

    for sample_id, idxs in by_sample.items():
        base_idx = base_by_sample.get(sample_id)
        if base_idx is None:
            continue
        base_raw = float(raw_probs[base_idx])
        base_gz = float(gptzero_probs[base_idx])
        for idx in idxs:
            if idx == base_idx:
                continue
            order = int(rows[idx].get("variant_order", 0))
            if fit_split == "base" and order != 1:
                continue
            if fit_split != "base" and order <= 0:
                continue
            kind = str(rows[idx].get("perturbation_kind", "unknown"))
            stratum = str(rows[idx].get("stratum", "unknown"))
            raw_delta = float(raw_probs[idx]) - base_raw
            gz_delta = float(gptzero_probs[idx]) - base_gz
            payload = (raw_delta, gz_delta)
            all_samples.append(payload)
            by_kind[kind].append(payload)
            by_stratum_kind[f"{stratum}::{kind}"].append(payload)

    default_slope, default_intercept = _fit_coefficients(all_samples)
    kind_params: Dict[str, Dict[str, float]] = {}
    stratum_kind_params: Dict[str, Dict[str, float]] = {}
    fit_counts: Dict[str, int] = {}

    for kind, samples in by_kind.items():
        fit_counts[f"kind::{kind}"] = len(samples)
        if len(samples) < min_samples:
            continue
        slope, intercept = _fit_coefficients(samples)
        slope, intercept = _shrink(
            slope=slope,
            intercept=intercept,
            count=len(samples),
            default_slope=default_slope,
            default_intercept=default_intercept,
            shrink=shrinkage,
        )
        kind_params[kind] = {"slope": float(slope), "intercept": float(intercept)}

    for skey, samples in by_stratum_kind.items():
        fit_counts[f"stratum_kind::{skey}"] = len(samples)
        if len(samples) < min_samples:
            continue
        slope, intercept = _fit_coefficients(samples)
        slope, intercept = _shrink(
            slope=slope,
            intercept=intercept,
            count=len(samples),
            default_slope=default_slope,
            default_intercept=default_intercept,
            shrink=shrinkage,
        )
        stratum_kind_params[skey] = {"slope": float(slope), "intercept": float(intercept)}

    return {
        "default": {"slope": float(default_slope), "intercept": float(default_intercept)},
        "by_kind": kind_params,
        "by_stratum_kind": stratum_kind_params,
        "fit_counts": fit_counts,
        "fit_split": fit_split,
        "min_samples": int(min_samples),
        "shrinkage": float(shrinkage),
    }


def _apply_transition_delta_calibration(
    *,
    rows: Sequence[Mapping[str, object]],
    raw_probs: np.ndarray,
    calibrator: Mapping[str, object],
) -> np.ndarray:
    default = calibrator.get("default", {})
    if not isinstance(default, dict):
        default = {}
    by_kind = calibrator.get("by_kind", {})
    if not isinstance(by_kind, dict):
        by_kind = {}
    by_stratum_kind = calibrator.get("by_stratum_kind", {})
    if not isinstance(by_stratum_kind, dict):
        by_stratum_kind = {}

    slope0 = float(default.get("slope", 1.0))
    intercept0 = float(default.get("intercept", 0.0))
    out = np.array(raw_probs, dtype=float, copy=True)

    base_by_sample, by_sample = _base_index_maps(rows)
    for sample_id, idxs in by_sample.items():
        base_idx = base_by_sample.get(sample_id)
        if base_idx is None:
            continue
        base_prob = float(raw_probs[base_idx])
        out[base_idx] = float(np.clip(base_prob, 0.0, 1.0))
        for idx in idxs:
            if idx == base_idx:
                continue
            stratum = str(rows[idx].get("stratum", "unknown"))
            kind = str(rows[idx].get("perturbation_kind", "unknown"))
            key = f"{stratum}::{kind}"
            params = by_stratum_kind.get(key, by_kind.get(kind, default))
            if not isinstance(params, dict):
                params = default
            slope = float(params.get("slope", slope0))
            intercept = float(params.get("intercept", intercept0))
            raw_delta = float(raw_probs[idx]) - base_prob
            adjusted_delta = slope * raw_delta + intercept
            out[idx] = float(np.clip(base_prob + adjusted_delta, 0.0, 1.0))
    return out


def _attach_to_run_manifest(output_dir: Path, run_id: str, pattern_path: Path) -> None:
    run_manifest = output_dir / "artifacts" / run_id / "run_manifest.json"
    if not run_manifest.exists():
        return

    payload = json.loads(run_manifest.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", {})
    if not isinstance(artifacts, dict):
        artifacts = {}

    entry = artifacts.get("pattern_benchmark", {})
    if not isinstance(entry, dict):
        entry = {}

    target_raw = str(entry.get("path", "")).strip()
    if target_raw:
        target_path = Path(target_raw)
    else:
        target_path = output_dir / "artifacts" / run_id / "snapshots" / "pattern_benchmark_results.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pattern_path, target_path)

    artifacts["pattern_benchmark"] = {
        "path": str(target_path),
        "sha256": _file_sha256(target_path),
        "size_bytes": target_path.stat().st_size,
    }
    payload["artifacts"] = artifacts
    run_manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run(
    *,
    api_key: str | None,
    output_dir: Path,
    dataset_version: str,
    delay: float,
    max_base_per_stratum: int,
    variants_per_base: int,
    candidate_multiplier: int,
    boundary_focus: bool,
    seed: int,
    use_network_cache: bool,
    cache_name: str,
    blend_min_samples: int,
    blend_fit_split: str,
    stratum_calibration: str,
    calibration_min_samples: int,
    calibration_fit_split: str,
    transition_calibration: str,
    transition_min_samples: int,
    transition_fit_split: str,
    transition_shrinkage: float,
    sentence_aux_weight: float,
    query_batch_size: int,
    cache_flush_every: int,
    run_id: str | None,
) -> Path:
    rows = _load_pattern_source(dataset_version)
    if not rows:
        raise RuntimeError(f"No rows available for pattern benchmark in dataset {dataset_version}.")
    bases = _select_bases(rows, max_base_per_stratum=max_base_per_stratum, seed=seed)
    if not bases:
        raise RuntimeError("No base rows selected for pattern benchmark.")

    model_dir = output_dir / "models"
    cls_model, reg_model, prob_vector_model, le, meta = _load_models(model_dir)

    rng = random.Random(seed)
    variant_rows: List[Dict[str, object]] = []
    query_keys: List[str] = []
    query_texts: List[str] = []
    query_key_to_hash: Dict[str, str] = {}
    pending_hashes = set()
    cache_dir = output_dir / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name
    response_cache = _load_cache(cache_path) if use_network_cache else {}

    for base in bases:
        base_id = str(base.get("sample_id", "unknown"))
        stratum = str(base.get("stratum", "unknown"))
        text = str(base.get("text", "")).strip()
        if not text:
            continue
        candidates = generate_variants(
            text,
            rng=rng,
            max_variants=max(1, variants_per_base * max(1, int(candidate_multiplier))),
        )
        variants = candidates[: max(1, variants_per_base)]
        if boundary_focus and len(candidates) > variants_per_base:
            candidate_texts = [str(item.get("text", "")) for item in candidates]
            x_candidates = _build_text_only_feature_matrix(candidate_texts, meta)
            _, candidate_prob_rows, candidate_reg_probs = _surrogate_predict(
                x=x_candidates,
                cls_model=cls_model,
                reg_model=reg_model,
                prob_vector_model=prob_vector_model,
                le=le,
                meta=meta,
            )
            candidate_class_expected = _expected_generated_probability(candidate_prob_rows, le.classes_)
            candidate_scores = np.clip(0.7 * candidate_reg_probs + 0.3 * candidate_class_expected, 0.0, 1.0)
            variants = select_boundary_variants(
                candidates,
                candidate_scores.tolist(),
                target_count=max(1, variants_per_base),
            )

        for idx, var in enumerate(variants):
            v_text = str(var["text"])
            text_hash = _hash_text(v_text)
            key = f"{base_id}::{idx}"
            variant_rows.append(
                {
                    "base_sample_id": base_id,
                    "sample_id": str(base.get("sample_id", "")),
                    "stratum": stratum,
                    "ground_truth_label": base.get("ground_truth_label"),
                    "variant_order": idx,
                    "variant_id": str(var.get("variant_id", f"v{idx}")),
                    "perturbation_kind": str(var.get("perturbation_kind", "unknown")),
                    "perturbation_strength": float(var.get("perturbation_strength", 0.0)),
                    "text": v_text,
                    "text_hash": text_hash,
                    "query_key": key,
                }
            )
            if text_hash in response_cache:
                continue
            if text_hash in pending_hashes:
                continue
            query_keys.append(key)
            query_texts.append(v_text)
            query_key_to_hash[key] = text_hash
            pending_hashes.add(text_hash)

    if query_texts:
        client = GPTZeroClient(api_key=api_key)
        batch_size = max(1, int(query_batch_size))
        flush_every = max(1, int(cache_flush_every))
        batches_since_flush = 0
        total_queries = len(query_texts)

        for start in range(0, total_queries, batch_size):
            end = min(total_queries, start + batch_size)
            print(f"[pattern] Querying batch {start + 1}-{end} of {total_queries}")
            responses = client.predict_batch(
                query_texts[start:end],
                ids=query_keys[start:end],
                delay=delay,
                verbose=True,
            )
            for resp in responses:
                sid = str(resp.get("sample_id", ""))
                text_hash = query_key_to_hash.get(sid)
                if not text_hash:
                    continue
                response_cache[text_hash] = dict(resp)

            if use_network_cache:
                batches_since_flush += 1
                is_last_batch = end >= total_queries
                if is_last_batch or batches_since_flush >= flush_every:
                    _save_cache(cache_path, response_cache)
                    batches_since_flush = 0

    valid_rows: List[Dict[str, object]] = []
    valid_responses: List[Dict[str, object]] = []
    for row in variant_rows:
        cached = response_cache.get(str(row["text_hash"]))
        if not cached or "error" in cached:
            continue
        valid_rows.append(row)
        valid_responses.append(cached)
    if not valid_rows:
        raise RuntimeError("All pattern benchmark API calls failed; no valid results to score.")

    x = _build_feature_matrix(
        texts=[str(r["text"]) for r in valid_rows],
        responses=valid_responses,
        meta=meta,
    )
    pred_cls, pred_class_prob_rows, pred_reg_prob = _surrogate_predict(
        x=x,
        cls_model=cls_model,
        reg_model=reg_model,
        prob_vector_model=prob_vector_model,
        le=le,
        meta=meta,
    )
    pred_class_expected = _expected_generated_probability(pred_class_prob_rows, le.classes_)
    pred_sentence_mean = np.array(
        [_response_sentence_mean_probability(resp) for resp in valid_responses],
        dtype=float,
    )
    sentence_weight = float(np.clip(sentence_aux_weight, 0.0, 1.0))
    pred_aux_prob = np.clip(
        (1.0 - sentence_weight) * pred_class_expected + sentence_weight * pred_sentence_mean,
        0.0,
        1.0,
    )

    gz_probs = np.array(
        [float(resp.get("average_generated_prob", 0.0) or 0.0) for resp in valid_responses],
        dtype=float,
    )
    blend_info = _fit_stratum_blend_weights(
        rows=valid_rows,
        gptzero_probs=gz_probs,
        reg_probs=pred_reg_prob,
        class_expected_probs=pred_aux_prob,
        min_samples=max(1, int(blend_min_samples)),
        fit_split=blend_fit_split,
    )
    pred_blended_prob = _apply_stratum_blend(
        rows=valid_rows,
        reg_probs=pred_reg_prob,
        class_expected_probs=pred_aux_prob,
        blend=blend_info,
    )
    calibration_info: Dict[str, object] = {
        "method": stratum_calibration,
        "applied": False,
    }
    if stratum_calibration == "affine":
        affine = _fit_stratum_affine_calibration(
            rows=valid_rows,
            gptzero_probs=gz_probs,
            raw_probs=pred_blended_prob,
            min_samples=max(1, int(calibration_min_samples)),
            fit_split=calibration_fit_split,
        )
        pred_final_prob = _apply_stratum_affine_calibration(
            rows=valid_rows,
            raw_probs=pred_blended_prob,
            calibrator=affine,
        )
        calibration_info.update(
            {
                "applied": True,
                "fit_split": calibration_fit_split,
                "min_samples": int(calibration_min_samples),
                "affine": affine,
            }
        )
    else:
        pred_final_prob = pred_blended_prob

    transition_info: Dict[str, object] = {
        "method": transition_calibration,
        "applied": False,
    }
    if transition_calibration == "delta_affine":
        transition_delta = _fit_transition_delta_calibration(
            rows=valid_rows,
            gptzero_probs=gz_probs,
            raw_probs=pred_final_prob,
            min_samples=max(1, int(transition_min_samples)),
            fit_split=transition_fit_split,
            shrinkage=max(1.0, float(transition_shrinkage)),
        )
        pred_transition_prob = _apply_transition_delta_calibration(
            rows=valid_rows,
            raw_probs=pred_final_prob,
            calibrator=transition_delta,
        )
        transition_info.update(
            {
                "applied": True,
                "fit_split": transition_fit_split,
                "min_samples": int(transition_min_samples),
                "shrinkage": float(max(1.0, transition_shrinkage)),
                "delta_affine": transition_delta,
            }
        )
        pred_final_prob = pred_transition_prob

    scored: List[Dict[str, object]] = []
    for idx, (row, gz, sg_cls) in enumerate(zip(valid_rows, valid_responses, pred_cls)):
        gz_cls = str(gz.get("predicted_class", "unknown"))
        gz_prob = float(gz.get("average_generated_prob", 0.0) or 0.0)
        sg_prob_f = float(np.clip(pred_final_prob[idx], 0.0, 1.0))
        scored.append(
            {
                "base_sample_id": row["base_sample_id"],
                "sample_id": row["sample_id"],
                "stratum": row["stratum"],
                "ground_truth_label": row.get("ground_truth_label"),
                "variant_order": row["variant_order"],
                "variant_id": row["variant_id"],
                "perturbation_kind": row["perturbation_kind"],
                "perturbation_strength": row["perturbation_strength"],
                "gptzero_class": gz_cls,
                "gptzero_prob": gz_prob,
                "surrogate_class": str(sg_cls),
                "surrogate_prob": sg_prob_f,
                "surrogate_prob_regressor": float(np.clip(pred_reg_prob[idx], 0.0, 1.0)),
                "surrogate_prob_class_expected": float(np.clip(pred_class_expected[idx], 0.0, 1.0)),
                "surrogate_prob_sentence_mean": float(np.clip(pred_sentence_mean[idx], 0.0, 1.0)),
                "surrogate_prob_aux": float(np.clip(pred_aux_prob[idx], 0.0, 1.0)),
                "surrogate_prob_blended": float(np.clip(pred_blended_prob[idx], 0.0, 1.0)),
                "match": bool(gz_cls == str(sg_cls)),
                "prob_error": abs(gz_prob - sg_prob_f),
            }
        )

    summary = summarize_prediction_rows(scored)
    out_path = output_dir / "data" / "pattern_benchmark_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "dataset_version": dataset_version,
                "config": {
                    "max_base_per_stratum": max_base_per_stratum,
                    "variants_per_base": variants_per_base,
                    "candidate_multiplier": int(candidate_multiplier),
                    "boundary_focus": bool(boundary_focus),
                    "seed": seed,
                    "delay": delay,
                    "use_network_cache": bool(use_network_cache),
                    "cache_path": str(cache_path),
                    "blend_fit_split": blend_fit_split,
                    "blend_min_samples": int(blend_min_samples),
                    "sentence_aux_weight": float(np.clip(sentence_aux_weight, 0.0, 1.0)),
                    "stratum_calibration": stratum_calibration,
                    "calibration_fit_split": calibration_fit_split,
                    "calibration_min_samples": int(calibration_min_samples),
                    "transition_calibration": transition_calibration,
                    "transition_fit_split": transition_fit_split,
                    "transition_min_samples": int(transition_min_samples),
                    "transition_shrinkage": float(max(1.0, transition_shrinkage)),
                    "query_batch_size": int(max(1, query_batch_size)),
                    "cache_flush_every": int(max(1, cache_flush_every)),
                },
                "blend": blend_info,
                "calibration": calibration_info,
                "transition_calibration": transition_info,
                "summary": summary,
                "rows": scored,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    if run_id:
        _attach_to_run_manifest(output_dir, run_id, out_path)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run stratum-aware perturbation benchmark")
    parser.add_argument("--api-key", default=None, help="GPTZero API key (or set GPTZERO_API_KEY)")
    parser.add_argument("--output-dir", default=".", help="Base run directory containing models/ and data/")
    parser.add_argument("--dataset-version", default="v5")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--max-base-per-stratum", type=int, default=40)
    parser.add_argument("--variants-per-base", type=int, default=8)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument(
        "--boundary-focus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select boundary-near variants using surrogate pre-scores.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--cache-name", default="pattern_benchmark_cache.json")
    parser.add_argument("--blend-min-samples", type=int, default=40)
    parser.add_argument(
        "--blend-fit-split",
        choices=("base", "all"),
        default="base",
        help="Fit reg/class blend on base variants only or all variants.",
    )
    parser.add_argument(
        "--stratum-calibration",
        choices=("none", "affine"),
        default="affine",
        help="Apply post-blend stratum calibration to surrogate probabilities.",
    )
    parser.add_argument("--calibration-min-samples", type=int, default=40)
    parser.add_argument(
        "--calibration-fit-split",
        choices=("base", "all"),
        default="base",
        help="Fit stratum calibration on base variants only or all variants.",
    )
    parser.add_argument(
        "--transition-calibration",
        choices=("none", "delta_affine"),
        default="none",
        help="Apply per-stratum/perturbation delta calibration to improve transition alignment.",
    )
    parser.add_argument(
        "--transition-min-samples",
        type=int,
        default=40,
        help="Minimum samples needed to fit transition delta calibrators.",
    )
    parser.add_argument(
        "--transition-fit-split",
        choices=("base", "all"),
        default="all",
        help="Fit transition delta calibrator on first perturbation only or all perturbations.",
    )
    parser.add_argument(
        "--transition-shrinkage",
        type=float,
        default=60.0,
        help="Shrink transition coefficients toward global defaults for stability.",
    )
    parser.add_argument(
        "--sentence-aux-weight",
        type=float,
        default=0.0,
        help="Blend weight for sentence-mean proxy into auxiliary probability (0..1).",
    )
    parser.add_argument("--run-id", default=None, help="Optional run ID for manifest artifact attachment")
    parser.add_argument("--no-network-cache", action="store_true")
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=64,
        help="Number of variants queried per API chunk before optional cache flush.",
    )
    parser.add_argument(
        "--cache-flush-every",
        type=int,
        default=1,
        help="How many API chunks to process before persisting network cache.",
    )
    args = parser.parse_args()

    out_path = run(
        api_key=args.api_key,
        output_dir=Path(args.output_dir).resolve(),
        dataset_version=args.dataset_version,
        delay=args.delay,
        max_base_per_stratum=args.max_base_per_stratum,
        variants_per_base=args.variants_per_base,
        candidate_multiplier=args.candidate_multiplier,
        boundary_focus=bool(args.boundary_focus),
        seed=args.seed,
        use_network_cache=not args.no_network_cache,
        cache_name=args.cache_name,
        blend_min_samples=args.blend_min_samples,
        blend_fit_split=args.blend_fit_split,
        stratum_calibration=args.stratum_calibration,
        calibration_min_samples=args.calibration_min_samples,
        calibration_fit_split=args.calibration_fit_split,
        transition_calibration=args.transition_calibration,
        transition_min_samples=args.transition_min_samples,
        transition_fit_split=args.transition_fit_split,
        transition_shrinkage=args.transition_shrinkage,
        sentence_aux_weight=args.sentence_aux_weight,
        query_batch_size=args.query_batch_size,
        cache_flush_every=args.cache_flush_every,
        run_id=args.run_id,
    )
    print(f"Pattern benchmark written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
