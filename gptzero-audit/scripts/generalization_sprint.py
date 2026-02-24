"""Targeted generalization sprint for weak real-world strata.

This pipeline focuses on domains where the surrogate underperformed in
real-world validation (notably news and technical casual writing):
  1) Build a domain-focused candidate pool (human corpus + frontier LLM outputs)
  2) Label candidates against live GPTZero in parallel with resumable checkpoints
  3) Compute disagreement against the current surrogate
  4) Retrain with weighted adaptation + holdout evaluation

Designed for fast iteration on Modal under time pressure.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pickle
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_THREAD_STATE = threading.local()


NEWS_TOPICS = [
    "global trade agreements",
    "energy market volatility",
    "public health policy changes",
    "technology regulation proposals",
    "transportation infrastructure funding",
    "education reform initiatives",
    "housing affordability trends",
    "workforce automation impacts",
    "cybersecurity incident response",
    "climate adaptation planning",
]

TECH_TOPICS = [
    "debugging flaky CI tests",
    "database migration rollback incident",
    "service latency regression after deploy",
    "feature flag rollout strategy",
    "refactoring a legacy API",
    "monitoring alert fatigue",
    "on-call incident handoff",
    "caching strategy tradeoffs",
    "async job reliability improvements",
    "schema evolution in production",
]

LLM_STYLES = [
    (
        "ai_news",
        "Write a neutral, realistic news article about {topic}. "
        "Use journalistic tone and concrete details. 120-280 words.",
    ),
    (
        "ai_technical_casual",
        "Write a candid first-person developer forum post about {topic}. "
        "Use casual engineering language, concrete debugging details, and uneven sentence lengths. "
        "120-260 words.",
    ),
    (
        "ai_productive_memo",
        "Write an internal team memo about {topic} with practical recommendations. "
        "Natural but professional voice. 120-240 words.",
    ),
]


def _get_key(name: str) -> str:
    key = os.environ.get(name)
    if key:
        return key
    for env_path in [PROJECT_ROOT.parent / ".env", PROJECT_ROOT / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip()
    return ""


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _openai_client(api_key: str):
    import openai

    client = getattr(_THREAD_STATE, "openai_client", None)
    if client is None:
        client = openai.OpenAI(api_key=api_key)
        _THREAD_STATE.openai_client = client
    return client


def _anthropic_client(api_key: str):
    import anthropic

    client = getattr(_THREAD_STATE, "anthropic_client", None)
    if client is None:
        client = anthropic.Anthropic(api_key=api_key)
        _THREAD_STATE.anthropic_client = client
    return client


def _gptzero_client(api_key: str):
    from src.client import GPTZeroClient

    client = getattr(_THREAD_STATE, "gptzero_client", None)
    if client is None:
        client = GPTZeroClient(
            api_key=api_key,
            max_retries=4,
            retry_base_delay=0.5,
            retry_max_delay=10.0,
            timeout_seconds=60,
        )
        _THREAD_STATE.gptzero_client = client
    return client


def _generate_openai(prompt: str, api_key: str) -> str:
    client = _openai_client(api_key)
    resp = client.chat.completions.create(
        model="gpt-5",
        max_completion_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        out = []
        for item in content:
            if isinstance(item, dict):
                out.append(str(item.get("text", "")))
        return "\n".join(out).strip()
    return str(content).strip()


def _generate_claude(prompt: str, api_key: str) -> str:
    client = _anthropic_client(api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _generate_llm_text(
    *,
    style_name: str,
    style_prompt: str,
    topic: str,
    model_name: str,
    anthropic_key: str,
    openai_key: str,
    max_attempts: int,
) -> Optional[Dict[str, Any]]:
    prompt = (
        style_prompt.format(topic=topic)
        + "\nReturn only the text body. No markdown, no title, no commentary."
    )
    for attempt in range(max(1, max_attempts)):
        try:
            if model_name == "claude":
                text = _generate_claude(prompt, anthropic_key)
            else:
                text = _generate_openai(prompt, openai_key)
            text = _normalize_text(text)
            wc = len(text.split())
            if 70 <= wc <= 420:
                return {
                    "text": text,
                    "domain": style_name,
                    "source_kind": f"{model_name}_{style_name}",
                    "candidate_id": f"{model_name}_{style_name}_{_text_hash(text)[:12]}",
                }
        except Exception:
            pass
        if attempt < max_attempts - 1:
            time.sleep(min(8.0, (2 ** attempt) + random.random()))
    return None


def _load_human_news(target_count: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    rng = random.Random(seed)
    out: List[Dict[str, Any]] = []
    try:
        ds = load_dataset("ag_news", split="train[:20000]")
        pool = []
        for row in ds:
            text = _normalize_text(str(row.get("text", "")))
            wc = len(text.split())
            if 55 <= wc <= 320:
                pool.append(text)
        rng.shuffle(pool)
        for text in pool[:target_count]:
            out.append(
                {
                    "text": text,
                    "domain": "human_news",
                    "source_kind": "ag_news",
                    "candidate_id": f"ag_news_{_text_hash(text)[:12]}",
                }
            )
    except Exception as exc:
        print(f"[warn] failed to load ag_news: {exc}")
    return out


def _load_human_reviews(target_count: int, seed: int) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    rng = random.Random(seed + 7)
    out: List[Dict[str, Any]] = []
    try:
        ds = load_dataset("imdb", split="train[:15000]")
        pool = []
        for row in ds:
            text = _normalize_text(str(row.get("text", "")).replace("<br />", " ").replace("<br/>", " "))
            wc = len(text.split())
            if 80 <= wc <= 380:
                pool.append(text)
        rng.shuffle(pool)
        for text in pool[:target_count]:
            out.append(
                {
                    "text": text,
                    "domain": "human_review",
                    "source_kind": "imdb",
                    "candidate_id": f"imdb_{_text_hash(text)[:12]}",
                }
            )
    except Exception as exc:
        print(f"[warn] failed to load imdb: {exc}")
    return out


def _load_sample_docs() -> List[Dict[str, Any]]:
    sample_dir = PROJECT_ROOT / "sample_docs"
    out: List[Dict[str, Any]] = []
    if not sample_dir.exists():
        return out
    for cat in sorted(sample_dir.iterdir()):
        if not cat.is_dir():
            continue
        for txt in sorted(cat.glob("*.txt")):
            text = _normalize_text(txt.read_text(encoding="utf-8"))
            if not text:
                continue
            out.append(
                {
                    "text": text,
                    "domain": f"sample_{cat.name}",
                    "source_kind": "sample_docs",
                    "candidate_id": f"sample_{cat.name}_{txt.stem}",
                }
            )
    return out


def _label_text_with_retry(
    *,
    text: str,
    gptzero_api_key: str,
    delay: float,
    max_attempts: int,
) -> Optional[Dict[str, Any]]:
    client = _gptzero_client(gptzero_api_key)
    for attempt in range(max(1, max_attempts)):
        resp = client.predict(text, delay=max(0.0, delay))
        if "error" not in resp:
            return resp
        if attempt < max_attempts - 1:
            time.sleep(min(10.0, (2 ** attempt) * 0.5 + random.random()))
    return None


def _predict_baseline(
    model_path: Path,
    records: List[Dict[str, Any]],
) -> None:
    from src.features import extract_features_batch

    if not model_path.exists():
        return
    with model_path.open("rb") as f:
        model_data = pickle.load(f)
    reg = model_data["model"]
    cls = model_data.get("class_model")
    texts = [r["text"] for r in records]
    X, _ = extract_features_batch(texts)
    probs = np.clip(reg.predict(X), 0.0, 1.0)
    if cls is not None:
        classes = cls.predict(X)
    else:
        classes = np.array(
            ["ai" if p >= 0.65 else ("human" if p <= 0.35 else "mixed") for p in probs],
            dtype=object,
        )
    for i, rec in enumerate(records):
        rec["baseline_prob"] = float(probs[i])
        rec["baseline_class"] = str(classes[i])


def _domain_weight(domain: str) -> float:
    if "news" in domain:
        return 3.0
    if "technical" in domain or "sample_adversarial_ai" in domain:
        return 3.5
    if domain.startswith("sample_"):
        return 2.5
    return 1.4


def _compute_weight(rec: Dict[str, Any]) -> float:
    w = _domain_weight(str(rec.get("domain", "")))
    gpt_prob = rec.get("gptzero_prob")
    base_prob = rec.get("baseline_prob")
    gpt_cls = rec.get("gptzero_class")
    base_cls = rec.get("baseline_class")

    if gpt_prob is not None and base_prob is not None:
        w += 8.0 * abs(float(gpt_prob) - float(base_prob))
    if gpt_cls is not None and base_cls is not None and str(gpt_cls) != str(base_cls):
        w += 4.0
    return float(min(18.0, max(1.0, w)))


def _load_training_records(path: Path) -> List[Dict[str, Any]]:
    rows = _load_jsonl(path)
    out = []
    for row in rows:
        if row.get("has_error"):
            continue
        if row.get("gptzero_prob") is None:
            continue
        if not row.get("text"):
            continue
        out.append(row)
    return out


def run(
    gptzero_api_key: str,
    output_dir: str = "/artifacts/gptzero-audit",
    seed: int = 42,
    news_target: int = 600,
    review_target: int = 300,
    llm_target: int = 800,
    gen_workers: int = 48,
    label_workers: int = 48,
    checkpoint_every: int = 25,
    delay: float = 0.10,
    generation_attempts: int = 4,
    label_attempts: int = 4,
    holdout_ratio: float = 0.2,
    reg_estimators: int = 3400,
    cls_estimators: int = 1200,
    reg_max_depth: int = 10,
    cls_max_depth: int = 6,
    reg_learning_rate: float = 0.01,
    cls_learning_rate: float = 0.045,
) -> Dict[str, Any]:
    from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
    from src.features import FEATURE_NAMES, extract_features_batch

    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    anthropic_key = _get_key("ANTHROPIC_API_KEY")
    openai_key = _get_key("OPENAI_API_KEY")
    model_names: List[str] = []
    if anthropic_key:
        model_names.append("claude")
    if openai_key:
        model_names.append("gpt5")
    if not model_names:
        raise RuntimeError("Need at least one of ANTHROPIC_API_KEY or OPENAI_API_KEY")

    print("=== Generalization Sprint ===")
    print(f"models={model_names} seed={seed}")

    # 1) Candidate assembly
    candidates_path = out / "generalization_candidates.jsonl"
    generated_llm_path = out / "generalization_llm_generated.jsonl"
    labeled_path = out / "generalization_labeled.jsonl"

    print("[1/5] Loading human-domain corpora...")
    candidates: List[Dict[str, Any]] = []
    candidates.extend(_load_human_news(news_target, seed))
    candidates.extend(_load_human_reviews(review_target, seed))
    candidates.extend(_load_sample_docs())

    print(
        f"  base candidates: total={len(candidates)} "
        f"(human_news={sum(1 for r in candidates if r['domain']=='human_news')}, "
        f"human_review={sum(1 for r in candidates if r['domain']=='human_review')}, "
        f"sample_docs={sum(1 for r in candidates if r['source_kind']=='sample_docs')})"
    )

    # 2) Frontier LLM generation (resumable)
    existing_llm = _load_jsonl(generated_llm_path)
    llm_by_hash = {_text_hash(str(r.get("text", ""))): r for r in existing_llm if r.get("text")}

    style_tasks: List[Tuple[str, str, str, str]] = []
    for i in range(llm_target):
        style_name, style_prompt = LLM_STYLES[i % len(LLM_STYLES)]
        topic_pool = NEWS_TOPICS if "news" in style_name else TECH_TOPICS
        topic = topic_pool[i % len(topic_pool)]
        model_name = model_names[i % len(model_names)]
        style_tasks.append((style_name, style_prompt, topic, model_name))
    rng.shuffle(style_tasks)

    print(f"[2/5] Generating LLM candidates: target={llm_target}, workers={max(1, gen_workers)}")
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, gen_workers)) as executor:
        futures = [
            executor.submit(
                _generate_llm_text,
                style_name=style_name,
                style_prompt=style_prompt,
                topic=topic,
                model_name=model_name,
                anthropic_key=anthropic_key,
                openai_key=openai_key,
                max_attempts=max(1, generation_attempts),
            )
            for (style_name, style_prompt, topic, model_name) in style_tasks
        ]
        for fut in concurrent.futures.as_completed(futures):
            completed += 1
            rec = fut.result()
            if rec is not None:
                h = _text_hash(rec["text"])
                if h not in llm_by_hash:
                    llm_by_hash[h] = rec
            if completed % max(1, checkpoint_every) == 0:
                _write_jsonl(generated_llm_path, list(llm_by_hash.values()))
                print(f"  generated progress: {completed}/{len(style_tasks)} saved={len(llm_by_hash)}")

    llm_records = list(llm_by_hash.values())
    _write_jsonl(generated_llm_path, llm_records)
    print(f"  llm unique records: {len(llm_records)}")
    candidates.extend(llm_records[:llm_target])

    # Dedupe candidate set
    uniq: Dict[str, Dict[str, Any]] = {}
    for rec in candidates:
        text = _normalize_text(str(rec.get("text", "")))
        if not text:
            continue
        rec["text"] = text
        h = _text_hash(text)
        rec["text_hash"] = h
        uniq[h] = rec
    candidates = list(uniq.values())
    _write_jsonl(candidates_path, candidates)
    print(f"  final candidate pool: {len(candidates)}")

    # 3) Baseline scoring + GPTZero labeling with resume
    current_model_path = out / "ppl_surrogate" / "ppl_surrogate.pkl"
    _predict_baseline(current_model_path, candidates)

    existing_labeled = _load_jsonl(labeled_path)
    labeled_by_hash = {str(r.get("text_hash", "")): r for r in existing_labeled if r.get("text_hash")}
    to_label = [r for r in candidates if r["text_hash"] not in labeled_by_hash]

    print(
        f"[3/5] Labeling with GPTZero: existing={len(labeled_by_hash)} "
        f"pending={len(to_label)} workers={max(1, label_workers)}"
    )

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, label_workers)) as executor:
        future_map = {
            executor.submit(
                _label_text_with_retry,
                text=rec["text"],
                gptzero_api_key=gptzero_api_key,
                delay=delay,
                max_attempts=max(1, label_attempts),
            ): rec
            for rec in to_label
        }
        for fut in concurrent.futures.as_completed(future_map):
            completed += 1
            rec = dict(future_map[fut])
            resp = fut.result()
            if resp is not None:
                rec["gptzero_prob"] = float(resp.get("average_generated_prob", 0.0))
                rec["gptzero_class"] = str(resp.get("predicted_class", "unknown"))
                rec["has_error"] = False
                labeled_by_hash[rec["text_hash"]] = rec
            if completed % max(1, checkpoint_every) == 0:
                _write_jsonl(labeled_path, list(labeled_by_hash.values()))
                print(f"  labeling progress: {completed}/{len(to_label)} saved={len(labeled_by_hash)}")

    labeled_rows = list(labeled_by_hash.values())
    _write_jsonl(labeled_path, labeled_rows)
    labeled_valid = [r for r in labeled_rows if not r.get("has_error") and r.get("gptzero_prob") is not None]
    print(f"  labeled valid rows: {len(labeled_valid)}")

    # Persist labeled checkpoints before training (paid API queries).
    try:
        from modal import Volume

        Volume.from_name("gptzero-audit-artifacts").commit()
        print("  committed labeled checkpoints to volume")
    except Exception:
        pass

    # 4) Train adaptation model with weighted rows + domain holdout
    print("[4/5] Training weighted adaptation model...")
    existing_train = _load_training_records(out / "combined_training_data.jsonl")
    existing_diverse = _load_training_records(out / "diverse_ai_corpus.jsonl")

    # Domain-stratified holdout from adaptation rows
    by_domain: Dict[str, List[Dict[str, Any]]] = {}
    for row in labeled_valid:
        by_domain.setdefault(str(row.get("domain", "unknown")), []).append(row)
    holdout_hashes = set()
    for domain, rows in by_domain.items():
        local = list(rows)
        rng.shuffle(local)
        k = int(round(len(local) * max(0.0, min(0.5, holdout_ratio))))
        if len(local) >= 3:
            k = max(1, k)
        for rec in local[:k]:
            holdout_hashes.add(rec["text_hash"])

    adapt_train = [r for r in labeled_valid if r["text_hash"] not in holdout_hashes]
    adapt_holdout = [r for r in labeled_valid if r["text_hash"] in holdout_hashes]

    train_records: List[Dict[str, Any]] = []
    train_weights: List[float] = []

    for row in existing_train:
        train_records.append(row)
        train_weights.append(1.0)
    for row in existing_diverse:
        train_records.append(row)
        train_weights.append(1.2)

    for row in adapt_train:
        train_records.append(
            {
                "base_sample_id": f"gen_{row['domain']}_{row['text_hash'][:10]}",
                "stratum": str(row.get("domain", "generalization")),
                "ground_truth_label": None,
                "source": str(row.get("source_kind", "generalization_sprint")),
                "variant_order": 0,
                "variant_id": "base",
                "perturbation_kind": "identity",
                "perturbation_strength": 0.0,
                "text": row["text"],
                "gptzero_prob": row["gptzero_prob"],
                "gptzero_class": row["gptzero_class"],
                "has_error": False,
            }
        )
        train_weights.append(_compute_weight(row))

    print(
        f"  train records={len(train_records)} "
        f"(existing={len(existing_train)}, diverse={len(existing_diverse)}, adapt={len(adapt_train)})"
    )

    texts = [r["text"] for r in train_records]
    probs = np.array([float(r["gptzero_prob"]) for r in train_records], dtype=float)
    classes = np.array([str(r.get("gptzero_class", "human")) for r in train_records], dtype=object)
    weights_np = np.array(train_weights, dtype=float)

    X, _ = extract_features_batch(texts)

    print(
        "  training config: "
        f"reg(n_estimators={reg_estimators}, depth={reg_max_depth}, lr={reg_learning_rate}) "
        f"cls(n_estimators={cls_estimators}, depth={cls_max_depth}, lr={cls_learning_rate})"
    )

    reg = GradientBoostingRegressor(
        n_estimators=max(200, int(reg_estimators)),
        max_depth=max(2, int(reg_max_depth)),
        learning_rate=max(0.001, float(reg_learning_rate)),
        subsample=0.85,
        min_samples_leaf=2,
        random_state=seed,
    )
    reg.fit(X, probs, sample_weight=weights_np)

    cls_model = GradientBoostingClassifier(
        n_estimators=max(100, int(cls_estimators)),
        max_depth=max(2, int(cls_max_depth)),
        learning_rate=max(0.001, float(cls_learning_rate)),
        subsample=0.85,
        min_samples_leaf=4,
        random_state=seed,
    )
    valid_cls = np.array([c in {"ai", "human", "mixed"} for c in classes])
    cls_model.fit(X[valid_cls], classes[valid_cls], sample_weight=weights_np[valid_cls])

    # Save model
    model_dir = out / "ppl_surrogate"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "ppl_surrogate.pkl"
    with model_path.open("wb") as f:
        pickle.dump(
            {
                "model": reg,
                "class_model": cls_model,
                "gpt2_model": "gpt2",
                "feature_names": list(FEATURE_NAMES),
                "use_text_features": True,
                "skip_perplexity": True,
            },
            f,
        )

    # 5) Evaluation on adaptation holdout and full adaptation set
    print("[5/5] Evaluating adaptation holdout...")

    def _eval(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {"count": 0, "class_agreement": 0.0, "mae": 0.0, "by_domain": {}}
        X_eval, _ = extract_features_batch([r["text"] for r in rows])
        p_pred = np.clip(reg.predict(X_eval), 0.0, 1.0)
        c_pred = cls_model.predict(X_eval)
        p_true = np.array([float(r["gptzero_prob"]) for r in rows], dtype=float)
        c_true = np.array([str(r["gptzero_class"]) for r in rows], dtype=object)
        mae = float(np.mean(np.abs(p_pred - p_true)))
        agr = float(np.mean(c_pred == c_true))
        domain_summary: Dict[str, Dict[str, Any]] = {}
        for i, row in enumerate(rows):
            d = str(row.get("domain", "unknown"))
            domain_summary.setdefault(d, {"count": 0, "matches": 0, "abs_err": []})
            domain_summary[d]["count"] += 1
            if str(c_pred[i]) == str(c_true[i]):
                domain_summary[d]["matches"] += 1
            domain_summary[d]["abs_err"].append(abs(float(p_pred[i]) - float(p_true[i])))
        by_domain = {
            d: {
                "count": stats["count"],
                "class_agreement": round(stats["matches"] / max(1, stats["count"]), 4),
                "mae": round(float(np.mean(stats["abs_err"])), 4) if stats["abs_err"] else 0.0,
            }
            for d, stats in sorted(domain_summary.items())
        }
        return {
            "count": len(rows),
            "class_agreement": round(agr, 4),
            "mae": round(mae, 4),
            "by_domain": by_domain,
        }

    full_eval = _eval(labeled_valid)
    holdout_eval = _eval(adapt_holdout)

    summary = {
        "seed": seed,
        "models": model_names,
        "candidate_count": len(candidates),
        "labeled_valid_count": len(labeled_valid),
        "adapt_train_count": len(adapt_train),
        "adapt_holdout_count": len(adapt_holdout),
        "full_adaptation_eval": full_eval,
        "holdout_eval": holdout_eval,
        "saved_model": str(model_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    summary_path = out / "generalization_sprint_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    try:
        from modal import Volume

        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    return summary
