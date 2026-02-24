"""Generate diverse AI text corpus using Claude, label with GPTZero, retrain surrogate.

The surrogate fails on real AI text because training data was dominated by formal/academic
AI style. This script generates AI text across 20+ writing styles to cover the full
distribution GPTZero sees in production.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import pickle
import random
import sys
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STYLES = [
    ("casual_blog", "Write a casual blog post about {topic}. Use informal language, contractions, short paragraphs."),
    ("personal_essay", "Write a personal essay about {topic}. First person, reflective, emotional, uses 'I' throughout."),
    ("product_review", "Write a product review about {topic}. Opinionated, specific details, star rating style."),
    ("news_article", "Write a short news article about {topic}. AP style, inverted pyramid, quotes from sources."),
    ("technical_doc", "Write technical documentation about {topic}. Code examples, numbered steps, developer audience."),
    ("persuasive_essay", "Write a persuasive essay arguing that {topic}. Strong thesis, evidence, counterarguments."),
    ("short_story", "Write a short creative fiction piece about {topic}. Dialogue, scene setting, character development."),
    ("text_message", "Write a series of text messages between friends discussing {topic}. Very casual, abbreviations, emojis described in words."),
    ("diary_entry", "Write a diary/journal entry about {topic}. Stream of consciousness, personal, raw."),
    ("how_to_guide", "Write a how-to guide about {topic}. Step by step, practical, beginner friendly."),
    ("listicle", "Write a listicle: '7 things about {topic}'. Punchy headers, short explanations."),
    ("academic_abstract", "Write an academic abstract about {topic}. Formal, structured, methodology and findings."),
    ("email", "Write a professional email about {topic}. Business tone, clear action items."),
    ("reddit_post", "Write a Reddit post about {topic}. Casual, uses Reddit conventions, asks for opinions."),
    ("code_explanation", "Explain {topic} as if teaching a junior developer. Mix of prose and pseudocode concepts."),
    ("opinion_piece", "Write an opinion/editorial about {topic}. Strong voice, provocative, argues a position."),
    ("cover_letter", "Write a cover letter mentioning experience with {topic}. Professional but personable."),
    ("tweet_thread", "Write a Twitter/X thread about {topic}. 5-7 tweets, numbered, each under 280 chars feel."),
    ("student_essay", "Write a college student's essay about {topic}. Good but not perfect, has a thesis, some filler."),
    ("fake_personal_anecdote", "Write a personal anecdote about experiencing {topic}. Specific details, sensory language, feels authentic."),
]

TOPICS = [
    "artificial intelligence in healthcare",
    "remote work productivity",
    "climate change adaptation",
    "social media and mental health",
    "electric vehicles",
    "cryptocurrency regulation",
    "machine learning in education",
    "sustainable agriculture",
    "cybersecurity threats",
    "space exploration",
    "urban planning and housing",
    "food delivery apps",
    "renewable energy storage",
    "online privacy",
    "generative AI and copyright",
    "fitness and wearable tech",
    "immigration policy",
    "self-driving cars",
    "streaming services vs cinema",
    "student loan debt",
    "hiking and outdoor recreation",
    "cooking at home vs eating out",
    "video game addiction",
    "working from coffee shops",
    "adopting a rescue dog",
]

_THREAD_STATE = threading.local()


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


def _generate_claude(prompt: str, api_key: str) -> str:
    import anthropic
    client = getattr(_THREAD_STATE, "anthropic_client", None)
    if client is None:
        client = anthropic.Anthropic(api_key=api_key)
        _THREAD_STATE.anthropic_client = client
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _generate_openai(prompt: str, api_key: str) -> str:
    import openai
    client = getattr(_THREAD_STATE, "openai_client", None)
    if client is None:
        client = openai.OpenAI(api_key=api_key)
        _THREAD_STATE.openai_client = client
    resp = client.chat.completions.create(
        model="gpt-5",
        max_completion_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "\n".join(parts).strip()
    return str(content).strip()


def _generate_text(style_prompt: str, topic: str, model: str,
                   anthropic_key: str, openai_key: str) -> str:
    prompt = style_prompt.format(topic=topic) + "\n\nWrite 100-300 words. Return only the text, no commentary."
    if model == "claude":
        return _generate_claude(prompt, anthropic_key)
    else:
        return _generate_openai(prompt, openai_key)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _make_key(style: str, topic: str, model: str) -> str:
    return f"{style}::{topic}::{model}"


def _generate_with_retry(
    style_name: str,
    style_prompt: str,
    topic: str,
    model: str,
    anthropic_key: str,
    openai_key: str,
    max_attempts: int,
) -> Optional[Dict[str, Any]]:
    for attempt in range(max(1, max_attempts)):
        try:
            text = _generate_text(style_prompt, topic, model, anthropic_key, openai_key)
            if len(text.split()) >= 30:
                return {
                    "key": _make_key(style_name, topic, model),
                    "text": text,
                    "style": style_name,
                    "topic": topic,
                    "model": model,
                }
        except Exception:
            pass
        if attempt < max_attempts - 1:
            time.sleep(min(8.0, (2 ** attempt) + random.random()))
    return None


def _label_with_retry(
    text: str,
    delay: float,
    gptzero_api_key: str,
    max_attempts: int,
) -> Optional[Dict[str, Any]]:
    from src.client import GPTZeroClient

    client = getattr(_THREAD_STATE, "gptzero_client", None)
    if client is None:
        client = GPTZeroClient(
            api_key=gptzero_api_key,
            max_retries=4,
            retry_base_delay=0.5,
            retry_max_delay=8.0,
            timeout_seconds=60,
        )
        _THREAD_STATE.gptzero_client = client

    for attempt in range(max(1, max_attempts)):
        resp = client.predict(text, delay=max(0.0, delay))
        if "error" not in resp:
            return resp
        if attempt < max_attempts - 1:
            time.sleep(min(10.0, (2 ** attempt) * 0.5 + random.random()))
    return None


def generate_and_retrain(
    gptzero_api_key: str,
    output_dir: str = "/artifacts/gptzero-audit",
    target_count: int = 500,
    delay: float = 0.3,
    seed: int = 42,
    gen_workers: int = 24,
    label_workers: int = 32,
    checkpoint_every: int = 25,
    generation_attempts: int = 4,
    label_attempts: int = 3,
) -> Dict[str, Any]:
    from src.features import extract_features_batch, FEATURE_NAMES
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

    rng = random.Random(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    anthropic_key = _get_key("ANTHROPIC_API_KEY")
    openai_key = _get_key("OPENAI_API_KEY")

    have_claude = bool(anthropic_key)
    have_openai = bool(openai_key)
    if not have_claude and not have_openai:
        raise RuntimeError("Need at least one of ANTHROPIC_API_KEY or OPENAI_API_KEY")

    models = []
    if have_claude:
        models.append("claude")
    if have_openai:
        models.append("gpt5")
    print(f"Using models: {models}")

    # Generate diverse AI texts, alternating models, with checkpoints for preemption recovery.
    print(f"Generating {target_count} diverse AI texts...")
    generated_path = out / "diverse_ai_generated_raw.jsonl"
    generated: List[Dict[str, Any]] = _load_jsonl(generated_path)
    generated_by_key = {str(r.get("key")): r for r in generated if r.get("key")}

    style_topic_pairs = []
    for style_name, style_prompt in STYLES:
        for topic in TOPICS:
            style_topic_pairs.append((style_name, style_prompt, topic))
    rng.shuffle(style_topic_pairs)

    tasks: List[Tuple[str, str, str, str]] = []
    for i, (style_name, style_prompt, topic) in enumerate(style_topic_pairs[:target_count]):
        model = models[i % len(models)]
        key = _make_key(style_name, topic, model)
        if key in generated_by_key:
            continue
        tasks.append((style_name, style_prompt, topic, model))

    if tasks:
        print(
            f"  Resume checkpoint found: {len(generated_by_key)} pre-generated, "
            f"submitting {len(tasks)} tasks with {max(1, gen_workers)} workers..."
        )
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, gen_workers)) as executor:
            futures = [
                executor.submit(
                    _generate_with_retry,
                    style_name,
                    style_prompt,
                    topic,
                    model,
                    anthropic_key,
                    openai_key,
                    max(1, generation_attempts),
                )
                for (style_name, style_prompt, topic, model) in tasks
            ]
            for fut in concurrent.futures.as_completed(futures):
                completed += 1
                rec = fut.result()
                if rec is not None and rec["key"] not in generated_by_key:
                    generated_by_key[rec["key"]] = rec
                if completed % max(1, checkpoint_every) == 0:
                    generated = list(generated_by_key.values())[:target_count]
                    _write_jsonl(generated_path, generated)
                    print(
                        f"  Generation progress {completed}/{len(tasks)} | "
                        f"saved={len(generated)}"
                    )

    generated = list(generated_by_key.values())[:target_count]
    _write_jsonl(generated_path, generated)

    print(f"Generated {len(generated)} texts across {len(set(g['style'] for g in generated))} styles")

    # Label with GPTZero
    print("\nLabeling with GPTZero API...")
    corpus_path = out / "diverse_ai_corpus.jsonl"
    labeled: List[Dict[str, Any]] = _load_jsonl(corpus_path)
    labeled_by_hash = {
        hashlib.sha256(str(r.get("text", "")).encode()).hexdigest(): r
        for r in labeled
        if r.get("text")
    }
    label_tasks = [g for g in generated if hashlib.sha256(g["text"].encode()).hexdigest() not in labeled_by_hash]
    if label_tasks:
        print(
            f"  Resume checkpoint found: {len(labeled_by_hash)} pre-labeled, "
            f"submitting {len(label_tasks)} texts with {max(1, label_workers)} workers..."
        )
        completed = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, label_workers)) as executor:
            future_map = {
                executor.submit(
                    _label_with_retry,
                    g["text"],
                    delay,
                    gptzero_api_key,
                    max(1, label_attempts),
                ): g
                for g in label_tasks
            }
            for fut in concurrent.futures.as_completed(future_map):
                completed += 1
                g = future_map[fut]
                resp = fut.result()
                if resp is not None:
                    prob = resp.get("average_generated_prob")
                    cls = resp.get("predicted_class")
                    text_hash = hashlib.sha256(g["text"].encode()).hexdigest()
                    if text_hash not in labeled_by_hash:
                        rec = {
                            "base_sample_id": f"diverse_ai_{g['style']}_{g['model']}_{text_hash[:10]}",
                            "stratum": f"ai_{g['style']}",
                            "ground_truth_label": "ai",
                            "source": f"{g['model']}_{g['style']}",
                            "variant_order": 0,
                            "variant_id": "base",
                            "perturbation_kind": "identity",
                            "perturbation_strength": 0.0,
                            "text": g["text"],
                            "gptzero_prob": prob,
                            "gptzero_class": cls,
                            "has_error": False,
                        }
                        labeled_by_hash[text_hash] = rec
                if completed % max(1, checkpoint_every) == 0:
                    labeled = list(labeled_by_hash.values())
                    _write_jsonl(corpus_path, labeled)
                    print(
                        f"  Label progress {completed}/{len(label_tasks)} | "
                        f"saved={len(labeled)}"
                    )

    labeled = list(labeled_by_hash.values())
    _write_jsonl(corpus_path, labeled)

    print(f"Labeled {len(labeled)} texts")

    # Load existing training data
    existing_path = out / "combined_training_data.jsonl"
    existing: List[Dict[str, Any]] = []
    if existing_path.exists():
        with existing_path.open("r") as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("gptzero_prob") is not None and not rec.get("has_error"):
                        existing.append(rec)
    print(f"Existing training records: {len(existing)}")

    # Also load targeted generalization labels if present (news/technical focus).
    generalization_path = out / "generalization_labeled.jsonl"
    generalization_records: List[Dict[str, Any]] = []
    if generalization_path.exists():
        try:
            raw_gen = _load_jsonl(generalization_path)
            by_hash: Dict[str, Dict[str, Any]] = {}
            for row in raw_gen:
                if row.get("has_error"):
                    continue
                if row.get("gptzero_prob") is None or row.get("gptzero_class") is None:
                    continue
                text = str(row.get("text", "")).strip()
                if not text:
                    continue
                h = str(row.get("text_hash") or hashlib.sha256(text.encode()).hexdigest())
                if h in by_hash:
                    continue
                domain = str(row.get("domain", "generalization"))
                source_kind = str(row.get("source_kind", "generalization_sprint"))
                by_hash[h] = {
                    "base_sample_id": f"gen_{domain}_{h[:10]}",
                    "stratum": f"gen_{domain}",
                    "ground_truth_label": None,
                    "source": source_kind,
                    "variant_order": 0,
                    "variant_id": "base",
                    "perturbation_kind": "identity",
                    "perturbation_strength": 0.0,
                    "text": text,
                    "gptzero_prob": row.get("gptzero_prob"),
                    "gptzero_class": row.get("gptzero_class"),
                    "has_error": False,
                }
            generalization_records = list(by_hash.values())
            print(f"Generalization sprint records: {len(generalization_records)}")
        except Exception as exc:
            print(f"  Failed to load generalization_labeled.jsonl: {exc}")

    # Also load sample_docs if labeled
    sample_docs_labeled = out / "sample_docs_labeled.jsonl"
    sample_records: List[Dict[str, Any]] = []
    # Label sample_docs too
    sample_dir = Path(output_dir).parent / "gptzero-audit" / "sample_docs"
    if not sample_dir.exists():
        sample_dir = PROJECT_ROOT / "sample_docs"
    if sample_dir.exists():
        print("\nAlso labeling sample_docs...")
        for cat_dir in sorted(sample_dir.iterdir()):
            if not cat_dir.is_dir():
                continue
            for txt_file in sorted(cat_dir.iterdir()):
                if txt_file.suffix != ".txt":
                    continue
                text = txt_file.read_text().strip()
                if not text:
                    continue
                resp = _label_with_retry(
                    text=text,
                    delay=delay,
                    gptzero_api_key=gptzero_api_key,
                    max_attempts=max(1, label_attempts),
                )
                if resp is None or "error" in resp:
                    continue
                sample_records.append({
                    "base_sample_id": f"sampledoc_{cat_dir.name}_{txt_file.stem}",
                    "stratum": cat_dir.name,
                    "source": "sample_docs",
                    "variant_order": 0,
                    "variant_id": "base",
                    "perturbation_kind": "identity",
                    "perturbation_strength": 0.0,
                    "text": text,
                    "gptzero_prob": resp.get("average_generated_prob"),
                    "gptzero_class": resp.get("predicted_class"),
                    "has_error": False,
                })
        print(f"Labeled {len(sample_records)} sample_docs texts")

    all_records = existing + labeled + generalization_records + sample_records
    print(f"\nTotal training records: {len(all_records)}")

    # Extract features and train
    texts = [r["text"] for r in all_records]
    probs = np.array([float(r["gptzero_prob"]) for r in all_records])
    classes = np.array([str(r.get("gptzero_class", "human")) for r in all_records])

    print("Extracting features...")
    X, _ = extract_features_batch(texts)

    print("Training probability regressor (3000 trees)...")
    reg = GradientBoostingRegressor(
        n_estimators=3000, max_depth=10, learning_rate=0.01,
        subsample=0.8, min_samples_leaf=2, random_state=42,
    )
    reg.fit(X, probs)

    print("Training class predictor (1000 trees)...")
    cls_model = GradientBoostingClassifier(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    cls_model.fit(X, classes)

    # Evaluate on the NEW data only (diverse + generalization + sample_docs)
    eval_records = labeled + generalization_records + sample_records
    eval_texts = [r["text"] for r in eval_records]
    eval_true_probs = np.array([float(r["gptzero_prob"]) for r in eval_records])
    eval_true_classes = np.array([str(r["gptzero_class"]) for r in eval_records])

    X_eval, _ = extract_features_batch(eval_texts)
    pred_probs = np.clip(reg.predict(X_eval), 0.0, 1.0)
    pred_classes = cls_model.predict(X_eval)

    mae = float(np.mean(np.abs(pred_probs - eval_true_probs)))
    agreement = float(np.mean(pred_classes == eval_true_classes))

    print(f"\nNew data evaluation:")
    print(f"  MAE: {mae:.4f}")
    print(f"  Class agreement: {agreement:.4f}")

    # Save
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
    print(f"Model saved to {model_path}")

    try:
        from modal import Volume
        Volume.from_name("gptzero-audit-artifacts").commit()
    except Exception:
        pass

    return {
        "generated": len(labeled),
        "sample_docs": len(sample_records),
        "total_training": len(all_records),
        "new_data_mae": round(mae, 4),
        "new_data_agreement": round(agreement, 4),
        "generation_workers": max(1, gen_workers),
        "label_workers": max(1, label_workers),
    }
