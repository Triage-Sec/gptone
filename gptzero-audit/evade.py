#!/usr/bin/env python3
"""GPTZero evasion tool.

Uses the extracted surrogate model to score, diagnose, and automatically
rewrite text to evade GPTZero's AI detection. Runs entirely offline.

Usage:
  python evade.py score "Your text here"
  python evade.py diagnose "Your text here"
  python evade.py rewrite "Your text here"
  echo "text" | python evade.py score --stdin
"""

from __future__ import annotations

import argparse
import difflib
import json
import pickle
import random
import re
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features import (
    FEATURE_NAMES,
    HEDGING_PHRASES,
    TRANSITION_WORDS,
    extract_features,
    extract_features_batch,
)
from src.boundary_mining import (
    contractionize,
    casualize,
    humanize_variants,
    normalize_text,
    split_sentences,
    join_sentences,
)

DEFAULT_MODEL_DIR = str(PROJECT_ROOT / "models" / "ppl_surrogate")

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _supports_color():
        return text
    return f"{code}{text}{_RESET}"


def _prob_bar(prob: float, width: int = 30) -> str:
    filled = int(prob * width)
    if prob >= 0.65:
        color = _RED
    elif prob >= 0.35:
        color = _YELLOW
    else:
        color = _GREEN
    bar = _c(color, "\u2588" * filled) + _c(_DIM, "\u2591" * (width - filled))
    return bar


def _verdict(cls: str) -> str:
    if cls == "ai":
        return _c(_RED, "DETECTED AS AI")
    if cls == "human":
        return _c(_GREEN, "PASSES AS HUMAN")
    return _c(_YELLOW, "AMBIGUOUS (MIXED)")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Surrogate:
    """Wrapper around the trained GBM surrogate."""

    def __init__(self, model_dir: str):
        model_path = Path(model_dir) / "ppl_surrogate.pkl"
        if not model_path.exists():
            print(f"{_c(_RED, 'Error:')} Model not found at {model_path}")
            print(f"Download the trained model from Modal first:")
            print(f"  python -c \"import modal_app; ...\"  # see README")
            sys.exit(1)
        with model_path.open("rb") as fh:
            data = pickle.load(fh)
        self.reg_model = data["model"]
        self.cls_model = data.get("class_model")
        self.feature_names = data.get("feature_names", FEATURE_NAMES)

    def predict_prob(self, text: str) -> float:
        vec, _ = extract_features(text)
        return float(np.clip(self.reg_model.predict(vec.reshape(1, -1))[0], 0.0, 1.0))

    def predict_prob_batch(self, texts: Sequence[str]) -> np.ndarray:
        X, _ = extract_features_batch(list(texts))
        return np.clip(self.reg_model.predict(X), 0.0, 1.0)

    def predict_class(self, text: str) -> str:
        if self.cls_model is not None:
            vec, _ = extract_features(text)
            return str(self.cls_model.predict(vec.reshape(1, -1))[0])
        p = self.predict_prob(text)
        if p >= 0.65:
            return "ai"
        if p <= 0.35:
            return "human"
        return "mixed"

    def feature_importances(self) -> Dict[str, float]:
        return dict(zip(self.feature_names, self.reg_model.feature_importances_.tolist()))

    def feature_vector(self, text: str) -> Tuple[np.ndarray, List[str]]:
        return extract_features(text)


# ---------------------------------------------------------------------------
# Score
# ---------------------------------------------------------------------------

def cmd_score(text: str, surrogate: Surrogate, as_json: bool = False) -> Dict[str, Any]:
    prob = surrogate.predict_prob(text)
    cls = surrogate.predict_class(text)
    wc = len(text.split())

    result = {"class": cls, "probability": round(prob, 6), "word_count": wc}

    if as_json:
        print(json.dumps(result, indent=2))
        return result

    print()
    print(f"  {_c(_BOLD, 'GPTZero Prediction')}  (surrogate)")
    print(f"  {'=' * 44}")
    print(f"  AI Probability   {prob:.4f}  {_prob_bar(prob)}")
    print(f"  Class            {cls}")
    print(f"  Words            {wc}")
    print(f"  Verdict          {_verdict(cls)}")
    print()
    return result


# ---------------------------------------------------------------------------
# Diagnose
# ---------------------------------------------------------------------------

_FEATURE_GUIDANCE: Dict[str, Tuple[str, str, str]] = {
    #  feature_name:  (direction, short_label, suggestion)
    "unique_word_ratio":           ("low",  "Vocabulary diversity",   "Use more varied word choices. Avoid reusing the same terms."),
    "contraction_count":           ("low",  "Contractions",          "Add contractions: don't, can't, it's, won't, I've, etc."),
    "hedging_score":               ("high", "Hedging phrases",       "Remove phrases like 'it is important to note', 'furthermore'."),
    "transition_word_score":       ("high", "Mechanical transitions","Replace 'moreover', 'consequently', 'therefore' with casual connectors."),
    "first_person_count":          ("low",  "First-person voice",    "Write in first person: 'I think', 'in my experience', 'we found'."),
    "passive_voice_ratio":         ("high", "Passive voice",         "Rewrite passive constructions in active voice."),
    "bigram_repeat_ratio":         ("high", "Bigram repetition",     "Vary phrasing. Don't repeat two-word patterns."),
    "trigram_repeat_ratio":        ("high", "Trigram repetition",    "Rephrase repeated three-word constructions."),
    "adjacent_sentence_similarity":("high", "Sentence similarity",   "Make consecutive sentences more distinct in topic and vocabulary."),
    "sentence_len_cv":             ("low",  "Sentence length variety","Mix short punchy sentences with longer ones."),
    "question_mark_count":         ("low",  "Questions",             "Add a rhetorical or genuine question."),
    "exclamation_count":           ("low",  "Exclamations",          "Add natural emphasis with exclamation marks where appropriate."),
    "starts_with_the_ratio":       ("high", "'The' sentence starts", "Vary sentence openers. Not every sentence should start with 'The'."),
    "starts_with_transition_ratio":("high", "Transition openers",    "Don't open sentences with 'Furthermore', 'Moreover', etc."),
    "repeated_sentence_starts":    ("high", "Repetitive openers",    "Use different first words for each sentence."),
    "avg_word_len":                ("high", "Word length",           "Use simpler, shorter words where possible."),
    "pronoun_density":             ("low",  "Pronoun usage",         "Add personal pronouns: he, she, they, I, we."),
    "windowed_ttr":                ("low",  "Type-token ratio",      "Your vocabulary is too narrow. Introduce synonym variety."),
    "determiner_density":          ("high", "Determiner density",    "Reduce overuse of 'the', 'a', 'this'. Typical of formulaic writing."),
}

_HUMAN_BASELINES: Dict[str, float] = {
    "unique_word_ratio": 0.70, "contraction_count": 3.0, "hedging_score": 0.05,
    "transition_word_score": 0.10, "first_person_count": 0.03, "passive_voice_ratio": 0.03,
    "bigram_repeat_ratio": 0.08, "trigram_repeat_ratio": 0.02,
    "adjacent_sentence_similarity": 0.10, "sentence_len_cv": 0.45,
    "question_mark_count": 1.0, "exclamation_count": 0.5,
    "starts_with_the_ratio": 0.10, "starts_with_transition_ratio": 0.05,
    "repeated_sentence_starts": 0.25, "avg_word_len": 4.5,
    "pronoun_density": 0.06, "determiner_density": 0.08, "windowed_ttr": 0.75,
}


def _feature_gap(name: str, value: float) -> float:
    if name not in _FEATURE_GUIDANCE:
        return 0.0
    direction = _FEATURE_GUIDANCE[name][0]
    baseline = _HUMAN_BASELINES.get(name, value)
    if direction == "high":
        return max(0.0, value - baseline)
    return max(0.0, baseline - value)


def cmd_diagnose(text: str, surrogate: Surrogate, as_json: bool = False) -> Dict[str, Any]:
    prob = surrogate.predict_prob(text)
    cls = surrogate.predict_class(text)
    vec, _ = surrogate.feature_vector(text)
    importances = surrogate.feature_importances()

    triggers: List[Dict[str, Any]] = []
    for i, name in enumerate(FEATURE_NAMES):
        if name not in _FEATURE_GUIDANCE:
            continue
        gap = _feature_gap(name, vec[i])
        imp = importances.get(name, 0.0)
        score = gap * imp
        if score > 0.0001:
            direction, label, suggestion = _FEATURE_GUIDANCE[name]
            triggers.append({
                "feature": name,
                "label": label,
                "value": round(float(vec[i]), 4),
                "baseline": round(_HUMAN_BASELINES.get(name, 0.0), 4),
                "importance": round(imp, 4),
                "impact": round(score, 6),
                "suggestion": suggestion,
            })
    triggers.sort(key=lambda t: -t["impact"])

    sentences = split_sentences(text)
    sent_scores: List[Dict[str, Any]] = []
    if sentences:
        probs = surrogate.predict_prob_batch(sentences)
        for i, (s, sp) in enumerate(zip(sentences, probs)):
            sent_scores.append({
                "index": i,
                "text": s,
                "ai_prob": round(float(sp), 4),
            })
        sent_scores.sort(key=lambda s: -s["ai_prob"])

    result = {
        "class": cls,
        "probability": round(prob, 6),
        "triggers": triggers[:10],
        "flagged_sentences": [
            {**s, "text": s["text"][:150]} for s in sent_scores[:5]
        ],
    }

    if as_json:
        print(json.dumps(result, indent=2))
        return result

    print()
    print(f"  {_c(_BOLD, 'Diagnosis')}  AI Prob: {prob:.4f}  {_prob_bar(prob)}  {_verdict(cls)}")
    print()

    if triggers:
        print(f"  {_c(_BOLD, 'Detection Triggers')}  (features pushing the score toward AI)")
        print(f"  {'-' * 70}")
        for t in triggers[:8]:
            impact_bar = _c(_RED, "\u25a0") * min(10, max(1, int(t["impact"] * 500)))
            print(f"  {impact_bar}  {_c(_CYAN, t['label'])}")
            print(f"       Value: {t['value']:.3f}  |  Human baseline: {t['baseline']:.3f}")
            print(f"       {_c(_DIM, 'Fix:')} {t['suggestion']}")
            print()

    if sent_scores:
        print(f"  {_c(_BOLD, 'Flagged Sentences')}  (ranked by individual AI probability)")
        print(f"  {'-' * 70}")
        for s in sent_scores[:5]:
            p = s["ai_prob"]
            if p >= 0.6:
                tag = _c(_RED, f"[{p:.2f}]")
            elif p >= 0.4:
                tag = _c(_YELLOW, f"[{p:.2f}]")
            else:
                tag = _c(_GREEN, f"[{p:.2f}]")
            wrapped = textwrap.shorten(s["text"], width=80, placeholder="...")
            print(f"  {tag}  {wrapped}")
        print()

    return result


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------

_HEDGING_PATTERNS = [re.compile(re.escape(p), re.IGNORECASE) for p in HEDGING_PHRASES]
_TRANSITION_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in TRANSITION_WORDS) + r")\b,?\s*",
    re.IGNORECASE,
)
_FIRST_PERSON_INSERTS = [
    "I think ", "In my experience, ", "From what I've seen, ",
    "Honestly, ", "I'd say ", "What I noticed is ",
    "I believe ", "The way I see it, ",
]
_QUESTIONS = [
    "But does that actually hold up?",
    "Is that really the case though?",
    "What does this mean in practice?",
    "Why does that matter?",
    "Sound familiar?",
    "Makes you think, right?",
    "But here's the thing -- does it work?",
]


def _strip_hedging(text: str) -> str:
    sents = split_sentences(text)
    cleaned = []
    for s in sents:
        out = s
        for pat in _HEDGING_PATTERNS:
            out = pat.sub("", out)
        out = re.sub(r"\s+", " ", out).strip()
        out = re.sub(r"^[,;:\s]+", "", out).strip()
        if out and out[0].islower():
            out = out[0].upper() + out[1:]
        if out:
            cleaned.append(out)
    return join_sentences(cleaned) if cleaned else text


def _strip_transitions(text: str) -> str:
    sents = split_sentences(text)
    rewritten = []
    for s in sents:
        cleaned = _TRANSITION_RE.sub("", s, count=1).strip()
        cleaned = re.sub(r"^[,;:\s]+", "", cleaned).strip()
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]
        rewritten.append(cleaned if cleaned else s)
    return join_sentences(rewritten)


def _inject_first_person(text: str, rng: random.Random) -> str:
    sents = split_sentences(text)
    if not sents:
        return text
    idx = rng.randrange(len(sents))
    prefix = rng.choice(_FIRST_PERSON_INSERTS)
    s = sents[idx]
    sents[idx] = prefix + (s[0].lower() + s[1:] if len(s) > 1 else s)
    return join_sentences(sents)


def _vary_sentence_lengths(text: str, rng: random.Random) -> str:
    sents = split_sentences(text)
    if len(sents) < 3:
        return text
    longest_idx = max(range(len(sents)), key=lambda i: len(sents[i].split()))
    words = sents[longest_idx].split()
    if len(words) > 12:
        mid = len(words) // 2
        part1 = " ".join(words[:mid]).rstrip(".,;:") + "."
        part2 = " ".join(words[mid:])
        if part2 and part2[0].islower():
            part2 = part2[0].upper() + part2[1:]
        sents[longest_idx] = part1
        sents.insert(longest_idx + 1, part2)
    return join_sentences(sents)


def _add_question(text: str, rng: random.Random) -> str:
    sents = split_sentences(text)
    if not sents:
        return text
    idx = rng.randint(1, max(1, len(sents) - 1))
    sents.insert(idx, rng.choice(_QUESTIONS))
    return join_sentences(sents)


_MUTATIONS: List[Tuple[str, Any]] = [
    ("Add contractions",       lambda t, r: contractionize(t)),
    ("Casualize vocabulary",   lambda t, r: casualize(t)),
    ("Remove hedging phrases", lambda t, r: _strip_hedging(t)),
    ("Remove transitions",     lambda t, r: _strip_transitions(t)),
    ("Inject first-person",    _inject_first_person),
    ("Vary sentence lengths",  _vary_sentence_lengths),
    ("Add rhetorical question",_add_question),
]


def cmd_rewrite(
    text: str,
    surrogate: Surrogate,
    target: float = 0.30,
    max_iterations: int = 20,
    as_json: bool = False,
    seed: int = 42,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    original = normalize_text(text)
    current = original
    current_prob = surrogate.predict_prob(current)
    original_prob = current_prob
    history: List[Dict[str, Any]] = []

    if not as_json:
        print()
        print(f"  {_c(_BOLD, 'Rewrite')}  Target: < {target:.2f}  |  Max iterations: {max_iterations}")
        print(f"  {'=' * 60}")
        print(f"  Start   {_prob_bar(current_prob)}  {current_prob:.4f}")
        print()

    for iteration in range(max_iterations):
        if current_prob <= target:
            break

        best_text = current
        best_prob = current_prob
        best_name = "none"

        for name, mutate_fn in _MUTATIONS:
            candidate = mutate_fn(current, rng)
            if candidate == current or not candidate.strip():
                continue
            cp = surrogate.predict_prob(candidate)
            if cp < best_prob:
                best_prob = cp
                best_text = candidate
                best_name = name

        if best_text == current:
            for v in humanize_variants(current):
                vp = surrogate.predict_prob(v)
                if vp < best_prob:
                    best_prob = vp
                    best_text = v
                    best_name = "Humanize variant"

        delta = current_prob - best_prob
        history.append({
            "iteration": iteration + 1,
            "mutation": best_name,
            "before": round(current_prob, 6),
            "after": round(best_prob, 6),
            "delta": round(delta, 6),
        })

        if not as_json:
            arrow = _c(_GREEN, "\u2193") if delta > 0.01 else _c(_YELLOW, "\u2192")
            print(f"  {iteration+1:>3d}.  {best_name:28s}  {arrow}  {best_prob:.4f}  ({delta:+.4f})")

        if best_text == current:
            if not as_json:
                print(f"\n  {_c(_DIM, 'No further improvement found.')}")
            break

        current = best_text
        current_prob = best_prob

    final_cls = surrogate.predict_class(current)
    passed = current_prob <= target

    diff_lines = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        current.splitlines(keepends=True),
        fromfile="original",
        tofile="rewritten",
        lineterm="",
    ))

    result = {
        "passed": passed,
        "original_prob": round(original_prob, 6),
        "final_prob": round(current_prob, 6),
        "final_class": final_cls,
        "iterations": len(history),
        "target": target,
        "rewritten_text": current,
        "diff": "\n".join(diff_lines) if diff_lines else "",
        "history": history,
    }

    if as_json:
        print(json.dumps(result, indent=2))
        return result

    print()
    print(f"  {'=' * 60}")
    status = _c(_GREEN, "PASSED") if passed else _c(_RED, "FAILED")
    print(f"  Result   {status}  {_prob_bar(current_prob)}  {current_prob:.4f}  ({final_cls})")
    print(f"  Dropped  {original_prob:.4f} -> {current_prob:.4f}  ({original_prob - current_prob:+.4f})")
    print()
    print(f"  {_c(_BOLD, 'Rewritten Text')}")
    print(f"  {'-' * 60}")
    for line in textwrap.wrap(current, width=72):
        print(f"  {line}")
    print()

    if diff_lines:
        print(f"  {_c(_BOLD, 'Changes')}")
        print(f"  {'-' * 60}")
        for line in diff_lines:
            if line.startswith("+") and not line.startswith("+++"):
                print(f"  {_c(_GREEN, line)}")
            elif line.startswith("-") and not line.startswith("---"):
                print(f"  {_c(_RED, line)}")
            else:
                print(f"  {_c(_DIM, line)}")
        print()

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_BANNER = r"""
   _____ _____ _____ _____
  |   __|  _  |_   _|__   |___ ___ ___
  |  |  |   __| | | |   -_| . |  _| . |
  |_____|__|    |_| |_____|___|_| |___|  evasion tool
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score, diagnose, or rewrite text to evade GPTZero AI detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              evade.py score "The structured methodology demonstrates significant potential."
              evade.py diagnose --stdin < essay.txt
              evade.py rewrite "Your AI text" --target 0.20 --max-iterations 15
              evade.py rewrite --stdin < essay.txt --json
        """),
    )
    parser.add_argument("mode", choices=["score", "diagnose", "rewrite"])
    parser.add_argument("text", nargs="?", help="Input text (or use --stdin)")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help=f"Surrogate model directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--target", type=float, default=0.30,
                        help="Rewrite target probability (default: 0.30)")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="Max rewrite iterations (default: 20)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for rewrite mutations (default: 42)")
    args = parser.parse_args()

    if args.stdin:
        text = sys.stdin.read().strip()
    elif args.text:
        text = args.text
    else:
        parser.error("Provide text as an argument or use --stdin")

    if not text:
        parser.error("Input text is empty")

    if not args.json:
        print(_c(_DIM, _BANNER))

    surrogate = Surrogate(args.model_dir)

    if args.mode == "score":
        cmd_score(text, surrogate, as_json=args.json)
    elif args.mode == "diagnose":
        cmd_diagnose(text, surrogate, as_json=args.json)
    elif args.mode == "rewrite":
        cmd_rewrite(
            text, surrogate,
            target=args.target,
            max_iterations=args.max_iterations,
            as_json=args.json,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
