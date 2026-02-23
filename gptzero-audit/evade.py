#!/usr/bin/env python3
"""GPTZero evasion tool.

Interactive CLI that uses the extracted surrogate model to score text,
diagnose which features trigger AI detection, and verify predictions
against GPTZero's live API.
"""

from __future__ import annotations

import getpass
import json
import os
import pickle
import re
import readline
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
    normalize_text,
    split_sentences,
)

MODEL_DIR = PROJECT_ROOT / "models" / "ppl_surrogate"
MODEL_PATH = MODEL_DIR / "ppl_surrogate.pkl"

# -- terminal formatting -----------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_RESET = "\033[0m"

_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOR else text


def _bar(prob: float, width: int = 30) -> str:
    filled = int(prob * width)
    color = _RED if prob >= 0.65 else (_YELLOW if prob >= 0.35 else _GREEN)
    return _c(color, "\u2588" * filled) + _c(_DIM, "\u2591" * (width - filled))


def _verdict(cls: str) -> str:
    if cls == "ai":
        return _c(_RED, "DETECTED AS AI")
    if cls == "human":
        return _c(_GREEN, "PASSES AS HUMAN")
    return _c(_YELLOW, "AMBIGUOUS (MIXED)")


def _hr(char: str = "\u2500", width: int = 64) -> str:
    return _c(_DIM, char * width)


BANNER = r"""
   _____ _____ _____ _____
  |   __|  _  |_   _|__   |___ ___ ___
  |  |  |   __| | | |   -_| . |  _| . |
  |_____|__|    |_| |_____|___|_| |___|  evasion tool
"""

# -- model -------------------------------------------------------------------

class Surrogate:

    def __init__(self, path: Path):
        with path.open("rb") as fh:
            data = pickle.load(fh)
        self.reg = data["model"]
        self.cls = data.get("class_model")
        self.names = data.get("feature_names", FEATURE_NAMES)

    def prob(self, text: str) -> float:
        vec, _ = extract_features(text)
        return float(np.clip(self.reg.predict(vec.reshape(1, -1))[0], 0.0, 1.0))

    def prob_batch(self, texts: Sequence[str]) -> np.ndarray:
        X, _ = extract_features_batch(list(texts))
        return np.clip(self.reg.predict(X), 0.0, 1.0)

    def classify(self, text: str) -> str:
        if self.cls is not None:
            vec, _ = extract_features(text)
            return str(self.cls.predict(vec.reshape(1, -1))[0])
        p = self.prob(text)
        return "ai" if p >= 0.65 else ("human" if p <= 0.35 else "mixed")

    def importances(self) -> Dict[str, float]:
        return dict(zip(self.names, self.reg.feature_importances_.tolist()))

    def features(self, text: str) -> Tuple[np.ndarray, List[str]]:
        return extract_features(text)


def _ensure_model() -> Surrogate:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1000:
        return Surrogate(MODEL_PATH)

    print()
    if MODEL_PATH.exists():
        print(f"  {_c(_YELLOW, 'Model file is a Git LFS pointer. Run: git lfs pull')}")
    else:
        print(f"  {_c(_YELLOW, 'Model not found at')} {MODEL_PATH}")
        print(f"  The model ships with the repo via Git LFS.")
        print(f"  If you cloned without LFS, run: git lfs pull")
        print()
        print(f"  Or download from Modal:")
        print(f"    python -c \"import modal_app; ...\"  # see README")
    sys.exit(1)


# -- feature analysis --------------------------------------------------------

_GUIDANCE: Dict[str, Tuple[str, str, str]] = {
    "unique_word_ratio":           ("low",  "Vocabulary diversity",    "Use more varied word choices."),
    "contraction_count":           ("low",  "Contractions",           "Add contractions: don't, can't, it's, etc."),
    "hedging_score":               ("high", "Hedging phrases",        "Remove 'it is important to note', 'furthermore', etc."),
    "transition_word_score":       ("high", "Mechanical transitions", "Remove 'moreover', 'consequently', 'therefore'."),
    "first_person_count":          ("low",  "First-person voice",     "Write in first person where natural."),
    "passive_voice_ratio":         ("high", "Passive voice",          "Rewrite passive constructions in active voice."),
    "bigram_repeat_ratio":         ("high", "Bigram repetition",      "Vary phrasing at the two-word level."),
    "trigram_repeat_ratio":        ("high", "Trigram repetition",     "Rephrase repeated three-word constructions."),
    "adjacent_sentence_similarity":("high", "Sentence similarity",    "Make consecutive sentences more distinct."),
    "sentence_len_cv":             ("low",  "Sentence length variety", "Mix short and long sentences."),
    "question_mark_count":         ("low",  "Questions",              "Add a question where natural."),
    "exclamation_count":           ("low",  "Exclamations",           "Add emphasis where appropriate."),
    "starts_with_the_ratio":       ("high", "'The' sentence starts",  "Vary sentence openers."),
    "starts_with_transition_ratio":("high", "Transition openers",     "Don't start sentences with formal transitions."),
    "repeated_sentence_starts":    ("high", "Repetitive openers",     "Use different first words for each sentence."),
    "avg_word_len":                ("high", "Word length",            "Use simpler, shorter words."),
    "pronoun_density":             ("low",  "Pronoun usage",          "Add personal pronouns."),
    "windowed_ttr":                ("low",  "Type-token ratio",       "Vocabulary is too narrow."),
    "determiner_density":          ("high", "Determiner density",     "Reduce overuse of 'the', 'a', 'this'."),
}

_BASELINES: Dict[str, float] = {
    "unique_word_ratio": 0.70, "contraction_count": 3.0, "hedging_score": 0.05,
    "transition_word_score": 0.10, "first_person_count": 0.03, "passive_voice_ratio": 0.03,
    "bigram_repeat_ratio": 0.08, "trigram_repeat_ratio": 0.02,
    "adjacent_sentence_similarity": 0.10, "sentence_len_cv": 0.45,
    "question_mark_count": 1.0, "exclamation_count": 0.5,
    "starts_with_the_ratio": 0.10, "starts_with_transition_ratio": 0.05,
    "repeated_sentence_starts": 0.25, "avg_word_len": 4.5,
    "pronoun_density": 0.06, "determiner_density": 0.08, "windowed_ttr": 0.75,
}


def _gap(name: str, value: float) -> float:
    if name not in _GUIDANCE:
        return 0.0
    direction = _GUIDANCE[name][0]
    baseline = _BASELINES.get(name, value)
    return max(0.0, value - baseline) if direction == "high" else max(0.0, baseline - value)


def _get_triggers(surrogate: Surrogate, text: str) -> List[Dict[str, Any]]:
    vec, _ = surrogate.features(text)
    imps = surrogate.importances()
    triggers: List[Dict[str, Any]] = []
    for i, name in enumerate(FEATURE_NAMES):
        if name not in _GUIDANCE:
            continue
        g = _gap(name, vec[i])
        imp = imps.get(name, 0.0)
        score = g * imp
        if score > 0.0001:
            direction, label, suggestion = _GUIDANCE[name]
            triggers.append({"feature": name, "label": label, "value": vec[i],
                             "baseline": _BASELINES.get(name, 0.0),
                             "impact": score, "suggestion": suggestion})
    triggers.sort(key=lambda t: -t["impact"])
    return triggers




# -- commands ----------------------------------------------------------------

def _read_text(prompt_msg: str = "Paste your text (enter a blank line to finish):") -> str:
    print(f"\n  {_c(_DIM, prompt_msg)}")
    lines: List[str] = []
    while True:
        try:
            line = input("  ")
        except EOFError:
            break
        if line == "":
            break
        lines.append(line)
    text = " ".join(lines).strip()
    if not text:
        print(f"  {_c(_YELLOW, 'No text entered.')}")
    return text


def do_score(surrogate: Surrogate, text: Optional[str] = None) -> None:
    if not text:
        text = _read_text()
    if not text:
        return
    prob = surrogate.prob(text)
    cls = surrogate.classify(text)
    wc = len(text.split())
    print()
    print(f"  {_c(_BOLD, 'GPTZero Prediction')}  {_c(_DIM, '(surrogate, offline)')}")
    print(f"  {_hr()}")
    print(f"  AI Probability   {prob:.4f}  {_bar(prob)}")
    print(f"  Class            {cls}")
    print(f"  Words            {wc}")
    print(f"  Verdict          {_verdict(cls)}")
    print(f"  {_c(_DIM, 'Use verify to check against GPTZero live API.')}")
    print()


def do_diagnose(surrogate: Surrogate, text: Optional[str] = None) -> None:
    if not text:
        text = _read_text()
    if not text:
        return
    prob = surrogate.prob(text)
    cls = surrogate.classify(text)
    triggers = _get_triggers(surrogate, text)

    sentences = split_sentences(text)
    sent_scores = []
    if sentences:
        probs = surrogate.prob_batch(sentences)
        sent_scores = sorted(
            [{"text": s, "prob": float(p)} for s, p in zip(sentences, probs)],
            key=lambda x: -x["prob"],
        )

    print()
    print(f"  {_c(_BOLD, 'Diagnosis')}  AI Prob: {prob:.4f}  {_bar(prob)}  {_verdict(cls)}")
    print()

    if triggers:
        print(f"  {_c(_BOLD, 'Detection Triggers')}")
        print(f"  {_hr()}")
        for t in triggers[:8]:
            n = min(10, max(1, int(t["impact"] * 500)))
            print(f"  {_c(_RED, chr(0x25a0)) * n}  {_c(_CYAN, t['label'])}")
            print(f"       Value: {t['value']:.3f}  |  Human baseline: {t['baseline']:.3f}")
            print(f"       {_c(_DIM, 'Fix:')} {t['suggestion']}")
            print()

    if sent_scores:
        print(f"  {_c(_BOLD, 'Flagged Sentences')}")
        print(f"  {_hr()}")
        for s in sent_scores[:5]:
            p = s["prob"]
            color = _RED if p >= 0.6 else (_YELLOW if p >= 0.4 else _GREEN)
            tag = _c(color, f"[{p:.2f}]")
            print(f"  {tag}  {textwrap.shorten(s['text'], width=72, placeholder='...')}")
        print()


def do_verify(surrogate: Surrogate, text: Optional[str] = None, api_key: Optional[str] = None) -> Optional[str]:
    """Query GPTZero's live API and show result alongside surrogate prediction."""
    if not text:
        text = _read_text()
    if not text:
        return api_key

    if not api_key:
        api_key = os.environ.get("GPTZERO_API_KEY") or os.environ.get("GPT_ZERO_API_KEY")
    if not api_key:
        try:
            api_key = getpass.getpass(f"  {_c(_CYAN, 'GPTZero API key:')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return api_key
    if not api_key:
        print(f"  {_c(_RED, 'No API key provided.')}")
        return api_key

    from src.client import GPTZeroClient
    client = GPTZeroClient(api_key=api_key)

    s_prob = surrogate.prob(text)
    s_cls = surrogate.classify(text)

    print(f"\n  {_c(_DIM, 'Querying GPTZero API...')}")
    resp = client.predict(text, delay=0)

    if "error" in resp:
        print(f"  {_c(_RED, 'API error:')} {resp['error']}")
        return api_key

    gz_prob = resp.get("average_generated_prob", 0.0)
    gz_cls = resp.get("predicted_class", "unknown")

    print()
    print(f"  {_c(_BOLD, 'Surrogate')} {_c(_DIM, '(offline)')}")
    print(f"  {_hr()}")
    print(f"  AI Probability   {s_prob:.4f}  {_bar(s_prob)}")
    print(f"  Class            {s_cls:8s}  {_verdict(s_cls)}")
    print()
    print(f"  {_c(_BOLD, 'GPTZero')} {_c(_DIM, '(live API)')}")
    print(f"  {_hr()}")
    print(f"  AI Probability   {gz_prob:.4f}  {_bar(gz_prob)}")
    print(f"  Class            {gz_cls:8s}  {_verdict(gz_cls)}")
    print()

    match = s_cls == gz_cls
    delta = abs(s_prob - gz_prob)
    if match and delta < 0.1:
        print(f"  {_c(_GREEN, 'Agreement:')} Both say {gz_cls}, prob delta {delta:.4f}")
    elif match:
        print(f"  {_c(_YELLOW, 'Partial agreement:')} Same class ({gz_cls}), prob delta {delta:.4f}")
    else:
        print(f"  {_c(_RED, 'Disagreement:')} Surrogate={s_cls}, GPTZero={gz_cls}, prob delta {delta:.4f}")
    print()
    return api_key






# -- interactive shell -------------------------------------------------------

def _interactive(surrogate: Surrogate) -> None:
    api_key: Optional[str] = os.environ.get("GPTZERO_API_KEY") or os.environ.get("GPT_ZERO_API_KEY")

    print(_c(_DIM, BANNER))
    print(f"  Surrogate model loaded. {_c(_GREEN, 'Ready.')}")
    print(f"  Type {_c(_CYAN, 'help')} for commands.\n")

    while True:
        try:
            raw = input(f"  {_c(_BOLD, 'gptzero')} {_c(_DIM, '>')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n  {_c(_DIM, 'Goodbye.')}")
            break

        if not raw:
            continue

        parts = raw.split(None, 1)
        cmd = parts[0].lower()
        inline = parts[1] if len(parts) > 1 else None

        if cmd in ("q", "quit", "exit"):
            print(f"  {_c(_DIM, 'Goodbye.')}")
            break

        elif cmd in ("h", "help", "?"):
            print(f"""
  {_c(_BOLD, 'Commands')}
  {_hr()}
  {_c(_CYAN, 'score')}  {_c(_DIM, '/ s')}      Score text against surrogate (offline)
  {_c(_CYAN, 'verify')} {_c(_DIM, '/ v')}      Score against GPTZero's live API + surrogate
  {_c(_CYAN, 'diagnose')} {_c(_DIM, '/ d')}    Identify detection triggers and how to fix them
  {_c(_CYAN, 'help')} {_c(_DIM, '/ h')}        Show this message
  {_c(_CYAN, 'quit')} {_c(_DIM, '/ q')}        Exit
""")

        elif cmd in ("s", "score"):
            do_score(surrogate, inline)

        elif cmd in ("v", "verify"):
            api_key = do_verify(surrogate, inline, api_key=api_key)

        elif cmd in ("d", "diagnose", "diag"):
            do_diagnose(surrogate, inline)

        else:
            do_score(surrogate, raw)


# -- entrypoint --------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GPTZero evasion tool. Interactive by default.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", nargs="?", choices=["score", "diagnose", "verify"],
                        help="Non-interactive mode (omit for interactive shell)")
    parser.add_argument("text", nargs="?", help="Input text")
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--api-key", help="GPTZero API key (for verify mode)")
    args = parser.parse_args()

    surrogate = _ensure_model()

    if args.mode is None and not args.stdin:
        _interactive(surrogate)
        return

    if args.stdin:
        text = sys.stdin.read().strip()
    elif args.text:
        text = args.text
    else:
        print(f"  {_c(_RED, 'Error:')} Provide text as argument or use --stdin")
        sys.exit(1)

    if not text:
        print(f"  {_c(_RED, 'Error:')} Input text is empty")
        sys.exit(1)

    mode = args.mode or "score"

    if args.json:
        prob = surrogate.prob(text)
        cls = surrogate.classify(text)
        if mode == "score":
            print(json.dumps({"class": cls, "probability": round(prob, 6), "word_count": len(text.split())}, indent=2))
        elif mode == "diagnose":
            triggers = _get_triggers(surrogate, text)
            print(json.dumps({"class": cls, "probability": round(prob, 6),
                               "triggers": [{"feature": t["feature"], "label": t["label"],
                                             "impact": round(t["impact"], 6), "suggestion": t["suggestion"]}
                                            for t in triggers[:10]]}, indent=2))
        elif mode == "verify":
            do_verify(surrogate, text, api_key=args.api_key)
        return

    print(_c(_DIM, BANNER))
    if mode == "score":
        do_score(surrogate, text)
    elif mode == "diagnose":
        do_diagnose(surrogate, text)
    elif mode == "verify":
        do_verify(surrogate, text, api_key=args.api_key)


if __name__ == "__main__":
    main()
