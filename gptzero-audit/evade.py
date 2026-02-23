#!/usr/bin/env python3
"""GPTZero evasion tool.

Interactive CLI that uses the extracted surrogate model to score, diagnose,
and rewrite text to evade GPTZero's AI detection. Fully offline after
initial model download.
"""

from __future__ import annotations

import difflib
import json
import os
import pickle
import random
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
    contractionize,
    casualize,
    humanize_variants,
    normalize_text,
    split_sentences,
    join_sentences,
)

MODEL_DIR = PROJECT_ROOT / "models" / "ppl_surrogate"
MODEL_PATH = MODEL_DIR / "ppl_surrogate.pkl"

# -- terminal formatting -----------------------------------------------------

_BOLD = "\033[1m"
_DIM = "\033[2m"
_ITALIC = "\033[3m"
_RED = "\033[91m"
_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_WHITE = "\033[97m"
_RESET = "\033[0m"
_CLEAR_LINE = "\033[2K\r"

_USE_COLOR = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOR else text


def _bar(prob: float, width: int = 30) -> str:
    filled = int(prob * width)
    if prob >= 0.65:
        color = _RED
    elif prob >= 0.35:
        color = _YELLOW
    else:
        color = _GREEN
    return _c(color, "\u2588" * filled) + _c(_DIM, "\u2591" * (width - filled))


def _verdict(cls: str) -> str:
    if cls == "ai":
        return _c(_RED, "DETECTED AS AI")
    if cls == "human":
        return _c(_GREEN, "PASSES AS HUMAN")
    return _c(_YELLOW, "AMBIGUOUS (MIXED)")


def _hr(char: str = "\u2500", width: int = 64) -> str:
    return _c(_DIM, char * width)


def _prompt(msg: str) -> str:
    return input(f"  {_c(_CYAN, '>')} {msg}")


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


# -- model download ----------------------------------------------------------

def _ensure_model() -> Surrogate:
    if MODEL_PATH.exists():
        return Surrogate(MODEL_PATH)

    print()
    print(f"  {_c(_YELLOW, 'Model not found locally.')}")
    print(f"  Downloading trained surrogate from Modal...")
    print()

    try:
        import modal_app
        with modal_app.app.run():
            data = modal_app.download_model.remote()
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_PATH.write_bytes(data)
        mb = len(data) / (1024 * 1024)
        print(f"  {_c(_GREEN, 'Downloaded')} {mb:.1f} MB to {MODEL_PATH}")
        print()
        return Surrogate(MODEL_PATH)
    except ImportError:
        print(f"  {_c(_RED, 'Error:')} modal package not installed.")
        print(f"  Install with: pip install modal && python3 -m modal setup")
        sys.exit(1)
    except Exception as e:
        print(f"  {_c(_RED, 'Error downloading model:')} {e}")
        print(f"  Make sure Modal is configured: python3 -m modal setup")
        sys.exit(1)


# -- feature analysis --------------------------------------------------------

_GUIDANCE: Dict[str, Tuple[str, str, str]] = {
    "unique_word_ratio":           ("low",  "Vocabulary diversity",    "Use more varied word choices. Avoid reusing the same terms."),
    "contraction_count":           ("low",  "Contractions",           "Add contractions: don't, can't, it's, won't, I've, etc."),
    "hedging_score":               ("high", "Hedging phrases",        "Remove phrases like 'it is important to note', 'furthermore'."),
    "transition_word_score":       ("high", "Mechanical transitions", "Replace 'moreover', 'consequently', 'therefore' with casual connectors."),
    "first_person_count":          ("low",  "First-person voice",     "Write in first person: 'I think', 'in my experience', 'we found'."),
    "passive_voice_ratio":         ("high", "Passive voice",          "Rewrite passive constructions in active voice."),
    "bigram_repeat_ratio":         ("high", "Bigram repetition",      "Vary phrasing. Don't repeat two-word patterns."),
    "trigram_repeat_ratio":        ("high", "Trigram repetition",     "Rephrase repeated three-word constructions."),
    "adjacent_sentence_similarity":("high", "Sentence similarity",    "Make consecutive sentences more distinct in topic and vocabulary."),
    "sentence_len_cv":             ("low",  "Sentence length variety", "Mix short punchy sentences with longer ones."),
    "question_mark_count":         ("low",  "Questions",              "Add a rhetorical or genuine question."),
    "exclamation_count":           ("low",  "Exclamations",           "Add natural emphasis where appropriate."),
    "starts_with_the_ratio":       ("high", "'The' sentence starts",  "Vary sentence openers. Not every sentence should start with 'The'."),
    "starts_with_transition_ratio":("high", "Transition openers",     "Don't open sentences with 'Furthermore', 'Moreover', etc."),
    "repeated_sentence_starts":    ("high", "Repetitive openers",     "Use different first words for each sentence."),
    "avg_word_len":                ("high", "Word length",            "Use simpler, shorter words where possible."),
    "pronoun_density":             ("low",  "Pronoun usage",          "Add personal pronouns: he, she, they, I, we."),
    "windowed_ttr":                ("low",  "Type-token ratio",       "Vocabulary is too narrow. Introduce synonym variety."),
    "determiner_density":          ("high", "Determiner density",     "Reduce overuse of 'the', 'a', 'this'. Typical of formulaic writing."),
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


# -- mutations ---------------------------------------------------------------

_HEDGING_RE = [re.compile(re.escape(p), re.IGNORECASE) for p in HEDGING_PHRASES]
_TRANSITION_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in TRANSITION_WORDS) + r")\b,?\s*",
    re.IGNORECASE,
)
_FP_INSERTS = [
    "I think ", "In my experience, ", "From what I've seen, ",
    "Honestly, ", "I'd say ", "What I noticed is ",
    "I believe ", "The way I see it, ",
]
_QUESTIONS = [
    "But does that actually hold up?", "Is that really the case though?",
    "What does this mean in practice?", "Why does that matter?",
    "Sound familiar?", "Makes you think, right?",
    "But here's the thing -- does it work?",
]


def _m_contractionize(t: str, r: random.Random) -> str:
    return contractionize(t)

def _m_casualize(t: str, r: random.Random) -> str:
    return casualize(t)

def _m_strip_hedging(t: str, r: random.Random) -> str:
    sents = split_sentences(t)
    out = []
    for s in sents:
        x = s
        for pat in _HEDGING_RE:
            x = pat.sub("", x)
        x = re.sub(r"\s+", " ", x).strip()
        x = re.sub(r"^[,;:\s]+", "", x).strip()
        if x and x[0].islower():
            x = x[0].upper() + x[1:]
        if x:
            out.append(x)
    return join_sentences(out) if out else t

def _m_strip_transitions(t: str, r: random.Random) -> str:
    sents = split_sentences(t)
    out = []
    for s in sents:
        x = _TRANSITION_RE.sub("", s, count=1).strip()
        x = re.sub(r"^[,;:\s]+", "", x).strip()
        if x and x[0].islower():
            x = x[0].upper() + x[1:]
        out.append(x if x else s)
    return join_sentences(out)

def _m_first_person(t: str, r: random.Random) -> str:
    sents = split_sentences(t)
    if not sents:
        return t
    i = r.randrange(len(sents))
    s = sents[i]
    sents[i] = r.choice(_FP_INSERTS) + (s[0].lower() + s[1:] if len(s) > 1 else s)
    return join_sentences(sents)

def _m_vary_lengths(t: str, r: random.Random) -> str:
    sents = split_sentences(t)
    if len(sents) < 3:
        return t
    li = max(range(len(sents)), key=lambda i: len(sents[i].split()))
    words = sents[li].split()
    if len(words) > 12:
        mid = len(words) // 2
        p1 = " ".join(words[:mid]).rstrip(".,;:") + "."
        p2 = " ".join(words[mid:])
        if p2 and p2[0].islower():
            p2 = p2[0].upper() + p2[1:]
        sents[li] = p1
        sents.insert(li + 1, p2)
    return join_sentences(sents)

def _m_question(t: str, r: random.Random) -> str:
    sents = split_sentences(t)
    if not sents:
        return t
    i = r.randint(1, max(1, len(sents) - 1))
    sents.insert(i, r.choice(_QUESTIONS))
    return join_sentences(sents)

_MUTATIONS: List[Tuple[str, Any]] = [
    ("Add contractions",        _m_contractionize),
    ("Casualize vocabulary",    _m_casualize),
    ("Remove hedging phrases",  _m_strip_hedging),
    ("Remove transitions",      _m_strip_transitions),
    ("Inject first-person",     _m_first_person),
    ("Vary sentence lengths",   _m_vary_lengths),
    ("Add rhetorical question", _m_question),
]


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
    print(f"  {_c(_BOLD, 'GPTZero Prediction')}  {_c(_DIM, '(surrogate)')}")
    print(f"  {_hr()}")
    print(f"  AI Probability   {prob:.4f}  {_bar(prob)}")
    print(f"  Class            {cls}")
    print(f"  Words            {wc}")
    print(f"  Verdict          {_verdict(cls)}")
    print()


def do_diagnose(surrogate: Surrogate, text: Optional[str] = None) -> None:
    if not text:
        text = _read_text()
    if not text:
        return
    prob = surrogate.prob(text)
    cls = surrogate.classify(text)
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
            triggers.append({"label": label, "value": vec[i], "baseline": _BASELINES.get(name, 0.0),
                             "impact": score, "suggestion": suggestion})
    triggers.sort(key=lambda t: -t["impact"])

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


def do_rewrite(surrogate: Surrogate, text: Optional[str] = None, target: float = 0.30, max_iter: int = 20) -> None:
    if not text:
        text = _read_text()
    if not text:
        return

    rng = random.Random(42)
    original = normalize_text(text)
    current = original
    current_prob = surrogate.prob(current)
    orig_prob = current_prob

    print()
    print(f"  {_c(_BOLD, 'Rewrite')}  Target: < {target:.2f}  |  Max iterations: {max_iter}")
    print(f"  {_hr('=')}")
    print(f"  Start   {_bar(current_prob)}  {current_prob:.4f}")
    print()

    for it in range(max_iter):
        if current_prob <= target:
            break

        best, bp, bn = current, current_prob, "none"
        for name, fn in _MUTATIONS:
            c = fn(current, rng)
            if c == current or not c.strip():
                continue
            cp = surrogate.prob(c)
            if cp < bp:
                bp, best, bn = cp, c, name

        if best == current:
            for v in humanize_variants(current):
                vp = surrogate.prob(v)
                if vp < bp:
                    bp, best, bn = vp, v, "Humanize variant"

        delta = current_prob - bp
        arrow = _c(_GREEN, "\u2193") if delta > 0.01 else _c(_YELLOW, "\u2192")
        print(f"  {it+1:>3d}.  {bn:28s}  {arrow}  {bp:.4f}  ({delta:+.4f})")

        if best == current:
            print(f"\n  {_c(_DIM, 'No further improvement found.')}")
            break
        current, current_prob = best, bp

    cls = surrogate.classify(current)
    passed = current_prob <= target

    print()
    print(f"  {_hr('=')}")
    status = _c(_GREEN, "PASSED") if passed else _c(_RED, "FAILED")
    print(f"  Result   {status}  {_bar(current_prob)}  {current_prob:.4f}  ({cls})")
    print(f"  Dropped  {orig_prob:.4f} -> {current_prob:.4f}  ({orig_prob - current_prob:+.4f})")
    print()
    print(f"  {_c(_BOLD, 'Rewritten Text')}")
    print(f"  {_hr()}")
    for line in textwrap.wrap(current, width=72):
        print(f"  {line}")
    print()

    diff = list(difflib.unified_diff(
        original.splitlines(keepends=True), current.splitlines(keepends=True),
        fromfile="original", tofile="rewritten", lineterm="",
    ))
    if diff:
        print(f"  {_c(_BOLD, 'Changes')}")
        print(f"  {_hr()}")
        for d in diff:
            if d.startswith("+") and not d.startswith("+++"):
                print(f"  {_c(_GREEN, d)}")
            elif d.startswith("-") and not d.startswith("---"):
                print(f"  {_c(_RED, d)}")
            else:
                print(f"  {_c(_DIM, d)}")
        print()


# -- interactive shell -------------------------------------------------------

_HELP = f"""
  {_c(_BOLD, 'Commands')}
  {_hr()}
  {_c(_CYAN, 'score')}      Score text against GPTZero (paste or type inline)
  {_c(_CYAN, 'diagnose')}   Identify what triggers AI detection and how to fix it
  {_c(_CYAN, 'rewrite')}    Auto-rewrite text to evade detection
  {_c(_CYAN, 'target')}     Set rewrite target probability  (current: {{target}})
  {_c(_CYAN, 'iterations')} Set max rewrite iterations      (current: {{max_iter}})
  {_c(_CYAN, 'help')}       Show this message
  {_c(_CYAN, 'quit')}       Exit
"""


def _interactive(surrogate: Surrogate) -> None:
    target = 0.30
    max_iter = 20

    print(_c(_DIM, BANNER))
    print(f"  Surrogate model loaded. {_c(_GREEN, 'Ready.')}")
    print(f"  Type {_c(_CYAN, 'help')} for commands or just start with {_c(_CYAN, 'score')}.\n")

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
            print(_HELP.replace("{target}", f"{target:.2f}").replace("{max_iter}", str(max_iter)))
        elif cmd in ("s", "score"):
            do_score(surrogate, inline)
        elif cmd in ("d", "diagnose", "diag"):
            do_diagnose(surrogate, inline)
        elif cmd in ("r", "rewrite"):
            do_rewrite(surrogate, inline, target=target, max_iter=max_iter)
        elif cmd in ("t", "target"):
            if inline:
                try:
                    target = float(inline)
                    print(f"  Target set to {_c(_CYAN, f'{target:.2f}')}")
                except ValueError:
                    print(f"  {_c(_RED, 'Invalid number.')}")
            else:
                try:
                    val = _prompt("Target probability: ")
                    target = float(val)
                    print(f"  Target set to {_c(_CYAN, f'{target:.2f}')}")
                except (ValueError, EOFError):
                    print(f"  {_c(_RED, 'Invalid number.')}")
        elif cmd in ("i", "iterations", "iter"):
            if inline:
                try:
                    max_iter = int(inline)
                    print(f"  Max iterations set to {_c(_CYAN, str(max_iter))}")
                except ValueError:
                    print(f"  {_c(_RED, 'Invalid number.')}")
            else:
                try:
                    val = _prompt("Max iterations: ")
                    max_iter = int(val)
                    print(f"  Max iterations set to {_c(_CYAN, str(max_iter))}")
                except (ValueError, EOFError):
                    print(f"  {_c(_RED, 'Invalid number.')}")
        else:
            do_score(surrogate, raw)


# -- entrypoint --------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="GPTZero evasion tool. Interactive by default.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", nargs="?", choices=["score", "diagnose", "rewrite"],
                        help="Non-interactive mode (omit for interactive shell)")
    parser.add_argument("text", nargs="?", help="Input text for non-interactive mode")
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    parser.add_argument("--json", action="store_true", help="JSON output (non-interactive only)")
    parser.add_argument("--target", type=float, default=0.30, help="Rewrite target (default: 0.30)")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max rewrite iterations (default: 20)")
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
            vec, _ = surrogate.features(text)
            imps = surrogate.importances()
            triggers = []
            for i, name in enumerate(FEATURE_NAMES):
                if name not in _GUIDANCE:
                    continue
                g = _gap(name, vec[i])
                imp = imps.get(name, 0.0)
                sc = g * imp
                if sc > 0.0001:
                    _, label, suggestion = _GUIDANCE[name]
                    triggers.append({"feature": name, "label": label, "impact": round(sc, 6), "suggestion": suggestion})
            triggers.sort(key=lambda t: -t["impact"])
            print(json.dumps({"class": cls, "probability": round(prob, 6), "triggers": triggers[:10]}, indent=2))
        elif mode == "rewrite":
            rng = random.Random(42)
            current = normalize_text(text)
            cp = surrogate.prob(current)
            for _ in range(args.max_iterations):
                if cp <= args.target:
                    break
                best, bp, bn = current, cp, "none"
                for n, fn in _MUTATIONS:
                    c = fn(current, rng)
                    if c == current:
                        continue
                    p = surrogate.prob(c)
                    if p < bp:
                        bp, best, bn = p, c, n
                if best == current:
                    break
                current, cp = best, bp
            print(json.dumps({
                "passed": cp <= args.target,
                "original_prob": round(surrogate.prob(normalize_text(text)), 6),
                "final_prob": round(cp, 6),
                "final_class": surrogate.classify(current),
                "rewritten_text": current,
            }, indent=2))
        return

    print(_c(_DIM, BANNER))
    if mode == "score":
        do_score(surrogate, text)
    elif mode == "diagnose":
        do_diagnose(surrogate, text)
    elif mode == "rewrite":
        do_rewrite(surrogate, text, target=args.target, max_iter=args.max_iterations)


if __name__ == "__main__":
    main()
