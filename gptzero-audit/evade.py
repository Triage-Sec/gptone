#!/usr/bin/env python3
"""GPTZero evasion tool.

Interactive CLI that uses the extracted surrogate model to score, diagnose,
verify against the live API, and rewrite text to evade GPTZero's AI detection.
"""

from __future__ import annotations

import difflib
import getpass
import json
import os
import pickle
import random
import re
import readline
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
    normalize_text,
    split_sentences,
    join_sentences,
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
        print(f"  {_c(_GREEN, 'Downloaded')} {mb:.1f} MB")
        print()
        return Surrogate(MODEL_PATH)
    except ImportError:
        print(f"  {_c(_RED, 'Error:')} modal package not installed.")
        print(f"  Run: pip install modal && python3 -m modal setup")
        sys.exit(1)
    except Exception as e:
        print(f"  {_c(_RED, 'Download failed:')} {e}")
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


# -- feature-targeted mutations ----------------------------------------------

_HEDGING_RE = [re.compile(re.escape(p), re.IGNORECASE) for p in HEDGING_PHRASES]
_TRANSITION_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in TRANSITION_WORDS) + r")\b,?\s*",
    re.IGNORECASE,
)

# Maps a trigger feature name to a mutation function.
# Each function takes (text) -> text. Applied at most once.

def _mut_contractions(text: str) -> str:
    return contractionize(text)


def _mut_casualize(text: str) -> str:
    return casualize(text)


def _mut_strip_hedging(text: str) -> str:
    sents = split_sentences(text)
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
    return join_sentences(out) if out else text


def _mut_strip_transitions(text: str) -> str:
    sents = split_sentences(text)
    out = []
    for s in sents:
        x = _TRANSITION_RE.sub("", s, count=1).strip()
        x = re.sub(r"^[,;:\s]+", "", x).strip()
        if x and x[0].islower():
            x = x[0].upper() + x[1:]
        out.append(x if x else s)
    return join_sentences(out)


def _mut_shorten_words(text: str) -> str:
    """Replace long formal words with shorter casual equivalents (case-preserving)."""
    from src.boundary_mining import _FORMAL_TO_CASUAL, _sub_wordwise
    out = text
    for pattern, replacement in _FORMAL_TO_CASUAL:
        out = _sub_wordwise(out, pattern, replacement)
    return normalize_text(out)


def _mut_split_longest(text: str) -> str:
    """Split the longest sentence at a natural comma or conjunction."""
    sents = split_sentences(text)
    if len(sents) < 2:
        return text
    li = max(range(len(sents)), key=lambda i: len(sents[i].split()))
    s = sents[li]
    words = s.split()
    if len(words) < 10:
        return text
    split_points = []
    for i, w in enumerate(words):
        wl = w.lower().rstrip(".,;:")
        if wl in ("and", "but", "which", "where", "while", "although", "because") and i > 3:
            split_points.append(i)
    if not split_points:
        for i, w in enumerate(words):
            if w.endswith(",") and i > 3 and i < len(words) - 3:
                split_points.append(i + 1)
    if not split_points:
        return text
    sp = split_points[len(split_points) // 2]
    p1 = " ".join(words[:sp]).rstrip(".,;:") + "."
    p2 = " ".join(words[sp:]).lstrip(",;: ")
    if p2 and p2[0].islower():
        p2 = p2[0].upper() + p2[1:]
    sents[li] = p1
    sents.insert(li + 1, p2)
    return join_sentences(sents)


def _mut_vary_starts(text: str) -> str:
    """Reorder sentences that share the same opening word."""
    sents = split_sentences(text)
    if len(sents) < 3:
        return text
    starts = [s.split()[0].lower() if s.split() else "" for s in sents]
    changed = False
    for i in range(len(sents) - 1):
        if starts[i] == starts[i + 1] and starts[i]:
            for j in range(i + 2, len(sents)):
                if starts[j] != starts[i]:
                    sents[i + 1], sents[j] = sents[j], sents[i + 1]
                    starts[i + 1], starts[j] = starts[j], starts[i + 1]
                    changed = True
                    break
    return join_sentences(sents) if changed else text


def _mut_swap_distant(text: str) -> str:
    """Swap two non-adjacent sentences to reduce local similarity."""
    sents = split_sentences(text)
    if len(sents) < 4:
        return text
    i, j = 1, len(sents) - 2
    if i >= j:
        return text
    sents[i], sents[j] = sents[j], sents[i]
    return join_sentences(sents)


# Feature -> mutation mapping. Order = priority when features tie.
_FEATURE_MUTATIONS: Dict[str, List[Tuple[str, Any]]] = {
    "contraction_count":            [("Add contractions",        _mut_contractions)],
    "avg_word_len":                 [("Shorten words",           _mut_shorten_words),
                                     ("Casualize vocabulary",    _mut_casualize)],
    "hedging_score":                [("Strip hedging phrases",   _mut_strip_hedging)],
    "transition_word_score":        [("Strip transition words",  _mut_strip_transitions)],
    "starts_with_transition_ratio": [("Strip transition openers",_mut_strip_transitions)],
    "sentence_len_cv":              [("Split longest sentence",  _mut_split_longest)],
    "starts_with_the_ratio":        [("Vary sentence starts",    _mut_vary_starts)],
    "repeated_sentence_starts":     [("Diversify openers",       _mut_vary_starts)],
    "adjacent_sentence_similarity": [("Reorder sentences",       _mut_swap_distant)],
    "unique_word_ratio":            [("Casualize vocabulary",    _mut_casualize)],
    "windowed_ttr":                 [("Shorten words",           _mut_shorten_words)],
    "passive_voice_ratio":          [("Casualize vocabulary",    _mut_casualize),
                                     ("Add contractions",        _mut_contractions)],
    "bigram_repeat_ratio":          [("Reorder sentences",       _mut_swap_distant),
                                     ("Vary sentence starts",    _mut_vary_starts)],
    "trigram_repeat_ratio":         [("Reorder sentences",       _mut_swap_distant)],
    "determiner_density":           [("Casualize vocabulary",    _mut_casualize)],
    "pronoun_density":              [("Add contractions",        _mut_contractions)],
}


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


def _build_rewrite_prompt(text: str, triggers: List[Dict[str, Any]], attempt: int) -> str:
    """Build the Claude prompt for rewriting text to evade AI detection."""
    trigger_lines = []
    for t in triggers[:6]:
        trigger_lines.append(f"- {t['label']}: your text scores {t['value']:.3f}, human baseline is {t['baseline']:.3f}. {t['suggestion']}")
    trigger_block = "\n".join(trigger_lines)

    strictness = ""
    if attempt > 1:
        strictness = (
            f"\nThis is attempt {attempt}. Previous rewrites did not reduce the score enough. "
            "Be MORE aggressive: use very casual language, contractions everywhere, "
            "short punchy sentences, personal anecdotes, slang where appropriate. "
            "Break up long sentences. Remove ALL formal academic phrasing."
        )

    return f"""Rewrite the following text so that an AI text detector classifies it as human-written.

RULES:
- Preserve the EXACT same meaning, arguments, and information
- Do NOT add new claims, facts, or ideas that weren't in the original
- Do NOT remove key information or arguments
- Make it sound like a real person wrote it naturally
- Use contractions (don't, can't, it's, won't)
- Vary sentence lengths -- mix short and long
- Use casual vocabulary where possible (big not significant, show not demonstrate, use not utilize)
- Add first-person perspective where natural (I think, in my view)
- Remove hedging phrases (it is important to note, furthermore, moreover)
- Remove mechanical transition words at sentence starts
- Write like a human who knows the topic, not a textbook
- Return ONLY the rewritten text, nothing else{strictness}

DETECTION TRIGGERS TO FIX:
{trigger_block}

ORIGINAL TEXT:
{text}

REWRITTEN TEXT:"""


def _call_claude(prompt: str) -> Optional[str]:
    """Call Claude 4.5 Haiku via Anthropic API."""
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        env_path = PROJECT_ROOT.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
                    break
    if not api_key:
        print(f"  {_c(_RED, 'Error:')} ANTHROPIC_API_KEY not found in environment or .env")
        return None

    client = anthropic.Anthropic(api_key=api_key)
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        print(f"  {_c(_RED, 'Claude API error:')} {e}")
        return None


def do_rewrite(surrogate: Surrogate, text: Optional[str] = None,
               target: float = 0.30, max_iter: int = 5) -> Optional[str]:
    """LLM-powered rewrite guided by surrogate feature diagnosis."""
    if not text:
        text = _read_text()
    if not text:
        return None

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

        triggers = _get_triggers(surrogate, current)
        if not triggers:
            print(f"  {_c(_DIM, 'No detection triggers identified.')}")
            break

        print(f"  {_c(_DIM, f'[{it+1}] Rewriting with Claude (targeting {len(triggers[:6])} triggers)...')}")

        prompt = _build_rewrite_prompt(current, triggers, attempt=it + 1)
        candidate = _call_claude(prompt)

        if not candidate:
            break

        candidate = normalize_text(candidate)
        new_prob = surrogate.prob(candidate)
        delta = current_prob - new_prob

        arrow = _c(_GREEN, "\u2193") if delta > 0.01 else (_c(_YELLOW, "\u2192") if delta > -0.01 else _c(_RED, "\u2191"))
        print(f"  {it+1:>3d}.  Claude rewrite               {arrow}  {new_prob:.4f}  ({delta:+.4f})")

        if new_prob >= current_prob:
            print(f"  {_c(_DIM, '      Rewrite did not improve score. Retrying...')}")
            continue

        current = candidate
        current_prob = new_prob

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

    return current


# -- interactive shell -------------------------------------------------------

def _interactive(surrogate: Surrogate) -> None:
    target = 0.30
    max_iter = 20
    api_key: Optional[str] = os.environ.get("GPTZERO_API_KEY") or os.environ.get("GPT_ZERO_API_KEY")
    last_rewrite: Optional[str] = None

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
  {_c(_CYAN, 'rewrite')} {_c(_DIM, '/ r')}     Auto-rewrite text to evade detection
  {_c(_CYAN, 'target')} {_c(_DIM, '/ t')}      Set rewrite target probability  {_c(_DIM, f'(current: {target:.2f})')}
  {_c(_CYAN, 'iterations')} {_c(_DIM, '/ i')}  Set max rewrite iterations      {_c(_DIM, f'(current: {max_iter})')}
  {_c(_CYAN, 'help')} {_c(_DIM, '/ h')}        Show this message
  {_c(_CYAN, 'quit')} {_c(_DIM, '/ q')}        Exit
""")

        elif cmd in ("s", "score"):
            do_score(surrogate, inline)

        elif cmd in ("v", "verify"):
            api_key = do_verify(surrogate, inline, api_key=api_key)

        elif cmd in ("d", "diagnose", "diag"):
            do_diagnose(surrogate, inline)

        elif cmd in ("r", "rewrite"):
            result = do_rewrite(surrogate, inline, target=target, max_iter=max_iter)
            if result:
                last_rewrite = result

        elif cmd in ("t", "target"):
            val = inline
            if not val:
                try:
                    val = input(f"  {_c(_CYAN, '>')} Target probability: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
            try:
                target = float(val)
                print(f"  Target set to {_c(_CYAN, f'{target:.2f}')}")
            except ValueError:
                print(f"  {_c(_RED, 'Invalid number.')}")

        elif cmd in ("i", "iterations", "iter"):
            val = inline
            if not val:
                try:
                    val = input(f"  {_c(_CYAN, '>')} Max iterations: ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    continue
            try:
                max_iter = int(val)
                print(f"  Max iterations set to {_c(_CYAN, str(max_iter))}")
            except ValueError:
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
    parser.add_argument("mode", nargs="?", choices=["score", "diagnose", "rewrite", "verify"],
                        help="Non-interactive mode (omit for interactive shell)")
    parser.add_argument("text", nargs="?", help="Input text")
    parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--target", type=float, default=0.30, help="Rewrite target (default: 0.30)")
    parser.add_argument("--max-iterations", type=int, default=20, help="Max rewrite iterations (default: 20)")
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
        elif mode == "rewrite":
            current = normalize_text(text)
            cp = surrogate.prob(current)
            for attempt in range(args.max_iterations):
                if cp <= args.target:
                    break
                trigs = _get_triggers(surrogate, current)
                if not trigs:
                    break
                prompt = _build_rewrite_prompt(current, trigs, attempt + 1)
                candidate = _call_claude(prompt)
                if not candidate:
                    break
                candidate = normalize_text(candidate)
                np_ = surrogate.prob(candidate)
                if np_ >= cp:
                    continue
                current, cp = candidate, np_
            print(json.dumps({
                "passed": cp <= args.target,
                "original_prob": round(surrogate.prob(normalize_text(text)), 6),
                "final_prob": round(cp, 6),
                "final_class": surrogate.classify(current),
                "rewritten_text": current,
            }, indent=2))
        elif mode == "verify":
            do_verify(surrogate, text, api_key=args.api_key)
        return

    print(_c(_DIM, BANNER))
    if mode == "score":
        do_score(surrogate, text)
    elif mode == "diagnose":
        do_diagnose(surrogate, text)
    elif mode == "rewrite":
        do_rewrite(surrogate, text, target=args.target, max_iter=args.max_iterations)
    elif mode == "verify":
        do_verify(surrogate, text, api_key=args.api_key)


if __name__ == "__main__":
    main()
