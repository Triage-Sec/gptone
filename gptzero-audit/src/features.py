"""
Feature extraction for the surrogate classifier.

Extracts linguistic features that approximate GPTZero's
internal signals: perplexity proxies, burstiness, vocabulary
complexity, structural patterns, and stylistic markers.

Feature count: 40 (expanded from 28 in v1).
Additions: Zipf coefficient, windowed type-token ratio,
bigram/trigram repetition, adjacent-sentence similarity,
pronoun density, conjunction patterns, sentence length
entropy, and paragraph structure features.
"""

import re
import math
from collections import Counter

import numpy as np


HEDGING_PHRASES = [
    "it is worth noting", "it is important to", "furthermore",
    "moreover", "additionally", "consequently", "nevertheless",
    "in conclusion", "it should be mentioned", "it is essential",
    "it is crucial", "it bears emphasizing", "it is reasonable",
    "it is generally understood", "significant potential",
    "has demonstrated", "plays a crucial role",
]

TRANSITION_WORDS = [
    "furthermore", "moreover", "additionally", "consequently",
    "nevertheless", "however", "therefore", "thus", "hence",
    "accordingly", "subsequently", "meanwhile", "nonetheless",
    "in conclusion", "ultimately", "similarly",
]

FEATURE_NAMES = [
    # Basic counts (4)
    "word_count",
    "char_count",
    "avg_word_len",
    "avg_word_len_std",
    # Sentence structure (5)
    "sentence_count",
    "avg_sentence_len_words",
    "sentence_len_std",
    "sentence_len_cv",
    "sentence_len_entropy",
    # Vocabulary complexity (5)
    "unique_word_ratio",
    "long_word_ratio",
    "zipf_coefficient",
    "windowed_ttr",
    "hapax_ratio",
    # Punctuation (6)
    "comma_density",
    "semicolon_count",
    "question_mark_count",
    "exclamation_count",
    "ellipsis_count",
    "quote_count",
    # Stylistic markers (6)
    "contraction_count",
    "parenthetical_count",
    "hedging_score",
    "transition_word_score",
    "first_person_count",
    "passive_voice_ratio",
    # Structural patterns (6)
    "avg_clause_depth",
    "starts_with_the_ratio",
    "starts_with_transition_ratio",
    "repeated_sentence_starts",
    "all_caps_words",
    "numeric_token_count",
    # N-gram repetition (3)
    "bigram_repeat_ratio",
    "trigram_repeat_ratio",
    "adjacent_sentence_similarity",
    # Pronoun / conjunction patterns (3)
    "pronoun_density",
    "conjunction_density",
    "determiner_density",
    # Paragraph structure (2)
    "paragraph_count",
    "avg_paragraph_len",
]


def _tokenize(text):
    """Lowercase word tokens stripped of trailing punctuation."""
    return [w.lower().strip(".,!?;:\"'()[]{}") for w in text.split()]


def _sentences(text):
    """Split on sentence-ending punctuation."""
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def _zipf_coefficient(word_freqs):
    """
    Estimate Zipf exponent via log-log regression of rank vs frequency.
    AI text tends toward a tighter Zipf fit; human text is noisier.
    Returns slope magnitude. Falls back to 0.0 if insufficient data.
    """
    if len(word_freqs) < 5:
        return 0.0
    counts = sorted(word_freqs.values(), reverse=True)
    ranks = np.arange(1, len(counts) + 1, dtype=float)
    log_ranks = np.log(ranks)
    log_counts = np.log(np.array(counts, dtype=float) + 1e-10)
    if np.std(log_ranks) < 1e-10:
        return 0.0
    slope = np.polyfit(log_ranks, log_counts, 1)[0]
    return abs(slope)


def _windowed_ttr(tokens, window=50):
    """
    Moving-window type-token ratio (MATTR).
    More stable than global TTR for varying text lengths.
    """
    if len(tokens) < window:
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)
    ratios = []
    for i in range(len(tokens) - window + 1):
        w = tokens[i:i + window]
        ratios.append(len(set(w)) / window)
    return float(np.mean(ratios))


def _ngram_repeat_ratio(tokens, n):
    """Fraction of n-grams that appear more than once."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(1 for c in counts.values() if c > 1)
    return repeated / len(counts) if counts else 0.0


def _jaccard(a, b):
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def _adjacent_sentence_similarity(sentences):
    """
    Average Jaccard similarity between consecutive sentences.
    AI text tends to have higher inter-sentence similarity.
    """
    if len(sentences) < 2:
        return 0.0
    sims = []
    for i in range(len(sentences) - 1):
        a = set(_tokenize(sentences[i]))
        b = set(_tokenize(sentences[i + 1]))
        sims.append(_jaccard(a, b))
    return float(np.mean(sims))


def _entropy(values):
    """Shannon entropy of a discrete distribution."""
    if not values:
        return 0.0
    total = sum(values)
    if total == 0:
        return 0.0
    probs = [v / total for v in values if v > 0]
    return -sum(p * math.log2(p) for p in probs)


PRONOUNS = {
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    "it", "its", "itself",
}

CONJUNCTIONS = {
    "and", "but", "or", "nor", "for", "yet", "so",
    "because", "although", "while", "since", "unless",
    "whereas", "whether", "though",
}

DETERMINERS = {
    "the", "a", "an", "this", "that", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "some", "any", "no", "every", "each", "all", "both",
}


def extract_features(text):
    """
    Extract a 40-dimensional feature vector from raw text.
    Returns (feature_vector, feature_names).
    """
    words = text.split()
    word_count = len(words) or 1
    chars = len(text)

    sentences = _sentences(text)
    sent_count = len(sentences) or 1
    sent_lens = [len(s.split()) for s in sentences]

    word_lens = [len(w.strip(".,!?;:\"'()")) for w in words]
    tokens = _tokenize(text)
    token_freqs = Counter(tokens)

    lower_text = text.lower()

    sent_starts = [s.split()[0].lower() if s.split() else "" for s in sentences]

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    para_count = max(len(paragraphs), 1)

    features = {
        # Basic counts
        "word_count": word_count,
        "char_count": chars,
        "avg_word_len": float(np.mean(word_lens)) if word_lens else 0.0,
        "avg_word_len_std": float(np.std(word_lens)) if word_lens else 0.0,

        # Sentence structure
        "sentence_count": sent_count,
        "avg_sentence_len_words": float(np.mean(sent_lens)) if sent_lens else 0.0,
        "sentence_len_std": float(np.std(sent_lens)) if len(sent_lens) > 1 else 0.0,
        "sentence_len_cv": (
            float(np.std(sent_lens) / np.mean(sent_lens))
            if sent_lens and np.mean(sent_lens) > 0 else 0.0
        ),
        "sentence_len_entropy": _entropy(sent_lens),

        # Vocabulary complexity
        "unique_word_ratio": len(set(tokens)) / word_count,
        "long_word_ratio": sum(1 for w in word_lens if w > 8) / word_count,
        "zipf_coefficient": _zipf_coefficient(token_freqs),
        "windowed_ttr": _windowed_ttr(tokens),
        "hapax_ratio": sum(1 for c in token_freqs.values() if c == 1) / max(len(token_freqs), 1),

        # Punctuation
        "comma_density": text.count(",") / word_count,
        "semicolon_count": text.count(";"),
        "question_mark_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "ellipsis_count": text.count("..."),
        "quote_count": text.count('"') + text.count("'"),

        # Stylistic markers
        "contraction_count": sum(1 for w in words if "'" in w and len(w) > 2),
        "parenthetical_count": text.count("("),
        "hedging_score": sum(1 for p in HEDGING_PHRASES if p in lower_text) / sent_count,
        "transition_word_score": sum(1 for w in tokens if w in TRANSITION_WORDS) / sent_count,
        "first_person_count": (
            sum(1 for w in tokens if w in {"i", "me", "my", "we", "our", "i'm", "i've"})
            / word_count
        ),
        "passive_voice_ratio": (
            sum(1 for w in tokens if w in {"was", "were", "been", "being", "is", "are"})
            / word_count
        ),

        # Structural patterns
        "avg_clause_depth": text.count(",") / sent_count,
        "starts_with_the_ratio": sum(1 for s in sent_starts if s == "the") / sent_count,
        "starts_with_transition_ratio": (
            sum(1 for s in sent_starts if s in TRANSITION_WORDS) / sent_count
        ),
        "repeated_sentence_starts": 1 - (len(set(sent_starts)) / sent_count),
        "all_caps_words": sum(1 for w in words if w.isupper() and len(w) > 1) / word_count,
        "numeric_token_count": (
            sum(1 for w in words if any(c.isdigit() for c in w)) / word_count
        ),

        # N-gram repetition
        "bigram_repeat_ratio": _ngram_repeat_ratio(tokens, 2),
        "trigram_repeat_ratio": _ngram_repeat_ratio(tokens, 3),
        "adjacent_sentence_similarity": _adjacent_sentence_similarity(sentences),

        # Pronoun / conjunction / determiner density
        "pronoun_density": sum(1 for w in tokens if w in PRONOUNS) / word_count,
        "conjunction_density": sum(1 for w in tokens if w in CONJUNCTIONS) / word_count,
        "determiner_density": sum(1 for w in tokens if w in DETERMINERS) / word_count,

        # Paragraph structure
        "paragraph_count": para_count,
        "avg_paragraph_len": word_count / para_count,
    }

    return np.array([features[name] for name in FEATURE_NAMES]), FEATURE_NAMES


def extract_features_batch(texts):
    """Extract features for a list of texts. Returns (X, feature_names)."""
    vecs = []
    for text in texts:
        vec, names = extract_features(text)
        vecs.append(vec)
    return np.array(vecs), names
