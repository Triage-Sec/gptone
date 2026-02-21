"""
Feature extraction for the surrogate classifier.

Extracts linguistic features that approximate what GPTZero
measures internally: perplexity proxies, burstiness, vocabulary
complexity, hedging, structural repetition, etc.
"""

import re
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
    "word_count",
    "char_count",
    "avg_word_len",
    "avg_word_len_std",
    "sentence_count",
    "avg_sentence_len_words",
    "sentence_len_std",          # burstiness proxy
    "sentence_len_cv",           # coefficient of variation
    "unique_word_ratio",
    "long_word_ratio",           # words > 8 chars
    "comma_density",
    "semicolon_count",
    "question_mark_count",
    "exclamation_count",
    "contraction_count",
    "parenthetical_count",
    "hedging_score",
    "transition_word_score",
    "first_person_count",
    "passive_voice_indicators",
    "avg_clause_depth",          # commas per sentence
    "starts_with_the_ratio",
    "starts_with_transition_ratio",
    "repeated_sentence_starts",
    "ellipsis_count",
    "all_caps_words",
    "numeric_token_count",
    "quote_count",
]


def extract_features(text):
    """
    Extract a feature vector from raw text.
    Returns (feature_vector, feature_names).
    """
    words = text.split()
    word_count = len(words) or 1
    chars = len(text)
    
    # Sentence splitting (simple but effective)
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    sent_count = len(sentences) or 1
    sent_lens = [len(s.split()) for s in sentences]
    
    word_lens = [len(w.strip(".,!?;:\"'()")) for w in words]
    
    lower_text = text.lower()
    lower_words = [w.lower().strip(".,!?;:\"'()") for w in words]
    
    # Sentence starts
    sent_starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
    
    features = {
        "word_count": word_count,
        "char_count": chars,
        "avg_word_len": np.mean(word_lens) if word_lens else 0,
        "avg_word_len_std": np.std(word_lens) if word_lens else 0,
        "sentence_count": sent_count,
        "avg_sentence_len_words": np.mean(sent_lens) if sent_lens else 0,
        "sentence_len_std": np.std(sent_lens) if len(sent_lens) > 1 else 0,
        "sentence_len_cv": (np.std(sent_lens) / np.mean(sent_lens)) if sent_lens and np.mean(sent_lens) > 0 else 0,
        "unique_word_ratio": len(set(lower_words)) / word_count,
        "long_word_ratio": sum(1 for w in word_lens if w > 8) / word_count,
        "comma_density": text.count(",") / word_count,
        "semicolon_count": text.count(";"),
        "question_mark_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "contraction_count": sum(1 for w in words if "'" in w and len(w) > 2),
        "parenthetical_count": text.count("("),
        "hedging_score": sum(1 for p in HEDGING_PHRASES if p in lower_text) / sent_count,
        "transition_word_score": sum(1 for w in lower_words if w in TRANSITION_WORDS) / sent_count,
        "first_person_count": sum(1 for w in lower_words if w in ["i", "me", "my", "we", "our", "i'm", "i've"]) / word_count,
        "passive_voice_indicators": sum(1 for w in lower_words if w in ["was", "were", "been", "being", "is", "are"]) / word_count,
        "avg_clause_depth": text.count(",") / sent_count,
        "starts_with_the_ratio": sum(1 for s in sent_starts if s == "the") / sent_count,
        "starts_with_transition_ratio": sum(1 for s in sent_starts if s in TRANSITION_WORDS) / sent_count,
        "repeated_sentence_starts": 1 - (len(set(sent_starts)) / sent_count),
        "ellipsis_count": text.count("..."),
        "all_caps_words": sum(1 for w in words if w.isupper() and len(w) > 1) / word_count,
        "numeric_token_count": sum(1 for w in words if any(c.isdigit() for c in w)) / word_count,
        "quote_count": text.count('"') + text.count("'"),
    }
    
    return np.array([features[name] for name in FEATURE_NAMES]), FEATURE_NAMES


def extract_features_batch(texts):
    """Extract features for a list of texts. Returns (X, feature_names)."""
    vecs = []
    for text in texts:
        vec, names = extract_features(text)
        vecs.append(vec)
    return np.array(vecs), names
