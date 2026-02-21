"""GPTZero API client with full response parsing and logging."""

import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path


API_URL = "https://api.gptzero.me/v2/predict/text"


class GPTZeroClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("GPTZERO_API_KEY")
        if not self.api_key:
            raise ValueError("Set GPTZERO_API_KEY or pass api_key=")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        self.call_count = 0
        self.total_words = 0

    def predict(self, text, delay=0.5):
        """
        Send text to GPTZero. Returns parsed response dict.
        
        Response fields (confirmed from API docs + R library):
          doc-level: average_generated_prob, completely_generated_prob,
                     overall_burstiness, predicted_class, class_probabilities,
                     confidence_category
          sentence-level: generated_prob, perplexity, highlight_sentence_for_ai
        """
        time.sleep(delay)
        
        try:
            resp = requests.post(
                API_URL,
                headers=self.headers,
                json={"document": text},
                timeout=30,
            )
            resp.raise_for_status()
            raw = resp.json()
        except requests.exceptions.HTTPError as e:
            return {"error": str(e), "status": resp.status_code, "body": resp.text}
        except Exception as e:
            return {"error": str(e)}

        self.call_count += 1
        self.total_words += len(text.split())

        return self._parse(raw, text)

    def _parse(self, raw, text):
        docs = raw.get("documents", [{}])
        doc = docs[0] if docs else {}

        sentences = []
        for s in doc.get("sentences", []):
            sentences.append({
                "text": s.get("sentence", ""),
                "generated_prob": s.get("generated_prob"),
                "perplexity": s.get("perplexity"),
                "highlighted": s.get("highlight_sentence_for_ai", False),
            })

        return {
            "input_text": text,
            "input_word_count": len(text.split()),
            # Document-level (these are the gradient leakage channels)
            "average_generated_prob": doc.get("average_generated_prob"),
            "completely_generated_prob": doc.get("completely_generated_prob"),
            "overall_burstiness": doc.get("overall_burstiness"),
            "predicted_class": doc.get("predicted_class"),
            "class_probabilities": doc.get("class_probabilities"),
            "confidence_category": doc.get("confidence_category"),
            # Paragraph-level
            "paragraphs": [
                {
                    "completely_generated_prob": p.get("completely_generated_prob"),
                    "num_sentences": p.get("num_sentences"),
                    "start_sentence_index": p.get("start_sentence_index"),
                }
                for p in doc.get("paragraphs", [])
            ],
            # Sentence-level (HMM / highlight leakage)
            "sentences": sentences,
            # Keep raw for anything we missed
            "raw": raw,
        }

    def predict_batch(self, texts, ids=None, delay=0.5, verbose=True):
        """Run a list of texts. Returns list of parsed responses."""
        results = []
        ids = ids or [f"sample_{i}" for i in range(len(texts))]
        for i, (sid, text) in enumerate(zip(ids, texts)):
            if verbose:
                print(f"  [{i+1}/{len(texts)}] {sid[:40]}...", end=" ", flush=True)
            r = self.predict(text, delay=delay)
            if "error" in r:
                if verbose:
                    print(f"ERROR: {r['error']}")
            else:
                if verbose:
                    cls = r.get("predicted_class", "?")
                    prob = r.get("average_generated_prob", "?")
                    print(f"{cls} (p={prob})")
            r["sample_id"] = sid
            results.append(r)
        return results
