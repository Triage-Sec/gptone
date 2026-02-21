"""GPTZero API client with full response parsing and logging."""

import os
import time
import random
import requests


API_URL = "https://api.gptzero.me/v2/predict/text"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class GPTZeroClient:
    def __init__(
        self,
        api_key=None,
        max_retries=3,
        retry_base_delay=1.0,
        retry_max_delay=8.0,
        timeout_seconds=30,
    ):
        self.api_key = api_key or os.environ.get("GPTZERO_API_KEY")
        if not self.api_key:
            raise ValueError("Set GPTZERO_API_KEY or pass api_key=")
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
        }
        self.max_retries = max(0, int(max_retries))
        self.retry_base_delay = float(retry_base_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.timeout_seconds = float(timeout_seconds)
        self.call_count = 0
        self.total_words = 0

    def _retry_delay_seconds(self, attempt_idx):
        """
        Return jittered exponential backoff delay for a retry attempt.

        attempt_idx is zero-based where 0 is the first retry.
        """
        raw = self.retry_base_delay * (2 ** attempt_idx)
        bounded = min(raw, self.retry_max_delay)
        jitter = random.uniform(0.0, bounded * 0.25)
        return bounded + jitter

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

        # Logical query counters (not per retry attempt).
        self.call_count += 1
        self.total_words += len(text.split())

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    API_URL,
                    headers=self.headers,
                    json={"document": text},
                    timeout=self.timeout_seconds,
                )
                status = resp.status_code
                if status >= 400:
                    should_retry = (
                        status in RETRYABLE_STATUS_CODES and attempt < self.max_retries
                    )
                    if should_retry:
                        time.sleep(self._retry_delay_seconds(attempt))
                        continue
                    return {
                        "error": f"HTTP {status}",
                        "status": status,
                        "body": resp.text,
                        "attempts": attempt + 1,
                    }

                try:
                    raw = resp.json()
                except ValueError as e:
                    return {
                        "error": f"Invalid JSON response: {e}",
                        "status": status,
                        "body": resp.text,
                        "attempts": attempt + 1,
                    }
                return self._parse(raw, text)

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.max_retries:
                    time.sleep(self._retry_delay_seconds(attempt))
                    continue
                return {
                    "error": str(e),
                    "status": "network_error",
                    "attempts": attempt + 1,
                }
            except requests.exceptions.RequestException as e:
                status = e.response.status_code if e.response is not None else None
                body = e.response.text if e.response is not None else None
                if (
                    status in RETRYABLE_STATUS_CODES
                    and attempt < self.max_retries
                ):
                    time.sleep(self._retry_delay_seconds(attempt))
                    continue
                return {
                    "error": str(e),
                    "status": status,
                    "body": body,
                    "attempts": attempt + 1,
                }
            except Exception as e:
                return {"error": str(e), "attempts": attempt + 1}

        return {
            "error": "Exhausted retries without response",
            "status": "retry_exhausted",
            "attempts": self.max_retries + 1,
        }

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
