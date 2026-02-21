import unittest
from unittest.mock import Mock, patch

from src.client import GPTZeroClient


def _mock_response(status_code: int, payload=None, text: str = "") -> Mock:
    resp = Mock()
    resp.status_code = status_code
    resp.text = text
    if payload is None:
        resp.json.side_effect = ValueError("No JSON")
    else:
        resp.json.return_value = payload
    return resp


class GPTZeroClientRetryTest(unittest.TestCase):
    @patch("src.client.time.sleep", return_value=None)
    def test_retries_transient_5xx_then_succeeds(self, _sleep_mock: Mock) -> None:
        success_payload = {
            "documents": [
                {
                    "average_generated_prob": 0.9,
                    "completely_generated_prob": 0.8,
                    "overall_burstiness": 0.2,
                    "predicted_class": "ai",
                    "class_probabilities": {"ai": 0.9, "human": 0.1, "mixed": 0.0},
                    "confidence_category": "high",
                    "sentences": [],
                    "paragraphs": [],
                }
            ]
        }
        responses = [
            _mock_response(502, payload={"error": "bad gateway"}, text="bad gateway"),
            _mock_response(200, payload=success_payload, text="ok"),
        ]

        with patch("src.client.requests.post", side_effect=responses) as post_mock:
            client = GPTZeroClient(
                api_key="test-key",
                max_retries=3,
                retry_base_delay=0.01,
                retry_max_delay=0.02,
            )
            result = client.predict("hello world", delay=0.0)

        self.assertEqual(post_mock.call_count, 2)
        self.assertNotIn("error", result)
        self.assertEqual(result["predicted_class"], "ai")

    @patch("src.client.time.sleep", return_value=None)
    def test_does_not_retry_non_retryable_4xx(self, _sleep_mock: Mock) -> None:
        with patch(
            "src.client.requests.post",
            return_value=_mock_response(400, payload={"error": "bad request"}, text="bad request"),
        ) as post_mock:
            client = GPTZeroClient(api_key="test-key", max_retries=3)
            result = client.predict("hello world", delay=0.0)

        self.assertEqual(post_mock.call_count, 1)
        self.assertIn("error", result)
        self.assertEqual(result["status"], 400)
        self.assertEqual(result["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
