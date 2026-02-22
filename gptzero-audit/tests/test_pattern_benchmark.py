import random
import unittest

from src.pattern_benchmark import (
    generate_variants,
    select_boundary_variants,
    summarize_prediction_rows,
    summarize_prediction_rows_by_stratum_kind,
)


class PatternBenchmarkTest(unittest.TestCase):
    def test_generate_variants_includes_base(self) -> None:
        text = "I like this project. It has clear goals. We should test edge cases too."
        variants = generate_variants(text, rng=random.Random(7), max_variants=6)
        self.assertGreaterEqual(len(variants), 1)
        self.assertEqual(variants[0]["variant_id"], "base")
        self.assertEqual(variants[0]["perturbation_kind"], "identity")
        texts = [v["text"] for v in variants]
        self.assertEqual(len(texts), len(set(texts)))

    def test_summarize_prediction_rows(self) -> None:
        rows = [
            {
                "base_sample_id": "s1",
                "stratum": "human_pre_llm",
                "variant_order": 0,
                "gptzero_prob": 0.2,
                "surrogate_prob": 0.2,
                "match": True,
            },
            {
                "base_sample_id": "s1",
                "stratum": "human_pre_llm",
                "variant_order": 1,
                "gptzero_prob": 0.4,
                "surrogate_prob": 0.38,
                "match": True,
            },
            {
                "base_sample_id": "s2",
                "stratum": "ai_instructional",
                "variant_order": 0,
                "gptzero_prob": 0.8,
                "surrogate_prob": 0.7,
                "match": False,
            },
            {
                "base_sample_id": "s2",
                "stratum": "ai_instructional",
                "variant_order": 1,
                "gptzero_prob": 0.7,
                "surrogate_prob": 0.65,
                "match": True,
            },
        ]
        summary = summarize_prediction_rows(rows)
        self.assertIn("overall", summary)
        self.assertIn("strata", summary)
        self.assertAlmostEqual(summary["overall"]["agreement_rate"], 0.75, places=6)
        self.assertEqual(summary["base_sample_count"], 2)
        self.assertIn("human_pre_llm", summary["strata"])
        self.assertIn("ai_instructional", summary["strata"])

    def test_select_boundary_variants_prefers_boundary_and_diversity(self) -> None:
        variants = [
            {"variant_id": "base", "perturbation_kind": "identity", "text": "a"},
            {"variant_id": "toward_human_0", "perturbation_kind": "toward_human", "text": "b"},
            {"variant_id": "toward_ai_0", "perturbation_kind": "toward_ai", "text": "c"},
            {"variant_id": "shuffle_sentences_1", "perturbation_kind": "shuffle_sentences", "text": "d"},
            {"variant_id": "case_jitter_2", "perturbation_kind": "case_jitter", "text": "e"},
        ]
        probs = [0.9, 0.55, 0.92, 0.48, 0.1]
        picked = select_boundary_variants(variants, probs, target_count=3)
        self.assertEqual(picked[0]["variant_id"], "base")
        picked_ids = {row["variant_id"] for row in picked}
        picked_probs = [probs[variants.index(row)] for row in picked]
        # Should include one below-base and one above-base candidate.
        self.assertTrue(any(p < probs[0] for p in picked_probs[1:]))
        self.assertTrue(any(p > probs[0] for p in picked_probs[1:]))

    def test_select_boundary_variants_applies_kind_quotas(self) -> None:
        variants = [
            {"variant_id": "base", "perturbation_kind": "identity", "text": "a"},
            {"variant_id": "toward_human_0", "perturbation_kind": "toward_human", "text": "b"},
            {"variant_id": "toward_human_1", "perturbation_kind": "toward_human", "text": "c"},
            {"variant_id": "toward_ai_0", "perturbation_kind": "toward_ai", "text": "d"},
            {"variant_id": "toward_ai_1", "perturbation_kind": "toward_ai", "text": "e"},
            {"variant_id": "shuffle_sentences_0", "perturbation_kind": "shuffle_sentences", "text": "f"},
        ]
        probs = [0.8, 0.45, 0.42, 0.88, 0.91, 0.5]
        picked = select_boundary_variants(
            variants,
            probs,
            target_count=5,
            kind_min_counts={"toward_human": 1, "toward_ai": 2},
        )
        picked_kinds = [row["perturbation_kind"] for row in picked]
        self.assertGreaterEqual(picked_kinds.count("toward_human"), 1)
        self.assertGreaterEqual(picked_kinds.count("toward_ai"), 2)

    def test_summarize_prediction_rows_by_stratum_kind(self) -> None:
        rows = [
            {
                "base_sample_id": "s1",
                "stratum": "human_pre_llm",
                "perturbation_kind": "identity",
                "variant_order": 0,
                "gptzero_prob": 0.20,
                "surrogate_prob": 0.22,
                "match": True,
            },
            {
                "base_sample_id": "s1",
                "stratum": "human_pre_llm",
                "perturbation_kind": "toward_ai",
                "variant_order": 1,
                "gptzero_prob": 0.40,
                "surrogate_prob": 0.38,
                "match": True,
            },
            {
                "base_sample_id": "s2",
                "stratum": "human_pre_llm",
                "perturbation_kind": "identity",
                "variant_order": 0,
                "gptzero_prob": 0.70,
                "surrogate_prob": 0.68,
                "match": True,
            },
            {
                "base_sample_id": "s2",
                "stratum": "human_pre_llm",
                "perturbation_kind": "toward_ai",
                "variant_order": 1,
                "gptzero_prob": 0.60,
                "surrogate_prob": 0.71,
                "match": False,
            },
        ]
        summary = summarize_prediction_rows_by_stratum_kind(rows)
        self.assertIn("human_pre_llm", summary)
        self.assertIn("toward_ai", summary["human_pre_llm"])
        toward_ai = summary["human_pre_llm"]["toward_ai"]
        self.assertEqual(toward_ai["sample_count"], 2.0)
        # First sample has same transition sign, second differs.
        self.assertAlmostEqual(toward_ai["transition_agreement"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
