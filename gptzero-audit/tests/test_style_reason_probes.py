import unittest

from src.style_reason_probes import (
    AI_REASON_TAGS,
    HUMAN_REASON_TAGS,
    build_style_reason_probe_records,
)


class StyleReasonProbeTest(unittest.TestCase):
    def test_records_have_unique_ids_and_expected_variants(self) -> None:
        records = build_style_reason_probe_records()
        sample_ids = [str(r["sample_id"]) for r in records]
        self.assertEqual(len(sample_ids), len(set(sample_ids)))
        self.assertEqual(len(records), 24)

        for row in records:
            self.assertEqual(row["collection"], "attack_train")
            self.assertIn(row["style_variant"], {"ai", "human", "mixed"})
            self.assertTrue(str(row["text"]).strip())
            self.assertIsNone(row["ground_truth_label"])

    def test_reason_tag_coverage_matches_taxonomy(self) -> None:
        records = build_style_reason_probe_records()
        seen_tags = set()
        for row in records:
            for tag in row.get("ui_reason_tags", []):
                seen_tags.add(str(tag))

        self.assertTrue(set(AI_REASON_TAGS).issubset(seen_tags))
        self.assertTrue(set(HUMAN_REASON_TAGS).issubset(seen_tags))


if __name__ == "__main__":
    unittest.main()
