import unittest

from src.corpus_strata import infer_stratum


class CorpusStrataTest(unittest.TestCase):
    def test_infer_from_verified_human_provenance(self) -> None:
        row = {
            "sample_id": "holdout_human_01",
            "ground_truth_label": "human",
            "provenance": "Reddit archived 2022",
        }
        self.assertEqual(infer_stratum(row), "human_pre_llm")

    def test_infer_from_boundary_prefix(self) -> None:
        row = {"sample_id": "bound_030pct_human", "ground_truth_label": None}
        self.assertEqual(infer_stratum(row), "boundary_near")

    def test_infer_from_mixed_variant(self) -> None:
        row = {"sample_id": "style_axis_x", "style_variant": "mixed"}
        self.assertEqual(infer_stratum(row), "mixed_blended")


if __name__ == "__main__":
    unittest.main()
