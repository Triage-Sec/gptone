import importlib.util
import unittest
from pathlib import Path

import numpy as np


def _load_run_pattern_module():
    project_root = Path(__file__).resolve().parents[1]
    module_path = project_root / "scripts" / "run_pattern_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_pattern_benchmark_module", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rpb = _load_run_pattern_module()


class RunPatternBenchmarkInternalsTest(unittest.TestCase):
    def test_blend_objective_can_prioritize_transition_alignment(self) -> None:
        rows = [
            {"base_sample_id": "s1", "variant_order": 0, "stratum": "boundary_near"},
            {"base_sample_id": "s1", "variant_order": 1, "stratum": "boundary_near"},
        ]
        gptzero_probs = np.array([0.8, 0.7], dtype=float)
        reg_probs = np.array([0.8, 0.82], dtype=float)  # Wrong transition sign but low MAE.
        class_expected_probs = np.array([0.2, 0.1], dtype=float)  # Correct transition sign, much higher MAE.

        mae_only = rpb._fit_stratum_blend_weights(
            rows=rows,
            gptzero_probs=gptzero_probs,
            reg_probs=reg_probs,
            class_expected_probs=class_expected_probs,
            min_samples=1,
            fit_split="all",
            mae_weight=1.0,
            transition_weight=0.0,
        )
        transition_weighted = rpb._fit_stratum_blend_weights(
            rows=rows,
            gptzero_probs=gptzero_probs,
            reg_probs=reg_probs,
            class_expected_probs=class_expected_probs,
            min_samples=1,
            fit_split="all",
            mae_weight=0.1,
            transition_weight=1.0,
        )

        self.assertGreater(mae_only["default_alpha"], transition_weighted["default_alpha"])
        self.assertGreaterEqual(mae_only["default_alpha"], 0.9)

    def test_transition_delta_calibration_respects_allowlist_and_bounds(self) -> None:
        rows = [
            {"base_sample_id": "b1", "variant_order": 0, "stratum": "boundary_near", "perturbation_kind": "identity"},
            {"base_sample_id": "b1", "variant_order": 1, "stratum": "boundary_near", "perturbation_kind": "toward_ai"},
            {"base_sample_id": "h1", "variant_order": 0, "stratum": "human_pre_llm", "perturbation_kind": "identity"},
            {"base_sample_id": "h1", "variant_order": 1, "stratum": "human_pre_llm", "perturbation_kind": "toward_ai"},
        ]
        gptzero_probs = np.array([0.5, 0.6, 0.5, 0.1], dtype=float)
        raw_probs = np.array([0.5, 0.9, 0.5, 0.9], dtype=float)

        calibrator = rpb._fit_transition_delta_calibration(
            rows=rows,
            gptzero_probs=gptzero_probs,
            raw_probs=raw_probs,
            min_samples=1,
            fit_split="all",
            shrinkage=10.0,
            stratum_min_samples=1,
            allowed_strata=["boundary_near"],
            slope_bounds=(-0.5, 0.5),
            intercept_bounds=(-0.1, 0.1),
            low_support_shrinkage_multiplier=2.0,
        )

        by_stratum_kind = calibrator.get("by_stratum_kind", {})
        self.assertIn("boundary_near::toward_ai", by_stratum_kind)
        self.assertNotIn("human_pre_llm::toward_ai", by_stratum_kind)
        params = by_stratum_kind["boundary_near::toward_ai"]
        self.assertLessEqual(abs(float(params["slope"])), 0.5 + 1e-9)
        self.assertLessEqual(abs(float(params["intercept"])), 0.1 + 1e-9)

        adjusted = rpb._apply_transition_delta_calibration(
            rows=rows,
            raw_probs=raw_probs,
            calibrator=calibrator,
        )
        # Non-allowed stratum should remain unchanged.
        self.assertAlmostEqual(float(adjusted[3]), float(raw_probs[3]), places=9)

    def test_stratum_kind_gate_diagnostics_ranks_blockers(self) -> None:
        summary = {
            "human_conversational": {
                "toward_ai": {
                    "agreement_rate": 0.95,
                    "probability_mae": 0.07,
                    "transition_agreement": 0.75,
                    "sample_count": 120,
                }
            },
            "ai_instructional": {
                "toward_human": {
                    "agreement_rate": 0.99,
                    "probability_mae": 0.01,
                    "transition_agreement": 0.96,
                    "sample_count": 120,
                }
            },
        }
        diagnostics = rpb._build_stratum_kind_gate_diagnostics(
            stratum_kind_summary=summary,
            targets={
                "stratum_agreement_min": 0.97,
                "stratum_probability_mae_max": 0.03,
                "stratum_transition_agreement_min": 0.94,
                "stratum_min_samples": 30,
            },
        )
        ranking = diagnostics["ranking"]
        self.assertGreater(len(ranking), 1)
        self.assertEqual(ranking[0]["stratum"], "human_conversational")
        self.assertTrue(ranking[0]["blocking"])


if __name__ == "__main__":
    unittest.main()
