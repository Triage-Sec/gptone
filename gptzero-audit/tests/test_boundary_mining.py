import unittest

from src.boundary_mining import (
    ai_formalize_variants,
    boundary_distance,
    contractionize,
    decontractionize,
    generate_directional_variants,
    humanize_variants,
    in_boundary_band,
)


class BoundaryMiningUtilsTest(unittest.TestCase):
    def test_contraction_round_trip_preserves_meaningful_tokens(self) -> None:
        text = "It is important that we do not ignore what they are saying."
        contracted = contractionize(text)
        self.assertIn("it's", contracted.lower())
        self.assertIn("don't", contracted.lower())
        decontracted = decontractionize(contracted)
        self.assertIn("it is", decontracted.lower())
        self.assertIn("do not", decontracted.lower())

    def test_directional_variant_generation_is_non_empty(self) -> None:
        text = "Furthermore, the implementation has demonstrated significant potential."
        to_human = humanize_variants(text)
        to_ai = ai_formalize_variants(text)
        self.assertTrue(to_human)
        self.assertTrue(to_ai)
        self.assertEqual(len(to_human), len(set(to_human)))
        self.assertEqual(len(to_ai), len(set(to_ai)))

    def test_generate_directional_variants_dispatch(self) -> None:
        text = "I think this works fine."
        self.assertTrue(generate_directional_variants(text, "toward_human"))
        self.assertTrue(generate_directional_variants(text, "toward_ai"))
        with self.assertRaises(ValueError):
            generate_directional_variants(text, "unknown")

    def test_boundary_helpers(self) -> None:
        self.assertAlmostEqual(boundary_distance(0.5), 0.0)
        self.assertAlmostEqual(boundary_distance(0.9), 0.4)
        self.assertTrue(in_boundary_band(0.5, 0.35, 0.65))
        self.assertFalse(in_boundary_band(0.9, 0.35, 0.65))


if __name__ == "__main__":
    unittest.main()
