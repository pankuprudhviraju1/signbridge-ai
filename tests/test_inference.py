import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from signbridge import TemporalRecognizer, normalize_landmarks


class InferenceTests(unittest.TestCase):
    def test_normalization_is_translation_and_scale_invariant(self):
        base = [(float(i), float(i % 4), 0.0) for i in range(21)]
        shifted = [(x * 3 + 10, y * 3 - 7, z) for x, y, z in base]
        first, second = normalize_landmarks(base), normalize_landmarks(shifted)
        for point_a, point_b in zip(first, second):
            for a, b in zip(point_a, point_b):
                self.assertAlmostEqual(a, b)

    def test_rejects_low_confidence(self):
        result = TemporalRecognizer().update({"HELLO": 0.51, "THANKS": 0.49})
        self.assertEqual(result.label, "UNKNOWN")

    def test_requires_temporal_consensus(self):
        model = TemporalRecognizer(window=3, consensus=2 / 3)
        model.update({"HELLO": 0.9})
        model.update({"THANKS": 0.9})
        result = model.update({"HELLO": 0.8})
        self.assertTrue(result.stable)
        self.assertEqual(result.label, "HELLO")

    def test_validation(self):
        with self.assertRaises(ValueError):
            normalize_landmarks([(0, 0, 0)])
        with self.assertRaises(ValueError):
            TemporalRecognizer(window=0)


if __name__ == "__main__":
    unittest.main()
