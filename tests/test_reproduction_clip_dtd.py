import importlib.util
from pathlib import Path
import unittest

import torch
from torch import nn
from torch.nn import functional as F


spec = importlib.util.spec_from_file_location(
    "reproduce_clip_dtd", Path(__file__).resolve().parents[1] / "scripts/reproduce_clip_dtd.py")
reproduction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reproduction)


class ReproductionTests(unittest.TestCase):
    def test_prompt_features_are_normalized_before_averaging(self):
        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))

            def encode_text(self, tokens):
                return torch.tensor([[3., 0.], [0., 1.]])

        actual = reproduction.ensemble_text_features(
            FakeEncoder(), ["class"], ["a {}", "the {}"], lambda texts: torch.zeros(2, 1))
        expected = F.normalize(torch.ones(2), dim=0).unsqueeze(1)
        self.assertTrue(torch.allclose(actual, expected))

    def test_percentage_points_and_relative_error_are_distinct(self):
        comparison = reproduction.compare_accuracy(46., 44.5)
        self.assertEqual(comparison["absolute_gap_percentage_points"], 1.5)
        self.assertTrue(comparison["within_2_percentage_points"])
        self.assertFalse(comparison["within_2_percent_relative"])

    def test_duplicate_test_images_and_class_reordering_are_rejected(self):
        classes = [str(i) for i in range(47)]
        rows = [{"path": f"{c}/{j}.jpg", "label": c} for c in range(47) for j in range(40)]
        reproduction.validate_test_rows(classes, classes, rows, 1880)
        with self.assertRaisesRegex(ValueError, "ordering"):
            reproduction.validate_test_rows(classes[::-1], classes, rows, 1880)
        rows[-1] = rows[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            reproduction.validate_test_rows(classes, classes, rows, 1880)


if __name__ == "__main__":
    unittest.main()
