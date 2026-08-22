"""Tests for the Training Log Analyzer adapter."""

import unittest

from src.tools.contracts import ToolResult
from src.tools.training_log_analyzer import analyze_training_log


class TrainingLogAnalyzerTests(unittest.TestCase):
    def test_normal_training_curves(self):
        result = analyze_training_log(
            {
                "epoch": [1, 2, 3, 4, 5],
                "train_loss": [1.2, 0.9, 0.7, 0.6, 0.5],
                "validation_loss": [1.1, 0.9, 0.8, 0.85, 0.95],
                "train_metric": [0.60, 0.72, 0.82, 0.89, 0.94],
                "validation_metric": [0.58, 0.69, 0.76, 0.78, 0.74],
                "metric_direction": "higher_is_better",
            }
        )

        self.assertEqual(result.status, "success")
        self.assertGreater(
            result.features["relative_generalization_gap"],
            0,
        )
        self.assertGreater(result.features["late_degradation"], 0)
        self.assertGreater(result.features["relative_improvement"], 0)
        self.assertEqual(result.provenance["epoch_count"], 5)

    def test_missing_required_field(self):
        result = analyze_training_log(
            {
                "epoch": [1, 2, 3],
                "train_metric": [0.60, 0.70, 0.80],
                "metric_direction": "higher_is_better",
            }
        )

        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.error)
        self.assertIn("validation_metric", result.error)

    def test_tool_result_contract(self):
        result = analyze_training_log(
            {
                "epoch": [1, 2, 3],
                "train_loss": [1.0, 0.8, 0.6],
                "validation_loss": [1.1, 0.9, 0.7],
            }
        )

        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tool_name, "training_log_analyzer")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.flags, {})
        self.assertEqual(
            result.provenance["module"],
            "src.evaluation.feature_calculator",
        )
        self.assertIsNone(result.error)

    def test_does_not_include_diagnosis_fields(self):
        result = analyze_training_log(
            {
                "epoch": [1, 2, 3],
                "train_loss": [1.0, 0.8, 0.7],
                "validation_loss": [1.1, 0.9, 0.85],
            }
        )

        prohibited = {
            "diagnosis",
            "primary_issue",
            "recommendation",
            "recommended_action_codes",
        }

        self.assertTrue(prohibited.isdisjoint(result.to_dict()))
        self.assertTrue(prohibited.isdisjoint(result.features))


if __name__ == "__main__":
    unittest.main()
