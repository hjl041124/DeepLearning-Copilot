"""Tests for the Metric Analysis Tool adapter."""

import unittest

from src.tools.contracts import ToolResult
from src.tools.metric_analysis_tool import analyze_metrics


class MetricAnalysisToolTests(unittest.TestCase):
    def test_normal_metric_input(self):
        result = analyze_metrics(
            {
                "accuracy": 0.90,
                "macro_f1": 0.75,
                "train_metric": 0.90,
                "validation_metric": 0.80,
                "metric_direction": "higher_is_better",
                "per_class_metric": [0.90, 0.70, 0.50],
            }
        )

        self.assertEqual(result.status, "success")
        self.assertAlmostEqual(
            result.features["accuracy_macro_f1_gap"],
            0.15,
        )
        self.assertAlmostEqual(
            result.features["relative_generalization_gap"],
            0.10 / 0.90,
        )
        self.assertAlmostEqual(
            result.features["class_performance_gap"],
            0.40,
        )
        self.assertAlmostEqual(
            result.features["class_performance_ratio"],
            0.50 / 0.90,
        )

    def test_missing_required_field(self):
        result = analyze_metrics({"accuracy": 0.90})

        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.error)
        self.assertIn("macro_f1", result.error)

    def test_tool_result_contract(self):
        result = analyze_metrics(
            {
                "accuracy": 0.88,
                "macro_f1": 0.80,
            }
        )

        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tool_name, "metric_analysis")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.flags, {})
        self.assertEqual(
            result.provenance["module"],
            "src.evaluation.feature_calculator",
        )
        self.assertIsNone(result.error)


if __name__ == "__main__":
    unittest.main()
