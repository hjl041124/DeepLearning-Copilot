"""Tests for the Dataset Checker adapter."""

import unittest

from src.tools.contracts import ToolResult
from src.tools.dataset_checker import check_dataset


class DatasetCheckerTests(unittest.TestCase):
    def test_normal_dataset_input(self):
        result = check_dataset(
            {
                "class_counts": [900, 90, 10],
                "per_class_metric": [0.90, 0.70, 0.40],
                "missing_value_rate": 0.02,
                "duplicate_rate": 0.10,
                "split_overlap_rate": 0.01,
            }
        )

        self.assertEqual(result.status, "success")
        self.assertAlmostEqual(
            result.features["class_imbalance_ratio"],
            10 / 900,
        )
        self.assertAlmostEqual(
            result.features["class_performance_gap"],
            0.50,
        )
        self.assertAlmostEqual(
            result.features["class_performance_ratio"],
            0.40 / 0.90,
        )
        self.assertEqual(result.features["missing_value_rate"], 0.02)
        self.assertEqual(result.features["duplicate_rate"], 0.10)
        self.assertEqual(result.features["split_overlap_rate"], 0.01)

    def test_missing_required_field(self):
        result = check_dataset({})

        self.assertEqual(result.status, "failed")
        self.assertIsNotNone(result.error)
        self.assertIn("supported dataset field", result.error)

    def test_tool_result_contract(self):
        result = check_dataset(
            {
                "class_counts": {
                    "class_a": 100,
                    "class_b": 50,
                }
            }
        )

        self.assertIsInstance(result, ToolResult)
        self.assertEqual(result.tool_name, "dataset_checker")
        self.assertEqual(result.status, "success")
        self.assertEqual(result.flags, {})
        self.assertEqual(
            result.provenance["module"],
            "src.evaluation.feature_calculator",
        )
        self.assertIsNone(result.error)

    def test_does_not_include_diagnosis_fields(self):
        result = check_dataset(
            {
                "class_counts": [100, 20],
                "missing_value_rate": 0.01,
            }
        )

        prohibited = {
            "diagnosis",
            "primary_issue",
            "recommendation",
            "recommended_action_codes",
            "label_noise_rate",
            "corrupted_sample_rate",
            "distribution_shift",
            "statistical_non_iid",
        }

        self.assertTrue(prohibited.isdisjoint(result.to_dict()))
        self.assertTrue(prohibited.isdisjoint(result.features))


if __name__ == "__main__":
    unittest.main()
