"""Tests for structured Agent context construction."""

import unittest

from src.agent.context_builder import build_context
from src.tools.contracts import ToolResult


class ContextBuilderTests(unittest.TestCase):
    def test_builds_required_context_sections(self):
        metric_result = ToolResult.success(
            "metric_analysis",
            features={"relative_generalization_gap": 0.2},
            provenance={"module": "metric_source"},
        )
        log_result = ToolResult.success(
            "training_log_analyzer",
            features={"late_degradation": 0.1},
            provenance={"module": "log_source"},
        )
        dataset_result = ToolResult.success(
            "dataset_checker",
            features={"class_imbalance_ratio": 0.5},
            provenance={"module": "dataset_source"},
        )

        context = build_context(
            metric_result,
            log_result,
            dataset_result,
            experiment_context={"model_name": "demo-model"},
        )

        self.assertEqual(
            set(context),
            {
                "experiment_context",
                "metric_features",
                "log_features",
                "dataset_features",
                "provenance",
            },
        )
        self.assertEqual(
            context["metric_features"],
            {"relative_generalization_gap": 0.2},
        )
        self.assertEqual(
            context["provenance"]["metric_analysis"]["status"],
            "success",
        )

        prohibited = {
            "diagnosis",
            "primary_issue",
            "recommendation",
            "recommended_action_codes",
        }
        self.assertTrue(prohibited.isdisjoint(context))

    def test_failed_tool_contributes_no_features(self):
        failed = ToolResult.failed("metric_analysis", "missing metrics")
        empty_log = ToolResult.success("training_log_analyzer")
        empty_dataset = ToolResult.success("dataset_checker")

        context = build_context(failed, empty_log, empty_dataset)

        self.assertEqual(context["metric_features"], {})
        self.assertEqual(
            context["provenance"]["metric_analysis"]["error"],
            "missing metrics",
        )


if __name__ == "__main__":
    unittest.main()
