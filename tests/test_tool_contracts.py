"""Tests for the shared Agent tool result contract."""

import json
import unittest

from src.tools.contracts import ToolResult


class ToolResultTests(unittest.TestCase):
    def test_success_result(self):
        result = ToolResult.success(
            tool_name="metric_tool",
            features={"generalization_gap": 0.12},
            flags={"nan_detected": False},
            provenance={"source": "test_input"},
            warnings=["example warning"],
        )

        self.assertEqual(result.status, "success")
        self.assertEqual(
            result.features,
            {"generalization_gap": 0.12},
        )
        self.assertIsNone(result.error)
        self.assertEqual(
            set(result.to_dict()),
            {
                "tool_name",
                "status",
                "features",
                "flags",
                "provenance",
                "warnings",
                "error",
            },
        )

    def test_failed_result(self):
        result = ToolResult.failed(
            tool_name="dataset_checker",
            error="dataset input is unavailable",
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(
            result.error,
            "dataset input is unavailable",
        )

    def test_json_serialization(self):
        result = ToolResult.success(
            tool_name="log_analyzer",
            features={"late_degradation": 0.2},
            provenance={"epochs": 5},
        )

        serialized = json.dumps(result.to_dict())
        restored = json.loads(serialized)

        self.assertEqual(restored["tool_name"], "log_analyzer")
        self.assertEqual(restored["status"], "success")
        self.assertEqual(
            restored["features"],
            {"late_degradation": 0.2},
        )


if __name__ == "__main__":
    unittest.main()
