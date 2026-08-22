"""Tests for system-level Agent workflow evaluation."""

import unittest

from src.evaluation.agent_evaluation import evaluate_agent_runs
from src.tools.contracts import ToolResult


def _agent_result(
    *,
    status="completed",
    primary_issue="overfitting",
    validation_errors=None,
    metric_status="success",
    log_status="success",
    dataset_status="success",
):
    def tool_result(tool_name, tool_status):
        if tool_status == "success":
            return ToolResult.success(tool_name)
        return ToolResult.failed(tool_name, "mock failure")

    return {
        "workflow_status": status,
        "metric_tool_result": tool_result(
            "metric_analysis",
            metric_status,
        ),
        "log_tool_result": tool_result(
            "training_log_analyzer",
            log_status,
        ),
        "dataset_tool_result": tool_result(
            "dataset_checker",
            dataset_status,
        ),
        "validation_errors": list(validation_errors or []),
        "diagnosis": (
            {"primary_issue": primary_issue}
            if primary_issue is not None
            else None
        ),
    }


class AgentEvaluationTests(unittest.TestCase):
    def test_multiple_successful_agent_results(self):
        result = evaluate_agent_runs(
            [
                _agent_result(primary_issue="overfitting"),
                _agent_result(primary_issue="class_imbalance"),
            ]
        )

        workflow = result["workflow_success_rate"]
        self.assertEqual(workflow["total_runs"], 2)
        self.assertEqual(workflow["completed_runs"], 2)
        self.assertEqual(workflow["failed_runs"], 0)
        self.assertEqual(workflow["success_rate"], 1.0)
        self.assertEqual(
            result["diagnosis_distribution"],
            {"overfitting": 1, "class_imbalance": 1},
        )

    def test_includes_failed_runs(self):
        result = evaluate_agent_runs(
            [
                _agent_result(),
                _agent_result(
                    status="failed",
                    primary_issue=None,
                    validation_errors=["invalid model output"],
                ),
            ]
        )

        workflow = result["workflow_success_rate"]
        self.assertEqual(workflow["total_runs"], 2)
        self.assertEqual(workflow["completed_runs"], 1)
        self.assertEqual(workflow["failed_runs"], 1)
        self.assertEqual(workflow["success_rate"], 0.5)

    def test_counts_tool_execution_statuses(self):
        result = evaluate_agent_runs(
            [
                _agent_result(),
                _agent_result(
                    metric_status="failed",
                    dataset_status="failed",
                ),
            ]
        )

        tools = result["tool_execution_statistics"]
        self.assertEqual(
            tools["metric_analysis"],
            {
                "call_count": 2,
                "success_count": 1,
                "failed_count": 1,
            },
        )
        self.assertEqual(
            tools["training_log_analyzer"],
            {
                "call_count": 2,
                "success_count": 2,
                "failed_count": 0,
            },
        )
        self.assertEqual(
            tools["dataset_checker"],
            {
                "call_count": 2,
                "success_count": 1,
                "failed_count": 1,
            },
        )

    def test_calculates_validation_pass_rate(self):
        result = evaluate_agent_runs(
            [
                _agent_result(),
                _agent_result(validation_errors=["schema error"]),
                _agent_result(),
            ]
        )

        validation = result["validation_pass_rate"]
        self.assertEqual(validation["total_runs"], 3)
        self.assertEqual(validation["evaluated_runs"], 3)
        self.assertEqual(validation["passed_runs"], 2)
        self.assertEqual(validation["failed_runs"], 1)
        self.assertAlmostEqual(validation["pass_rate"], 2 / 3)

    def test_supports_sqlite_execution_record_shape(self):
        sqlite_records = [
            {
                "execution_id": "execution-1",
                "status": "completed",
                "tool_results": {
                    "metric_analysis": {
                        "tool_name": "metric_analysis",
                        "status": "success",
                    },
                    "dataset_checker": {
                        "tool_name": "dataset_checker",
                        "status": "failed",
                    },
                },
                "validation_errors": [],
                "diagnosis": {"primary_issue": "class_imbalance"},
            }
        ]

        result = evaluate_agent_runs(sqlite_records)

        self.assertEqual(
            result["workflow_success_rate"]["completed_runs"],
            1,
        )
        tools = result["tool_execution_statistics"]
        self.assertEqual(tools["metric_analysis"]["success_count"], 1)
        self.assertEqual(tools["dataset_checker"]["failed_count"], 1)
        self.assertEqual(
            result["diagnosis_distribution"],
            {"class_imbalance": 1},
        )

    def test_missing_validation_data_is_not_counted_as_pass(self):
        result = evaluate_agent_runs(
            [{"status": "completed", "tool_results": {}}]
        )

        validation = result["validation_pass_rate"]
        self.assertEqual(validation["evaluated_runs"], 0)
        self.assertEqual(validation["passed_runs"], 0)
        self.assertEqual(validation["not_recorded_runs"], 1)
        self.assertEqual(validation["pass_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
