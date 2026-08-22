"""Tests for the Agent CLI demo without loading the real QLoRA model."""

from pathlib import Path
import unittest

from scripts.demo_agent import (
    format_diagnosis_output,
    load_demo_input,
    run_demo,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_INPUT = PROJECT_ROOT / "examples" / "demo_experiment.json"


def _mock_agent_result(experiment_id):
    return {
        "experiment_id": experiment_id,
        "workflow_status": "completed",
        "error": None,
        "diagnosis": {
            "task_type": "experiment_diagnosis",
            "primary_issue": "class_imbalance",
            "severity": "medium",
            "evidence_codes": [
                "strong_class_distribution_skew",
                "large_class_performance_gap",
            ],
            "recommended_action_codes": [
                "use_class_weighting",
                "use_balanced_sampling",
            ],
            "explanation": "Class performance differs across classes.",
        },
    }


class DemoAgentTests(unittest.TestCase):
    def test_loads_complete_demo_input(self):
        payload = load_demo_input(DEMO_INPUT)

        self.assertEqual(
            payload["experiment_id"],
            "demo-cifar10-resnet18-001",
        )
        self.assertEqual(
            set(payload).intersection(
                {
                    "experiment_context",
                    "metrics",
                    "training_log",
                    "dataset_info",
                }
            ),
            {
                "experiment_context",
                "metrics",
                "training_log",
                "dataset_info",
            },
        )

    def test_calls_agent_interface_with_loaded_input(self):
        calls = []

        def mock_runner(experiment_id, user_input):
            calls.append((experiment_id, user_input))
            return _mock_agent_result(experiment_id)

        output = run_demo(DEMO_INPUT, diagnosis_runner=mock_runner)

        self.assertEqual(len(calls), 1)
        experiment_id, user_input = calls[0]
        self.assertEqual(
            experiment_id,
            "demo-cifar10-resnet18-001",
        )
        self.assertNotIn("experiment_id", user_input)
        self.assertEqual(
            set(user_input),
            {
                "experiment_context",
                "metrics",
                "training_log",
                "dataset_info",
            },
        )
        self.assertIn("Primary Issue: class_imbalance", output)

    def test_formats_required_output_fields(self):
        output = format_diagnosis_output(
            _mock_agent_result("format-test")
        )

        self.assertIn("Experiment ID: format-test", output)
        self.assertIn("Task Type: experiment_diagnosis", output)
        self.assertIn("Primary Issue: class_imbalance", output)
        self.assertIn("Severity: medium", output)
        self.assertIn(
            "Evidence Codes: strong_class_distribution_skew, "
            "large_class_performance_gap",
            output,
        )
        self.assertIn(
            "Recommended Action Codes: use_class_weighting, "
            "use_balanced_sampling",
            output,
        )
        self.assertIn(
            "Explanation: Class performance differs across classes.",
            output,
        )


if __name__ == "__main__":
    unittest.main()
