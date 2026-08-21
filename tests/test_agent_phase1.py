"""Minimal end-to-end tests for the Phase 1 Agent skeleton."""

import unittest

from src.agent.service import run_agent


class AgentPhase1Tests(unittest.TestCase):
    def test_fixed_workflow_completes(self):
        user_input = {
            "model_name": "demo-model",
            "metrics": {"train_accuracy": 0.95},
        }

        result = run_agent("experiment-phase1", user_input)

        self.assertEqual(result["workflow_status"], "completed")
        self.assertIsNone(result["error"])
        self.assertEqual(result["experiment_context"], user_input)
        self.assertEqual(
            result["diagnosis"],
            {
                "task_type": "experiment_diagnosis",
                "primary_issue": "overfitting",
                "severity": "medium",
                "evidence_codes": ["mock_evidence"],
                "recommended_action_codes": ["mock_action"],
                "explanation": "mock diagnosis",
            },
        )
        self.assertIn("mock_evidence", result["report"])
        self.assertIn("mock_action", result["report"])

    def test_invalid_input_finishes_as_failed(self):
        result = run_agent("", {})

        self.assertEqual(result["workflow_status"], "failed")
        self.assertIsNotNone(result["error"])
        self.assertIsNone(result["diagnosis"])
        self.assertIsNone(result["report"])


if __name__ == "__main__":
    unittest.main()
