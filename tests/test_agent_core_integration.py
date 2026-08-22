"""End-to-end Agent Core test with an injected mock QLoRA model."""

import json
import unittest

from src.agent.service import run_agent
from src.inference.qlora_diagnosis_model import QLoRADiagnosisModel


class MockQLoRAModel:
    def __init__(self):
        self.received_context = None

    def generate(self, combined_context):
        self.received_context = combined_context
        return json.dumps(
            {
                "task_type": "experiment_diagnosis",
                "primary_issue": "overfitting",
                "severity": "medium",
                "evidence_codes": [
                    "strong_generalization_gap",
                    "late_validation_degradation",
                ],
                "recommended_action_codes": [
                    "increase_regularization",
                    "use_early_stopping",
                ],
                "explanation": "Validation performance degrades.",
            }
        )


class AgentCoreIntegrationTests(unittest.TestCase):
    def test_full_workflow_with_mock_qlora(self):
        model = MockQLoRAModel()
        user_input = {
            "experiment_context": {
                "model_name": "demo-model",
                "dataset_name": "demo-dataset",
            },
            "metrics": {
                "accuracy": 0.82,
                "macro_f1": 0.75,
                "train_metric": 0.95,
                "validation_metric": 0.78,
                "metric_direction": "higher_is_better",
            },
            "training_log": {
                "epoch": [1, 2, 3, 4],
                "train_metric": [0.70, 0.82, 0.90, 0.95],
                "validation_metric": [0.68, 0.78, 0.81, 0.76],
                "metric_direction": "higher_is_better",
            },
            "dataset_info": {
                "class_counts": [100, 90],
                "per_class_metric": [0.80, 0.76],
                "missing_value_rate": 0.0,
            },
        }

        result = run_agent(
            "phase3-integration",
            user_input,
            diagnosis_model=model,
        )

        self.assertEqual(result["workflow_status"], "completed")
        self.assertIsNone(result["error"])
        self.assertIsNotNone(model.received_context)
        self.assertEqual(
            set(model.received_context),
            {
                "experiment_context",
                "metric_features",
                "log_features",
                "dataset_features",
                "provenance",
            },
        )
        self.assertEqual(
            result["diagnosis"]["primary_issue"],
            "overfitting",
        )
        self.assertEqual(result["validation_errors"], [])
        self.assertIn("Primary Issue: overfitting", result["report"])

    def test_real_wrapper_is_lazy(self):
        model = QLoRADiagnosisModel()

        self.assertFalse(model.is_loaded)


if __name__ == "__main__":
    unittest.main()
