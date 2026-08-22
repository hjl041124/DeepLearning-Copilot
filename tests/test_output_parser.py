"""Tests for diagnosis JSON extraction and validation."""

import json
import unittest

from src.inference.output_parser import parse_model_output


class OutputParserTests(unittest.TestCase):
    def test_extracts_and_validates_output(self):
        diagnosis = {
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
        raw = "Model output:\n" + json.dumps(diagnosis) + "\n"

        parsed = parse_model_output(raw)

        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.raw_model_output, raw)
        self.assertEqual(parsed.diagnosis, diagnosis)
        self.assertEqual(parsed.validation_errors, [])

    def test_invalid_output_is_not_repaired(self):
        diagnosis = {
            "task_type": "experiment_diagnosis",
            "primary_issue": "invented_issue",
            "severity": "medium",
            "evidence_codes": ["invented_evidence"],
            "recommended_action_codes": ["invented_action"],
            "explanation": "Invalid vocabulary test.",
        }
        raw = json.dumps(diagnosis)

        parsed = parse_model_output(raw)

        self.assertFalse(parsed.is_valid)
        self.assertEqual(parsed.raw_model_output, raw)
        self.assertEqual(
            parsed.diagnosis["primary_issue"],
            "invented_issue",
        )
        self.assertTrue(parsed.validation_errors)


if __name__ == "__main__":
    unittest.main()
