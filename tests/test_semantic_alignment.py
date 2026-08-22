"""Tests for deterministic diagnosis-output semantic alignment."""

import json
import unittest

from src.inference.output_parser import parse_model_output
from src.inference.semantic_alignment import align_model_output


def _diagnosis(**overrides):
    output = {
        "task_type": "experiment_diagnosis",
        "primary_issue": "class_imbalance",
        "severity": "medium",
        "evidence_codes": ["strong_class_distribution_skew"],
        "recommended_action_codes": ["use_class_weighting"],
        "explanation": "Class distribution is strongly skewed.",
    }
    output.update(overrides)
    return output


class SemanticAlignmentTests(unittest.TestCase):
    def test_aligns_confirmed_primary_issue_alias(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(primary_issue="class_imbalance_issue")
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["primary_issue"],
            "class_imbalance",
        )

    def test_synchronizes_exact_alias_in_explanation(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    primary_issue="class_imbalance_issue",
                    explanation=(
                        "Primary issue: class_imbalance_issue; "
                        "not class_imbalance_issue_extra."
                    ),
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["explanation"],
            "Primary issue: class_imbalance; "
            "not class_imbalance_issue_extra.",
        )

    def test_aligns_confirmed_evidence_alias(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    evidence_codes=[
                        "strong_class_distribution_skew",
                        "small_majority_class_f1",
                    ],
                    explanation=(
                        "Evidence: small_majority_class_f1."
                    ),
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["evidence_codes"],
            [
                "strong_class_distribution_skew",
                "large_class_performance_gap",
            ],
        )
        self.assertEqual(
            parsed.diagnosis["explanation"],
            "Evidence: large_class_performance_gap.",
        )

    def test_evidence_alias_mapping_deduplicates_array(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    evidence_codes=[
                        "large_class_performance_gap",
                        "small_majority_class_f1",
                    ]
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["evidence_codes"],
            ["large_class_performance_gap"],
        )

    def test_unknown_evidence_still_fails_validation(self):
        unknown_code = "unregistered_evidence_code"
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(evidence_codes=[unknown_code])
            )
        )

        self.assertFalse(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["evidence_codes"],
            [unknown_code],
        )
        self.assertIn(
            f"unknown evidence code: {unknown_code}",
            parsed.validation_errors,
        )

    def test_raw_model_output_remains_unchanged(self):
        raw = json.dumps(
            _diagnosis(
                primary_issue="class_imbalance_issue",
                evidence_codes=[
                    "large_class_performance_gap",
                    "small_majority_class_f1",
                ],
                explanation=(
                    "class_imbalance_issue with "
                    "small_majority_class_f1."
                ),
            )
        )

        parsed = parse_model_output(raw)

        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.raw_model_output, raw)
        self.assertIn("class_imbalance_issue", parsed.raw_model_output)
        self.assertIn("small_majority_class_f1", parsed.raw_model_output)

    def test_does_not_change_valid_output(self):
        output = _diagnosis()

        aligned = align_model_output(output)

        self.assertEqual(aligned, output)
        self.assertIsNot(aligned, output)

    def test_deduplicates_code_arrays_without_adding_values(self):
        output = _diagnosis(
            evidence_codes=[
                "strong_class_distribution_skew",
                "strong_class_distribution_skew",
            ],
            recommended_action_codes=[
                "use_class_weighting",
                "use_class_weighting",
            ],
        )

        aligned = align_model_output(output)

        self.assertEqual(
            aligned["evidence_codes"],
            ["strong_class_distribution_skew"],
        )
        self.assertEqual(
            aligned["recommended_action_codes"],
            ["use_class_weighting"],
        )


if __name__ == "__main__":
    unittest.main()
