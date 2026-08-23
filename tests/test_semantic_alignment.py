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
    def test_aligns_data_quality_assessment_task_type_alias(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    task_type="data_quality_assessment",
                    explanation=(
                        "Task type: data_quality_assessment."
                    ),
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["task_type"],
            "experiment_diagnosis",
        )
        self.assertEqual(
            parsed.diagnosis["explanation"],
            "Task type: experiment_diagnosis.",
        )

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

    def test_aligns_no_detected_issue_primary_issue_alias(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    primary_issue="no_detected_issue",
                    explanation=(
                        "Primary issue: no_detected_issue."
                    ),
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["primary_issue"],
            "no_clear_issue",
        )
        self.assertEqual(
            parsed.diagnosis["explanation"],
            "Primary issue: no_clear_issue.",
        )

    def test_aligns_strong_class_imbalance_primary_issue_alias(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    primary_issue="strong_class_imbalance",
                    explanation=(
                        "Primary issue: strong_class_imbalance."
                    ),
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["primary_issue"],
            "class_imbalance",
        )
        self.assertEqual(
            parsed.diagnosis["explanation"],
            "Primary issue: class_imbalance.",
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

    def test_aligns_high_class_imbalance_evidence_alias(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    evidence_codes=["high_class_imbalance"],
                    explanation="Evidence: high_class_imbalance.",
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["evidence_codes"],
            ["strong_class_distribution_skew"],
        )
        self.assertEqual(
            parsed.diagnosis["explanation"],
            "Evidence: strong_class_distribution_skew.",
        )

    def test_high_class_imbalance_mapping_deduplicates_array(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    evidence_codes=[
                        "strong_class_distribution_skew",
                        "high_class_imbalance",
                    ]
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["evidence_codes"],
            ["strong_class_distribution_skew"],
        )

    def test_aligns_generalization_evidence_aliases(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    evidence_codes=[
                        "strong_generalization_gap",
                        "relative_generalization_gap",
                        "late_degradation",
                    ],
                    explanation=(
                        "relative_generalization_gap with "
                        "late_degradation."
                    ),
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["evidence_codes"],
            [
                "strong_generalization_gap",
                "late_validation_degradation",
            ],
        )
        self.assertEqual(
            parsed.diagnosis["explanation"],
            "strong_generalization_gap with "
            "late_validation_degradation.",
        )

    def test_aligns_healthy_evidence_alias_and_explanation(self):
        raw = json.dumps(
            _diagnosis(
                primary_issue="no_clear_issue",
                severity="low",
                evidence_codes=[
                    "all_primary_indicators_within_threshold"
                ],
                recommended_action_codes=["continue_monitoring"],
                explanation=(
                    "Evidence: "
                    "all_primary_indicators_within_threshold."
                ),
            )
        )

        parsed = parse_model_output(raw)

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["evidence_codes"],
            ["no_strong_diagnostic_rule_triggered"],
        )
        self.assertEqual(
            parsed.diagnosis["explanation"],
            "Evidence: no_strong_diagnostic_rule_triggered.",
        )
        self.assertEqual(parsed.raw_model_output, raw)
        self.assertIn(
            "all_primary_indicators_within_threshold",
            parsed.raw_model_output,
        )

    def test_aligns_action_alias_and_deduplicates_array(self):
        parsed = parse_model_output(
            json.dumps(
                _diagnosis(
                    recommended_action_codes=[
                        "inspect_generalization_gap",
                        "monitor_generalization",
                    ],
                    explanation="Action: monitor_generalization.",
                )
            )
        )

        self.assertTrue(parsed.is_valid)
        self.assertEqual(
            parsed.diagnosis["recommended_action_codes"],
            ["inspect_generalization_gap"],
        )
        self.assertEqual(
            parsed.diagnosis["explanation"],
            "Action: inspect_generalization_gap.",
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

    def test_new_aliases_leave_raw_model_output_unchanged(self):
        raw = json.dumps(
            _diagnosis(
                task_type="data_quality_assessment",
                primary_issue="no_detected_issue",
                evidence_codes=[
                    "relative_generalization_gap",
                    "late_degradation",
                ],
                recommended_action_codes=["monitor_generalization"],
                explanation=(
                    "data_quality_assessment found no_detected_issue with "
                    "relative_generalization_gap and late_degradation; "
                    "monitor_generalization."
                ),
            )
        )

        parsed = parse_model_output(raw)

        self.assertTrue(parsed.is_valid)
        self.assertEqual(parsed.raw_model_output, raw)
        self.assertIn("data_quality_assessment", parsed.raw_model_output)
        self.assertIn("no_detected_issue", parsed.raw_model_output)
        self.assertIn("relative_generalization_gap", parsed.raw_model_output)
        self.assertIn("late_degradation", parsed.raw_model_output)
        self.assertIn("monitor_generalization", parsed.raw_model_output)
        self.assertEqual(
            parsed.diagnosis["task_type"],
            "experiment_diagnosis",
        )
        self.assertEqual(
            parsed.diagnosis["primary_issue"],
            "no_clear_issue",
        )

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
