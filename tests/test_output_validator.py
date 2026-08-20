from src.evaluation.output_validator import validate_output


valid_case = {
    "task_type": "experiment_diagnosis",
    "primary_issue": "overfitting",
    "severity": "high",
    "evidence_codes": [
        "strong_generalization_gap",
        "late_validation_degradation"
    ],
    "recommended_action_codes": [
        "increase_regularization",
        "use_early_stopping"
    ],
    "explanation": "Training performance is substantially better than validation performance."
}

errors = validate_output(valid_case)
assert errors == [], errors


invalid_code_case = {
    "task_type": "experiment_diagnosis",
    "primary_issue": "overfitting",
    "severity": "high",
    "evidence_codes": [
        "invented_evidence"
    ],
    "recommended_action_codes": [
        "invented_action"
    ],
    "explanation": "Invalid vocabulary test."
}

errors = validate_output(invalid_code_case)
assert len(errors) == 2, errors


invalid_task_case = {
    "task_type": "metric_interpretation",
    "primary_issue": "overfitting",
    "severity": "high",
    "evidence_codes": [],
    "recommended_action_codes": [],
    "explanation": "This should fail."
}

errors = validate_output(invalid_task_case)
assert errors, "Expected schema validation failure."


extra_field_case = {
    "task_type": "experiment_diagnosis",
    "primary_issue": "no_clear_issue",
    "severity": "low",
    "evidence_codes": [
        "no_strong_diagnostic_rule_triggered"
    ],
    "recommended_action_codes": [
        "continue_monitoring"
    ],
    "explanation": "No strong diagnostic signal was detected.",
    "fake_metric": 0.99
}

errors = validate_output(extra_field_case)
assert errors, "Extra fields must be rejected."


print("OUTPUT SCHEMA VALIDATION TESTS PASSED")
