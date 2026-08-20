import json
from pathlib import Path


path = Path(
    "configs/evaluation_metrics_v1.json"
)

data = json.loads(
    path.read_text(
        encoding="utf-8"
    )
)


required_standard = {
    "parse_success_rate",
    "json_schema_valid_rate",
    "core_exact_match_rate",
    "primary_issue_accuracy",
    "primary_issue_macro_f1",
    "severity_accuracy",
    "evidence_exact_set_accuracy",
    "evidence_micro_f1",
    "recommendation_exact_set_accuracy",
    "recommendation_micro_f1",
    "structural_hallucination_rate",
}


required_hard = {
    "hard_core_exact_match_rate",
    "directional_pair_success_rate",
    "invariance_consistency_rate",
    "invariance_correct_pair_rate",
    "invariance_violation_rate",
    "priority_composition_accuracy",
}


errors = []


if data["version"] != "1.0":
    errors.append(
        "version error"
    )


if set(data["standard_metrics"]) != required_standard:
    errors.append(
        "standard metrics mismatch"
    )


if set(data["hard_test_metrics"]) != required_hard:
    errors.append(
        "hard metrics mismatch"
    )


if data["policy"]["explanation_exact_match"]:
    errors.append(
        "explanation exact match should be false"
    )


if errors:
    print(
        "DAY4 EVALUATION METRIC SPEC VALIDATION FAILED"
    )

    for e in errors:
        print("-", e)

    raise SystemExit(1)


print(
    "DAY4 EVALUATION METRIC SPEC VALIDATION PASSED"
)

print(
    "Standard Metrics:",
    len(data["standard_metrics"])
)

print(
    "Hard Test Metrics:",
    len(data["hard_test_metrics"])
)

print(
    "Core Fields:",
    len(data["core_prediction_fields"])
)

print(
    "Deferred Metrics:",
    data["deferred_metrics"]
)
