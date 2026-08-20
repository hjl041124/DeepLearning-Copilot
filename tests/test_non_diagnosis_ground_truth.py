import json
from pathlib import Path

from jsonschema import validate

from src.evaluation.non_diagnosis_ground_truth import (
    build_non_diagnosis_ground_truth,
)


ROOT = Path.cwd()

SCHEMA = json.loads(
    (
        ROOT
        / "configs"
        / "output_schema_v1.json"
    ).read_text(encoding="utf-8")
)

VOCAB = json.loads(
    (
        ROOT
        / "configs"
        / "output_vocabulary_v1.json"
    ).read_text(encoding="utf-8")
)


def validate_output(output):
    validate(
        instance=output,
        schema=SCHEMA
    )

    for code in output["evidence_codes"]:
        assert code in VOCAB["evidence_codes"], code

    for code in output[
        "recommended_action_codes"
    ]:
        assert code in VOCAB[
            "recommended_action_codes"
        ], code


def run_case(name, record, expected_evidence):
    output = build_non_diagnosis_ground_truth(
        record
    )

    validate_output(output)

    for code in expected_evidence:
        assert code in output[
            "evidence_codes"
        ]

    print(
        name,
        "->",
        output["task_type"],
        output["evidence_codes"]
    )


def main():
    run_case(
        "metric_accuracy_macro_gap",
        {
            "task_type": "metric_interpretation",
            "scenario_family_id":
                "MI_ACCURACY_MACRO_F1_GAP",
            "accuracy": 0.90,
            "macro_f1": 0.68,
            "class_counts": {
                "A": 1000,
                "B": 100,
                "C": 80
            }
        },
        [
            "large_accuracy_macro_f1_gap"
        ]
    )

    run_case(
        "metric_precision_recall",
        {
            "task_type": "metric_interpretation",
            "scenario_family_id":
                "MI_PRECISION_RECALL_TRADEOFF",
            "precision": 0.92,
            "recall": 0.60
        },
        [
            "precision_dominates_recall"
        ]
    )

    run_case(
        "metric_classwise_gap",
        {
            "task_type": "metric_interpretation",
            "scenario_family_id":
                "MI_CLASSWISE_PERFORMANCE_GAP",
            "per_class_metric": {
                "A": 0.91,
                "B": 0.55,
                "C": 0.83
            }
        },
        [
            "large_class_performance_gap"
        ]
    )

    run_case(
        "metric_train_validation_gap",
        {
            "task_type": "metric_interpretation",
            "scenario_family_id":
                "MI_TRAIN_VALIDATION_GAP",
            "train_metric": 0.90,
            "validation_metric": 0.70,
            "metric_direction":
                "higher_is_better"
        },
        [
            "strong_generalization_gap"
        ]
    )

    run_case(
        "model_clear_winner",
        {
            "task_type": "model_comparison",
            "scenario_family_id":
                "MC_CLEAR_QUALITY_WINNER",
            "primary_metric": "macro_f1",
            "metric_direction":
                "higher_is_better",
            "model_a_value": 0.90,
            "model_b_value": 0.82
        },
        [
            "model_a_higher_primary_metric"
        ]
    )

    run_case(
        "model_quality_efficiency_tradeoff",
        {
            "task_type": "model_comparison",
            "scenario_family_id":
                "MC_QUALITY_EFFICIENCY_TRADEOFF",
            "quality_metric": "macro_f1",
            "quality_direction":
                "higher_is_better",
            "model_a_quality": 0.91,
            "model_b_quality": 0.85,
            "model_a_latency_ms": 120.0,
            "model_b_latency_ms": 60.0
        },
        [
            "quality_efficiency_tradeoff",
            "model_b_lower_latency"
        ]
    )

    run_case(
        "model_imbalanced_metric",
        {
            "task_type": "model_comparison",
            "scenario_family_id":
                "MC_IMBALANCED_METRIC_COMPARISON",
            "model_a_accuracy": 0.88,
            "model_a_macro_f1": 0.82,
            "model_b_accuracy": 0.91,
            "model_b_macro_f1": 0.74
        },
        [
            "model_a_higher_macro_f1"
        ]
    )

    run_case(
        "model_no_clear_winner",
        {
            "task_type": "model_comparison",
            "scenario_family_id":
                "MC_NO_CLEAR_WINNER",
            "primary_metric": "macro_f1",
            "metric_direction":
                "higher_is_better",
            "model_a_value": 0.880,
            "model_b_value": 0.875
        },
        [
            "no_material_model_difference"
        ]
    )

    print(
        "NON-DIAGNOSIS GROUND TRUTH TESTS PASSED"
    )


if __name__ == "__main__":
    main()
