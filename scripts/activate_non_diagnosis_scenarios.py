import json
from pathlib import Path


ROOT = Path.cwd()
PATH = ROOT / "configs" / "scenario_families_v1.json"


REQUIRED_INPUTS = {
    "MI_ACCURACY_MACRO_F1_GAP": [
        "accuracy",
        "macro_f1",
        "class_counts"
    ],
    "MI_PRECISION_RECALL_TRADEOFF": [
        "precision",
        "recall"
    ],
    "MI_CLASSWISE_PERFORMANCE_GAP": [
        "per_class_metric"
    ],
    "MI_TRAIN_VALIDATION_GAP": [
        "train_metric",
        "validation_metric",
        "metric_direction"
    ],
    "MC_CLEAR_QUALITY_WINNER": [
        "primary_metric",
        "metric_direction",
        "model_a_value",
        "model_b_value"
    ],
    "MC_QUALITY_EFFICIENCY_TRADEOFF": [
        "quality_metric",
        "quality_direction",
        "model_a_quality",
        "model_b_quality",
        "model_a_latency_ms",
        "model_b_latency_ms"
    ],
    "MC_IMBALANCED_METRIC_COMPARISON": [
        "model_a_accuracy",
        "model_a_macro_f1",
        "model_b_accuracy",
        "model_b_macro_f1"
    ],
    "MC_NO_CLEAR_WINNER": [
        "primary_metric",
        "metric_direction",
        "model_a_value",
        "model_b_value"
    ]
}


def main():
    data = json.loads(
        PATH.read_text(encoding="utf-8")
    )

    for task_type in [
        "metric_interpretation",
        "model_comparison"
    ]:
        task = data["task_types"][task_type]

        task["status"] = "ready_for_template_design"

        for scenario in task["scenario_families"]:
            scenario_id = scenario[
                "scenario_family_id"
            ]

            scenario["status"] = "ready"
            scenario["standard_set_allowed"] = True
            scenario["hard_test_only"] = False
            scenario["required_inputs"] = REQUIRED_INPUTS[
                scenario_id
            ]

            scenario[
                "ground_truth_strategy"
            ] = (
                "Use deterministic Python logic in "
                "src/evaluation/"
                "non_diagnosis_ground_truth.py."
            )

    PATH.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2
        ) + "\n",
        encoding="utf-8"
    )

    print(
        "NON-DIAGNOSIS SCENARIO ACTIVATION PASSED"
    )


if __name__ == "__main__":
    main()
