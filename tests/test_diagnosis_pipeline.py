from src.evaluation.diagnosis_pipeline import diagnose_record


cases = [
    {
        "name": "overfitting_case",
        "record": {
            "metric_direction": "higher_is_better",
            "train_metric": 0.95,
            "validation_metric": 0.78,
            "validation_curve": [
                0.70, 0.78, 0.84, 0.81, 0.74
            ],
        },
        "expected": "overfitting",
    },

    {
        "name": "underfitting_case",
        "record": {
            "metric_direction": "higher_is_better",
            "train_metric": 0.63,
            "validation_metric": 0.61,
            "reference_performance": 0.85,
        },
        "expected": "underfitting",
    },

    {
        "name": "optimization_case",
        "record": {
            "metric_direction": "lower_is_better",
            "training_curve": [
                2.0, 1.0, 2.0, 1.0, 2.0
            ],
        },
        "expected": "optimization_problem",
    },

    {
        "name": "class_imbalance_case",
        "record": {
            "class_counts": [
                9000, 700, 300
            ],
            "per_class_metric": [
                0.94, 0.79, 0.43
            ],
            "accuracy": 0.92,
            "macro_f1": 0.67,
        },
        "expected": "class_imbalance",
    },

    {
        "name": "data_quality_case",
        "record": {
            "data_quality": {
                "label_noise_rate": 0.25
            }
        },
        "expected": "data_quality_issue",
    },

    {
        "name": "healthy_case",
        "record": {
            "metric_direction": "higher_is_better",
            "train_metric": 0.89,
            "validation_metric": 0.87,
            "validation_curve": [
                0.78, 0.82, 0.85, 0.86, 0.87
            ],
            "class_counts": [
                1000, 950, 1050
            ],
            "per_class_metric": [
                0.87, 0.85, 0.86
            ],
            "accuracy": 0.87,
            "macro_f1": 0.86,
        },
        "expected": "no_clear_issue",
    },
]


for case in cases:
    result = diagnose_record(case["record"])

    predicted = result["diagnosis"]["primary_issue"]

    print(
        f'{case["name"]}: '
        f'{predicted} '
        f'(expected={case["expected"]})'
    )

    assert predicted == case["expected"], {
        "case": case["name"],
        "result": result,
    }


print("\nEND-TO-END DIAGNOSIS TESTS PASSED")
