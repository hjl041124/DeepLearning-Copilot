from src.evaluation.ground_truth_builder import (
    build_ground_truth,
)


cases = [
    {
        "name": "overfitting",
        "record": {
            "metric_direction": "higher_is_better",
            "train_metric": 0.96,
            "validation_metric": 0.77,
            "validation_curve": [
                0.70, 0.79, 0.85, 0.80, 0.75
            ]
        },
        "expected_issue": "overfitting"
    },

    {
        "name": "optimization",
        "record": {
            "metric_direction": "lower_is_better",
            "training_curve": [
                2.0, 1.0, 2.1, 0.9, 2.2
            ]
        },
        "expected_issue": "optimization_problem"
    },

    {
        "name": "data_quality",
        "record": {
            "data_quality": {
                "label_noise_rate": 0.25
            }
        },
        "expected_issue": "data_quality_issue"
    },

    {
        "name": "healthy",
        "record": {
            "metric_direction": "higher_is_better",
            "train_metric": 0.89,
            "validation_metric": 0.87,
            "validation_curve": [
                0.80, 0.83, 0.85, 0.86, 0.87
            ]
        },
        "expected_issue": "no_clear_issue"
    }
]


for case in cases:

    result = build_ground_truth(
        case["record"]
    )

    output = result["ground_truth"]

    print("\nCASE:", case["name"])
    print("Issue:", output["primary_issue"])
    print("Severity:", output["severity"])
    print("Evidence:", output["evidence_codes"])
    print(
        "Actions:",
        output["recommended_action_codes"]
    )

    assert (
        output["primary_issue"]
        == case["expected_issue"]
    )

    assert output["recommended_action_codes"]

    assert output["explanation"]


print(
    "\nGROUND TRUTH BUILDER TESTS PASSED"
)
