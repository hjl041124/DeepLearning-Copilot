import copy
import json
import math
from pathlib import Path

from src.evaluation.diagnosis_pipeline import (
    build_features,
)

from src.evaluation.ground_truth_builder import (
    build_ground_truth,
)

from src.evaluation.non_diagnosis_ground_truth import (
    build_non_diagnosis_ground_truth,
)


ROOT = Path.cwd()

THRESHOLD_PATH = (
    ROOT
    / "configs"
    / "threshold_bands_v1.json"
)


BOUNDARY_FAMILIES = {
    "HT_DIR_GENERALIZATION_GAP",
    "HT_DIR_CLASS_IMBALANCE_RATIO",
    "HT_DIR_LABEL_NOISE",
}


INVARIANCE_FAMILIES = {
    "HT_INV_OVERFITTING_METADATA",
    "HT_INV_DATA_QUALITY_METADATA",
    "HT_INV_MODEL_COMPARISON_METADATA",
}


PRIORITY_FAMILIES = {
    "HT_PC_DQ_OVERFITTING",
    "HT_PC_OPT_OVERFITTING",
    "HT_PC_CI_OVERFITTING",
    "HT_PC_DQ_OPTIMIZATION",
    "HT_PC_DQ_CLASS_IMBALANCE",
    "HT_PC_OPT_UNDERFITTING",
}


SUPPORTED_HARD_TEST_FAMILIES = (
    BOUNDARY_FAMILIES
    | INVARIANCE_FAMILIES
    | PRIORITY_FAMILIES
)


EXPECTED_PRIORITY = [
    "data_quality_issue",
    "optimization_problem",
    "class_imbalance",
    "overfitting",
    "underfitting",
    "no_clear_issue",
]


def _load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def _find_feature_config(
    obj,
    feature_name,
):
    if isinstance(obj, dict):
        if (
            feature_name in obj
            and isinstance(
                obj[feature_name],
                dict,
            )
        ):
            return obj[
                feature_name
            ]

        for value in obj.values():
            result = _find_feature_config(
                value,
                feature_name,
            )

            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = _find_feature_config(
                value,
                feature_name,
            )

            if result is not None:
                return result

    return None


def get_threshold(
    feature_name,
    threshold_key,
):
    config = _load_json(
        THRESHOLD_PATH
    )

    feature_config = (
        _find_feature_config(
            config,
            feature_name,
        )
    )

    if feature_config is None:
        raise KeyError(
            f"Cannot find threshold config: "
            f"{feature_name}"
        )

    if (
        threshold_key
        not in feature_config
    ):
        raise KeyError(
            f"{feature_name} has no "
            f"{threshold_key}"
        )

    return float(
        feature_config[
            threshold_key
        ]
    )


def neutral_data_quality():
    return {
        "label_noise_rate": 0.0,
        "duplicate_rate": 0.0,
        "missing_value_rate": 0.0,
        "corrupted_sample_rate": 0.0,
        "split_overlap_rate": 0.0,
    }


def neutral_flags():
    return {
        "nan_or_inf": False,
        "gradient_instability": False,
        "loss_divergence": False,
        "preprocessing_mismatch": False,
        "distribution_shift_detected": False,
    }


def neutral_diagnosis_record():
    return {
        "task_type":
            "experiment_diagnosis",
        "data_quality":
            neutral_data_quality(),
        "flags":
            neutral_flags(),
    }


def extract_ground_truth(result):
    if (
        isinstance(result, dict)
        and isinstance(
            result.get(
                "ground_truth"
            ),
            dict,
        )
    ):
        return result[
            "ground_truth"
        ]

    return result


def build_deterministic_ground_truth(
    record,
):
    task_type = record.get(
        "task_type",
        "experiment_diagnosis",
    )

    if (
        task_type
        == "experiment_diagnosis"
    ):
        return extract_ground_truth(
            build_ground_truth(
                record
            )
        )

    if task_type in {
        "metric_interpretation",
        "model_comparison",
    }:
        return (
            build_non_diagnosis_ground_truth(
                record
            )
        )

    raise ValueError(
        f"Unsupported task_type: "
        f"{task_type}"
    )


def decision_signature(
    ground_truth,
):
    return (
        ground_truth[
            "task_type"
        ],
        ground_truth[
            "primary_issue"
        ],
        ground_truth[
            "severity"
        ],
        tuple(
            ground_truth[
                "evidence_codes"
            ]
        ),
        tuple(
            ground_truth[
                "recommended_action_codes"
            ]
        ),
    )


# ============================================================
# Directional Boundary（方向性边界）
# ============================================================

def build_generalization_boundary_pair(
    epsilon,
):
    threshold = get_threshold(
        "relative_generalization_gap",
        "strong_min",
    )

    epsilon = float(
        epsilon
    )

    if not (
        0.0
        < epsilon
        < threshold
    ):
        raise ValueError(
            "epsilon must be positive "
            "and smaller than threshold"
        )

    below_gap = (
        threshold
        - epsilon
    )

    above_gap = (
        threshold
        + epsilon
    )

    train_metric = 0.90

    def make_record(
        target_gap,
    ):
        validation_metric = (
            train_metric
            * (
                1.0
                - target_gap
            )
        )

        record = (
            neutral_diagnosis_record()
        )

        record.update(
            {
                "metric_name":
                    "accuracy",

                "metric_direction":
                    "higher_is_better",

                "train_metric":
                    train_metric,

                "validation_metric":
                    validation_metric,

                "validation_curve": [
                    0.68,
                    0.76,
                    0.83,
                    0.88,
                    0.85,
                    0.81,
                    validation_metric,
                ],
            }
        )

        return record

    return {
        "threshold":
            threshold,

        "epsilon":
            epsilon,

        "below":
            make_record(
                below_gap
            ),

        "above":
            make_record(
                above_gap
            ),
    }


def build_class_imbalance_boundary_pair(
    epsilon,
):
    threshold = get_threshold(
        "class_imbalance_ratio",
        "strong_max",
    )

    epsilon = float(
        epsilon
    )

    if not (
        0.0
        < epsilon
        < threshold
    ):
        raise ValueError(
            "epsilon must be positive "
            "and smaller than threshold"
        )

    max_count = 1000

    below_target = (
        threshold
        - epsilon
    )

    above_target = (
        threshold
        + epsilon
    )

    below_min = max(
        1,
        math.floor(
            max_count
            * below_target
        ),
    )

    above_min = max(
        1,
        math.ceil(
            max_count
            * above_target
        ),
    )

    per_class_metric = [
        0.92,
        0.50,
        0.84,
    ]

    def make_record(
        minority_count,
    ):
        record = (
            neutral_diagnosis_record()
        )

        record.update(
            {
                "class_counts": [
                    max_count,
                    minority_count,
                    700,
                ],

                "per_class_metric":
                    list(
                        per_class_metric
                    ),
            }
        )

        return record

    return {
        "threshold":
            threshold,

        "epsilon":
            epsilon,

        "below":
            make_record(
                below_min
            ),

        "above":
            make_record(
                above_min
            ),
    }


def build_label_noise_boundary_pair(
    epsilon,
):
    threshold = get_threshold(
        "label_noise_rate",
        "strong_min",
    )

    epsilon = float(
        epsilon
    )

    if not (
        0.0
        < epsilon
        < threshold
    ):
        raise ValueError(
            "epsilon must be positive "
            "and smaller than threshold"
        )

    below_rate = (
        threshold
        - epsilon
    )

    above_rate = (
        threshold
        + epsilon
    )

    def make_record(
        rate,
    ):
        record = (
            neutral_diagnosis_record()
        )

        record[
            "data_quality"
        ][
            "label_noise_rate"
        ] = rate

        return record

    return {
        "threshold":
            threshold,

        "epsilon":
            epsilon,

        "below":
            make_record(
                below_rate
            ),

        "above":
            make_record(
                above_rate
            ),
    }


# ============================================================
# Invariance Distractor（不变性干扰）
# ============================================================

def strong_overfitting_record():
    record = (
        neutral_diagnosis_record()
    )

    record.update(
        {
            "metric_name":
                "accuracy",

            "metric_direction":
                "higher_is_better",

            "train_metric":
                0.95,

            "validation_metric":
                0.71,

            "validation_curve": [
                0.65,
                0.75,
                0.83,
                0.88,
                0.84,
                0.78,
                0.71,
            ],
        }
    )

    return record


def strong_data_quality_record():
    record = (
        neutral_diagnosis_record()
    )

    threshold = get_threshold(
        "label_noise_rate",
        "strong_min",
    )

    record[
        "data_quality"
    ][
        "label_noise_rate"
    ] = (
        threshold
        + 0.08
    )

    return record


def strong_model_comparison_record():
    return {
        "task_type":
            "model_comparison",

        "scenario_family_id":
            "MC_CLEAR_QUALITY_WINNER",

        "primary_metric":
            "macro_f1",

        "metric_direction":
            "higher_is_better",

        "model_a_value":
            0.91,

        "model_b_value":
            0.81,
    }


def add_irrelevant_metadata(
    record,
    metadata,
):
    output = copy.deepcopy(
        record
    )

    output[
        "metadata"
    ] = copy.deepcopy(
        metadata
    )

    return output


def build_invariance_pair(
    family_id,
    metadata,
):
    if (
        family_id
        == "HT_INV_OVERFITTING_METADATA"
    ):
        base = strong_overfitting_record()

    elif (
        family_id
        == "HT_INV_DATA_QUALITY_METADATA"
    ):
        base = strong_data_quality_record()

    elif (
        family_id
        == "HT_INV_MODEL_COMPARISON_METADATA"
    ):
        base = (
            strong_model_comparison_record()
        )

    else:
        raise ValueError(
            f"Unsupported invariance family: "
            f"{family_id}"
        )

    perturbed = (
        add_irrelevant_metadata(
            base,
            metadata,
        )
    )

    return {
        "base":
            base,

        "perturbed":
            perturbed,
    }


# ============================================================
# Priority Composition（优先级组合）
# ============================================================

def component_data_quality():
    return (
        strong_data_quality_record()
    )


def component_optimization():
    record = (
        neutral_diagnosis_record()
    )

    record[
        "flags"
    ][
        "nan_or_inf"
    ] = True

    return record


def component_class_imbalance():
    record = (
        neutral_diagnosis_record()
    )

    record.update(
        {
            "class_counts": [
                1200,
                50,
                700,
            ],

            "per_class_metric": [
                0.93,
                0.42,
                0.84,
            ],
        }
    )

    return record


def component_overfitting():
    return (
        strong_overfitting_record()
    )


def component_underfitting():
    record = (
        neutral_diagnosis_record()
    )

    record.update(
        {
            "metric_name":
                "accuracy",

            "metric_direction":
                "higher_is_better",

            "train_metric":
                0.59,

            "validation_metric":
                0.57,

            "reference_performance":
                0.88,
        }
    )

    return record


COMPONENT_BUILDERS = {
    "data_quality_issue":
        component_data_quality,

    "optimization_problem":
        component_optimization,

    "class_imbalance":
        component_class_imbalance,

    "overfitting":
        component_overfitting,

    "underfitting":
        component_underfitting,
}


PRIORITY_COMPOSITION_CONFIG = {
    "HT_PC_DQ_OVERFITTING": {
        "components": [
            "data_quality_issue",
            "overfitting",
        ],
        "expected":
            "data_quality_issue",
    },

    "HT_PC_OPT_OVERFITTING": {
        "components": [
            "optimization_problem",
            "overfitting",
        ],
        "expected":
            "optimization_problem",
    },

    "HT_PC_CI_OVERFITTING": {
        "components": [
            "class_imbalance",
            "overfitting",
        ],
        "expected":
            "class_imbalance",
    },

    "HT_PC_DQ_OPTIMIZATION": {
        "components": [
            "data_quality_issue",
            "optimization_problem",
        ],
        "expected":
            "data_quality_issue",
    },

    "HT_PC_DQ_CLASS_IMBALANCE": {
        "components": [
            "data_quality_issue",
            "class_imbalance",
        ],
        "expected":
            "data_quality_issue",
    },

    "HT_PC_OPT_UNDERFITTING": {
        "components": [
            "optimization_problem",
            "underfitting",
        ],
        "expected":
            "optimization_problem",
    },
}


def merge_diagnosis_records(
    records,
):
    output = (
        neutral_diagnosis_record()
    )

    for record in records:
        for key, value in (
            record.items()
        ):
            if key == "task_type":
                continue

            if key == "data_quality":
                for (
                    sub_key,
                    sub_value,
                ) in value.items():
                    if (
                        float(
                            sub_value
                        )
                        != 0.0
                    ):
                        output[
                            "data_quality"
                        ][
                            sub_key
                        ] = copy.deepcopy(
                            sub_value
                        )

                continue

            if key == "flags":
                for (
                    sub_key,
                    sub_value,
                ) in value.items():
                    if sub_value:
                        output[
                            "flags"
                        ][
                            sub_key
                        ] = True

                continue

            output[
                key
            ] = copy.deepcopy(
                value
            )

    return output


def build_priority_composition(
    family_id,
):
    if (
        family_id
        not in PRIORITY_COMPOSITION_CONFIG
    ):
        raise ValueError(
            f"Unsupported priority family: "
            f"{family_id}"
        )

    config = (
        PRIORITY_COMPOSITION_CONFIG[
            family_id
        ]
    )

    component_records = {}

    for issue in config[
        "components"
    ]:
        component_records[
            issue
        ] = (
            COMPONENT_BUILDERS[
                issue
            ]()
        )

    composed = (
        merge_diagnosis_records(
            list(
                component_records.values()
            )
        )
    )

    return {
        "family_id":
            family_id,

        "components":
            component_records,

        "composed":
            composed,

        "expected_primary_issue":
            config[
                "expected"
            ],
    }


def compute_diagnosis_features(
    record,
):
    return build_features(
        record
    )
