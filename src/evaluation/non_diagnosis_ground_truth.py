import json
from pathlib import Path


ROOT = Path.cwd()

RULES_PATH = ROOT / "configs" / "non_diagnosis_rules_v1.json"
THRESHOLD_PATH = ROOT / "configs" / "threshold_bands_v1.json"


def _load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _relative_difference(a, b):
    denominator = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denominator


def _find_feature_config(obj, feature_name):
    if isinstance(obj, dict):
        if feature_name in obj and isinstance(
            obj[feature_name],
            dict
        ):
            return obj[feature_name]

        for value in obj.values():
            result = _find_feature_config(
                value,
                feature_name
            )
            if result is not None:
                return result

    elif isinstance(obj, list):
        for value in obj:
            result = _find_feature_config(
                value,
                feature_name
            )
            if result is not None:
                return result

    return None


def _get_existing_strong_min(feature_name):
    thresholds = _load_json(THRESHOLD_PATH)

    config = _find_feature_config(
        thresholds,
        feature_name
    )

    if config is None:
        raise ValueError(
            f"Cannot find threshold config for "
            f"{feature_name}"
        )

    if "strong_min" not in config:
        raise ValueError(
            f"{feature_name} has no strong_min threshold"
        )

    return float(config["strong_min"])


def _base_output(
    task_type,
    evidence_codes,
    action_codes,
    explanation
):
    return {
        "task_type": task_type,
        "primary_issue": "not_applicable",
        "severity": "not_applicable",
        "evidence_codes": evidence_codes,
        "recommended_action_codes": action_codes,
        "explanation": explanation
    }


def _build_metric_interpretation(record):
    rules = _load_json(RULES_PATH)[
        "metric_interpretation"
    ]

    scenario_id = record["scenario_family_id"]

    if scenario_id == "MI_ACCURACY_MACRO_F1_GAP":
        accuracy = float(record["accuracy"])
        macro_f1 = float(record["macro_f1"])

        gap = max(
            accuracy - macro_f1,
            0.0
        )

        threshold = _get_existing_strong_min(
            "accuracy_macro_f1_gap"
        )

        if gap < threshold:
            raise ValueError(
                "MI_ACCURACY_MACRO_F1_GAP requires "
                f"accuracy_macro_f1_gap >= {threshold}"
            )

        return _base_output(
            "metric_interpretation",
            ["large_accuracy_macro_f1_gap"],
            ["evaluate_macro_metrics"],
            (
                f"Accuracy ({accuracy:.3f}) is materially higher "
                f"than macro-F1 ({macro_f1:.3f}). "
                "The aggregate accuracy may therefore hide "
                "uneven class-level performance."
            )
        )

    if scenario_id == "MI_PRECISION_RECALL_TRADEOFF":
        precision = float(record["precision"])
        recall = float(record["recall"])

        gap = abs(
            precision - recall
        )

        threshold = float(
            rules[
                "precision_recall_absolute_gap"
            ]["strong_min"]
        )

        if gap < threshold:
            raise ValueError(
                "MI_PRECISION_RECALL_TRADEOFF requires "
                f"|precision-recall| >= {threshold}"
            )

        if precision > recall:
            return _base_output(
                "metric_interpretation",
                ["precision_dominates_recall"],
                ["inspect_false_negatives"],
                (
                    f"Precision ({precision:.3f}) is substantially "
                    f"higher than recall ({recall:.3f}). "
                    "The model is comparatively selective and may "
                    "be missing a meaningful number of positives."
                )
            )

        return _base_output(
            "metric_interpretation",
            ["recall_dominates_precision"],
            ["inspect_false_positives"],
            (
                f"Recall ({recall:.3f}) is substantially higher "
                f"than precision ({precision:.3f}). "
                "The model captures positives more aggressively "
                "but may produce more false positives."
            )
        )

    if scenario_id == "MI_CLASSWISE_PERFORMANCE_GAP":
        per_class_metric = record[
            "per_class_metric"
        ]

        if not isinstance(
            per_class_metric,
            dict
        ) or len(per_class_metric) < 2:
            raise ValueError(
                "per_class_metric must contain at least 2 classes"
            )

        values = [
            float(value)
            for value in per_class_metric.values()
        ]

        gap = max(values) - min(values)

        threshold = _get_existing_strong_min(
            "class_performance_gap"
        )

        if gap < threshold:
            raise ValueError(
                "MI_CLASSWISE_PERFORMANCE_GAP requires "
                f"class_performance_gap >= {threshold}"
            )

        worst_class = min(
            per_class_metric,
            key=per_class_metric.get
        )

        best_class = max(
            per_class_metric,
            key=per_class_metric.get
        )

        return _base_output(
            "metric_interpretation",
            ["large_class_performance_gap"],
            ["inspect_worst_class"],
            (
                f"Class-wise performance differs substantially: "
                f"{best_class} is strongest while {worst_class} "
                "is weakest. Aggregate metrics should not be "
                "interpreted without inspecting per-class results."
            )
        )

    if scenario_id == "MI_TRAIN_VALIDATION_GAP":
        train_metric = float(
            record["train_metric"]
        )
        validation_metric = float(
            record["validation_metric"]
        )
        direction = record[
            "metric_direction"
        ]

        denominator = max(
            abs(train_metric),
            abs(validation_metric),
            1e-12
        )

        if direction == "higher_is_better":
            signed_gap = (
                train_metric - validation_metric
            ) / denominator

        elif direction == "lower_is_better":
            signed_gap = (
                validation_metric - train_metric
            ) / denominator

        else:
            raise ValueError(
                "metric_direction must be higher_is_better "
                "or lower_is_better"
            )

        threshold = _get_existing_strong_min(
            "relative_generalization_gap"
        )

        if signed_gap < threshold:
            raise ValueError(
                "MI_TRAIN_VALIDATION_GAP requires "
                f"relative_generalization_gap >= {threshold}"
            )

        return _base_output(
            "metric_interpretation",
            ["strong_generalization_gap"],
            ["inspect_generalization_gap"],
            (
                f"The direction-aware train-validation gap is "
                f"{signed_gap:.3f}, indicating materially better "
                "training-set performance than validation-set "
                "performance."
            )
        )

    raise ValueError(
        f"Unsupported metric_interpretation scenario: "
        f"{scenario_id}"
    )


def _better_model(
    model_a,
    model_b,
    direction
):
    if direction == "higher_is_better":
        return "a" if model_a > model_b else "b"

    if direction == "lower_is_better":
        return "a" if model_a < model_b else "b"

    raise ValueError(
        "metric_direction must be higher_is_better "
        "or lower_is_better"
    )


def _build_model_comparison(record):
    rules = _load_json(RULES_PATH)[
        "model_comparison"
    ]

    scenario_id = record["scenario_family_id"]

    if scenario_id == "MC_CLEAR_QUALITY_WINNER":
        metric_name = record[
            "primary_metric"
        ]
        direction = record[
            "metric_direction"
        ]

        model_a = float(
            record["model_a_value"]
        )
        model_b = float(
            record["model_b_value"]
        )

        margin = _relative_difference(
            model_a,
            model_b
        )

        threshold = float(
            rules[
                "clear_quality_relative_margin"
            ]["strong_min"]
        )

        if margin < threshold:
            raise ValueError(
                "MC_CLEAR_QUALITY_WINNER requires "
                f"relative difference >= {threshold}"
            )

        winner = _better_model(
            model_a,
            model_b,
            direction
        )

        if winner == "a":
            evidence = [
                "model_a_higher_primary_metric"
            ]
            action = [
                "prefer_model_a"
            ]
            winner_name = "Model A"
        else:
            evidence = [
                "model_b_higher_primary_metric"
            ]
            action = [
                "prefer_model_b"
            ]
            winner_name = "Model B"

        return _base_output(
            "model_comparison",
            evidence,
            action,
            (
                f"{winner_name} is the clear winner on the "
                f"specified primary metric {metric_name}. "
                f"The relative difference is {margin:.3f}."
            )
        )

    if scenario_id == "MC_QUALITY_EFFICIENCY_TRADEOFF":
        direction = record[
            "quality_direction"
        ]

        model_a_quality = float(
            record["model_a_quality"]
        )
        model_b_quality = float(
            record["model_b_quality"]
        )

        model_a_latency = float(
            record["model_a_latency_ms"]
        )
        model_b_latency = float(
            record["model_b_latency_ms"]
        )

        quality_margin = _relative_difference(
            model_a_quality,
            model_b_quality
        )

        latency_margin = _relative_difference(
            model_a_latency,
            model_b_latency
        )

        quality_threshold = float(
            rules[
                "quality_efficiency_quality_margin"
            ]["strong_min"]
        )

        latency_threshold = float(
            rules[
                "quality_efficiency_latency_margin"
            ]["strong_min"]
        )

        if quality_margin < quality_threshold:
            raise ValueError(
                "Quality difference is not strong enough "
                "for MC_QUALITY_EFFICIENCY_TRADEOFF"
            )

        if latency_margin < latency_threshold:
            raise ValueError(
                "Latency difference is not strong enough "
                "for MC_QUALITY_EFFICIENCY_TRADEOFF"
            )

        quality_winner = _better_model(
            model_a_quality,
            model_b_quality,
            direction
        )

        latency_winner = (
            "a"
            if model_a_latency < model_b_latency
            else "b"
        )

        if quality_winner == latency_winner:
            raise ValueError(
                "MC_QUALITY_EFFICIENCY_TRADEOFF requires "
                "different quality and latency winners"
            )

        evidence = [
            "quality_efficiency_tradeoff"
        ]

        if quality_winner == "a":
            evidence.append(
                "model_a_higher_primary_metric"
            )
        else:
            evidence.append(
                "model_b_higher_primary_metric"
            )

        if latency_winner == "a":
            evidence.append(
                "model_a_lower_latency"
            )
        else:
            evidence.append(
                "model_b_lower_latency"
            )

        return _base_output(
            "model_comparison",
            evidence,
            ["choose_by_deployment_constraint"],
            (
                "The models present a clear quality-efficiency "
                "tradeoff: one model has stronger predictive "
                "quality while the other has lower latency. "
                "The final choice depends on deployment constraints."
            )
        )

    if scenario_id == "MC_IMBALANCED_METRIC_COMPARISON":
        model_a_macro_f1 = float(
            record["model_a_macro_f1"]
        )
        model_b_macro_f1 = float(
            record["model_b_macro_f1"]
        )

        margin = abs(
            model_a_macro_f1
            - model_b_macro_f1
        )

        threshold = float(
            rules[
                "imbalanced_macro_f1_absolute_margin"
            ]["strong_min"]
        )

        if margin < threshold:
            raise ValueError(
                "MC_IMBALANCED_METRIC_COMPARISON requires "
                f"macro-F1 difference >= {threshold}"
            )

        if model_a_macro_f1 > model_b_macro_f1:
            evidence = [
                "model_a_higher_macro_f1"
            ]
            actions = [
                "prioritize_macro_f1",
                "prefer_model_a"
            ]
            winner = "Model A"
        else:
            evidence = [
                "model_b_higher_macro_f1"
            ]
            actions = [
                "prioritize_macro_f1",
                "prefer_model_b"
            ]
            winner = "Model B"

        return _base_output(
            "model_comparison",
            evidence,
            actions,
            (
                f"{winner} has materially higher macro-F1. "
                "For an imbalance-aware evaluation objective, "
                "macro-F1 should be prioritized over accuracy alone."
            )
        )

    if scenario_id == "MC_NO_CLEAR_WINNER":
        direction = record[
            "metric_direction"
        ]

        model_a = float(
            record["model_a_value"]
        )
        model_b = float(
            record["model_b_value"]
        )

        if direction not in {
            "higher_is_better",
            "lower_is_better"
        }:
            raise ValueError(
                "metric_direction must be higher_is_better "
                "or lower_is_better"
            )

        margin = _relative_difference(
            model_a,
            model_b
        )

        threshold = float(
            rules[
                "no_clear_winner_relative_margin"
            ]["normal_max"]
        )

        if margin > threshold:
            raise ValueError(
                "MC_NO_CLEAR_WINNER requires "
                f"relative difference <= {threshold}"
            )

        return _base_output(
            "model_comparison",
            ["no_material_model_difference"],
            ["collect_more_evaluation_evidence"],
            (
                f"The relative difference between the models is "
                f"only {margin:.3f}. There is no clear winner on "
                "the supplied primary metric."
            )
        )

    raise ValueError(
        f"Unsupported model_comparison scenario: "
        f"{scenario_id}"
    )


def build_non_diagnosis_ground_truth(record):
    task_type = record.get("task_type")

    if task_type == "metric_interpretation":
        return _build_metric_interpretation(
            record
        )

    if task_type == "model_comparison":
        return _build_model_comparison(
            record
        )

    raise ValueError(
        "build_non_diagnosis_ground_truth only supports "
        "metric_interpretation and model_comparison"
    )
