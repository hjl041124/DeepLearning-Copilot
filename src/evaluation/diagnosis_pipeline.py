from typing import Any, Dict

from src.evaluation.feature_calculator import (
    relative_generalization_gap,
    late_degradation,
    relative_improvement,
    plateau_streak,
    oscillation_score,
    relative_amplitude,
    class_imbalance_ratio,
    class_performance_gap,
    class_performance_ratio,
    accuracy_macro_f1_gap,
    reference_performance_gap,
)

from src.evaluation.rule_engine import diagnose_experiment


def build_features(record: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert raw experiment data into deterministic numeric features
    （將原始實驗資料轉換為確定性數值特徵）.
    """

    features: Dict[str, float] = {}

    direction = record.get(
        "metric_direction",
        "higher_is_better",
    )

    # Generalization（泛化）
    if "train_metric" in record and "validation_metric" in record:
        features["relative_generalization_gap"] = (
            relative_generalization_gap(
                record["train_metric"],
                record["validation_metric"],
                direction,
            )
        )

    # Validation trend（驗證趨勢）
    if "validation_curve" in record:
        features["late_degradation"] = late_degradation(
            record["validation_curve"],
            direction,
        )

    # Training trend（訓練趨勢）
    if "training_curve" in record:
        features["relative_improvement"] = relative_improvement(
            record["training_curve"],
            direction,
        )

        features["plateau_streak"] = plateau_streak(
            record["training_curve"],
            direction,
        )

        features["oscillation_score"] = oscillation_score(
            record["training_curve"]
        )

        features["relative_amplitude"] = relative_amplitude(
            record["training_curve"]
        )

    # Underfitting reference（欠擬合參考）
    if (
        "train_metric" in record
        and "reference_performance" in record
    ):
        features["reference_performance_gap"] = (
            reference_performance_gap(
                record["train_metric"],
                record["reference_performance"],
                direction,
            )
        )

    # Class imbalance（類別不平衡）
    if "class_counts" in record:
        features["class_imbalance_ratio"] = (
            class_imbalance_ratio(
                record["class_counts"]
            )
        )

    if "per_class_metric" in record:
        features["class_performance_gap"] = (
            class_performance_gap(
                record["per_class_metric"]
            )
        )

        features["class_performance_ratio"] = (
            class_performance_ratio(
                record["per_class_metric"]
            )
        )

    if "accuracy" in record and "macro_f1" in record:
        features["accuracy_macro_f1_gap"] = (
            accuracy_macro_f1_gap(
                record["accuracy"],
                record["macro_f1"],
            )
        )

    # Data quality（資料品質）
    data_quality = record.get("data_quality", {})

    for key in [
        "label_noise_rate",
        "duplicate_rate",
        "missing_value_rate",
        "corrupted_sample_rate",
        "split_overlap_rate",
    ]:
        if key in data_quality:
            features[key] = data_quality[key]

    return features


def diagnose_record(
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Full deterministic diagnosis pipeline
    （完整確定性診斷流程）.
    """

    features = build_features(record)

    diagnosis = diagnose_experiment(
        features=features,
        flags=record.get("flags", {}),
    )

    return {
        "features": features,
        "diagnosis": diagnosis,
    }
