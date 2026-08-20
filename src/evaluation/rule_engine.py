import json
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
THRESHOLD_PATH = ROOT / "configs" / "threshold_bands_v1.json"

with THRESHOLD_PATH.open(encoding="utf-8") as f:
    T = json.load(f)


def diagnose_experiment(
    features: Dict[str, float],
    flags: Dict[str, bool] | None = None,
) -> Dict[str, Any]:

    flags = flags or {}

    # ==================================================
    # 1. Data quality issue（資料品質問題）
    # ==================================================

    dq_cfg = T["data_quality"]

    checks = {
        "high_label_noise":
            features.get("label_noise_rate", 0.0)
            >= dq_cfg["label_noise_rate"]["strong_min"],

        "high_duplicate_rate":
            features.get("duplicate_rate", 0.0)
            >= dq_cfg["duplicate_rate"]["strong_min"],

        "high_missing_value_rate":
            features.get("missing_value_rate", 0.0)
            >= dq_cfg["missing_value_rate"]["strong_min"],

        "high_corrupted_sample_rate":
            features.get("corrupted_sample_rate", 0.0)
            >= dq_cfg["corrupted_sample_rate"]["strong_min"],

        "split_overlap_detected":
            features.get("split_overlap_rate", 0.0)
            >= dq_cfg["split_overlap_rate"]["strong_min"],

        "preprocessing_mismatch":
            flags.get("preprocessing_mismatch", False),

        "distribution_shift_detected":
            flags.get("distribution_shift_detected", False),
    }

    evidence = [
        name for name, triggered in checks.items()
        if triggered
    ]

    if evidence:
        high = (
            features.get("label_noise_rate", 0.0)
            >= dq_cfg["label_noise_rate"]["high_min"]
            or
            features.get("duplicate_rate", 0.0)
            >= dq_cfg["duplicate_rate"]["high_min"]
            or
            features.get("missing_value_rate", 0.0)
            >= dq_cfg["missing_value_rate"]["high_min"]
            or
            features.get("corrupted_sample_rate", 0.0)
            >= dq_cfg["corrupted_sample_rate"]["high_min"]
            or
            features.get("split_overlap_rate", 0.0)
            >= dq_cfg["split_overlap_rate"]["high_min"]
        )

        return {
            "primary_issue": "data_quality_issue",
            "severity": "high" if high else "medium",
            "evidence_codes": evidence,
        }

    # ==================================================
    # 2. Optimization problem（優化問題）
    # ==================================================

    evidence = []

    for flag_name in [
        "nan_or_inf",
        "gradient_instability",
        "loss_divergence",
    ]:
        if flags.get(flag_name, False):
            evidence.append(flag_name)

    osc_cfg = T["optimization"]["oscillation"]

    oscillation = features.get("oscillation_score", 0.0)
    amplitude = features.get("relative_amplitude", 0.0)

    if (
        oscillation >= osc_cfg["oscillation_score_strong_min"]
        and
        amplitude >= osc_cfg["relative_amplitude_strong_min"]
    ):
        evidence.append("strong_loss_oscillation")

    if evidence:
        high = (
            flags.get("nan_or_inf", False)
            or flags.get("gradient_instability", False)
            or (
                oscillation
                >= osc_cfg["oscillation_score_high_min"]
                and
                amplitude
                >= osc_cfg["relative_amplitude_high_min"]
            )
        )

        return {
            "primary_issue": "optimization_problem",
            "severity": "high" if high else "medium",
            "evidence_codes": evidence,
        }

    # ==================================================
    # 3. Class imbalance（類別不平衡）
    # ==================================================

    ci_cfg = T["class_imbalance"]

    class_ratio = features.get("class_imbalance_ratio", 1.0)
    class_gap = features.get("class_performance_gap", 0.0)
    acc_f1_gap = features.get("accuracy_macro_f1_gap", 0.0)

    distribution_skew = (
        class_ratio
        <= ci_cfg["class_imbalance_ratio"]["strong_max"]
    )

    performance_affected = (
        class_gap
        >= ci_cfg["class_performance_gap"]["strong_min"]
        or
        acc_f1_gap
        >= ci_cfg["accuracy_macro_f1_gap"]["strong_min"]
    )

    if distribution_skew and performance_affected:
        evidence = ["strong_class_distribution_skew"]

        if (
            class_gap
            >= ci_cfg["class_performance_gap"]["strong_min"]
        ):
            evidence.append("large_class_performance_gap")

        if (
            acc_f1_gap
            >= ci_cfg["accuracy_macro_f1_gap"]["strong_min"]
        ):
            evidence.append("large_accuracy_macro_f1_gap")

        high = (
            class_ratio
            <= ci_cfg["class_imbalance_ratio"]["high_max"]
            and (
                class_gap
                >= ci_cfg["class_performance_gap"]["high_min"]
                or
                acc_f1_gap
                >= ci_cfg["accuracy_macro_f1_gap"]["high_min"]
            )
        )

        return {
            "primary_issue": "class_imbalance",
            "severity": "high" if high else "medium",
            "evidence_codes": evidence,
        }

    # ==================================================
    # 4. Overfitting（過擬合）
    # ==================================================

    gen_cfg = T["generalization"]

    gen_gap = features.get(
        "relative_generalization_gap", 0.0
    )
    late_deg = features.get(
        "late_degradation", 0.0
    )

    if (
        gen_gap
        >= gen_cfg["relative_generalization_gap"]["strong_min"]
        and
        late_deg
        >= gen_cfg["late_degradation"]["strong_min"]
    ):
        evidence = [
            "strong_generalization_gap",
            "late_validation_degradation",
        ]

        high = (
            gen_gap
            >= gen_cfg["relative_generalization_gap"]["high_min"]
            or
            late_deg
            >= gen_cfg["late_degradation"]["high_min"]
        )

        return {
            "primary_issue": "overfitting",
            "severity": "high" if high else "medium",
            "evidence_codes": evidence,
        }

    # ==================================================
    # 5. Underfitting（欠擬合）
    # ==================================================

    under_cfg = T["underfitting"][
        "reference_performance_gap"
    ]

    reference_gap = features.get(
        "reference_performance_gap", 0.0
    )

    normal_gap = (
        gen_gap
        <= gen_cfg[
            "relative_generalization_gap"
        ]["normal_max"]
    )

    if (
        reference_gap >= under_cfg["strong_min"]
        and normal_gap
    ):
        evidence = [
            "large_reference_performance_shortfall",
            "small_train_validation_gap",
        ]

        plateau = features.get("plateau_streak", 0)

        if (
            plateau
            >= T["optimization"]["plateau"]["strong_streak_min"]
        ):
            evidence.append("training_plateau")

        return {
            "primary_issue": "underfitting",
            "severity": (
                "high"
                if reference_gap >= under_cfg["high_min"]
                else "medium"
            ),
            "evidence_codes": evidence,
        }

    # ==================================================
    # 6. No clear issue（沒有明顯問題）
    # ==================================================

    return {
        "primary_issue": "no_clear_issue",
        "severity": "low",
        "evidence_codes": [
            "no_strong_diagnostic_rule_triggered"
        ],
    }
