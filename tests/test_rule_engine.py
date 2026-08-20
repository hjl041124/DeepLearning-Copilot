from src.evaluation.rule_engine import diagnose_experiment


# 1. Overfitting（過擬合）
result = diagnose_experiment({
    "relative_generalization_gap": 0.20,
    "late_degradation": 0.14,
})
assert result["primary_issue"] == "overfitting"


# 2. Underfitting（欠擬合）
result = diagnose_experiment({
    "relative_generalization_gap": 0.03,
    "reference_performance_gap": 0.22,
    "plateau_streak": 5,
})
assert result["primary_issue"] == "underfitting"


# 3. Optimization problem（優化問題）
result = diagnose_experiment({
    "oscillation_score": 0.75,
    "relative_amplitude": 0.30,
})
assert result["primary_issue"] == "optimization_problem"


# 4. Class imbalance（類別不平衡）
result = diagnose_experiment({
    "class_imbalance_ratio": 0.05,
    "class_performance_gap": 0.40,
    "accuracy_macro_f1_gap": 0.22,
})
assert result["primary_issue"] == "class_imbalance"


# 5. Data quality issue（資料品質問題）
result = diagnose_experiment({
    "label_noise_rate": 0.25,
})
assert result["primary_issue"] == "data_quality_issue"


# 6. No clear issue（沒有明顯問題）
result = diagnose_experiment({
    "relative_generalization_gap": 0.03,
    "late_degradation": 0.01,
    "class_imbalance_ratio": 0.80,
    "class_performance_gap": 0.04,
    "accuracy_macro_f1_gap": 0.02,
})
assert result["primary_issue"] == "no_clear_issue"


# 7. Priority test（優先級測試）
# 同時有過擬合特徵和資料品質問題，
# 必須優先判 data_quality_issue。
result = diagnose_experiment({
    "label_noise_rate": 0.30,
    "relative_generalization_gap": 0.25,
    "late_degradation": 0.20,
})
assert result["primary_issue"] == "data_quality_issue"


# 8. Optimization priority（優化問題優先）
result = diagnose_experiment(
    {
        "relative_generalization_gap": 0.25,
        "late_degradation": 0.20,
    },
    {
        "nan_or_inf": True,
    },
)
assert result["primary_issue"] == "optimization_problem"


print("RULE ENGINE TESTS PASSED")
