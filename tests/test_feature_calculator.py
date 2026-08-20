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


def approx(value, expected, tol=1e-6):
    assert abs(value - expected) <= tol, (value, expected)


# Overfitting-like example（類過擬合案例）
gap = relative_generalization_gap(
    0.95,
    0.80,
    "higher_is_better",
)
approx(gap, 0.15 / 0.95)


# Late validation degradation（後期驗證退化）
late = late_degradation(
    [0.70, 0.78, 0.84, 0.82, 0.77],
    "higher_is_better",
)
assert late > 0


# Learning improvement（學習改善）
improvement = relative_improvement(
    [2.0, 1.7, 1.3, 1.0, 0.8],
    "lower_is_better",
)
assert improvement > 0


# Plateau（停滯）
streak = plateau_streak(
    [1.0, 0.8, 0.70, 0.699, 0.698, 0.698],
    "lower_is_better",
    relative_min_delta=0.005,
)
assert streak >= 3


# Monotonic curve（單調曲線） should have low oscillation
monotonic = oscillation_score(
    [2.0, 1.8, 1.6, 1.4, 1.2]
)
assert monotonic < 0.05


# Oscillating curve（震盪曲線）
oscillating = oscillation_score(
    [2.0, 1.0, 2.0, 1.0, 2.0]
)
assert oscillating > 0.9


amplitude = relative_amplitude(
    [2.0, 1.0, 2.0, 1.0, 2.0]
)
assert amplitude > 0


# Class imbalance（類別不平衡）
ratio = class_imbalance_ratio([9000, 800, 200])
approx(ratio, 200 / 9000)


# Class-wise performance（逐類性能）
perf_gap = class_performance_gap([0.94, 0.81, 0.42])
approx(perf_gap, 0.52)

perf_ratio = class_performance_ratio([0.94, 0.81, 0.42])
approx(perf_ratio, 0.42 / 0.94)


# Accuracy vs Macro-F1
metric_gap = accuracy_macro_f1_gap(0.94, 0.61)
approx(metric_gap, 0.33)


# Underfitting reference（欠擬合參考性能）
reference_gap = reference_performance_gap(
    observed_training_performance=0.65,
    reference_performance=0.85,
)
assert reference_gap > 0.20


print("FEATURE CALCULATOR TESTS PASSED")
