import math
from statistics import mean
from typing import Sequence


EPS = 1e-8

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"


def _check_direction(direction: str) -> None:
    if direction not in {HIGHER_IS_BETTER, LOWER_IS_BETTER}:
        raise ValueError(f"Unsupported metric direction: {direction}")


def _check_series(values: Sequence[float], min_len: int = 1) -> None:
    if len(values) < min_len:
        raise ValueError(f"Expected at least {min_len} values.")


def signed_train_validation_gap(
    train_value: float,
    validation_value: float,
    metric_direction: str,
) -> float:
    """
    Direction-aware train-validation gap（方向感知的訓練-驗證差距）.

    Positive value:
        validation performance is worse than training performance.

    Negative value:
        validation performance is better than training performance.
    """
    _check_direction(metric_direction)

    if metric_direction == HIGHER_IS_BETTER:
        return train_value - validation_value

    return validation_value - train_value


def relative_generalization_gap(
    train_value: float,
    validation_value: float,
    metric_direction: str,
    eps: float = EPS,
) -> float:
    """
    Relative generalization gap（相對泛化差距）.
    """
    gap = signed_train_validation_gap(
        train_value,
        validation_value,
        metric_direction,
    )

    denominator = max(
        abs(train_value),
        abs(validation_value),
        eps,
    )

    return max(gap, 0.0) / denominator


def late_degradation(
    metric_series: Sequence[float],
    metric_direction: str,
    eps: float = EPS,
) -> float:
    """
    Relative degradation from best epoch to final epoch
    （最佳輪次到最後輪次的相對退化）.
    """
    _check_direction(metric_direction)
    _check_series(metric_series)

    final_value = metric_series[-1]

    if metric_direction == HIGHER_IS_BETTER:
        best_value = max(metric_series)
        degradation = best_value - final_value
    else:
        best_value = min(metric_series)
        degradation = final_value - best_value

    denominator = max(
        abs(best_value),
        abs(final_value),
        eps,
    )

    return max(degradation, 0.0) / denominator


def relative_improvement(
    metric_series: Sequence[float],
    metric_direction: str,
    window_fraction: float = 0.2,
    eps: float = EPS,
) -> float:
    """
    Relative improvement between early and late windows
    （前後窗口的相對改善率）.
    """
    _check_direction(metric_direction)
    _check_series(metric_series, min_len=2)

    if not 0 < window_fraction <= 0.5:
        raise ValueError("window_fraction must be in (0, 0.5].")

    window_size = max(
        1,
        math.ceil(len(metric_series) * window_fraction),
    )

    early_mean = mean(metric_series[:window_size])
    late_mean = mean(metric_series[-window_size:])

    if metric_direction == HIGHER_IS_BETTER:
        improvement = late_mean - early_mean
    else:
        improvement = early_mean - late_mean

    denominator = max(
        abs(early_mean),
        abs(late_mean),
        eps,
    )

    return improvement / denominator


def plateau_streak(
    metric_series: Sequence[float],
    metric_direction: str,
    relative_min_delta: float = 0.005,
    eps: float = EPS,
) -> int:
    """
    Number of consecutive final epochs without meaningful improvement
    （訓練結尾連續沒有有效改善的輪數）.

    Adapted from min_delta + patience（最小改善量 + 容忍輪數） concept.
    """
    _check_direction(metric_direction)
    _check_series(metric_series)

    if relative_min_delta < 0:
        raise ValueError("relative_min_delta must be >= 0.")

    best = metric_series[0]
    streak = 0

    for current in metric_series[1:]:
        if metric_direction == HIGHER_IS_BETTER:
            raw_improvement = current - best
        else:
            raw_improvement = best - current

        denominator = max(abs(best), abs(current), eps)
        relative_change = raw_improvement / denominator

        if relative_change > relative_min_delta:
            best = current
            streak = 0
        else:
            streak += 1

    return streak


def oscillation_score(
    metric_series: Sequence[float],
    eps: float = EPS,
) -> float:
    """
    Oscillation score（震盪分數）.

    0 -> approximately monotonic
    1 -> strong back-and-forth movement
    """
    _check_series(metric_series, min_len=2)

    total_variation = sum(
        abs(metric_series[i] - metric_series[i - 1])
        for i in range(1, len(metric_series))
    )

    if total_variation <= eps:
        return 0.0

    net_change = abs(metric_series[-1] - metric_series[0])

    score = 1.0 - net_change / (total_variation + eps)

    return min(max(score, 0.0), 1.0)


def relative_amplitude(
    metric_series: Sequence[float],
    eps: float = EPS,
) -> float:
    """
    Relative amplitude（相對振幅）.
    """
    _check_series(metric_series)

    amplitude = max(metric_series) - min(metric_series)
    mean_value = mean(metric_series)

    return amplitude / max(abs(mean_value), eps)


def class_imbalance_ratio(
    class_counts: Sequence[int],
) -> float:
    """
    Least frequent class / most frequent class
    （最少類樣本數 / 最多類樣本數）.
    """
    _check_series(class_counts)

    if any(count < 0 for count in class_counts):
        raise ValueError("Class counts cannot be negative.")

    maximum = max(class_counts)

    if maximum == 0:
        raise ValueError("At least one class must contain samples.")

    return min(class_counts) / maximum


def class_performance_gap(
    per_class_metric: Sequence[float],
) -> float:
    """
    Maximum class metric - minimum class metric
    （最佳類別與最差類別的性能差距）.
    """
    _check_series(per_class_metric)

    return max(per_class_metric) - min(per_class_metric)


def class_performance_ratio(
    per_class_metric: Sequence[float],
    eps: float = EPS,
) -> float:
    """
    Minimum class metric / maximum class metric
    （最差類別性能 / 最佳類別性能）.
    """
    _check_series(per_class_metric)

    maximum = max(per_class_metric)
    minimum = min(per_class_metric)

    if abs(maximum) <= eps:
        return 1.0

    return minimum / maximum


def accuracy_macro_f1_gap(
    accuracy: float,
    macro_f1: float,
) -> float:
    """
    Positive Accuracy - Macro-F1 gap
    （Accuracy 與 Macro-F1 的正向差距）.
    """
    return max(accuracy - macro_f1, 0.0)


def reference_performance_gap(
    observed_training_performance: float,
    reference_performance: float,
    metric_direction: str = HIGHER_IS_BETTER,
    eps: float = EPS,
) -> float:
    """
    Relative shortfall from an explicit reference performance
    （相對參考性能的不足程度）.
    """
    _check_direction(metric_direction)

    if metric_direction == HIGHER_IS_BETTER:
        shortfall = reference_performance - observed_training_performance
    else:
        shortfall = observed_training_performance - reference_performance

    denominator = max(
        abs(reference_performance),
        abs(observed_training_performance),
        eps,
    )

    return max(shortfall, 0.0) / denominator
