"""Training-log adapter backed by the existing feature calculator."""

from typing import Any, Sequence

from src.evaluation.feature_calculator import (
    late_degradation,
    oscillation_score,
    plateau_streak,
    relative_amplitude,
    relative_generalization_gap,
    relative_improvement,
)
from src.tools.contracts import ToolResult


TOOL_NAME = "training_log_analyzer"
FEATURE_SOURCE = "src.evaluation.feature_calculator"


def _series(log_data: dict[str, Any], name: str) -> Sequence[float]:
    value = log_data[name]

    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")

    if len(value) < 2:
        raise ValueError(f"{name} must contain at least two values")

    return value


def analyze_training_log(log_data: dict[str, Any]) -> ToolResult:
    """Convert structured epoch series into canonical numeric features."""

    if not isinstance(log_data, dict):
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error="log_data must be a dictionary",
        )

    if "epoch" not in log_data:
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error="epoch is required",
        )

    has_train_loss = "train_loss" in log_data
    has_validation_loss = "validation_loss" in log_data
    has_train_metric = "train_metric" in log_data
    has_validation_metric = "validation_metric" in log_data

    if has_train_loss != has_validation_loss:
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error=(
                "train_loss and validation_loss "
                "must be provided together"
            ),
        )

    if has_train_metric != has_validation_metric:
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error=(
                "train_metric and validation_metric "
                "must be provided together"
            ),
        )

    if not has_train_loss and not has_train_metric:
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error=(
                "a complete train/validation loss or metric "
                "pair is required"
            ),
        )

    if has_train_metric and "metric_direction" not in log_data:
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error="metric_direction is required for metric curves",
        )

    try:
        epochs = _series(log_data, "epoch")
        epoch_count = len(epochs)

        provided_series = [
            name
            for name in (
                "train_loss",
                "validation_loss",
                "train_metric",
                "validation_metric",
            )
            if name in log_data
        ]

        series = {
            name: _series(log_data, name)
            for name in provided_series
        }

        for name, values in series.items():
            if len(values) != epoch_count:
                raise ValueError(
                    f"{name} length must match epoch length"
                )

        if has_train_metric:
            comparison_train = series["train_metric"]
            comparison_validation = series["validation_metric"]
            comparison_direction = log_data["metric_direction"]
            validation_curve = series["validation_metric"]
            validation_direction = comparison_direction
            comparison_source = "metric"
        else:
            comparison_train = series["train_loss"]
            comparison_validation = series["validation_loss"]
            comparison_direction = "lower_is_better"
            validation_curve = series["validation_loss"]
            validation_direction = "lower_is_better"
            comparison_source = "loss"

        if has_train_loss:
            training_curve = series["train_loss"]
            training_direction = "lower_is_better"
            training_source = "train_loss"
        else:
            training_curve = series["train_metric"]
            training_direction = log_data["metric_direction"]
            training_source = "train_metric"

        features = {
            "relative_generalization_gap": (
                relative_generalization_gap(
                    comparison_train[-1],
                    comparison_validation[-1],
                    comparison_direction,
                )
            ),
            "late_degradation": late_degradation(
                validation_curve,
                validation_direction,
            ),
            "relative_improvement": relative_improvement(
                training_curve,
                training_direction,
            ),
            "plateau_streak": plateau_streak(
                training_curve,
                training_direction,
            ),
            "oscillation_score": oscillation_score(training_curve),
            "relative_amplitude": relative_amplitude(training_curve),
        }
    except (TypeError, ValueError) as exc:
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error=f"training log feature calculation failed: {exc}",
            provenance={"module": FEATURE_SOURCE},
        )

    return ToolResult.success(
        tool_name=TOOL_NAME,
        features=features,
        provenance={
            "module": FEATURE_SOURCE,
            "functions": [
                "relative_generalization_gap",
                "late_degradation",
                "relative_improvement",
                "plateau_streak",
                "oscillation_score",
                "relative_amplitude",
            ],
            "epoch_count": epoch_count,
            "comparison_source": comparison_source,
            "training_curve_source": training_source,
        },
    )
