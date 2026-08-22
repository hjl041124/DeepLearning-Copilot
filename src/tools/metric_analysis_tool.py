"""Metric analysis adapter backed by the existing feature calculator."""

from typing import Any

from src.evaluation.feature_calculator import (
    accuracy_macro_f1_gap,
    class_performance_gap,
    class_performance_ratio,
    relative_generalization_gap,
)
from src.tools.contracts import ToolResult


TOOL_NAME = "metric_analysis"
FEATURE_SOURCE = "src.evaluation.feature_calculator"


def analyze_metrics(input_data: dict[str, Any]) -> ToolResult:
    """Convert supported experiment metrics into canonical features."""

    if not isinstance(input_data, dict):
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error="input_data must be a dictionary",
        )

    if ("accuracy" in input_data) != ("macro_f1" in input_data):
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error="accuracy and macro_f1 must be provided together",
        )

    if ("train_metric" in input_data) != (
        "validation_metric" in input_data
    ):
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error=(
                "train_metric and validation_metric "
                "must be provided together"
            ),
        )

    features: dict[str, Any] = {}
    functions_used: list[str] = []
    warnings: list[str] = []

    try:
        if "accuracy" in input_data:
            features["accuracy_macro_f1_gap"] = (
                accuracy_macro_f1_gap(
                    input_data["accuracy"],
                    input_data["macro_f1"],
                )
            )
            functions_used.append("accuracy_macro_f1_gap")

        if "train_metric" in input_data:
            metric_direction = input_data.get(
                "metric_direction",
                "higher_is_better",
            )

            if "metric_direction" not in input_data:
                warnings.append(
                    "metric_direction was not provided; "
                    "defaulted to higher_is_better"
                )

            features["relative_generalization_gap"] = (
                relative_generalization_gap(
                    input_data["train_metric"],
                    input_data["validation_metric"],
                    metric_direction,
                )
            )
            functions_used.append("relative_generalization_gap")

        if "per_class_metric" in input_data:
            per_class_metric = input_data["per_class_metric"]
            features["class_performance_gap"] = (
                class_performance_gap(per_class_metric)
            )
            features["class_performance_ratio"] = (
                class_performance_ratio(per_class_metric)
            )
            functions_used.extend(
                [
                    "class_performance_gap",
                    "class_performance_ratio",
                ]
            )
    except (TypeError, ValueError) as exc:
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error=f"metric feature calculation failed: {exc}",
            provenance={"module": FEATURE_SOURCE},
        )

    if not features:
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error="no complete supported metric input was provided",
            provenance={"module": FEATURE_SOURCE},
        )

    return ToolResult.success(
        tool_name=TOOL_NAME,
        features=features,
        provenance={
            "module": FEATURE_SOURCE,
            "functions": functions_used,
        },
        warnings=warnings,
    )
