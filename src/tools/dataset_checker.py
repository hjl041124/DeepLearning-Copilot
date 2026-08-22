"""Dataset-information adapter backed by existing feature logic."""

from typing import Any, Sequence

from src.evaluation.feature_calculator import (
    class_imbalance_ratio,
    class_performance_gap,
    class_performance_ratio,
)
from src.tools.contracts import ToolResult


TOOL_NAME = "dataset_checker"
FEATURE_SOURCE = "src.evaluation.feature_calculator"
EXPLICIT_RATE_FIELDS = (
    "missing_value_rate",
    "duplicate_rate",
    "split_overlap_rate",
)


def _values(
    dataset_info: dict[str, Any],
    name: str,
) -> Sequence[Any]:
    value = dataset_info[name]

    if isinstance(value, dict):
        values = list(value.values())
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ValueError(f"{name} must be a list, tuple, or dictionary")

    if not values:
        raise ValueError(f"{name} must not be empty")

    return values


def _rate(dataset_info: dict[str, Any], name: str) -> float:
    value = dataset_info[name]

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")

    value = float(value)

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")

    return value


def check_dataset(dataset_info: dict[str, Any]) -> ToolResult:
    """Convert explicit dataset statistics into canonical features."""

    if not isinstance(dataset_info, dict):
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error="dataset_info must be a dictionary",
        )

    supported_fields = {
        "class_counts",
        "per_class_metric",
        *EXPLICIT_RATE_FIELDS,
    }

    if not supported_fields.intersection(dataset_info):
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error="no supported dataset field was provided",
        )

    features: dict[str, Any] = {}
    functions_used: list[str] = []
    explicit_statistics: list[str] = []

    try:
        if "class_counts" in dataset_info:
            features["class_imbalance_ratio"] = (
                class_imbalance_ratio(
                    _values(dataset_info, "class_counts")
                )
            )
            functions_used.append("class_imbalance_ratio")

        if "per_class_metric" in dataset_info:
            per_class_metric = _values(
                dataset_info,
                "per_class_metric",
            )
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

        for name in EXPLICIT_RATE_FIELDS:
            if name in dataset_info:
                features[name] = _rate(dataset_info, name)
                explicit_statistics.append(name)
    except (TypeError, ValueError) as exc:
        return ToolResult.failed(
            tool_name=TOOL_NAME,
            error=f"dataset feature calculation failed: {exc}",
            provenance={"module": FEATURE_SOURCE},
        )

    return ToolResult.success(
        tool_name=TOOL_NAME,
        features=features,
        provenance={
            "module": FEATURE_SOURCE,
            "functions": functions_used,
            "explicit_statistics": explicit_statistics,
        },
    )
