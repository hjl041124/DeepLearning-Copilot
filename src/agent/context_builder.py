"""Build the structured diagnosis context from Agent tool results."""

from typing import Any

from src.tools.contracts import ToolResult


def _tool_provenance(result: ToolResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "details": dict(result.provenance),
        "warnings": list(result.warnings),
        "error": result.error,
    }


def _successful_features(result: ToolResult) -> dict[str, Any]:
    if result.status != "success":
        return {}

    return dict(result.features)


def build_context(
    metric_result: ToolResult,
    log_result: ToolResult,
    dataset_result: ToolResult,
    *,
    experiment_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Combine evidence without producing any diagnosis or action."""

    for result in (metric_result, log_result, dataset_result):
        if not isinstance(result, ToolResult):
            raise TypeError("all tool results must be ToolResult instances")

    return {
        "experiment_context": dict(experiment_context or {}),
        "metric_features": _successful_features(metric_result),
        "log_features": _successful_features(log_result),
        "dataset_features": _successful_features(dataset_result),
        "provenance": {
            metric_result.tool_name: _tool_provenance(metric_result),
            log_result.tool_name: _tool_provenance(log_result),
            dataset_result.tool_name: _tool_provenance(dataset_result),
        },
    }
