"""System-level statistics for Agent workflow executions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


TRACKED_TOOLS = (
    "metric_analysis",
    "training_log_analyzer",
    "dataset_checker",
)

AGENT_TOOL_FIELDS = (
    "metric_tool_result",
    "log_tool_result",
    "dataset_tool_result",
)


def _as_mapping(record: Any) -> Mapping[str, Any]:
    if isinstance(record, Mapping):
        return record

    try:
        return dict(record)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "each Agent result must be a mapping or SQLite record"
        ) from exc


def _tool_identity(
    tool_result: Any,
    fallback_name: str | None = None,
) -> tuple[str | None, str | None]:
    if isinstance(tool_result, Mapping):
        return (
            tool_result.get("tool_name") or fallback_name,
            tool_result.get("status"),
        )

    return (
        getattr(tool_result, "tool_name", fallback_name),
        getattr(tool_result, "status", None),
    )


def _iter_tool_results(
    record: Mapping[str, Any],
) -> Iterable[tuple[str | None, str | None]]:
    agent_results = [
        record.get(field)
        for field in AGENT_TOOL_FIELDS
        if record.get(field) is not None
    ]

    if agent_results:
        for tool_result in agent_results:
            yield _tool_identity(tool_result)
        return

    stored_results = record.get("tool_results")
    if isinstance(stored_results, Mapping):
        for tool_name, tool_result in stored_results.items():
            yield _tool_identity(tool_result, str(tool_name))
    elif isinstance(stored_results, list):
        for tool_result in stored_results:
            yield _tool_identity(tool_result)


def evaluate_agent_runs(
    agent_results: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate workflow statistics without judging diagnosis quality."""

    records = [_as_mapping(result) for result in agent_results]
    total_runs = len(records)

    completed_runs = sum(
        1
        for record in records
        if record.get("workflow_status", record.get("status"))
        == "completed"
    )
    failed_runs = total_runs - completed_runs

    tool_statistics = {
        tool_name: {
            "call_count": 0,
            "success_count": 0,
            "failed_count": 0,
        }
        for tool_name in TRACKED_TOOLS
    }

    for record in records:
        for tool_name, status in _iter_tool_results(record):
            if not tool_name:
                continue

            statistics = tool_statistics.setdefault(
                tool_name,
                {
                    "call_count": 0,
                    "success_count": 0,
                    "failed_count": 0,
                },
            )
            statistics["call_count"] += 1
            if status == "success":
                statistics["success_count"] += 1
            elif status == "failed":
                statistics["failed_count"] += 1

    validation_evaluated_runs = 0
    validation_passed_runs = 0
    for record in records:
        if "validation_errors" not in record:
            continue

        validation_evaluated_runs += 1
        if not record["validation_errors"]:
            validation_passed_runs += 1

    validation_failed_runs = (
        validation_evaluated_runs - validation_passed_runs
    )

    diagnosis_distribution: dict[str, int] = {}
    for record in records:
        diagnosis = record.get("diagnosis")
        if not isinstance(diagnosis, Mapping):
            continue

        primary_issue = diagnosis.get("primary_issue")
        if isinstance(primary_issue, str):
            diagnosis_distribution[primary_issue] = (
                diagnosis_distribution.get(primary_issue, 0) + 1
            )

    return {
        "workflow_success_rate": {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "success_rate": (
                completed_runs / total_runs if total_runs else 0.0
            ),
        },
        "tool_execution_statistics": tool_statistics,
        "validation_pass_rate": {
            "total_runs": total_runs,
            "evaluated_runs": validation_evaluated_runs,
            "passed_runs": validation_passed_runs,
            "failed_runs": validation_failed_runs,
            "not_recorded_runs": total_runs - validation_evaluated_runs,
            "pass_rate": (
                validation_passed_runs / validation_evaluated_runs
                if validation_evaluated_runs
                else 0.0
            ),
        },
        "diagnosis_distribution": diagnosis_distribution,
    }
