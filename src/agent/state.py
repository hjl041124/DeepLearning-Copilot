"""State definition for the integrated Agent workflow."""

from typing import Any, TypedDict

from src.tools.contracts import ToolResult


class AgentState(TypedDict):
    """State shared by the fixed Phase 3 LangGraph nodes."""

    experiment_id: str
    execution_id: str
    started_at: str
    completed_at: str | None
    user_input: dict[str, Any]
    experiment_context: dict[str, Any]
    metric_tool_result: ToolResult | None
    log_tool_result: ToolResult | None
    dataset_tool_result: ToolResult | None
    combined_context: dict[str, Any]
    raw_model_output: str | None
    diagnosis: dict[str, Any] | None
    validation_errors: list[str]
    report: str | None
    workflow_status: str
    error: str | None
    persistence_error: str | None
