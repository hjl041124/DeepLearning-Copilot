"""State definition for the Phase 1 Agent workflow."""

from typing import Any, TypedDict


class AgentState(TypedDict):
    """Minimal state shared by all Phase 1 LangGraph nodes."""

    experiment_id: str
    user_input: dict[str, Any]
    experiment_context: dict[str, Any]
    diagnosis: dict[str, Any] | None
    report: str | None
    workflow_status: str
    error: str | None
