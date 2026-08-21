"""Public service entry point for the Phase 1 Agent skeleton."""

from typing import Any

from src.agent.state import AgentState
from src.agent.workflow import build_workflow


_WORKFLOW = build_workflow()


def run_agent(
    experiment_id: str,
    user_input: dict[str, Any],
) -> AgentState:
    """Run one complete Phase 1 workflow execution."""

    initial_state: AgentState = {
        "experiment_id": experiment_id,
        "user_input": user_input,
        "experiment_context": {},
        "diagnosis": None,
        "report": None,
        "workflow_status": "initialized",
        "error": None,
    }

    result = _WORKFLOW.invoke(initial_state)
    return dict(result)  # type: ignore[return-value]
