"""Nodes for the fixed Phase 1 Agent workflow."""

from typing import Any

from src.agent.state import AgentState
from src.inference.mock_diagnosis_model import MockDiagnosisModel


_MOCK_MODEL = MockDiagnosisModel()


def receive_input(state: AgentState) -> dict[str, Any]:
    """Validate the minimal input required by the skeleton."""

    experiment_id = state.get("experiment_id")
    user_input = state.get("user_input")

    if not isinstance(experiment_id, str) or not experiment_id.strip():
        return {
            "workflow_status": "failed",
            "error": "experiment_id must be a non-empty string",
        }

    if not isinstance(user_input, dict):
        return {
            "workflow_status": "failed",
            "error": "user_input must be a dictionary",
        }

    return {
        "workflow_status": "input_received",
        "error": None,
    }


def prepare_state(state: AgentState) -> dict[str, Any]:
    """Create the structured context passed to later phases."""

    if state.get("error"):
        return {}

    return {
        "experiment_context": dict(state["user_input"]),
        "workflow_status": "state_prepared",
    }


def mock_model_call(state: AgentState) -> dict[str, Any]:
    """Call the deterministic Phase 1 mock model."""

    if state.get("error"):
        return {}

    try:
        diagnosis = _MOCK_MODEL.diagnose(
            state["experiment_context"]
        )
    except Exception as exc:
        return {
            "workflow_status": "failed",
            "error": f"mock model failed: {exc}",
        }

    return {
        "diagnosis": diagnosis,
        "workflow_status": "diagnosis_generated",
    }


def generate_report(state: AgentState) -> dict[str, Any]:
    """Format the diagnosis without changing any diagnosis fields."""

    if state.get("error"):
        return {}

    diagnosis = state.get("diagnosis")

    if diagnosis is None:
        return {
            "workflow_status": "failed",
            "error": "diagnosis is missing",
        }

    report = "\n".join(
        [
            f"Experiment ID: {state['experiment_id']}",
            f"Task Type: {diagnosis['task_type']}",
            f"Primary Issue: {diagnosis['primary_issue']}",
            f"Severity: {diagnosis['severity']}",
            "Evidence Codes: "
            + ", ".join(diagnosis["evidence_codes"]),
            "Recommended Action Codes: "
            + ", ".join(
                diagnosis["recommended_action_codes"]
            ),
            f"Explanation: {diagnosis['explanation']}",
        ]
    )

    return {
        "report": report,
        "workflow_status": "report_generated",
    }


def finish(state: AgentState) -> dict[str, Any]:
    """Mark the fixed workflow as completed or failed."""

    if state.get("error"):
        return {"workflow_status": "failed"}

    return {"workflow_status": "completed"}
