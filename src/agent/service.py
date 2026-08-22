"""Public service entry points for mock and real Agent execution."""

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.agent.state import AgentState
from src.agent.workflow import build_workflow
from src.inference.mock_diagnosis_model import MockDiagnosisModel
from src.inference.qlora_diagnosis_model import QLoRADiagnosisModel
from src.storage.sqlite_store import SQLiteExperimentStore


_MOCK_MODEL = MockDiagnosisModel()
_MOCK_WORKFLOW = build_workflow(_MOCK_MODEL)
_QLORA_MODEL = QLoRADiagnosisModel()
_QLORA_WORKFLOW = build_workflow(_QLORA_MODEL)


def _initial_state(
    experiment_id: str,
    user_input: dict[str, Any],
) -> AgentState:
    started_at = datetime.now(timezone.utc).isoformat()
    return {
        "experiment_id": experiment_id,
        "execution_id": str(uuid4()),
        "started_at": started_at,
        "completed_at": None,
        "user_input": user_input,
        "experiment_context": {},
        "metric_tool_result": None,
        "log_tool_result": None,
        "dataset_tool_result": None,
        "combined_context": {},
        "raw_model_output": None,
        "diagnosis": None,
        "validation_errors": [],
        "report": None,
        "workflow_status": "initialized",
        "error": None,
        "persistence_error": None,
    }


def run_agent(
    experiment_id: str,
    user_input: dict[str, Any],
    diagnosis_model: Any | None = None,
    store: SQLiteExperimentStore | None = None,
) -> AgentState:
    """Run with an injected model, or the Phase 1 mock by default."""

    if diagnosis_model is None and store is None:
        workflow = _MOCK_WORKFLOW
    else:
        workflow = build_workflow(
            diagnosis_model or _MOCK_MODEL,
            store,
        )
    result = workflow.invoke(_initial_state(experiment_id, user_input))
    return dict(result)  # type: ignore[return-value]


def run_diagnosis(
    experiment_id: str,
    user_input: dict[str, Any],
    store: SQLiteExperimentStore | None = None,
) -> AgentState:
    """Run the integrated workflow with the final lazy QLoRA model."""

    workflow = (
        _QLORA_WORKFLOW
        if store is None
        else build_workflow(_QLORA_MODEL, store)
    )
    result = workflow.invoke(
        _initial_state(experiment_id, user_input)
    )
    return dict(result)  # type: ignore[return-value]
