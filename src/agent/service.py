"""Public service entry points for mock and real Agent execution."""

from typing import Any

from src.agent.state import AgentState
from src.agent.workflow import build_workflow
from src.inference.mock_diagnosis_model import MockDiagnosisModel
from src.inference.qlora_diagnosis_model import QLoRADiagnosisModel


_MOCK_MODEL = MockDiagnosisModel()
_MOCK_WORKFLOW = build_workflow(_MOCK_MODEL)
_QLORA_MODEL = QLoRADiagnosisModel()
_QLORA_WORKFLOW = build_workflow(_QLORA_MODEL)


def _initial_state(
    experiment_id: str,
    user_input: dict[str, Any],
) -> AgentState:
    return {
        "experiment_id": experiment_id,
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
    }


def run_agent(
    experiment_id: str,
    user_input: dict[str, Any],
    diagnosis_model: Any | None = None,
) -> AgentState:
    """Run with an injected model, or the Phase 1 mock by default."""

    workflow = (
        _MOCK_WORKFLOW
        if diagnosis_model is None
        else build_workflow(diagnosis_model)
    )
    result = workflow.invoke(_initial_state(experiment_id, user_input))
    return dict(result)  # type: ignore[return-value]


def run_diagnosis(
    experiment_id: str,
    user_input: dict[str, Any],
) -> AgentState:
    """Run the integrated workflow with the final lazy QLoRA model."""

    result = _QLORA_WORKFLOW.invoke(
        _initial_state(experiment_id, user_input)
    )
    return dict(result)  # type: ignore[return-value]
