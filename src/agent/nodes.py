"""Nodes for the fixed Phase 3 Agent workflow."""

import json
from typing import Any

from src.agent.context_builder import build_context
from src.agent.state import AgentState
from src.inference.output_parser import parse_model_output
from src.tools.dataset_checker import check_dataset
from src.tools.metric_analysis_tool import analyze_metrics
from src.tools.training_log_analyzer import analyze_training_log


def receive_input(state: AgentState) -> dict[str, Any]:
    """Validate input and initialize the experiment context."""

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

    experiment_context = user_input.get("experiment_context")

    if experiment_context is None:
        experiment_context = dict(user_input)
    elif not isinstance(experiment_context, dict):
        return {
            "workflow_status": "failed",
            "error": "experiment_context must be a dictionary",
        }

    return {
        "experiment_context": dict(experiment_context),
        "workflow_status": "input_received",
        "error": None,
    }


def execute_tools(state: AgentState) -> dict[str, Any]:
    """Execute the three fixed Tool Adapters."""

    if state.get("error"):
        return {}

    return {
        "metric_tool_result": analyze_metrics(
            state["user_input"].get("metrics", {})
        ),
        "log_tool_result": analyze_training_log(
            state["user_input"].get("training_log", {})
        ),
        "dataset_tool_result": check_dataset(
            state["user_input"].get("dataset_info", {})
        ),
        "workflow_status": "tools_executed",
    }


def build_context_node(state: AgentState) -> dict[str, Any]:
    """Build the combined model context from ToolResult objects."""

    if state.get("error"):
        return {}

    try:
        combined_context = build_context(
            state["metric_tool_result"],
            state["log_tool_result"],
            state["dataset_tool_result"],
            experiment_context=state["experiment_context"],
        )
    except Exception as exc:
        return {
            "workflow_status": "failed",
            "error": f"context construction failed: {exc}",
        }

    return {
        "combined_context": combined_context,
        "workflow_status": "context_built",
    }


def invoke_qlora(
    state: AgentState,
    diagnosis_model: Any,
) -> dict[str, Any]:
    """Invoke an injected QLoRA-compatible model interface."""

    if state.get("error"):
        return {}

    try:
        if hasattr(diagnosis_model, "generate"):
            raw_output = diagnosis_model.generate(
                state["combined_context"]
            )

            if not isinstance(raw_output, str):
                raise TypeError("model generate() must return a string")

            return {
                "raw_model_output": raw_output,
                "workflow_status": "model_invoked",
            }

        if hasattr(diagnosis_model, "diagnose"):
            diagnosis = diagnosis_model.diagnose(
                state["combined_context"]
            )
            return {
                "raw_model_output": json.dumps(
                    diagnosis,
                    ensure_ascii=False,
                ),
                "diagnosis": diagnosis,
                "workflow_status": "model_invoked",
            }

        raise TypeError(
            "diagnosis model must define generate() or diagnose()"
        )
    except Exception as exc:
        return {
            "workflow_status": "failed",
            "error": f"model invocation failed: {exc}",
        }


def validate_output_node(state: AgentState) -> dict[str, Any]:
    """Parse and validate model output without repairing it."""

    if state.get("error"):
        return {}

    # Preserve compatibility with the Phase 1 structured mock interface.
    if state.get("diagnosis") is not None:
        return {
            "validation_errors": [],
            "workflow_status": "output_validated",
        }

    parsed = parse_model_output(state.get("raw_model_output") or "")

    if not parsed.is_valid:
        return {
            "diagnosis": None,
            "validation_errors": parsed.validation_errors,
            "workflow_status": "failed",
            "error": (
                "model output validation failed: "
                + "; ".join(parsed.validation_errors)
            ),
        }

    return {
        "diagnosis": parsed.diagnosis,
        "validation_errors": [],
        "workflow_status": "output_validated",
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
