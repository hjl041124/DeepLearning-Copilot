"""LangGraph construction for the integrated Agent workflow."""

from typing import Any

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    build_context_node,
    execute_tools,
    finish,
    generate_report,
    invoke_qlora,
    persist_result,
    receive_input,
    validate_output_node,
)
from src.agent.state import AgentState
from src.inference.mock_diagnosis_model import MockDiagnosisModel
from src.storage.sqlite_store import SQLiteExperimentStore


def build_workflow(
    diagnosis_model: Any | None = None,
    store: SQLiteExperimentStore | None = None,
):
    """Build the fixed, single-agent Phase 3 workflow."""

    model = diagnosis_model or MockDiagnosisModel()

    graph = StateGraph(AgentState)

    graph.add_node("receive_input", receive_input)
    graph.add_node("execute_tools", execute_tools)
    graph.add_node("build_context", build_context_node)
    graph.add_node(
        "invoke_qlora",
        lambda state: invoke_qlora(state, model),
    )
    graph.add_node("validate_output", validate_output_node)
    graph.add_node("generate_report", generate_report)
    graph.add_node(
        "persist_result",
        lambda state: persist_result(state, store),
    )
    graph.add_node("finish", finish)

    graph.add_edge(START, "receive_input")
    graph.add_edge("receive_input", "execute_tools")
    graph.add_edge("execute_tools", "build_context")
    graph.add_edge("build_context", "invoke_qlora")
    graph.add_edge("invoke_qlora", "validate_output")
    graph.add_edge("validate_output", "generate_report")
    graph.add_edge("generate_report", "persist_result")
    graph.add_edge("persist_result", "finish")
    graph.add_edge("finish", END)

    return graph.compile()
