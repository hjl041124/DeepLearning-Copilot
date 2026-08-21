"""LangGraph construction for the Phase 1 Agent skeleton."""

from langgraph.graph import END, START, StateGraph

from src.agent.nodes import (
    finish,
    generate_report,
    mock_model_call,
    prepare_state,
    receive_input,
)
from src.agent.state import AgentState


def build_workflow():
    """Build the fixed, single-agent Phase 1 workflow."""

    graph = StateGraph(AgentState)

    graph.add_node("receive_input", receive_input)
    graph.add_node("prepare_state", prepare_state)
    graph.add_node("mock_model_call", mock_model_call)
    graph.add_node("generate_report", generate_report)
    graph.add_node("finish", finish)

    graph.add_edge(START, "receive_input")
    graph.add_edge("receive_input", "prepare_state")
    graph.add_edge("prepare_state", "mock_model_call")
    graph.add_edge("mock_model_call", "generate_report")
    graph.add_edge("generate_report", "finish")
    graph.add_edge("finish", END)

    return graph.compile()
