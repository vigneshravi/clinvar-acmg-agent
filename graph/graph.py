"""LangGraph graph definition — multi-agent variant classification pipeline."""

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.acmg_classifier import acmg_classifier_node
from agents.alphafold_agent import alphafold_agent_node
from agents.clinvar_agent import clinvar_agent_node
from agents.gnomad_agent import gnomad_agent_node
from agents.input_parser import input_parser_node
from agents.pathway_agent import pathway_agent_node
from agents.pubmed_agent import pubmed_agent_node
from agents.rag_guideline_agent import rag_guideline_agent_node
from agents.tcga_agent import tcga_agent_node
from graph.state import VariantState, make_initial_state
from graph.supervisor import supervisor_node, supervisor_route

logger = logging.getLogger(__name__)


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph StateGraph.

    Graph topology (post-2026-04-30 SVI integration — 10 nodes):
        START → input_parser → supervisor ─┬─→ clinvar_agent → gnomad_agent
                                            │    → pubmed_agent → alphafold_agent
                                            │    → tcga_agent → pathway_agent
                                            │    → rag_guideline_agent
                                            │    → acmg_classifier → END
                                            │
                                            └─→ END  (on parse error)
    """
    graph = StateGraph(VariantState)

    # --- Add nodes ---
    graph.add_node("input_parser", input_parser_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("clinvar_agent", clinvar_agent_node)
    graph.add_node("gnomad_agent", gnomad_agent_node)
    graph.add_node("pubmed_agent", pubmed_agent_node)
    graph.add_node("alphafold_agent", alphafold_agent_node)
    graph.add_node("tcga_agent", tcga_agent_node)
    graph.add_node("pathway_agent", pathway_agent_node)
    graph.add_node("rag_guideline_agent", rag_guideline_agent_node)
    graph.add_node("acmg_classifier", acmg_classifier_node)

    # --- Add edges ---
    # START → input_parser
    graph.add_edge(START, "input_parser")

    # input_parser → supervisor
    graph.add_edge("input_parser", "supervisor")

    # supervisor → conditional routing
    graph.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "clinvar_agent": "clinvar_agent",
            "__end__": END,
        },
    )

    # Evidence gathering pipeline (sequential)
    graph.add_edge("clinvar_agent", "gnomad_agent")
    graph.add_edge("gnomad_agent", "pubmed_agent")
    graph.add_edge("pubmed_agent", "alphafold_agent")
    graph.add_edge("alphafold_agent", "tcga_agent")
    graph.add_edge("tcga_agent", "pathway_agent")
    # SVI integration (2026-04-30): RAG guideline agent injected before classifier
    graph.add_edge("pathway_agent", "rag_guideline_agent")
    graph.add_edge("rag_guideline_agent", "acmg_classifier")

    # acmg_classifier → END
    graph.add_edge("acmg_classifier", END)

    return graph


# Compile the graph once at module level
_compiled_graph = build_graph().compile()


def run_graph(state: VariantState) -> VariantState:
    """Execute the full graph pipeline on a VariantState.

    Args:
        state: Initial VariantState dict (use make_initial_state() to create).

    Returns:
        Final VariantState dict with all fields populated.
    """
    logger.info("run_graph: starting pipeline for '%s'", state.get("raw_input"))
    result = _compiled_graph.invoke(state)
    logger.info(
        "run_graph: completed — classification=%s",
        result.get("classification", "N/A"),
    )
    return result


def run_graph_stream(state: VariantState):
    """Execute the graph with streaming, yielding after each node completes.

    Yields (node_name, updated_state) tuples for progress tracking.
    """
    logger.info("run_graph_stream: starting pipeline for '%s'", state.get("raw_input"))
    for event in _compiled_graph.stream(state):
        # Each event is a dict {node_name: node_output}
        for node_name, node_output in event.items():
            logger.info("run_graph_stream: completed node '%s'", node_name)
            yield node_name, node_output
