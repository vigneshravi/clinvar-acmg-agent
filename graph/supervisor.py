"""Supervisor router — pure Python, no LLM.

Inspects the state after input_parser runs and routes:
- If input_parse_error is set → route to END
- If required fields are present → route to clinvar_agent
"""

import logging
from typing import Literal

from graph.state import VariantState

logger = logging.getLogger(__name__)


def supervisor_route(state: VariantState) -> Literal["clinvar_agent", "__end__"]:
    """Route after input parsing based on state contents.

    Returns:
        "clinvar_agent" if input was parsed successfully
        "__end__" if there was a fatal parse error
    """
    # Fatal error: input could not be parsed at all
    if state.get("input_parse_error") and not state.get("gene_symbol"):
        logger.info(
            "supervisor: routing to END — input_parse_error: %s",
            state["input_parse_error"],
        )
        return "__end__"

    # Check required fields for downstream processing
    has_gene = state.get("gene_symbol") is not None
    has_hgvs = state.get("hgvs_on_transcript") is not None

    if has_gene or has_hgvs:
        logger.info(
            "supervisor: routing to clinvar_agent — gene=%s, hgvs=%s",
            state.get("gene_symbol"),
            state.get("hgvs_on_transcript"),
        )
        return "clinvar_agent"

    # Fallback: we have no gene and no HGVS — cannot proceed
    logger.warning(
        "supervisor: routing to END — missing gene_symbol and hgvs_on_transcript"
    )
    return "__end__"


def supervisor_node(state: VariantState) -> dict:
    """Supervisor passthrough node — updates current_node for tracking.

    The actual routing decision is made by supervisor_route() which is
    used as a conditional edge function in the graph definition.
    """
    logger.info("supervisor_node: evaluating routing")

    warnings = list(state.get("warnings", []))

    # Add routing warnings for missing optional fields
    if not state.get("selected_transcript"):
        warnings.append(
            "No specific transcript selected — using gene-level query"
        )

    return {
        "current_node": "supervisor",
        "warnings": warnings,
    }
