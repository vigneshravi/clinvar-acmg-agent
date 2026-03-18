"""Node 6: TCGA Agent — somatic cancer data (stub).

Phase 5 implementation will query:
- GDC API for somatic mutation frequency
- Gene expression data across cancer types
- Supports PM1 (hotspot), PS4 (prevalence in affected)
"""

import logging
from typing import Any

from graph.state import VariantState

logger = logging.getLogger(__name__)


def tcga_agent_node(state: VariantState) -> dict[str, Any]:
    """Stub: Query TCGA/GDC for somatic cancer evidence.

    TODO (Phase 5):
    - Query GDC somatic mutation endpoint
    - Get mutation frequency across cancer types
    - Retrieve gene expression data
    - Return structured TCGA evidence dict
    """
    logger.info("tcga_agent_node: called (stub — not yet implemented)")

    return {
        "current_node": "tcga_agent",
        "tcga_somatic": None,
        "tcga_expression": None,
        "tcga_error": "TCGA integration not yet implemented",
        "warnings": state.get("warnings", []) + [
            "TCGA data not available — somatic evidence cannot be assessed"
        ],
    }
