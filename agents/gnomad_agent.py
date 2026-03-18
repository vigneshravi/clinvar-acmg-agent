"""Node 3: gnomAD Agent — population allele frequency data (stub).

Phase 2 implementation will query the gnomAD API for:
- Global and population-specific allele frequencies
- Filtering allele frequency (FAF)
- Homozygote/hemizygote counts
- Relevant for ACMG criteria: BA1, BS1, PM2
"""

import logging
from typing import Any

from graph.state import VariantState

logger = logging.getLogger(__name__)


def gnomad_agent_node(state: VariantState) -> dict[str, Any]:
    """Stub: Query gnomAD for population allele frequency data.

    TODO (Phase 2):
    - Query gnomAD GraphQL API with variant coordinates
    - Extract allele frequencies across populations
    - Compute filtering allele frequency
    - Return structured gnomAD evidence dict
    """
    logger.info("gnomad_agent_node: called (stub — not yet implemented)")

    return {
        "current_node": "gnomad_agent",
        "gnomad": None,
        "gnomad_error": "gnomAD integration not yet implemented",
        "warnings": state.get("warnings", []) + [
            "gnomAD data not available — allele frequency criteria (BA1, BS1, PM2) cannot be assessed"
        ],
    }
