"""Node 5: AlphaFold Agent — structural evidence (stub).

Phase 4 implementation will query:
- AlphaFold Database for predicted structures
- RCSB PDB for experimental structures
- Assess PM1 (mutational hotspot / functional domain)
"""

import logging
from typing import Any

from graph.state import VariantState

logger = logging.getLogger(__name__)


def alphafold_agent_node(state: VariantState) -> dict[str, Any]:
    """Stub: Query AlphaFold/PDB for structural evidence.

    TODO (Phase 4):
    - Fetch AlphaFold predicted structure for the protein
    - Check if variant position is in a functional domain
    - Retrieve pLDDT confidence scores
    - Query PDB for experimental structures
    - Return structural evidence dict
    """
    logger.info("alphafold_agent_node: called (stub — not yet implemented)")

    return {
        "current_node": "alphafold_agent",
        "alphafold": None,
        "pdb": None,
        "structural_error": "AlphaFold/PDB integration not yet implemented",
        "warnings": state.get("warnings", []) + [
            "Structural data not available — domain criteria (PM1) cannot be assessed"
        ],
    }
