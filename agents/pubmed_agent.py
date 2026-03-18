"""Node 4: PubMed Agent — literature evidence (stub).

Phase 3 implementation will query PubMed for:
- Functional studies (PS3/BS3)
- Case-control studies (PS4)
- Co-segregation data (PP1/BS4)
- De novo reports (PS2/PM6)
"""

import logging
from typing import Any

from graph.state import VariantState

logger = logging.getLogger(__name__)


def pubmed_agent_node(state: VariantState) -> dict[str, Any]:
    """Stub: Query PubMed for relevant literature evidence.

    TODO (Phase 3):
    - Search PubMed via Entrez for variant-specific publications
    - Extract functional study results
    - Identify case reports and case-control studies
    - Return structured PubMed evidence dict
    """
    logger.info("pubmed_agent_node: called (stub — not yet implemented)")

    return {
        "current_node": "pubmed_agent",
        "pubmed": None,
        "pubmed_error": "PubMed integration not yet implemented",
        "warnings": state.get("warnings", []) + [
            "PubMed data not available — functional evidence criteria (PS3, BS3) cannot be assessed"
        ],
    }
