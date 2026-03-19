"""Node 8: Pathway & Druggability Agent — via BioMCP.

Queries BioMCP for:
- Reactome pathway memberships for the gene
- DGIdb druggability categories and drug-gene interactions

Provides contextual evidence for variant interpretation:
- Pathway involvement helps assess biological impact
- Druggability informs clinical actionability
"""

import logging
from typing import Any

from graph.state import VariantState
from tools.biomcp import get_gene_druggability, get_gene_pathways

logger = logging.getLogger(__name__)


def pathway_agent_node(state: VariantState) -> dict[str, Any]:
    """Query BioMCP for pathway and druggability data.

    Steps:
    1. Query Reactome pathways for the gene
    2. Query DGIdb for druggability information
    """
    logger.info("pathway_agent_node: starting pathway/druggability lookup via BioMCP")

    updates: dict[str, Any] = {"current_node": "pathway_agent"}
    warnings: list[str] = list(state.get("warnings", []))

    gene_symbol = state.get("gene_symbol")

    if not gene_symbol:
        updates["pathways"] = None
        updates["druggability"] = None
        updates["pathway_error"] = "No gene symbol available"
        warnings.append("Pathway/druggability lookup skipped — no gene symbol")
        updates["warnings"] = warnings
        return updates

    # --- Pathways ---
    pathway_data = None
    try:
        pathway_data = get_gene_pathways(gene_symbol)
        if pathway_data:
            logger.info(
                "pathway_agent: %d pathways for %s",
                len(pathway_data), gene_symbol,
            )
        else:
            logger.info("pathway_agent: no pathways for %s", gene_symbol)
    except Exception as e:
        logger.warning("pathway_agent: pathway query failed: %s", e)
        warnings.append(f"Pathway query failed: {str(e)}")

    updates["pathways"] = pathway_data

    # --- Druggability ---
    drug_data = None
    try:
        drug_data = get_gene_druggability(gene_symbol)
        if drug_data:
            logger.info(
                "pathway_agent: druggability for %s — categories: %s",
                gene_symbol,
                ", ".join(drug_data.get("categories", [])[:3]),
            )
        else:
            logger.info("pathway_agent: no druggability data for %s", gene_symbol)
    except Exception as e:
        logger.warning("pathway_agent: druggability query failed: %s", e)
        warnings.append(f"Druggability query failed: {str(e)}")

    updates["druggability"] = drug_data

    if not pathway_data and not drug_data:
        updates["pathway_error"] = (
            f"No pathway or druggability data for {gene_symbol}"
        )

    updates["warnings"] = warnings
    return updates
