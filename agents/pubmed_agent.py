"""Node 4: Literature Agent — LitVar-powered publication evidence.

Queries NCBI LitVar for variant-specific publications, disease associations,
and related entities. Enriches with PubMed metadata to classify publications
as case reports, functional studies, or reviews.

Supports ACMG criteria:
- PS3: Well-established functional studies (Strong Pathogenic)
- PS4: Prevalence in affected significantly increased (Strong Pathogenic)
- Literature volume and disease associations for classifier context
"""

import logging
from typing import Any

from graph.state import VariantState
from tools.litvar import query_litvar

logger = logging.getLogger(__name__)


def pubmed_agent_node(state: VariantState) -> dict[str, Any]:
    """Query LitVar for literature evidence.

    Uses rsID from ClinVar record to search LitVar, retrieves publication
    counts, disease associations, and enriched PubMed metadata.
    """
    logger.info("pubmed_agent_node: starting literature lookup")

    updates: dict[str, Any] = {"current_node": "pubmed_agent"}
    warnings: list[str] = list(state.get("warnings", []))

    # Get rsID from ClinVar or gnomAD
    clinvar = state.get("clinvar") or {}
    gnomad = state.get("gnomad") or {}
    rsid = clinvar.get("rsid") or gnomad.get("rsid") or ""

    if not rsid:
        updates["pubmed"] = None
        updates["pubmed_error"] = "No rsID available — cannot query LitVar"
        warnings.append("Literature evidence unavailable (no rsID)")
        updates["warnings"] = warnings
        return updates

    try:
        litvar_result = query_litvar(rsid, max_publications=15)

        if not litvar_result.get("available"):
            updates["pubmed"] = None
            updates["pubmed_error"] = f"No LitVar data for {rsid}"
            warnings.append(f"No literature evidence found for {rsid}")
            updates["warnings"] = warnings
            return updates

        updates["pubmed"] = litvar_result
        logger.info(
            "pubmed_agent: %s — %d publications, %d case reports, %d diseases",
            rsid,
            litvar_result.get("pmids_count", 0),
            litvar_result.get("case_report_count", 0),
            len(litvar_result.get("diseases", [])),
        )

    except Exception as e:
        error_msg = f"LitVar query failed: {str(e)}"
        updates["pubmed"] = None
        updates["pubmed_error"] = error_msg
        updates["errors"] = state.get("errors", []) + [error_msg]
        warnings.append(error_msg)
        logger.exception("pubmed_agent: %s", error_msg)

    updates["warnings"] = warnings
    return updates
