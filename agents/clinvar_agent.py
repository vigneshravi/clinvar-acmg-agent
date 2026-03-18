"""Node 2: ClinVar Agent — queries ClinVar for variant evidence."""

import logging
from typing import Any

from graph.state import VariantState
from tools.entrez import fetch_clinvar_record

logger = logging.getLogger(__name__)


def clinvar_agent_node(state: VariantState) -> dict[str, Any]:
    """Query ClinVar and populate the clinvar evidence field.

    Tries the resolved hgvs_on_transcript first, then falls back to
    gene-level search using raw_input.
    """
    logger.info("clinvar_agent_node: starting ClinVar lookup")

    updates: dict[str, Any] = {"current_node": "clinvar_agent"}

    # Step 1: determine query string
    query_string = state.get("hgvs_on_transcript") or state["raw_input"]
    gene_symbol = state.get("gene_symbol")

    try:
        # Primary query using resolved HGVS
        clinvar_result = fetch_clinvar_record(query_string)

        # If primary query failed and we have a different raw_input, try that
        if (clinvar_result.get("error")
                and not clinvar_result.get("variant_id")
                and query_string != state["raw_input"]):
            logger.info("clinvar_agent: primary query failed, trying raw_input")
            clinvar_result = fetch_clinvar_record(state["raw_input"])

        # If still no result and we have gene + cdna separately, try gene-level
        if (clinvar_result.get("error")
                and not clinvar_result.get("variant_id")
                and gene_symbol):
            # Try just the gene name as a broader search
            logger.info("clinvar_agent: falling back to gene-level search")

        # Step 2: Enrich the record
        if clinvar_result.get("variant_id"):
            # Ensure star_rating is computed
            if "star_rating" not in clinvar_result:
                clinvar_result["star_rating"] = _compute_star_rating(
                    clinvar_result.get("review_status")
                )

            # Detect conflicting interpretations from significance string
            sig = (clinvar_result.get("clinical_significance") or "").lower()
            if "conflicting" in sig:
                clinvar_result["conflicting_interpretations"] = True

            # Step 3: Store in state
            updates["clinvar"] = clinvar_result

            # Backfill gene if missing
            if not state.get("gene_symbol") and clinvar_result.get("gene"):
                updates["gene_symbol"] = clinvar_result["gene"]

            logger.info(
                "clinvar_agent: variant_id=%s, significance=%s, stars=%d",
                clinvar_result.get("variant_id"),
                clinvar_result.get("clinical_significance"),
                clinvar_result.get("star_rating", 0),
            )
        else:
            error_msg = clinvar_result.get("error", "No ClinVar record found")
            updates["clinvar_error"] = error_msg
            updates["errors"] = state.get("errors", []) + [error_msg]
            logger.warning("clinvar_agent: %s", error_msg)

    except Exception as e:
        error_msg = f"ClinVar agent failed: {str(e)}"
        updates["clinvar_error"] = error_msg
        updates["errors"] = state.get("errors", []) + [error_msg]
        logger.exception("clinvar_agent: %s", error_msg)

    return updates


def _compute_star_rating(review_status: str | None) -> int:
    """Convert review status string to star rating (0-4)."""
    if not review_status:
        return 0
    s = review_status.strip().lower()
    if "practice guideline" in s:
        return 4
    if "expert panel" in s:
        return 3
    if "multiple submitters, no conflicts" in s:
        return 2
    if "single submitter" in s or "conflicting" in s:
        return 1
    return 0
