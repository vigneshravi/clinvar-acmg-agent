"""Node 2: ClinVar Agent — queries ClinVar for variant evidence.

Includes HGVS match verification to prevent returning the wrong variant
when ClinVar's search returns imprecise results.
"""

import logging
import re
from typing import Any, Optional

from graph.state import VariantState
from tools.entrez import fetch_clinvar_record

logger = logging.getLogger(__name__)


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


def _extract_cdna_position(hgvs: str) -> Optional[str]:
    """Extract the core cDNA position number from an HGVS string.

    Examples:
        "NM_007294.4(BRCA1):c.5266dup (p.Gln1756fs)" → "5266"
        "BRCA1 c.5266dupC" → "5266"
        "NM_000546.6(TP53):c.817C>T (p.Arg273Cys)" → "817"
    """
    # Match c.DIGITS pattern (the core position)
    m = re.search(r"c\.(\d+)", hgvs)
    if m:
        return m.group(1)
    return None


def _extract_variant_type(hgvs: str) -> Optional[str]:
    """Extract the variant type keyword from HGVS notation.

    Examples:
        "c.5266dupC" → "dup"
        "c.5266dup"  → "dup"
        "c.817C>T"   → ">"
        "c.1114A>C"  → ">"
        "c.206_207delinsTG" → "delins"
    """
    hgvs_lower = hgvs.lower()
    for vtype in ["delins", "dup", "del", "ins", ">"]:
        if vtype in hgvs_lower:
            return vtype
    return None


def _hgvs_matches(query_hgvs: str, result_hgvs: str) -> bool:
    """Check if a ClinVar result's HGVS plausibly matches the query.

    Compares the cDNA position number and variant type to detect mismatches
    like querying c.5266dupC but getting c.206_207delinsTG.
    """
    if not query_hgvs or not result_hgvs:
        return True  # can't verify, assume OK

    query_pos = _extract_cdna_position(query_hgvs)
    result_pos = _extract_cdna_position(result_hgvs)

    if query_pos and result_pos and query_pos != result_pos:
        logger.warning(
            "HGVS mismatch: query position c.%s vs result position c.%s",
            query_pos, result_pos,
        )
        return False

    query_type = _extract_variant_type(query_hgvs)
    result_type = _extract_variant_type(result_hgvs)

    if query_type and result_type and query_type != result_type:
        logger.warning(
            "HGVS type mismatch: query '%s' vs result '%s'",
            query_type, result_type,
        )
        return False

    return True


def _pick_best_result(
    candidates: list[dict[str, Any]],
    query_hgvs: str,
) -> Optional[dict[str, Any]]:
    """Pick the best ClinVar result from multiple candidates.

    Priority:
    1. HGVS matches AND highest star rating
    2. HGVS matches AND has variant_id
    3. Any result with variant_id (fallback)
    """
    matching = []
    non_matching = []

    for r in candidates:
        if not r.get("variant_id"):
            continue
        result_hgvs = r.get("hgvs", "")
        if _hgvs_matches(query_hgvs, result_hgvs):
            matching.append(r)
        else:
            non_matching.append(r)

    if matching:
        # Sort by star rating descending
        matching.sort(key=lambda r: r.get("star_rating", 0), reverse=True)
        return matching[0]

    if non_matching:
        non_matching.sort(key=lambda r: r.get("star_rating", 0), reverse=True)
        return non_matching[0]

    return None


def clinvar_agent_node(state: VariantState) -> dict[str, Any]:
    """Query ClinVar and populate the clinvar evidence field.

    Strategy:
    1. Try resolved hgvs_on_transcript (e.g. NM_007294.4:c.5266dupC)
    2. Try raw_input (e.g. BRCA1 c.5266dupC)
    3. Try gene[gene] AND cdna[Variant name] if available
    4. Validate all results against input HGVS — reject mismatches
    5. Pick the best matching result by star rating
    """
    logger.info("clinvar_agent_node: starting ClinVar lookup")

    updates: dict[str, Any] = {"current_node": "clinvar_agent"}
    warnings: list[str] = list(state.get("warnings", []))

    hgvs_on_tx = state.get("hgvs_on_transcript") or ""
    raw_input = state["raw_input"]
    gene_symbol = state.get("gene_symbol")

    # If gene_symbol is missing, try to extract from ClinVar result later
    # or from the transcript HGVS pattern
    cdna_part = ""
    if ":" in hgvs_on_tx:
        cdna_part = hgvs_on_tx.split(":")[-1]

    # Build list of query strings to try (most specific first)
    queries_to_try: list[str] = []
    if hgvs_on_tx:
        queries_to_try.append(hgvs_on_tx)
    if raw_input and raw_input != hgvs_on_tx:
        queries_to_try.append(raw_input)
    # Gene-level field-tagged search
    if gene_symbol and cdna_part:
        queries_to_try.append(f"{gene_symbol} {cdna_part}")
    # If no gene but we have a transcript, try extracting gene from
    # transcript accession by querying ClinVar with just the cdna part
    if not gene_symbol and cdna_part:
        # Common pattern: try without transcript prefix using gene search
        # This will be tried after the first query reveals the gene
        pass

    try:
        candidates: list[dict[str, Any]] = []
        input_hgvs = hgvs_on_tx or raw_input
        found_good_match = False

        for query in queries_to_try:
            logger.info("clinvar_agent: trying query '%s'", query)
            result = fetch_clinvar_record(query)
            if result.get("variant_id"):
                if "star_rating" not in result:
                    result["star_rating"] = _compute_star_rating(result.get("review_status"))
                candidates.append(result)

                result_hgvs = result.get("hgvs", "")
                if _hgvs_matches(input_hgvs, result_hgvs) and result["star_rating"] >= 2:
                    logger.info(
                        "clinvar_agent: good match on query '%s' (stars=%d)",
                        query, result["star_rating"],
                    )
                    found_good_match = True
                    break

                # If first result revealed a gene, add gene-level fallback
                if not gene_symbol and result.get("gene") and cdna_part:
                    gene_from_result = result["gene"]
                    gene_query = f"{gene_from_result} {cdna_part}"
                    if gene_query not in queries_to_try:
                        queries_to_try.append(gene_query)
                        logger.info(
                            "clinvar_agent: discovered gene '%s', adding fallback query",
                            gene_from_result,
                        )

        # If no good match yet, and first result was a mismatch, try more
        if not found_good_match and candidates:
            first = candidates[0]
            if first.get("gene") and cdna_part:
                gene_fallback = f"{first['gene']} {cdna_part}"
                if gene_fallback not in [q for q in queries_to_try]:
                    logger.info("clinvar_agent: extra gene-level fallback '%s'", gene_fallback)
                    result = fetch_clinvar_record(gene_fallback)
                    if result.get("variant_id"):
                        if "star_rating" not in result:
                            result["star_rating"] = _compute_star_rating(result.get("review_status"))
                        candidates.append(result)
        best = _pick_best_result(candidates, input_hgvs)

        if best:
            # Final HGVS match check — add warning if mismatch
            if not _hgvs_matches(input_hgvs, best.get("hgvs", "")):
                warnings.append(
                    f"ClinVar HGVS mismatch: queried '{input_hgvs}' "
                    f"but ClinVar returned '{best.get('hgvs', '')}'. "
                    f"Result may not be for the exact queried variant."
                )

            # Detect conflicting interpretations
            sig = (best.get("clinical_significance") or "").lower()
            if "conflicting" in sig:
                best["conflicting_interpretations"] = True

            updates["clinvar"] = best
            updates["warnings"] = warnings

            # Backfill gene if missing
            if not state.get("gene_symbol") and best.get("gene"):
                updates["gene_symbol"] = best["gene"]

            logger.info(
                "clinvar_agent: final result — id=%s, hgvs=%s, significance=%s, stars=%d",
                best.get("variant_id"),
                best.get("hgvs"),
                best.get("clinical_significance"),
                best.get("star_rating", 0),
            )
        else:
            error_msg = f"No ClinVar records found for '{input_hgvs}'"
            updates["clinvar_error"] = error_msg
            updates["errors"] = state.get("errors", []) + [error_msg]
            updates["warnings"] = warnings
            logger.warning("clinvar_agent: %s", error_msg)

    except Exception as e:
        error_msg = f"ClinVar agent failed: {str(e)}"
        updates["clinvar_error"] = error_msg
        updates["errors"] = state.get("errors", []) + [error_msg]
        updates["warnings"] = warnings
        logger.exception("clinvar_agent: %s", error_msg)

    return updates
