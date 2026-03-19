"""Node 4: Literature Agent — LitVar + PubTator3 + BioMCP Europe PMC.

Queries three literature sources:
1. NCBI LitVar: variant-specific publications, disease associations (rsID-based)
2. PubTator3: NLP-annotated article search with entity annotations (direct API)
3. BioMCP article search: Europe PMC federated search (via biomcp CLI)

Supports ACMG criteria:
- PS3: Well-established functional studies (Strong Pathogenic)
- PS4: Prevalence in affected significantly increased (Strong Pathogenic)
- Literature volume and disease associations for classifier context
"""

import logging
from typing import Any

from graph.state import VariantState
from tools.biomcp import search_articles
from tools.litvar import query_litvar
from tools.pubtator3 import (
    get_article_annotations,
    search_variant_articles,
)

logger = logging.getLogger(__name__)


def pubmed_agent_node(state: VariantState) -> dict[str, Any]:
    """Query LitVar, PubTator3, and BioMCP for literature evidence.

    Uses rsID from ClinVar record to search LitVar, gene+variant to search
    PubTator3 for NLP-annotated articles, and BioMCP for Europe PMC coverage.
    """
    logger.info("pubmed_agent_node: starting literature lookup")

    updates: dict[str, Any] = {"current_node": "pubmed_agent"}
    warnings: list[str] = list(state.get("warnings", []))

    # Get identifiers
    clinvar = state.get("clinvar") or {}
    gnomad = state.get("gnomad") or {}
    rsid = clinvar.get("rsid") or gnomad.get("rsid") or ""
    gene_symbol = state.get("gene_symbol") or ""
    hgvs = state.get("hgvs_on_transcript") or ""

    # --- Source 1: LitVar (rsID-based, existing) ---
    litvar_result = None
    if rsid:
        try:
            litvar_result = query_litvar(rsid, max_publications=15)
            if not litvar_result.get("available"):
                litvar_result = None
                logger.info("pubmed_agent: no LitVar data for %s", rsid)
            else:
                logger.info(
                    "pubmed_agent: LitVar — %d publications, %d case reports",
                    litvar_result.get("pmids_count", 0),
                    litvar_result.get("case_report_count", 0),
                )
        except Exception as e:
            logger.warning("pubmed_agent: LitVar query failed: %s", e)
            warnings.append(f"LitVar query failed: {str(e)}")
    else:
        warnings.append("LitVar lookup skipped — no rsID available")

    # --- Source 2: PubTator3 (direct API — NLP annotations) ---
    pubtator_articles = None
    pubtator_annotations = None

    # Build variant notation for search
    variant_notation = ""
    if hgvs and ":" in hgvs:
        variant_notation = hgvs.split(":")[-1]  # e.g. "c.5266dupC"
    elif rsid:
        variant_notation = rsid

    if gene_symbol and variant_notation:
        try:
            pubtator_articles = search_variant_articles(
                gene_symbol, variant_notation, max_results=10,
            )
            if pubtator_articles:
                logger.info(
                    "pubmed_agent: PubTator3 — %d articles for %s %s",
                    len(pubtator_articles), gene_symbol, variant_notation,
                )
                # Get NLP annotations for top articles
                top_pmids = [a["pmid"] for a in pubtator_articles[:5]]
                try:
                    pubtator_annotations = get_article_annotations(top_pmids)
                    logger.info(
                        "pubmed_agent: PubTator3 annotations — %d articles annotated",
                        len(pubtator_annotations),
                    )
                except Exception as e:
                    logger.warning(
                        "pubmed_agent: PubTator3 annotation fetch failed: %s", e
                    )
            else:
                logger.info(
                    "pubmed_agent: PubTator3 — no results for %s %s",
                    gene_symbol, variant_notation,
                )
        except Exception as e:
            logger.warning("pubmed_agent: PubTator3 search failed: %s", e)
            warnings.append(f"PubTator3 search failed: {str(e)}")
    else:
        logger.info("pubmed_agent: PubTator3 skipped — need gene + variant notation")

    # --- Source 3: BioMCP article search (Europe PMC) ---
    biomcp_articles = None
    if gene_symbol:
        try:
            biomcp_articles = search_articles(
                gene=gene_symbol,
                query=variant_notation or None,
                max_results=10,
            )
            if biomcp_articles:
                logger.info(
                    "pubmed_agent: BioMCP/Europe PMC — %d articles",
                    len(biomcp_articles),
                )
            else:
                logger.info("pubmed_agent: BioMCP/Europe PMC — no results")
        except Exception as e:
            logger.warning("pubmed_agent: BioMCP article search failed: %s", e)
            warnings.append(f"BioMCP article search failed: {str(e)}")

    # --- Merge results ---
    if litvar_result:
        merged = dict(litvar_result)
        if pubtator_articles:
            merged["pubtator3_articles"] = pubtator_articles
        if pubtator_annotations:
            merged["pubtator3_annotations"] = pubtator_annotations
        if biomcp_articles:
            merged["biomcp_articles"] = biomcp_articles
        updates["pubmed"] = merged
    elif pubtator_articles or biomcp_articles:
        # LitVar unavailable — use PubTator3/BioMCP as primary
        updates["pubmed"] = {
            "rsid": rsid,
            "available": True,
            "pmids_count": len(pubtator_articles or []) + len(biomcp_articles or []),
            "publications": [],
            "diseases": [],
            "related_genes": [],
            "related_chemicals": [],
            "clinical_significance": "",
            "first_published": None,
            "publication_types": {},
            "functional_study_count": 0,
            "case_report_count": 0,
            "review_count": 0,
            "pubtator3_articles": pubtator_articles or [],
            "pubtator3_annotations": pubtator_annotations or [],
            "biomcp_articles": biomcp_articles or [],
            "source": "PubTator3 + BioMCP (Europe PMC)",
        }
    else:
        # All sources failed
        updates["pubmed"] = None
        updates["pubmed_error"] = (
            "No literature data from LitVar, PubTator3, or BioMCP"
        )
        if not rsid:
            warnings.append("Literature evidence unavailable (no rsID)")
        else:
            warnings.append(f"No literature evidence found for {rsid}")

    updates["warnings"] = warnings
    return updates
