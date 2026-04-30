"""Node 8.5: RAG Guideline Agent — ClinGen Dosage + FAISS retrieval.

Sits between pathway_agent and acmg_classifier. Two responsibilities:

1. Look up the ClinGen Haploinsufficiency Score for the gene (Riggs et al.
   2020) — this is the SVI-aware gate for PVS1 applicability.

2. Build/load a FAISS index over the SVI/VCEP guideline corpus and retrieve
   the top-k most relevant chunks for the variant under review. These are
   handed to the acmg_classifier so the LLM can ground its judgments in
   citable text rather than parametric knowledge.

Both steps are best-effort: if the ClinGen TSV fetch or the FAISS embedding
model fails, the node populates an `*_error` field on the state and lets the
downstream classifier degrade gracefully.
"""

from __future__ import annotations

import logging
from typing import Any

from graph.state import VariantState
from tools.gene_constraint import clingen_dosage_lookup

logger = logging.getLogger(__name__)


def rag_guideline_agent_node(state: VariantState) -> dict[str, Any]:
    """Populate state.clingen_dosage, state.rag_query, state.rag_chunks."""
    logger.info("rag_guideline_agent_node: starting ClinGen + RAG retrieval")
    updates: dict[str, Any] = {"current_node": "rag_guideline_agent"}
    warnings: list[str] = list(state.get("warnings", []))

    # --- 1) ClinGen Dosage Sensitivity (HI Score) ---
    gene_symbol = state.get("gene_symbol")
    clingen_dosage: dict[str, Any] | None = None
    if gene_symbol:
        try:
            clingen_dosage = clingen_dosage_lookup(gene_symbol)
            logger.info(
                "rag_guideline_agent: ClinGen dosage for %s -> available=%s, hi_score=%s",
                gene_symbol,
                clingen_dosage.get("available"),
                clingen_dosage.get("hi_score"),
            )
        except Exception as e:
            logger.warning("rag_guideline_agent: ClinGen dosage failed: %s", e)
            warnings.append(f"ClinGen dosage lookup failed: {e}")
            clingen_dosage = {"available": False, "error": str(e)}
    else:
        clingen_dosage = {"available": False, "error": "no gene symbol"}

    updates["clingen_dosage"] = clingen_dosage

    # --- 2) FAISS retrieval over SVI/VCEP corpus ---
    rag_chunks: list[dict[str, Any]] = []
    rag_query: str | None = None
    rag_error: str | None = None
    try:
        from svi import rag as _rag

        rag_query = _rag.build_query(state)
        try:
            vs = _rag.build_index()
            rag_chunks = _rag.retrieve(vs, rag_query, k=8)
            logger.info(
                "rag_guideline_agent: retrieved %d chunks for query: %s",
                len(rag_chunks), rag_query,
            )
        except Exception as ie:
            rag_error = f"FAISS index build/retrieve failed: {ie}"
            logger.warning("rag_guideline_agent: %s", rag_error)
            warnings.append(rag_error)
    except Exception as e:
        rag_error = f"RAG module load failed: {e}"
        logger.warning("rag_guideline_agent: %s", rag_error)
        warnings.append(rag_error)

    updates["rag_query"] = rag_query
    updates["rag_chunks"] = rag_chunks
    if rag_error:
        updates["rag_error"] = rag_error

    updates["warnings"] = warnings
    return updates
