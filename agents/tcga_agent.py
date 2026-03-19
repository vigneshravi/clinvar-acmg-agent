"""Node 6: Clinical Evidence Agent — CIViC + ClinGen + GWAS via BioMCP.

Replaces the TCGA stub with curated clinical evidence sources:
- CIViC: Clinical interpretations of variants in cancer (diagnostic,
  predictive, prognostic evidence with therapies)
- ClinGen: Gene-disease validity, dosage sensitivity, haploinsufficiency
- GWAS Catalog: Genome-wide association study trait associations

Supports ACMG criteria:
- PS3/PS4: Functional/clinical evidence from CIViC
- Gene-disease validity strengthens overall confidence
"""

import logging
from typing import Any

from graph.state import VariantState
from tools.biomcp import (
    get_gene_clingen,
    get_variant_civic,
    get_variant_gwas,
)

logger = logging.getLogger(__name__)


def _build_civic_query(state: VariantState) -> str | None:
    """Build a CIViC-compatible variant query from state.

    BioMCP's variant endpoint accepts:
    - rsID: rs113488022
    - HGVS genomic: chr7:g.140453136A>T
    - Gene + protein change: BRAF V600E
    """
    # Try rsID first (most reliable for CIViC matching)
    clinvar = state.get("clinvar") or {}
    gnomad = state.get("gnomad") or {}
    rsid = clinvar.get("rsid") or gnomad.get("rsid")
    if rsid:
        return rsid

    # Try gene + protein change (for missense variants)
    gene = state.get("gene_symbol")
    if gene:
        # Check if we have amino acid change info from the transcript
        for tx in state.get("all_transcripts") or []:
            tid = tx.get("nm_accession") or tx.get("enst_accession")
            if tid == state.get("selected_transcript"):
                aa = tx.get("amino_acids", "")
                protein_pos = tx.get("protein_start")
                if aa and protein_pos and "/" in aa:
                    ref_aa, alt_aa = aa.split("/")
                    # Build "GENE RefPosAlt" format e.g. "BRAF V600E"
                    return f"{gene} {ref_aa}{protein_pos}{alt_aa}"
                break

    # Try genomic HGVS notation from coordinates
    chrom = state.get("chrom")
    pos = state.get("pos")
    ref = state.get("ref")
    alt = state.get("alt")
    if chrom and pos and ref and alt:
        # Build chr:g.pos format
        if len(ref) == 1 and len(alt) == 1:
            return f"chr{str(chrom).lstrip('chr')}:g.{pos}{ref}>{alt}"

    return None


def tcga_agent_node(state: VariantState) -> dict[str, Any]:
    """Query BioMCP for CIViC, ClinGen, and GWAS evidence.

    Steps:
    1. Query CIViC for clinical interpretations (via variant rsID or gene+change)
    2. Query ClinGen for gene-disease validity and dosage sensitivity
    3. Query GWAS Catalog for trait associations
    """
    logger.info("tcga_agent_node: starting clinical evidence lookup via BioMCP")

    updates: dict[str, Any] = {"current_node": "tcga_agent"}
    warnings: list[str] = list(state.get("warnings", []))

    gene_symbol = state.get("gene_symbol")

    # --- CIViC ---
    civic_query = _build_civic_query(state)
    civic_data = None

    if civic_query:
        try:
            civic_data = get_variant_civic(civic_query)
            if civic_data:
                logger.info(
                    "tcga_agent: CIViC returned data for '%s'", civic_query
                )
            else:
                logger.info(
                    "tcga_agent: no CIViC data for '%s'", civic_query
                )
        except Exception as e:
            logger.warning("tcga_agent: CIViC query failed: %s", e)
            warnings.append(f"CIViC query failed: {str(e)}")
    else:
        logger.info("tcga_agent: no suitable query for CIViC")
        warnings.append(
            "CIViC lookup skipped — no rsID, protein change, or coordinates available"
        )

    updates["civic"] = civic_data

    # --- ClinGen ---
    clingen_data = None
    if gene_symbol:
        try:
            clingen_data = get_gene_clingen(gene_symbol)
            if clingen_data:
                logger.info(
                    "tcga_agent: ClinGen returned data for %s", gene_symbol
                )
            else:
                logger.info(
                    "tcga_agent: no ClinGen data for %s", gene_symbol
                )
        except Exception as e:
            logger.warning("tcga_agent: ClinGen query failed: %s", e)
            warnings.append(f"ClinGen query failed: {str(e)}")
    else:
        warnings.append("ClinGen lookup skipped — no gene symbol")

    updates["clingen"] = clingen_data

    # --- GWAS Catalog ---
    gwas_data = None
    if civic_query:
        try:
            gwas_data = get_variant_gwas(civic_query)
            if gwas_data:
                logger.info(
                    "tcga_agent: GWAS returned %d associations", len(gwas_data)
                )
            else:
                logger.info("tcga_agent: no GWAS data for '%s'", civic_query)
        except Exception as e:
            logger.warning("tcga_agent: GWAS query failed: %s", e)
            warnings.append(f"GWAS query failed: {str(e)}")

    updates["gwas"] = gwas_data

    # Keep legacy fields empty (no longer using TCGA directly)
    updates["tcga_somatic"] = None
    updates["tcga_expression"] = None

    # Set error only if ALL sources failed
    if not civic_data and not clingen_data and not gwas_data:
        updates["clinical_evidence_error"] = (
            "No clinical evidence from CIViC, ClinGen, or GWAS Catalog"
        )
    else:
        updates["clinical_evidence_error"] = None

    updates["warnings"] = warnings
    return updates
