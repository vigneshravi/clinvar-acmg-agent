"""Node 1: Input Parser — resolves variant input into transcripts with
NM↔ENST mapping, VEP annotation, and ranked transcript selection."""

import logging
import re
from typing import Any

from cache.cache_manager import CacheManager
from graph.state import VariantState
from tools.ensembl import resolve_transcripts
from tools.entrez import count_pathogenic_submissions_per_transcript, get_gene_aliases
from tools.variant_utils import parse_variant_input

logger = logging.getLogger(__name__)

_cache = CacheManager()


def _rank_transcripts(transcripts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Score and sort transcripts by clinical relevance.

    Priority order:
    1. MANE Select/Clinical + Most Reported Pathogenic + Canonical
    2. MANE Select/Clinical + (Most Reported OR Canonical)
    3. MANE Select/Clinical OR Most Reported
    4. Canonical
    5. Others
    """
    for tx in transcripts:
        score = 0
        if tx.get("is_mane_select"):
            score += 4
        if tx.get("is_mane_plus_clinical"):
            score += 4
        if tx.get("is_most_reported_pathogenic"):
            score += 2
        if tx.get("is_canonical"):
            score += 1
        if tx.get("nm_accession"):
            score += 1  # prefer transcripts with NM_
        tx["annotation_score"] = score

    transcripts.sort(
        key=lambda t: (-t["annotation_score"], t.get("nm_accession") or "zzz"),
    )
    return transcripts


def input_parser_node(state: VariantState) -> dict[str, Any]:
    """Parse raw_input, resolve transcripts via VEP, map NM↔ENST, annotate."""
    logger.info("input_parser_node: parsing '%s'", state["raw_input"])
    genome_build = state.get("genome_build", "GRCh38")
    updates: dict[str, Any] = {"current_node": "input_parser"}
    warnings: list[str] = list(state.get("warnings", []))

    # If the state already has a selected transcript (user chose from dropdown)
    if (state.get("selected_transcript")
            and state.get("hgvs_on_transcript")
            and state.get("gene_symbol")):
        logger.info("input_parser_node: using pre-selected transcript %s",
                     state["selected_transcript"])
        updates["input_mode"] = "hgvs"
        updates["warnings"] = warnings
        return updates

    # ---- Step 1: Parse raw input ----
    parsed = parse_variant_input(state["raw_input"])

    if parsed.get("error") and not parsed.get("gene") and not parsed.get("transcript"):
        updates["input_parse_error"] = parsed["error"]
        updates["errors"] = state.get("errors", []) + [parsed["error"]]
        return updates

    updates["input_mode"] = parsed.get("input_mode", "hgvs")

    # ---- Step 2: Build VEP query parameters ----
    gene_symbol = parsed.get("gene")
    transcript = parsed.get("transcript")
    cdna = parsed.get("cdna")
    chrom = parsed.get("chrom")
    pos = parsed.get("pos")
    ref = parsed.get("ref")
    alt = parsed.get("alt")

    # Construct HGVS string for VEP
    # VEP requires transcript:cdna format (e.g. NM_007294.4:c.5266dupC)
    hgvs_for_vep = None
    if transcript and cdna:
        hgvs_for_vep = f"{transcript}:{cdna}"
    elif gene_symbol and cdna and not chrom:
        # No transcript specified — look up the primary NM_ from ClinVar
        # pathogenic submission counts (this tells us the clinically
        # relevant transcript)
        try:
            path_counts = count_pathogenic_submissions_per_transcript(
                gene_symbol.upper()
            )
            if path_counts:
                primary_nm = path_counts.most_common(1)[0][0]
                hgvs_for_vep = f"{primary_nm}:{cdna}"
                logger.info(
                    "input_parser: using primary NM_ %s for VEP query",
                    primary_nm,
                )
        except Exception as e:
            logger.warning("Could not get primary NM_ for VEP: %s", e)

    # ---- Step 3: Resolve all transcripts ----
    logger.info("input_parser_node: resolving transcripts via VEP")
    resolution = resolve_transcripts(
        gene_symbol=gene_symbol.upper() if gene_symbol else None,
        hgvs=hgvs_for_vep,
        chrom=chrom,
        pos=pos,
        ref=ref,
        alt=alt,
        genome_build=genome_build,
    )

    if resolution.get("error"):
        # VEP failed — fall back to basic parsing
        logger.warning("VEP resolution failed: %s", resolution["error"])
        warnings.append(resolution["error"])
        if gene_symbol:
            updates["gene_symbol"] = gene_symbol.upper()
        if cdna:
            updates["hgvs_on_transcript"] = (
                f"{transcript}:{cdna}" if transcript
                else f"{gene_symbol} {cdna}" if gene_symbol
                else state["raw_input"]
            )
        else:
            updates["hgvs_on_transcript"] = state["raw_input"]
        updates["warnings"] = warnings
        return updates

    transcripts = resolution.get("transcripts", [])
    resolved_gene = resolution.get("gene_symbol", gene_symbol or "")

    if resolved_gene:
        updates["gene_symbol"] = resolved_gene.upper()

    if resolution.get("gene_full_name"):
        updates["gene_full_name"] = resolution["gene_full_name"]

    if not transcripts:
        warnings.append("No protein-coding transcripts found for this variant")
        if cdna and gene_symbol:
            updates["hgvs_on_transcript"] = f"{gene_symbol} {cdna}"
        else:
            updates["hgvs_on_transcript"] = state["raw_input"]
        updates["warnings"] = warnings
        return updates

    # ---- Step 4: Get gene aliases ----
    gene_upper = (resolved_gene or "").upper()
    try:
        gene_info = get_gene_aliases(gene_upper)
        updates["gene_aliases"] = gene_info.get("aliases", [])
        if gene_info.get("full_name"):
            updates["gene_full_name"] = gene_info["full_name"]
    except Exception as e:
        logger.warning("Gene alias lookup failed: %s", e)

    # ---- Step 5: Get ClinVar pathogenic counts → most reported transcript ----
    try:
        path_counts = count_pathogenic_submissions_per_transcript(gene_upper)
        if path_counts:
            top_nm = path_counts.most_common(1)[0][0]
            for tx in transcripts:
                if tx.get("nm_accession") == top_nm:
                    tx["is_most_reported_pathogenic"] = True
                    break
            else:
                # top_nm not in VEP results — check if canonical maps to it
                for tx in transcripts:
                    if tx.get("is_canonical") and not tx.get("nm_accession"):
                        tx["nm_accession"] = top_nm
                        tx["is_most_reported_pathogenic"] = True
                        break
    except Exception as e:
        logger.warning("Pathogenic count lookup failed: %s", e)

    # ---- Step 6: Rank transcripts ----
    transcripts = _rank_transcripts(transcripts)
    updates["all_transcripts"] = transcripts

    # ---- Step 7: Select best transcript ----
    selected = transcripts[0]
    sel_id = selected.get("nm_accession") or selected.get("enst_accession", "")
    updates["selected_transcript"] = sel_id

    # ---- Step 8: Set hgvs_on_transcript ----
    sel_hgvsc = selected.get("hgvsc", "")
    if sel_hgvsc:
        updates["hgvs_on_transcript"] = sel_hgvsc
    elif selected.get("nm_accession") and cdna:
        updates["hgvs_on_transcript"] = f"{selected['nm_accession']}:{cdna}"
    elif cdna and gene_symbol:
        updates["hgvs_on_transcript"] = f"{gene_symbol} {cdna}"
    else:
        updates["hgvs_on_transcript"] = state["raw_input"]

    # Set coordinates if available
    if chrom:
        updates["chrom"] = str(chrom).lstrip("chr").upper()
        updates["pos"] = pos
        updates["ref"] = ref
        updates["alt"] = alt

    updates["warnings"] = warnings
    logger.info(
        "input_parser_node: gene=%s, tx=%s, hgvs=%s, %d transcripts",
        updates.get("gene_symbol"), sel_id,
        updates.get("hgvs_on_transcript"), len(transcripts),
    )
    return updates
