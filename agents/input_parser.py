"""Node 1: Input Parser — resolves raw variant input into structured fields
with transcript list, gene info, and HGVS on selected transcript."""

import logging
import re
from typing import Any

from cache.cache_manager import CacheManager
from graph.state import VariantState
from tools.ensembl import get_transcripts_for_gene, recode_variant
from tools.entrez import count_pathogenic_submissions_per_transcript, get_gene_aliases
from tools.variant_utils import parse_variant_input

logger = logging.getLogger(__name__)

_cache = CacheManager()


def input_parser_node(state: VariantState) -> dict[str, Any]:
    """Parse raw_input and resolve to gene, transcripts, and HGVS notation."""
    logger.info("input_parser_node: parsing '%s'", state["raw_input"])
    genome_build = state.get("genome_build", "GRCh38")
    updates: dict[str, Any] = {"current_node": "input_parser"}
    warnings: list[str] = list(state.get("warnings", []))

    # ---- Step 1: Parse raw input ----
    parsed = parse_variant_input(state["raw_input"])

    if parsed.get("error") and not parsed.get("gene") and not parsed.get("transcript"):
        updates["input_parse_error"] = parsed["error"]
        updates["errors"] = state.get("errors", []) + [parsed["error"]]
        return updates

    updates["input_mode"] = parsed.get("input_mode", "hgvs")

    # Handle coordinates mode
    if parsed.get("input_mode") == "coordinates":
        chrom = (parsed.get("chrom") or "").lstrip("chr").upper()
        updates["chrom"] = chrom
        updates["pos"] = parsed.get("pos")
        updates["ref"] = parsed.get("ref")
        updates["alt"] = parsed.get("alt")
        # For coordinates, we cannot do transcript resolution without VEP
        # Set a minimal state so ClinVar can try a positional search
        updates["hgvs_on_transcript"] = state["raw_input"]
        warnings.append("Coordinate input: transcript resolution limited")
        updates["warnings"] = warnings
        return updates

    gene_symbol = parsed.get("gene")
    transcript_from_input = parsed.get("transcript")
    cdna = parsed.get("cdna")

    # If input already has NM_ accession (e.g. NM_007294.4:c.5266dupC)
    has_explicit_transcript = transcript_from_input is not None

    if has_explicit_transcript:
        updates["selected_transcript"] = transcript_from_input
        updates["hgvs_on_transcript"] = f"{transcript_from_input}:{cdna}"
        # Try to extract gene from transcript via ClinVar later
        if gene_symbol:
            updates["gene_symbol"] = gene_symbol.upper()
        updates["warnings"] = warnings
        return updates

    if not gene_symbol:
        updates["input_parse_error"] = "Could not determine gene symbol from input"
        updates["errors"] = state.get("errors", []) + [
            "Could not determine gene symbol from input"
        ]
        return updates

    gene_symbol = gene_symbol.upper()
    updates["gene_symbol"] = gene_symbol

    # ---- Step 2: Fetch transcripts (cache-first) ----
    transcripts = _cache.get_transcripts(gene_symbol, genome_build)
    if transcripts is None:
        logger.info("Cache miss for %s — calling Ensembl", gene_symbol)
        transcripts = get_transcripts_for_gene(gene_symbol, genome_build)
        if transcripts:
            _cache.set_transcripts(gene_symbol, genome_build, transcripts)
    else:
        logger.info("Cache hit for %s — %d transcripts", gene_symbol, len(transcripts))

    if not transcripts:
        # Proceed without transcript resolution
        warnings.append(f"No Ensembl transcripts found for {gene_symbol}")
        updates["hgvs_on_transcript"] = f"{gene_symbol} {cdna}" if cdna else gene_symbol
        updates["warnings"] = warnings
        return updates

    # ---- Step 3: Gene aliases and full name ----
    try:
        gene_info = get_gene_aliases(gene_symbol)
        updates["gene_aliases"] = gene_info.get("aliases", [])
        updates["gene_full_name"] = gene_info.get("full_name", "")
        # Propagate aliases to transcript records
        for tx in transcripts:
            tx["gene_aliases"] = gene_info.get("aliases", [])
            tx["gene_full_name"] = gene_info.get("full_name", "")
    except Exception as e:
        logger.warning("Gene alias lookup failed: %s", e)

    # ---- Step 4: Count pathogenic submissions per transcript ----
    try:
        path_counts = count_pathogenic_submissions_per_transcript(gene_symbol)
        if path_counts:
            top_nm = path_counts.most_common(1)[0][0]
            for tx in transcripts:
                if tx["nm_accession"] == top_nm:
                    tx["is_most_reported_pathogenic"] = True
    except Exception as e:
        logger.warning("Pathogenic count lookup failed: %s", e)

    # ---- Step 5: Score and sort transcripts ----
    for tx in transcripts:
        score = 0
        if tx.get("is_mane_select"):
            score += 2
        if tx.get("is_mane_plus_clinical"):
            score += 2
        if tx.get("is_most_reported_pathogenic"):
            score += 1
        tx["annotation_score"] = score

    transcripts.sort(
        key=lambda t: (-t["annotation_score"], t.get("nm_accession", "")),
    )

    updates["all_transcripts"] = transcripts

    # ---- Step 6: Recode variant to all transcripts ----
    if cdna:
        hgvs_query = f"{gene_symbol} {cdna}"
        # Try variant recoder for proper multi-transcript mapping
        try:
            recoded = recode_variant(hgvs_query, genome_build)
            if recoded:
                for tx in transcripts:
                    nm = tx.get("nm_accession", "")
                    enst = tx.get("enst_accession", "")
                    # Look up by both NM_ and ENST
                    equiv = recoded.get(nm) or recoded.get(enst) or ""
                    tx["equivalent_hgvs"] = equiv
            else:
                # Fallback: assume the cdna notation is valid on all transcripts
                for tx in transcripts:
                    if tx.get("nm_accession"):
                        tx["equivalent_hgvs"] = f"{tx['nm_accession']}:{cdna}"
        except Exception as e:
            logger.warning("Variant recoder failed: %s — using fallback", e)
            for tx in transcripts:
                if tx.get("nm_accession"):
                    tx["equivalent_hgvs"] = f"{tx['nm_accession']}:{cdna}"

    # ---- Step 7: Select best transcript ----
    selected = transcripts[0] if transcripts else None
    if selected:
        updates["selected_transcript"] = (
            selected.get("nm_accession") or selected.get("enst_accession")
        )

    # ---- Step 8: Set hgvs_on_transcript ----
    if selected and selected.get("equivalent_hgvs"):
        updates["hgvs_on_transcript"] = selected["equivalent_hgvs"]
    elif selected and cdna:
        nm = selected.get("nm_accession")
        if nm:
            updates["hgvs_on_transcript"] = f"{nm}:{cdna}"
        else:
            updates["hgvs_on_transcript"] = f"{gene_symbol} {cdna}"
    else:
        updates["hgvs_on_transcript"] = state["raw_input"]

    updates["warnings"] = warnings
    logger.info(
        "input_parser_node: gene=%s, transcript=%s, hgvs=%s, %d transcripts",
        updates.get("gene_symbol"),
        updates.get("selected_transcript"),
        updates.get("hgvs_on_transcript"),
        len(transcripts),
    )
    return updates
