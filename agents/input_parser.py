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
        # No transcript specified — build a list of candidate NM_ transcripts
        # to try with VEP. The cdna notation may only be valid on specific
        # transcript isoforms, so we try multiple.
        candidate_nms: list[str] = []

        # Source 1: ClinVar pathogenic submission counts
        try:
            path_counts = count_pathogenic_submissions_per_transcript(
                gene_symbol.upper()
            )
            if path_counts:
                candidate_nms.extend(nm for nm, _ in path_counts.most_common(5))
        except Exception as e:
            logger.warning("Could not get NMs from ClinVar: %s", e)

        # Source 2: Ensembl canonical transcript
        try:
            from tools.ensembl import _ensembl_get, _base_url, enst_to_nm
            base = _base_url(genome_build)
            gene_data = _ensembl_get(
                f"{base}/lookup/symbol/homo_sapiens/{gene_symbol.upper()}?expand=1"
            )
            if gene_data:
                for tx in gene_data.get("Transcript", []):
                    if tx.get("is_canonical") and tx.get("biotype") == "protein_coding":
                        enst = tx.get("id", "")
                        nm = enst_to_nm(enst, genome_build)
                        if nm and nm not in candidate_nms:
                            candidate_nms.append(nm)
                        # Keep ENST as last resort
                        if enst not in candidate_nms:
                            candidate_nms.append(enst)
                        break
        except Exception as e:
            logger.warning("Ensembl canonical lookup failed: %s", e)

        # Try each candidate until VEP accepts one
        from tools.ensembl import vep_annotate_hgvs
        for nm_candidate in candidate_nms:
            test_hgvs = f"{nm_candidate}:{cdna}"
            test_results = vep_annotate_hgvs(test_hgvs, genome_build)
            if test_results:
                hgvs_for_vep = test_hgvs
                logger.info("input_parser: VEP accepted %s", test_hgvs)
                break
            else:
                logger.info("input_parser: VEP rejected %s, trying next", test_hgvs)

        if not hgvs_for_vep:
            logger.warning("input_parser: no transcript accepted by VEP for %s %s",
                           gene_symbol, cdna)

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

    # Store genomic coordinates
    # For coordinate input: use the user's original values (already VCF-normalized)
    # For HGVS input: use left-aligned coordinates from VEP
    if chrom and pos and ref and alt:
        # User provided coordinates — trust them as-is
        updates["chrom"] = str(chrom).lstrip("chr")
        updates["pos"] = pos
        updates["ref"] = ref
        updates["alt"] = alt
    elif resolution.get("chrom"):
        # VEP-resolved coordinates (already left-aligned by resolve_transcripts)
        updates["chrom"] = resolution["chrom"]
        updates["pos"] = resolution["pos"]
        updates["ref"] = resolution["ref"]
        updates["alt"] = resolution["alt"]

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
    # Prefer NM_-based HGVS over ENST for ClinVar compatibility.
    # VEP returns ENST-based hgvsc, but ClinVar searches work better with NM_.
    if selected.get("nm_accession") and cdna:
        updates["hgvs_on_transcript"] = f"{selected['nm_accession']}:{cdna}"
    elif selected.get("hgvsc"):
        updates["hgvs_on_transcript"] = selected["hgvsc"]
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
