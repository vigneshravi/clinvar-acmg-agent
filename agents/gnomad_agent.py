"""Node 3: gnomAD + In Silico Agent — population frequencies and computational
predictions for ACMG criteria BA1, BS1, PM2, PP3, BP4.

Data sources:
- gnomAD GraphQL API (GRCh38-native allele frequencies by population)
- MyVariant.info (dbNSFP scores: REVEL, CADD, MetaRNN, BayesDel,
  AlphaMissense, SIFT, PolyPhen2, phyloP, phastCons, GERP++)
"""

import logging
import re
from typing import Any, Optional

from graph.state import VariantState
from tools.gnomad_graphql import (
    GNOMAD_DATASETS,
    query_gnomad_by_rsid,
    query_gnomad_variant,
)
from tools.vcf_normalize import normalize_vep_vcf_string
from tools.myvariant import query_myvariant

logger = logging.getLogger(__name__)

# ACMG thresholds
BA1_THRESHOLD = 0.05      # >5% in any population → stand-alone benign
BS1_THRESHOLD = 0.01      # >1% → strong benign (for rare disease genes)
PM2_THRESHOLD = 0.0001    # <0.01% or absent → supporting pathogenic


def _extract_coordinates_from_hgvs(hgvs: str) -> Optional[tuple[str, int, str, str]]:
    """Try to extract genomic coordinates from HGVS notation.

    This is a fallback — the input_parser should set chrom/pos/ref/alt
    but may not for all input modes.
    """
    # Can't reliably parse genomic coords from transcript HGVS
    # This needs VEP, which the input_parser already called
    return None


def _compute_frequency_criteria(
    gnomad_data: Optional[dict],
) -> dict[str, Any]:
    """Compute ACMG frequency-based criteria from gnomAD data."""
    criteria = {
        "BA1_met": False,
        "BA1_detail": "",
        "BS1_met": False,
        "BS1_detail": "",
        "PM2_met": False,
        "PM2_detail": "",
    }

    if gnomad_data is None:
        criteria["PM2_met"] = True
        criteria["PM2_detail"] = "Variant absent from gnomAD — supports PM2"
        return criteria

    global_af = gnomad_data.get("global_af") or 0
    max_pop_af = gnomad_data.get("max_pop_af", 0)
    max_pop_name = gnomad_data.get("max_pop_name", "")
    hom = gnomad_data.get("hom", 0)

    # BA1: AF > 5% in any population
    check_af = max(global_af, max_pop_af)
    if check_af > BA1_THRESHOLD:
        criteria["BA1_met"] = True
        criteria["BA1_detail"] = (
            f"Allele frequency {check_af:.4f} exceeds 5% threshold "
            f"(max population: {max_pop_name})"
        )

    # BS1: AF > 1% — greater than expected for rare disease
    elif check_af > BS1_THRESHOLD:
        criteria["BS1_met"] = True
        criteria["BS1_detail"] = (
            f"Allele frequency {check_af:.4f} exceeds 1% threshold "
            f"for rare disease genes"
        )

    # PM2: Absent or extremely rare in controls
    if global_af < PM2_THRESHOLD and max_pop_af < PM2_THRESHOLD:
        criteria["PM2_met"] = True
        if global_af == 0:
            criteria["PM2_detail"] = "Variant absent from gnomAD"
        else:
            criteria["PM2_detail"] = (
                f"Variant extremely rare in gnomAD (AF={global_af:.6f})"
            )

    return criteria


def _compute_insilico_consensus(
    predictors: dict[str, Any],
) -> dict[str, Any]:
    """Compute PP3/BP4 from in silico predictor consensus."""
    result = {
        "PP3_met": False,
        "PP3_detail": "",
        "BP4_met": False,
        "BP4_detail": "",
        "consensus": "",
        "damaging_count": 0,
        "benign_count": 0,
        "total_count": 0,
    }

    damaging = []
    benign = []

    # REVEL (best single predictor per ClinGen SVI)
    revel = predictors.get("revel", {})
    if isinstance(revel, dict) and revel.get("score") is not None:
        if revel["score"] > 0.75:
            damaging.append(f"REVEL={revel['score']}")
        elif revel["score"] < 0.15:
            benign.append(f"REVEL={revel['score']}")

    # CADD
    cadd = predictors.get("cadd_phred")
    if cadd is not None:
        if cadd >= 25:
            damaging.append(f"CADD={cadd}")
        elif cadd < 15:
            benign.append(f"CADD={cadd}")

    # AlphaMissense
    am = predictors.get("alphamissense", {})
    if isinstance(am, dict) and am.get("score") is not None:
        if am["score"] > 0.564:
            damaging.append(f"AlphaMissense={am['score']}")
        elif am["score"] < 0.34:
            benign.append(f"AlphaMissense={am['score']}")

    # BayesDel
    bd = predictors.get("bayesdel_af", {})
    if isinstance(bd, dict) and bd.get("score") is not None:
        if bd["score"] > 0.0692:
            damaging.append(f"BayesDel={bd['score']}")
        elif bd["score"] < -0.0570:
            benign.append(f"BayesDel={bd['score']}")

    # MetaRNN
    mr = predictors.get("metarnn", {})
    if isinstance(mr, dict) and mr.get("pred"):
        if mr["pred"] == "Damaging":
            damaging.append(f"MetaRNN=D")
        else:
            benign.append(f"MetaRNN=T")

    # SIFT
    sift = predictors.get("sift", {})
    if isinstance(sift, dict) and sift.get("score") is not None:
        if sift["score"] < 0.05:
            damaging.append(f"SIFT={sift['score']}")
        else:
            benign.append(f"SIFT={sift['score']}")

    # PolyPhen2
    pp2 = predictors.get("polyphen2", {})
    if isinstance(pp2, dict) and pp2.get("score") is not None:
        if pp2["score"] > 0.85:
            damaging.append(f"PolyPhen2={pp2['score']}")
        elif pp2["score"] < 0.15:
            benign.append(f"PolyPhen2={pp2['score']}")

    result["damaging_count"] = len(damaging)
    result["benign_count"] = len(benign)
    result["total_count"] = len(damaging) + len(benign)

    # PP3: majority of tools predict damaging
    if len(damaging) >= 3 and len(damaging) > len(benign):
        result["PP3_met"] = True
        result["PP3_detail"] = (
            f"{len(damaging)}/{result['total_count']} predictors support "
            f"damaging effect: {', '.join(damaging)}"
        )
        result["consensus"] = "Damaging"

    # BP4: majority of tools predict benign
    elif len(benign) >= 3 and len(benign) > len(damaging):
        result["BP4_met"] = True
        result["BP4_detail"] = (
            f"{len(benign)}/{result['total_count']} predictors support "
            f"benign effect: {', '.join(benign)}"
        )
        result["consensus"] = "Benign"
    else:
        result["consensus"] = "Uncertain"

    return result


def gnomad_agent_node(state: VariantState) -> dict[str, Any]:
    """Fetch gnomAD frequencies and in silico predictions.

    Steps:
    1. Get genomic coordinates (from state or parse from HGVS)
    2. Query gnomAD GraphQL for population allele frequencies
    3. Query MyVariant.info for dbNSFP + CADD scores
    4. Compute ACMG criteria (BA1, BS1, PM2, PP3, BP4)
    5. Return structured annotation dict
    """
    logger.info("gnomad_agent_node: starting annotation lookup")

    updates: dict[str, Any] = {"current_node": "gnomad_agent"}
    warnings: list[str] = list(state.get("warnings", []))

    chrom = state.get("chrom")
    pos = state.get("pos")
    ref_allele = state.get("ref")
    alt_allele = state.get("alt")
    genome_build = state.get("genome_build", "GRCh38")

    # Extract rsID from ClinVar if available (most reliable for gnomAD lookup)
    clinvar = state.get("clinvar") or {}
    rsid_from_clinvar = clinvar.get("rsid")

    # VCF-style gnomAD variant ID (chrom-pos-ref-alt format)
    gnomad_variant_id = None

    # If coordinates already present (from user input or input_parser),
    # use them directly — they're already in VCF format, no normalization needed
    if chrom and pos and ref_allele and alt_allele:
        chrom_clean = str(chrom).lstrip("chr")
        gnomad_variant_id = f"{chrom_clean}-{pos}-{ref_allele}-{alt_allele}"
        logger.info("gnomad_agent: using pre-resolved coordinates %s", gnomad_variant_id)

    # If we don't have coordinates, resolve via VEP
    if not (chrom and pos and ref_allele and alt_allele):
        hgvs = state.get("hgvs_on_transcript", "")
        if hgvs:
            logger.info("gnomad_agent: resolving coordinates via VEP for %s", hgvs)
            try:
                from tools.ensembl import _ensembl_get, _base_url
                from urllib.parse import quote
                base = _base_url(genome_build)
                encoded = quote(hgvs, safe="")
                url = f"{base}/vep/homo_sapiens/hgvs/{encoded}?vcf_string=1"
                vep_raw = _ensembl_get(url)
                if vep_raw and isinstance(vep_raw, list) and vep_raw:
                    entry = vep_raw[0]
                    chrom = str(entry.get("seq_region_name", ""))
                    pos = entry.get("start")
                    allele_string = entry.get("allele_string", "")
                    if "/" in allele_string:
                        parts = allele_string.split("/")
                        ref_allele = parts[0]
                        alt_allele = parts[1] if len(parts) > 1 else ""

                    # Get vcf_string and left-align for correct gnomAD lookup
                    vcf_string = entry.get("vcf_string", "")
                    if vcf_string and "-" in vcf_string:
                        # Left-align to match gnomAD's normalization
                        norm_chrom, norm_pos, norm_ref, norm_alt = (
                            normalize_vep_vcf_string(vcf_string, genome_build)
                        )
                        chrom = norm_chrom
                        pos = norm_pos
                        ref_allele = norm_ref
                        alt_allele = norm_alt
                        gnomad_variant_id = f"{norm_chrom}-{norm_pos}-{norm_ref}-{norm_alt}"
                        logger.info("gnomad_agent: left-aligned %s → %s",
                                    vcf_string, gnomad_variant_id)

                    # Write coordinates back to state
                    updates["chrom"] = chrom
                    updates["pos"] = pos
                    updates["ref"] = ref_allele
                    updates["alt"] = alt_allele

                    logger.info("gnomad_agent: coordinates %s:%s %s>%s",
                                chrom, pos, ref_allele, alt_allele)
            except Exception as e:
                logger.warning("gnomad_agent: VEP resolution failed: %s", e)

    if not (chrom and pos):
        warnings.append("Could not determine genomic coordinates — gnomAD/dbNSFP lookup skipped")
        updates["gnomad"] = None
        updates["gnomad_error"] = "No genomic coordinates available"
        updates["warnings"] = warnings
        return updates

    # ---- Step 1: Query gnomAD GraphQL ----
    # Strategy: try multiple variant ID formats since gnomAD requires
    # left-aligned VCF normalization which may differ from VEP output
    gnomad_data = None
    rsid = rsid_from_clinvar

    # Build list of variant IDs to try
    gnomad_ids_to_try = []
    if gnomad_variant_id:
        # VEP vcf_string (forward strand)
        gnomad_ids_to_try.append(gnomad_variant_id)
    if chrom and pos and ref_allele and alt_allele:
        # Direct from coordinates
        direct_id = f"{str(chrom).lstrip('chr')}-{pos}-{ref_allele}-{alt_allele}"
        if direct_id not in gnomad_ids_to_try:
            gnomad_ids_to_try.append(direct_id)

    try:
        # Try each variant ID
        for vid in gnomad_ids_to_try:
            parts = vid.split("-")
            if len(parts) == 4:
                gnomad_data = query_gnomad_variant(
                    parts[0], int(parts[1]), parts[2], parts[3], genome_build
                )
                if gnomad_data:
                    rsid = gnomad_data.get("rsid") or rsid
                    logger.info("gnomAD: found via ID %s, AF=%s",
                                vid, gnomad_data.get("global_af"))
                    break

        # If not found by coordinate, try rsID lookup
        if not gnomad_data and rsid:
            gnomad_data = query_gnomad_by_rsid(rsid, genome_build)
            if gnomad_data:
                rsid = gnomad_data.get("rsid") or rsid
                logger.info("gnomAD: found via rsID %s, AF=%s",
                            rsid, gnomad_data.get("global_af"))

        if not gnomad_data:
            logger.info("gnomAD: variant not found")
    except Exception as e:
        logger.warning("gnomAD query failed: %s", e)
        warnings.append(f"gnomAD query failed: {str(e)}")

    # ---- Step 2: Query MyVariant.info for dbNSFP + CADD ----
    myvariant_data = None
    try:
        myvariant_data = query_myvariant(
            chrom, pos, ref_allele or "", alt_allele or "",
            genome_build=genome_build,
            rsid=rsid,
        )
        if myvariant_data:
            logger.info("MyVariant.info: got dbNSFP/CADD data")
            # Use MyVariant rsID if gnomAD didn't have one
            if not rsid and myvariant_data.get("rsid"):
                rsid = myvariant_data["rsid"]
        else:
            logger.info("MyVariant.info: variant not found")
    except Exception as e:
        logger.warning("MyVariant.info query failed: %s", e)
        warnings.append(f"MyVariant.info query failed: {str(e)}")

    # ---- Step 3: Merge and compute criteria ----
    # Use gnomAD GraphQL as primary AF source (GRCh38-native),
    # fall back to MyVariant.info gnomAD data
    af_source = gnomad_data
    if not af_source and myvariant_data:
        af_source = myvariant_data.get("allele_frequency")
        if af_source:
            # Reshape to match gnomad_data format
            af_source = {
                "global_af": af_source.get("global_af"),
                "max_pop_af": max(af_source.get("populations", {}).values(), default=0)
                    if isinstance(list(af_source.get("populations", {}).values())[0] if af_source.get("populations") else 0, (int, float))
                    else 0,
                "max_pop_name": "",
                "hom": af_source.get("hom", 0),
                "populations": af_source.get("populations", {}),
            }

    freq_criteria = _compute_frequency_criteria(af_source if af_source else None)

    # In silico consensus
    predictors = {}
    if myvariant_data:
        predictors = myvariant_data.get("insilico_predictors", {})
    insilico_criteria = _compute_insilico_consensus(predictors)

    conservation = {}
    if myvariant_data:
        conservation = myvariant_data.get("conservation", {})

    # ---- Step 4: Build the gnomad state dict ----
    # Determine which dataset was actually used
    from tools.gnomad_graphql import _default_dataset
    used_dataset = (gnomad_data or {}).get("dataset") or _default_dataset(genome_build)

    gnomad_result: dict[str, Any] = {
        "source": "gnomAD GraphQL + MyVariant.info",
        "rsid": rsid,
        "dataset": used_dataset,
        "coordinates": f"chr{str(chrom).lstrip('chr')}:{pos} {ref_allele}>{alt_allele}" if chrom and pos else None,
        "chrom": str(chrom).lstrip("chr") if chrom else None,
        "pos": pos,
        "ref": ref_allele,
        "alt": alt_allele,
        "gnomad_variant_id": gnomad_variant_id or ((gnomad_data or {}).get("variant_id")),
        # Allele frequencies
        "allele_frequency": {
            "global_af": (gnomad_data or {}).get("global_af"),
            "exome": (gnomad_data or {}).get("exome"),
            "genome": (gnomad_data or {}).get("genome"),
            "populations": (gnomad_data or {}).get("populations", {}),
            "hom": (gnomad_data or {}).get("hom", 0),
            "max_pop_af": (gnomad_data or {}).get("max_pop_af", 0),
            "max_pop_name": (gnomad_data or {}).get("max_pop_name", ""),
            "variant_in_gnomad": gnomad_data is not None,
        },
        # In silico predictors
        "insilico_predictors": predictors,
        "insilico_consensus": insilico_criteria["consensus"],
        # Conservation
        "conservation": conservation,
        # Pre-computed ACMG criteria
        "acmg_criteria": {
            **freq_criteria,
            **{k: v for k, v in insilico_criteria.items()
               if k.startswith(("PP3", "BP4"))},
            "insilico_damaging_count": insilico_criteria["damaging_count"],
            "insilico_benign_count": insilico_criteria["benign_count"],
        },
    }

    updates["gnomad"] = gnomad_result
    updates["warnings"] = warnings

    logger.info(
        "gnomad_agent: AF=%s, PM2=%s, BA1=%s, PP3=%s, BP4=%s, consensus=%s",
        gnomad_result["allele_frequency"].get("global_af"),
        freq_criteria["PM2_met"],
        freq_criteria["BA1_met"],
        insilico_criteria["PP3_met"],
        insilico_criteria["BP4_met"],
        insilico_criteria["consensus"],
    )

    return updates
