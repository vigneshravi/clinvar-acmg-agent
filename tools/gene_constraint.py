"""Gene constraint metrics and protein domain annotations.

Queries gnomAD for missense/LOF constraint and UniProt for functional
domain positions. Used for ACMG criteria PM1 (mutational hotspot /
functional domain) and PM4/BP3 (protein length change).

Also hosts the ClinGen Dosage Sensitivity (Haploinsufficiency Score) loader
ported from FinalTermProject_28Apr2026/src/tools.py on 2026-04-30 for the
SVI-aware PVS1 gate (Riggs et al. 2020).
"""

import json as _json
import logging
import re
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

# `requests` is used by the ClinGen TSV loader below. It's a transitive
# dependency of langchain/langchain-anthropic in PathoMAN's existing
# requirements, so this import is safe in practice; if it ever isn't,
# the import lives inside _load_clingen_dosage_tsv() rather than at the
# top so the rest of this module is not impacted.

logger = logging.getLogger(__name__)

GNOMAD_API = "https://gnomad.broadinstitute.org/api"


# ---------------------------------------------------------------------------
# gnomAD gene constraint
# ---------------------------------------------------------------------------

def get_gene_constraint(
    gene_symbol: str, genome_build: str = "GRCh38"
) -> Optional[dict[str, Any]]:
    """Fetch gnomAD constraint metrics for a gene.

    Returns dict with missense Z-score, o/e ratios, pLI, LOEUF.
    """
    ref = "GRCh38" if genome_build != "GRCh37" else "GRCh37"
    query = (
        f'{{gene(gene_symbol: "{gene_symbol}", reference_genome: {ref}) {{'
        f'gnomad_constraint {{'
        f'mis_z oe_mis oe_mis_lower oe_mis_upper '
        f'pli oe_lof oe_lof_upper lof_z'
        f'}}}}}}'
    )
    payload = _json.dumps({"query": query}).encode("utf-8")
    try:
        req = Request(GNOMAD_API, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning("gnomAD constraint query failed: %s", e)
        return None

    gene_data = (data.get("data") or {}).get("gene")
    if not gene_data:
        return None

    c = gene_data.get("gnomad_constraint")
    if not c:
        return None

    loeuf = c.get("oe_lof_upper", 1.0)
    mis_z = c.get("mis_z", 0)
    oe_mis = c.get("oe_mis", 1.0)

    return {
        "mis_z": mis_z,
        "oe_mis": oe_mis,
        "oe_mis_lower": c.get("oe_mis_lower"),
        "oe_mis_upper": c.get("oe_mis_upper"),
        "pli": c.get("pli"),
        "oe_lof": c.get("oe_lof"),
        "loeuf": loeuf,
        "lof_z": c.get("lof_z"),
        # Interpretations
        "missense_constrained": mis_z > 3.09,
        "lof_intolerant": loeuf < 0.35,
        "lof_intolerant_moderate": loeuf < 0.6,
        "missense_interpretation": (
            "Highly constrained (Z > 3.09)" if mis_z > 3.09
            else "Moderately constrained" if mis_z > 2.0
            else "Not constrained"
        ),
        "lof_interpretation": (
            "Highly LOF intolerant (LOEUF < 0.35)" if loeuf < 0.35
            else "LOF intolerant (LOEUF < 0.6)" if loeuf < 0.6
            else "LOF tolerant"
        ),
    }


# ---------------------------------------------------------------------------
# UniProt protein domains
# ---------------------------------------------------------------------------

def get_uniprot_domains(gene_symbol: str) -> dict[str, Any]:
    """Fetch functional domain and repeat annotations from UniProt.

    Returns dict with domains, repeats, zinc_fingers, dna_binding regions,
    and the UniProt accession.
    """
    url = (
        f"https://rest.uniprot.org/uniprotkb/search?"
        f"query=gene:{gene_symbol}+AND+organism_id:9606+AND+reviewed:true"
        f"&fields=ft_domain,ft_repeat,ft_zn_fing,ft_dna_bind,ft_motif,"
        f"ft_region,accession,sequence&format=json&size=1"
    )
    try:
        req = Request(url)
        req.add_header("Accept", "application/json")
        with urlopen(req, timeout=15) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        logger.warning("UniProt query failed for %s: %s", gene_symbol, e)
        return {"domains": [], "accession": ""}

    results = data.get("results", [])
    if not results:
        return {"domains": [], "accession": ""}

    entry = results[0]
    accession = entry.get("primaryAccession", "")

    # Get protein length
    seq_data = entry.get("sequence", {})
    protein_length = seq_data.get("length", 0)

    # Parse features into functional regions
    domains = []
    repeats = []

    for f in entry.get("features", []):
        ftype = f.get("type", "")
        desc = f.get("description", "")
        loc = f.get("location", {})
        start = loc.get("start", {}).get("value")
        end = loc.get("end", {}).get("value")

        if start is None or end is None:
            continue

        region = {
            "type": ftype,
            "description": desc,
            "start": int(start),
            "end": int(end),
        }

        if ftype in ("Domain", "Zinc finger", "DNA binding"):
            domains.append(region)
        elif ftype == "Repeat":
            repeats.append(region)
        elif ftype == "Motif":
            domains.append(region)
        # Skip generic "Region" annotations (too noisy)

    return {
        "accession": accession,
        "protein_length": protein_length,
        "domains": domains,
        "repeats": repeats,
        "has_domains": len(domains) > 0,
        "has_repeats": len(repeats) > 0,
    }


def check_domain_overlap(
    protein_position: Optional[int],
    domains: list[dict],
) -> Optional[dict]:
    """Check if a protein position falls within a functional domain."""
    if protein_position is None:
        return None
    for d in domains:
        if d["start"] <= protein_position <= d["end"]:
            return d
    return None


def check_repeat_overlap(
    protein_position: Optional[int],
    repeats: list[dict],
) -> bool:
    """Check if a protein position falls within a repeat region."""
    if protein_position is None:
        return False
    for r in repeats:
        if r["start"] <= protein_position <= r["end"]:
            return True
    return False


# ---------------------------------------------------------------------------
# ACMG criteria assessment
# ---------------------------------------------------------------------------

def assess_pm1(
    constraint: Optional[dict],
    domain_hit: Optional[dict],
    consequence_terms: list[str],
) -> dict[str, Any]:
    """Assess PM1: located in a mutational hotspot / functional domain.

    PM1 applies when:
    - Variant is in a well-established functional domain (UniProt) AND
    - Gene is missense-constrained (gnomAD Z > 2.0 or domain is critical)
    - Only for missense variants
    """
    result = {
        "met": False,
        "strength": "Moderate",
        "justification": "",
        "domain": None,
        "constraint_info": "",
    }

    is_missense = "missense_variant" in consequence_terms
    if not is_missense:
        result["justification"] = "Not a missense variant — PM1 does not apply"
        return result

    has_domain = domain_hit is not None
    has_constraint = (constraint and constraint.get("mis_z", 0) > 2.0)

    if has_domain and has_constraint:
        result["met"] = True
        result["domain"] = domain_hit
        result["justification"] = (
            f"Missense variant in {domain_hit['type']} domain "
            f"'{domain_hit['description']}' (aa {domain_hit['start']}-{domain_hit['end']}) "
            f"in gene with missense constraint Z={constraint['mis_z']:.2f}"
        )
    elif has_domain:
        result["met"] = True
        result["strength"] = "Supporting"
        result["domain"] = domain_hit
        result["justification"] = (
            f"Missense variant in {domain_hit['type']} domain "
            f"'{domain_hit['description']}' (aa {domain_hit['start']}-{domain_hit['end']}); "
            f"gene is not highly missense-constrained — PM1 downgraded to Supporting"
        )
    elif has_constraint:
        result["justification"] = (
            f"Gene is missense-constrained (Z={constraint['mis_z']:.2f}) "
            f"but variant is not in a known functional domain"
        )
    else:
        result["justification"] = "No functional domain overlap and gene not missense-constrained"

    if constraint:
        result["constraint_info"] = constraint.get("missense_interpretation", "")

    return result


def assess_pm4_bp3(
    consequence_terms: list[str],
    protein_start: Optional[int],
    protein_end: Optional[int],
    in_repeat: bool,
) -> dict[str, Any]:
    """Assess PM4 and BP3 for in-frame protein length changes.

    PM4: In-frame deletion/insertion in a non-repeat region (Moderate Pathogenic)
    BP3: In-frame deletion/insertion in a repetitive region (Supporting Benign)
    """
    result = {
        "pm4_met": False,
        "bp3_met": False,
        "strength": "",
        "justification": "",
        "aa_change_size": None,
    }

    inframe_types = {"inframe_deletion", "inframe_insertion"}
    is_inframe = bool(inframe_types & set(consequence_terms))

    if not is_inframe:
        result["justification"] = "Not an in-frame insertion/deletion — PM4/BP3 do not apply"
        return result

    # Compute size of protein change
    if protein_start is not None and protein_end is not None:
        result["aa_change_size"] = abs(protein_end - protein_start) + 1

    if in_repeat:
        result["bp3_met"] = True
        result["strength"] = "Supporting"
        result["justification"] = (
            "In-frame insertion/deletion in a repetitive region — "
            "supports BP3 (Supporting Benign)"
        )
    else:
        result["pm4_met"] = True
        result["strength"] = "Moderate"
        result["justification"] = (
            "In-frame insertion/deletion in a non-repeat region, "
            "causing protein length change — supports PM4 (Moderate Pathogenic)"
        )
        if result["aa_change_size"]:
            result["justification"] += f" ({result['aa_change_size']} amino acids affected)"

    return result


# ---------------------------------------------------------------------------
# ClinGen Dosage Sensitivity — Haploinsufficiency Score (PVS1 applicability)
# ---------------------------------------------------------------------------
# Ported from FinalTermProject_28Apr2026/src/tools.py on 2026-04-30 to support
# the SVI-aware PVS1 evaluator (Abou Tayoun 2018 + Riggs 2020 HI gate).
#
# Implementation: fetch the ClinGen TSV at first invocation, parse into a
# module-level dict, subsequent lookups are local (sub-second). The cache
# survives the Python process; a Streamlit restart re-fetches.
#
# Source TSV: https://ftp.clinicalgenome.org/ClinGen_gene_curation_list_GRCh38.tsv
# ClinGen does not provide a JSON REST API for these scores; the FTP TSV is
# the canonical source. Keys (per ClinGen scoring rubric):
#   3  = Sufficient evidence for HI / dosage pathogenicity (PVS1 APPLICABLE)
#   2  = Some evidence for HI                            (PVS1 with caveat)
#   1  = Little evidence for HI                          (PVS1 NOT applicable)
#   0  = No evidence for HI                              (PVS1 NOT applicable)
#   30 = Gene associated with autosomal recessive phenotype
#        (PVS1 only applicable in trans with another pathogenic variant)
#   40 = Dosage sensitivity unlikely                     (PVS1 NOT applicable)
# ---------------------------------------------------------------------------

_CLINGEN_DOSAGE_TSV_URL = (
    "https://ftp.clinicalgenome.org/ClinGen_gene_curation_list_GRCh38.tsv"
)
_CLINGEN_DOSAGE_CACHE: Optional[dict[str, dict[str, Any]]] = None


def _load_clingen_dosage_tsv() -> dict[str, dict[str, Any]]:
    """Fetch + parse the ClinGen dosage TSV. Returns dict keyed by gene_symbol.

    Source: https://ftp.clinicalgenome.org/ClinGen_gene_curation_list_GRCh38.tsv
    Refreshed by ClinGen — last-modified header surfaced in result.
    """
    import requests  # local import — see top-of-module note
    logger.info("Fetching ClinGen dosage TSV from %s", _CLINGEN_DOSAGE_TSV_URL)
    r = requests.get(_CLINGEN_DOSAGE_TSV_URL, timeout=60)
    r.raise_for_status()
    last_modified = r.headers.get("Last-Modified", "unknown")

    out: dict[str, dict[str, Any]] = {}
    for line in r.text.splitlines():
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) < 6:
            continue
        gene = cols[0]
        try:
            hi_score = int(cols[4]) if cols[4].lstrip("-").isdigit() else None
        except (ValueError, IndexError):
            hi_score = None
        out[gene] = {
            "gene_symbol": gene,
            "gene_id": cols[1],
            "cytoband": cols[2],
            "genomic_location": cols[3],
            "hi_score": hi_score,
            "hi_description": cols[5] if len(cols) > 5 else "",
            "hi_pmids": [c for c in cols[6:12] if c],
            "ts_score": int(cols[12]) if len(cols) > 12 and cols[12].lstrip("-").isdigit() else None,
            "ts_description": cols[13] if len(cols) > 13 else "",
            "date_last_evaluated": cols[20] if len(cols) > 20 else "",
            "_source_last_modified": last_modified,
        }
    logger.info(
        "Parsed %d ClinGen dosage entries (TSV last-modified: %s)",
        len(out), last_modified,
    )
    return out


def clingen_dosage_lookup(gene_symbol: str) -> dict[str, Any]:
    """Return ClinGen Haploinsufficiency Score + metadata for a gene.

    Loads the TSV once per Python session (~265 KB, sub-second) and caches
    in memory. Subsequent calls are dict lookups.

    Returns dict with hi_score (int or None), hi_description, pmids, etc.
    Used by svi.acmg_rules.evaluate_PVS1 — see svi/constants.py:
        hi_score=3   -> PVS1 applicable
        hi_score=2   -> PVS1 applicable with caveat
        hi_score=30  -> AR gene (PVS1 only in trans)
        hi_score in {0, 1, 40} -> PVS1 not applicable
    """
    global _CLINGEN_DOSAGE_CACHE
    if _CLINGEN_DOSAGE_CACHE is None:
        try:
            _CLINGEN_DOSAGE_CACHE = _load_clingen_dosage_tsv()
        except Exception as e:
            return {
                "available": False,
                "error": f"ClinGen TSV fetch failed: {e}",
                "gene_symbol": gene_symbol,
            }

    rec = _CLINGEN_DOSAGE_CACHE.get(gene_symbol)
    if not rec:
        return {
            "available": False,
            "gene_symbol": gene_symbol,
            "error": "gene not in ClinGen dosage curation list",
        }
    return {"available": True, **rec}
