"""Ensembl REST API calls for transcript resolution, VEP annotation,
and NM↔ENST mapping."""

import json as _json
import logging
import re
import time
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

ENSEMBL_REST_GRCH38 = "https://rest.ensembl.org"
ENSEMBL_REST_GRCH37 = "https://grch37.rest.ensembl.org"


def _base_url(genome_build: str) -> str:
    return ENSEMBL_REST_GRCH37 if genome_build == "GRCh37" else ENSEMBL_REST_GRCH38


def _ensembl_get(url: str, retries: int = 2) -> Optional[Any]:
    """GET from Ensembl REST with retry on 429."""
    for attempt in range(retries + 1):
        try:
            req = Request(url)
            req.add_header("Content-Type", "application/json")
            with urlopen(req, timeout=20) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = float(e.headers.get("Retry-After", "1"))
                logger.warning("Ensembl 429 — retrying in %.1fs", wait)
                time.sleep(wait)
                continue
            if e.code in (400, 404):
                return None
            logger.error("Ensembl HTTP %d for %s", e.code, url)
            return None
        except Exception as exc:
            logger.error("Ensembl request failed: %s", exc)
            return None
    return None


def _ensembl_post(url: str, body: dict, retries: int = 2) -> Optional[Any]:
    """POST to Ensembl REST with retry on 429."""
    data = _json.dumps(body).encode("utf-8")
    for attempt in range(retries + 1):
        try:
            req = Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            with urlopen(req, timeout=25) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = float(e.headers.get("Retry-After", "1"))
                time.sleep(wait)
                continue
            logger.error("Ensembl POST HTTP %d for %s", e.code, url)
            return None
        except Exception as exc:
            logger.error("Ensembl POST failed: %s", exc)
            return None
    return None


# ---------------------------------------------------------------------------
# NM ↔ ENST mapping
# ---------------------------------------------------------------------------

def nm_to_enst(nm_accession: str, genome_build: str = "GRCh38") -> str:
    """Map a RefSeq NM_ accession to an Ensembl ENST transcript ID."""
    base = _base_url(genome_build)
    url = f"{base}/xrefs/symbol/homo_sapiens/{nm_accession}?content-type=application/json"
    data = _ensembl_get(url)
    if not data or not isinstance(data, list):
        return ""
    for entry in data:
        if entry.get("type") == "transcript" and str(entry.get("id", "")).startswith("ENST"):
            return entry["id"]
    return ""


def enst_to_nm(enst_id: str, genome_build: str = "GRCh38") -> str:
    """Map an Ensembl ENST transcript to the primary RefSeq NM_ accession."""
    base = _base_url(genome_build)
    url = f"{base}/xrefs/id/{enst_id}?content-type=application/json&external_db=RefSeq_mRNA"
    data = _ensembl_get(url)
    if not data or not isinstance(data, list):
        return ""
    nms = [x["display_id"] for x in data if x.get("display_id", "").startswith("NM_")]
    if not nms:
        return ""
    # Prefer shortest accession number (primary transcript, e.g. NM_007294 < NM_001407598)
    nms.sort(key=lambda x: (len(x.split(".")[0]), x))
    return nms[0]


# ---------------------------------------------------------------------------
# VEP — variant annotation on all transcripts
# ---------------------------------------------------------------------------

def vep_annotate_hgvs(
    hgvs: str, genome_build: str = "GRCh38"
) -> list[dict[str, Any]]:
    """Annotate a variant (HGVS notation) using Ensembl VEP.

    Accepts NM_ or ENST HGVS like "NM_007294.4:c.5266dupC".
    Returns list of transcript consequences with exon, consequence type, etc.
    """
    from urllib.parse import quote
    base = _base_url(genome_build)
    encoded = quote(hgvs, safe="")
    url = f"{base}/vep/homo_sapiens/hgvs/{encoded}?hgvs=1&canonical=1&numbers=1&protein=1"
    data = _ensembl_get(url)
    if not data or not isinstance(data, list):
        return []
    results = []
    for entry in data:
        for tc in entry.get("transcript_consequences", []):
            results.append({
                "transcript_id": tc.get("transcript_id", ""),
                "gene_symbol": tc.get("gene_symbol", ""),
                "hgvsc": tc.get("hgvsc", ""),
                "hgvsp": tc.get("hgvsp", ""),
                "consequence_terms": tc.get("consequence_terms", []),
                "impact": tc.get("impact", ""),
                "exon": tc.get("exon", ""),         # e.g. "19/23"
                "intron": tc.get("intron", ""),       # e.g. "5/22"
                "biotype": tc.get("biotype", ""),
                "is_canonical": bool(tc.get("canonical")),
                "protein_start": tc.get("protein_start"),
                "protein_end": tc.get("protein_end"),
                "amino_acids": tc.get("amino_acids", ""),
                "codons": tc.get("codons", ""),
                "strand": tc.get("strand"),
            })
    return results


def vep_annotate_region(
    chrom: str, pos: int, ref: str, alt: str, genome_build: str = "GRCh38"
) -> list[dict[str, Any]]:
    """Annotate a variant (genomic coordinates) using Ensembl VEP.

    Returns same structure as vep_annotate_hgvs.
    """
    base = _base_url(genome_build)
    chrom = str(chrom).lstrip("chr")
    ref_len, alt_len = len(ref), len(alt)

    if ref_len == 1 and alt_len == 1:
        region = f"{chrom}:{pos}-{pos}:1/{alt}"
    elif ref_len == 1 and alt_len > 1:
        region = f"{chrom}:{pos + 1}-{pos}:1/{alt[1:]}"
    elif ref_len > 1 and alt_len == 1:
        region = f"{chrom}:{pos + 1}-{pos + ref_len - 1}:1/-"
    else:
        inserted = alt[1:] if alt_len > 1 else "-"
        region = f"{chrom}:{pos + 1}-{pos + ref_len - 1}:1/{inserted}"

    url = f"{base}/vep/homo_sapiens/region/{region}?hgvs=1&canonical=1&numbers=1&protein=1"
    data = _ensembl_get(url)
    if not data or not isinstance(data, list):
        return []

    results = []
    for entry in data:
        for tc in entry.get("transcript_consequences", []):
            hgvsc = tc.get("hgvsc")
            if not hgvsc:
                continue
            results.append({
                "transcript_id": tc.get("transcript_id", ""),
                "gene_symbol": tc.get("gene_symbol", ""),
                "hgvsc": hgvsc,
                "hgvsp": tc.get("hgvsp", ""),
                "consequence_terms": tc.get("consequence_terms", []),
                "impact": tc.get("impact", ""),
                "exon": tc.get("exon", ""),
                "intron": tc.get("intron", ""),
                "biotype": tc.get("biotype", ""),
                "is_canonical": bool(tc.get("canonical")),
                "protein_start": tc.get("protein_start"),
                "protein_end": tc.get("protein_end"),
                "amino_acids": tc.get("amino_acids", ""),
                "codons": tc.get("codons", ""),
                "strand": tc.get("strand"),
            })
    return results


# ---------------------------------------------------------------------------
# Full transcript resolution
# ---------------------------------------------------------------------------

def resolve_transcripts(
    gene_symbol: Optional[str] = None,
    hgvs: Optional[str] = None,
    chrom: Optional[str] = None,
    pos: Optional[int] = None,
    ref: Optional[str] = None,
    alt: Optional[str] = None,
    genome_build: str = "GRCh38",
) -> dict[str, Any]:
    """Resolve all transcripts for a variant with NM↔ENST mapping.

    Either (gene_symbol + hgvs) or (chrom + pos + ref + alt) must be provided.

    Returns dict with:
        gene_symbol: str
        gene_full_name: str
        transcripts: list of dicts with NM_, ENST, annotations, VEP consequences
        vep_input_coords: genomic coordinates used for VEP
    """
    result: dict[str, Any] = {
        "gene_symbol": gene_symbol or "",
        "gene_full_name": "",
        "transcripts": [],
        "error": None,
    }

    # Step 1: Get VEP annotations
    vep_results = []
    if hgvs:
        # Try VEP with the HGVS notation directly
        vep_results = vep_annotate_hgvs(hgvs, genome_build)
    if not vep_results and chrom and pos and ref and alt:
        vep_results = vep_annotate_region(chrom, pos, ref, alt, genome_build)
    if not vep_results and gene_symbol and hgvs:
        # Try constructing a query VEP might accept
        query = hgvs if ":" in hgvs else f"{gene_symbol} {hgvs}"
        vep_results = vep_annotate_hgvs(query, genome_build)

    if not vep_results:
        result["error"] = "VEP could not annotate this variant on any transcript"
        return result

    # Extract gene symbol from first result if not provided
    if not result["gene_symbol"]:
        for vr in vep_results:
            if vr.get("gene_symbol"):
                result["gene_symbol"] = vr["gene_symbol"]
                break

    # Step 2: Filter to protein-coding transcripts for the target gene
    gene = result["gene_symbol"].upper()
    pc_vep = [
        vr for vr in vep_results
        if vr.get("biotype") == "protein_coding"
        and vr.get("gene_symbol", "").upper() == gene
    ]
    if not pc_vep:
        pc_vep = [vr for vr in vep_results if vr.get("gene_symbol", "").upper() == gene]

    # Step 3: For each ENST, get the NM_ mapping
    transcripts = []
    seen_enst = set()
    for vr in pc_vep:
        enst = vr.get("transcript_id", "")
        if enst in seen_enst:
            continue
        seen_enst.add(enst)

        # Get NM_ via xrefs (with rate limiting)
        nm = enst_to_nm(enst, genome_build)
        time.sleep(0.1)

        # Determine position type
        exon_str = vr.get("exon", "")
        intron_str = vr.get("intron", "")
        if exon_str:
            position_type = "exonic"
            position_detail = f"Exon {exon_str}"
        elif intron_str:
            position_type = "intronic"
            position_detail = f"Intron {intron_str}"
        else:
            position_type = "unknown"
            position_detail = ""

        # Format consequence type
        consequences = vr.get("consequence_terms", [])
        consequence_display = ", ".join(
            c.replace("_", " ") for c in consequences
        )

        transcripts.append({
            "nm_accession": nm,
            "enst_accession": enst,
            "gene_symbol": gene,
            "hgvsc": vr.get("hgvsc", ""),
            "hgvsp": vr.get("hgvsp", ""),
            "consequence_terms": consequences,
            "consequence_display": consequence_display,
            "impact": vr.get("impact", ""),
            "position_type": position_type,
            "position_detail": position_detail,
            "exon": exon_str,
            "intron": intron_str,
            "is_canonical": vr.get("is_canonical", False),
            "is_mane_select": False,        # set below
            "is_mane_plus_clinical": False,
            "is_most_reported_pathogenic": False,  # set by caller
            "annotation_score": 0,                 # set by caller
            "amino_acids": vr.get("amino_acids", ""),
            "codons": vr.get("codons", ""),
            "biotype": vr.get("biotype", ""),
        })

    # Step 4: Mark MANE Select (canonical = MANE Select for GRCh38)
    for tx in transcripts:
        if tx["is_canonical"]:
            tx["is_mane_select"] = True

    # Step 5: Get gene full name from Ensembl
    base = _base_url(genome_build)
    if gene:
        gene_data = _ensembl_get(f"{base}/lookup/symbol/homo_sapiens/{gene}")
        if gene_data:
            result["gene_full_name"] = gene_data.get("description", "")

    result["transcripts"] = transcripts
    logger.info(
        "resolve_transcripts: %d transcripts for %s (%d with NM_)",
        len(transcripts), gene, sum(1 for t in transcripts if t["nm_accession"]),
    )
    return result


# ---------------------------------------------------------------------------
# Gene info (simple)
# ---------------------------------------------------------------------------

def get_gene_info(gene_symbol: str, genome_build: str = "GRCh38") -> dict[str, Any]:
    """Fetch gene metadata from Ensembl REST API."""
    base = _base_url(genome_build)
    data = _ensembl_get(f"{base}/lookup/symbol/homo_sapiens/{gene_symbol}")
    if not data:
        return {}
    return {
        "gene_full_name": data.get("description", ""),
        "strand": data.get("strand"),
        "chromosome": data.get("seq_region_name"),
        "start": data.get("start"),
        "end": data.get("end"),
    }
