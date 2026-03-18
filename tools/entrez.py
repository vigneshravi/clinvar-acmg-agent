"""NCBI Entrez API calls for ClinVar, Gene, and related databases."""

import logging
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any, Optional

from Bio import Entrez

logger = logging.getLogger(__name__)

# Star rating mapping
REVIEW_STATUS_STARS: dict[str, int] = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, single submitter": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
}


def configure_entrez() -> None:
    """Configure Entrez with credentials from environment."""
    Entrez.email = os.getenv("NCBI_EMAIL", "user@example.com")
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def _get_star_rating(review_status: Optional[str]) -> int:
    """Convert review status string to star rating (0-4)."""
    if not review_status:
        return 0
    status_lower = review_status.strip().lower()
    for key, stars in REVIEW_STATUS_STARS.items():
        if key in status_lower:
            return stars
    return 0


# ---------------------------------------------------------------------------
# Gene info
# ---------------------------------------------------------------------------

def get_gene_aliases(gene_symbol: str) -> dict[str, Any]:
    """Fetch gene aliases and full name from NCBI Gene.

    Returns dict with official_symbol, full_name, aliases list.
    """
    configure_entrez()
    result = {"official_symbol": gene_symbol, "full_name": "", "aliases": []}

    try:
        handle = Entrez.esearch(
            db="gene",
            term=f"{gene_symbol}[GENE] AND human[ORGN]",
            retmax=1,
        )
        search = Entrez.read(handle)
        handle.close()

        ids = search.get("IdList", [])
        if not ids:
            return result

        handle = Entrez.esummary(db="gene", id=ids[0])
        summary_text = handle.read()
        handle.close()

        if isinstance(summary_text, bytes):
            summary_text = summary_text.decode("utf-8")

        root = ET.fromstring(summary_text)
        for doc in root.findall(".//DocumentSummary"):
            name_elem = doc.find("Description")
            if name_elem is not None and name_elem.text:
                result["full_name"] = name_elem.text

            alias_elem = doc.find("OtherAliases")
            if alias_elem is not None and alias_elem.text:
                result["aliases"] = [
                    a.strip() for a in alias_elem.text.split(",") if a.strip()
                ]

            official = doc.find("NomenclatureSymbol")
            if official is not None and official.text:
                result["official_symbol"] = official.text

    except Exception as e:
        logger.warning("get_gene_aliases failed for %s: %s", gene_symbol, e)

    return result


# ---------------------------------------------------------------------------
# Pathogenic submission counting per transcript
# ---------------------------------------------------------------------------

def count_pathogenic_submissions_per_transcript(
    gene_symbol: str,
) -> Counter:
    """Count pathogenic ClinVar submissions per NM_ transcript accession.

    Returns Counter mapping NM_ accession → number of pathogenic submissions.
    """
    configure_entrez()
    counter: Counter = Counter()

    try:
        term = (
            f"{gene_symbol}[GENE] AND "
            f"(pathogenic[CLNSIG] OR likely pathogenic[CLNSIG])"
        )
        handle = Entrez.esearch(db="clinvar", term=term, retmax=200)
        search = Entrez.read(handle)
        handle.close()

        ids = search.get("IdList", [])
        if not ids:
            return counter

        # Fetch esummaries in batches
        for i in range(0, len(ids), 50):
            batch = ids[i : i + 50]
            handle = Entrez.esummary(db="clinvar", id=",".join(batch))
            xml_text = handle.read()
            handle.close()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode("utf-8")

            root = ET.fromstring(xml_text)
            for doc in root.findall(".//DocumentSummary"):
                title = doc.find("title")
                if title is not None and title.text:
                    # Extract NM_ accession from title like "NM_007294.4(BRCA1):c.5266dup"
                    nm_match = re.search(r"(NM_\d+\.\d+)", title.text)
                    if nm_match:
                        counter[nm_match.group(1)] += 1

    except Exception as e:
        logger.warning(
            "count_pathogenic_submissions failed for %s: %s", gene_symbol, e
        )

    return counter


# ---------------------------------------------------------------------------
# ClinVar record fetch
# ---------------------------------------------------------------------------

def _expand_dup_notation(desc: str) -> list[str]:
    """Expand bare 'dup' notation to include each possible base.

    ClinVar often stores duplications with the base (e.g. c.5266dupC)
    but HGVS v2 normalizes to bare 'dup' (e.g. c.5266dup).
    Returns list of variants with each base appended.
    """
    # Only expand if the description ends with 'dup' and no base follows
    if re.search(r"dup$", desc, re.IGNORECASE):
        return [f"{desc}{base}" for base in "ACGT"]
    return []


def _build_clinvar_search_queries(variant: str) -> list[str]:
    """Build ranked ClinVar search queries from a variant string."""
    queries = []
    variant = variant.strip()

    # HGVS-tagged search
    if ":" in variant:
        queries.append(f"{variant}[HGVS]")
        queries.append(f"{variant}[Variant name]")
        queries.append(variant)
        # For bare dup notation, also try with each base
        if ":" in variant:
            cdna = variant.split(":")[-1]
            tx_part = variant.split(":")[0]
            for expanded in _expand_dup_notation(cdna):
                queries.append(f"{tx_part}:{expanded}[Variant name]")

    # Gene + variant pattern
    match = re.match(r"^([A-Za-z0-9_-]+)\s+(.+)$", variant)
    if match:
        gene = match.group(1)
        desc = match.group(2).strip()
        queries.append(f"{gene}[gene] AND {desc}[Variant name]")
        if desc.startswith("c."):
            queries.append(f"{gene}[gene] AND {desc[2:]}[Variant name]")
        # Also try dup base expansion
        for expanded in _expand_dup_notation(desc):
            queries.append(f"{gene}[gene] AND {expanded}[Variant name]")
        queries.append(f"{gene}[gene] AND {desc}")

    if variant not in queries:
        queries.append(variant)

    return queries


def _parse_clinvar_esummary(xml_text: str) -> dict[str, Any]:
    """Parse ClinVar eSummary XML into a structured dict."""
    root = ET.fromstring(xml_text)

    result: dict[str, Any] = {
        "variant_id": None,
        "gene": None,
        "hgvs": None,
        "clinical_significance": None,
        "review_status": None,
        "star_rating": 0,
        "submitter_count": 0,
        "conflicting_interpretations": False,
        "condition": None,
        "last_evaluated": None,
        "raw_submissions": [],
    }

    doc_summaries = root.findall(".//DocumentSummary")
    if not doc_summaries:
        return result

    doc = doc_summaries[0]

    variation_id = doc.get("uid")
    if variation_id:
        result["variant_id"] = variation_id

    title = doc.find("title")
    if title is not None and title.text:
        result["hgvs"] = title.text

    if result["hgvs"]:
        gene_match = re.search(r"\(([A-Za-z0-9_-]+)\)", result["hgvs"])
        if gene_match:
            result["gene"] = gene_match.group(1)

    variation = doc.find(".//variation_set/variation")
    if variation is not None and not result["gene"]:
        var_name = variation.find("variation_name")
        if var_name is not None and var_name.text:
            gene_match = re.search(r"\(([A-Za-z0-9_-]+)\)", var_name.text)
            if gene_match:
                result["gene"] = gene_match.group(1)

    # Germline classification (current format)
    germline = doc.find("germline_classification")
    if germline is not None:
        desc = germline.find("description")
        if desc is not None and desc.text:
            result["clinical_significance"] = desc.text
        review = germline.find("review_status")
        if review is not None and review.text:
            result["review_status"] = review.text
        last_eval = germline.find("last_evaluated")
        if last_eval is not None and last_eval.text:
            result["last_evaluated"] = last_eval.text

        traits = germline.findall(".//trait_set/trait")
        conditions = []
        for trait in traits:
            trait_name = trait.find("trait_name")
            if trait_name is not None and trait_name.text:
                conditions.append(trait_name.text)
        if conditions:
            result["condition"] = "; ".join(conditions)

    # Fallback: older format
    if not result["clinical_significance"]:
        clin_sig = doc.find("clinical_significance")
        if clin_sig is not None:
            desc = clin_sig.find("description")
            if desc is not None and desc.text:
                result["clinical_significance"] = desc.text
            review = clin_sig.find("review_status")
            if review is not None and review.text:
                result["review_status"] = review.text
            last_eval = clin_sig.find("last_evaluated")
            if last_eval is not None and last_eval.text:
                result["last_evaluated"] = last_eval.text

    # Conditions fallback
    if not result["condition"]:
        traits = doc.findall(".//trait_set/trait")
        conditions = []
        for trait in traits:
            trait_name = trait.find("trait_name")
            if trait_name is not None and trait_name.text:
                conditions.append(trait_name.text)
        if conditions:
            seen = set()
            unique = []
            for c in conditions:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            result["condition"] = "; ".join(unique)

    # Submitter count
    supporting = doc.find("supporting_submissions")
    if supporting is not None:
        scv_elem = supporting.find("scv")
        if scv_elem is not None:
            scv_strings = scv_elem.findall("string")
            result["submitter_count"] = len(scv_strings)

    # Compute star rating
    result["star_rating"] = _get_star_rating(result["review_status"])

    # Extract rsID from dbSNP xref
    for xref in doc.findall(".//variation_xref"):
        db_source = xref.find("db_source")
        db_id = xref.find("db_id")
        if (db_source is not None and db_id is not None
                and "dbsnp" in (db_source.text or "").lower()
                and db_id.text):
            result["rsid"] = f"rs{db_id.text}"
            break

    return result


def _extract_cdna_position(hgvs: str) -> str:
    """Extract the core cDNA position from an HGVS string (e.g. '5266')."""
    m = re.search(r"c\.(\d+)", hgvs)
    return m.group(1) if m else ""


def _extract_variant_type(hgvs: str) -> str:
    """Extract variant type from HGVS (dup/del/>/delins)."""
    for vtype in ["delins", "dup", "del", "ins", ">"]:
        if vtype in hgvs.lower():
            return vtype
    return ""


def _validate_clinvar_result_strict(result: dict[str, Any], variant: str) -> bool:
    """Strictly validate that a ClinVar result matches the query.

    Checks both the cDNA position AND variant type to catch cases where
    ClinVar returns a different variant at a nearby position.
    """
    result_hgvs = (result.get("hgvs") or "").lower()
    variant_lower = variant.lower().strip()

    # Extract cdna part from query
    query_cdna = ""
    if ":" in variant_lower:
        query_cdna = variant_lower.split(":")[-1]
    else:
        m = re.match(r"^[A-Za-z0-9_-]+\s+(.+)$", variant_lower)
        if m:
            query_cdna = m.group(1).strip()

    if not query_cdna:
        return True  # can't validate

    query_pos = _extract_cdna_position(query_cdna)
    result_pos = _extract_cdna_position(result_hgvs)

    if query_pos and result_pos and query_pos != result_pos:
        return False

    query_type = _extract_variant_type(query_cdna)
    result_type = _extract_variant_type(result_hgvs)

    if query_type and result_type and query_type != result_type:
        return False

    return True


def fetch_clinvar_record(variant: str) -> dict[str, Any]:
    """Query ClinVar for a variant and return structured data.

    Uses ranked search queries for precision with fallbacks.
    Enriches result with star_rating and conflicting_interpretations.
    """
    configure_entrez()

    empty_result: dict[str, Any] = {
        "error": f"No ClinVar records found for '{variant}'",
        "variant_id": None,
        "gene": None,
        "hgvs": None,
        "clinical_significance": None,
        "review_status": None,
        "star_rating": 0,
        "submitter_count": 0,
        "conflicting_interpretations": False,
        "condition": None,
        "last_evaluated": None,
        "raw_submissions": [],
    }

    try:
        queries = _build_clinvar_search_queries(variant)
        seen_ids: set[str] = set()
        best_match = None
        fallback = None

        for query in queries:
            handle = Entrez.esearch(db="clinvar", term=query, retmax=5)
            search_results = Entrez.read(handle)
            handle.close()
            id_list = search_results.get("IdList", [])
            if not id_list:
                continue

            for clinvar_id in id_list[:3]:
                if clinvar_id in seen_ids:
                    continue
                seen_ids.add(clinvar_id)

                handle = Entrez.esummary(db="clinvar", id=clinvar_id)
                xml_text = handle.read()
                handle.close()
                if isinstance(xml_text, bytes):
                    xml_text = xml_text.decode("utf-8")
                candidate = _parse_clinvar_esummary(xml_text)

                if not candidate.get("variant_id"):
                    continue

                # Keep first result as fallback
                if fallback is None:
                    fallback = candidate

                # Strict validation: position + type must match
                if _validate_clinvar_result_strict(candidate, variant):
                    if best_match is None:
                        best_match = candidate
                    elif candidate.get("star_rating", 0) > best_match.get("star_rating", 0):
                        best_match = candidate

            # If we found a validated match with decent stars, stop
            if best_match and best_match.get("star_rating", 0) >= 2:
                break

        result = best_match or fallback
        if result is None:
            return empty_result

        if not result["variant_id"]:
            result["variant_id"] = list(seen_ids)[0] if seen_ids else None

        # Detect conflicting interpretations
        sig = (result.get("clinical_significance") or "").lower()
        if "conflicting" in sig:
            result["conflicting_interpretations"] = True

        return result

    except Exception as e:
        logger.exception("fetch_clinvar_record failed: %s", e)
        empty_result["error"] = f"ClinVar query failed: {str(e)}"
        return empty_result


# Keep backward compat alias
query_clinvar = fetch_clinvar_record
