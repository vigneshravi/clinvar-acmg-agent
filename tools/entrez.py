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

def _build_clinvar_search_queries(variant: str) -> list[str]:
    """Build ranked ClinVar search queries from a variant string."""
    queries = []
    variant = variant.strip()

    # HGVS-tagged search
    if ":" in variant:
        queries.append(f"{variant}[HGVS]")
        queries.append(f"{variant}[Variant name]")
        queries.append(variant)

    # Gene + variant pattern
    match = re.match(r"^([A-Za-z0-9_-]+)\s+(.+)$", variant)
    if match:
        gene = match.group(1)
        desc = match.group(2).strip()
        queries.append(f"{gene}[gene] AND {desc}[Variant name]")
        if desc.startswith("c."):
            queries.append(f"{gene}[gene] AND {desc[2:]}[Variant name]")
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

    return result


def _validate_clinvar_result(result: dict[str, Any], variant: str) -> bool:
    """Check if a ClinVar result plausibly matches the queried variant."""
    hgvs = (result.get("hgvs") or "").lower()
    variant_lower = variant.lower().strip()

    # For transcript-level queries like NM_007294.4:c.5266dupC
    if ":" in variant_lower:
        # Check if position digits appear
        digits = re.findall(r"\d+", variant_lower.split(":")[-1])
        if digits and digits[0] in hgvs:
            return True

    match = re.match(r"^([A-Za-z0-9_-]+)\s+(.+)$", variant_lower)
    if match:
        gene = match.group(1)
        desc = match.group(2).strip()
        if gene not in hgvs:
            return False
        desc_core = desc.lstrip("c.")
        desc_digits = re.findall(r"\d+", desc_core)
        if desc_digits and desc_digits[0] in hgvs:
            return True
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
        id_list: list[str] = []

        for query in queries:
            handle = Entrez.esearch(db="clinvar", term=query, retmax=5)
            search_results = Entrez.read(handle)
            handle.close()
            id_list = search_results.get("IdList", [])
            if id_list:
                break

        if not id_list:
            return empty_result

        # Fetch and validate
        result = None
        for clinvar_id in id_list[:3]:
            handle = Entrez.esummary(db="clinvar", id=clinvar_id)
            xml_text = handle.read()
            handle.close()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode("utf-8")
            candidate = _parse_clinvar_esummary(xml_text)
            if _validate_clinvar_result(candidate, variant):
                result = candidate
                break

        if result is None:
            handle = Entrez.esummary(db="clinvar", id=id_list[0])
            xml_text = handle.read()
            handle.close()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode("utf-8")
            result = _parse_clinvar_esummary(xml_text)

        if not result["variant_id"]:
            result["variant_id"] = id_list[0]

        # Detect conflicting interpretations from significance string
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
