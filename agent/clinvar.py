"""ClinVar Entrez API calls and XML parsing."""

import os
import re
import xml.etree.ElementTree as ET
from typing import Any

from Bio import Entrez


def _configure_entrez() -> None:
    """Configure Entrez with credentials from environment."""
    Entrez.email = os.getenv("NCBI_EMAIL", "user@example.com")
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def _build_search_queries(variant: str) -> list[str]:
    """Build a ranked list of ClinVar search queries from a variant string.

    Tries structured field-tagged queries first (most precise), then falls
    back to progressively looser searches.
    """
    queries = []
    variant = variant.strip()

    # Try to parse "GENE variant_desc" pattern (e.g. "BRCA1 c.5266dupC")
    match = re.match(r"^([A-Za-z0-9_-]+)\s+(.+)$", variant)
    if match:
        gene = match.group(1)
        desc = match.group(2).strip()
        # Most precise: gene field + variant name field
        queries.append(f"{gene}[gene] AND {desc}[Variant name]")
        # Without the "c." prefix if present
        if desc.startswith("c."):
            queries.append(f"{gene}[gene] AND {desc[2:]}[Variant name]")
        # Gene field + free text variant
        queries.append(f"{gene}[gene] AND {desc}")

    # If it looks like an HGVS expression with transcript (NM_...)
    if ":" in variant:
        queries.append(f"{variant}[Variant name]")
        queries.append(variant)

    # Raw query as final fallback
    if variant not in queries:
        queries.append(variant)

    return queries


def _parse_esummary_xml(xml_text: str) -> dict[str, Any]:
    """Parse ClinVar eSummary XML and extract structured data."""
    root = ET.fromstring(xml_text)

    result: dict[str, Any] = {
        "variant_id": None,
        "gene": None,
        "hgvs": None,
        "clinical_significance": None,
        "review_status": None,
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

    # Variation ID (the uid attribute)
    variation_id = doc.get("uid")
    if variation_id:
        result["variant_id"] = variation_id

    # Title contains HGVS notation, e.g. "NM_007294.4(BRCA1):c.5266dup ..."
    title = doc.find("title")
    if title is not None and title.text:
        result["hgvs"] = title.text

    # Gene: extract from title pattern "NM_...(GENE):..." or variation_name
    if result["hgvs"]:
        gene_match = re.search(r"\(([A-Za-z0-9_-]+)\)", result["hgvs"])
        if gene_match:
            result["gene"] = gene_match.group(1)

    # Also try the variation element for gene and cdna_change
    variation = doc.find(".//variation_set/variation")
    if variation is not None:
        if not result["gene"]:
            var_name = variation.find("variation_name")
            if var_name is not None and var_name.text:
                gene_match = re.search(r"\(([A-Za-z0-9_-]+)\)", var_name.text)
                if gene_match:
                    result["gene"] = gene_match.group(1)

    # Germline classification (current ClinVar XML format)
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

        # Conditions from germline_classification > trait_set > trait
        traits = germline.findall(".//trait_set/trait")
        conditions = []
        for trait in traits:
            trait_name = trait.find("trait_name")
            if trait_name is not None and trait_name.text:
                conditions.append(trait_name.text)
        if conditions:
            result["condition"] = "; ".join(conditions)

    # Fallback: older clinical_significance element
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

    # Conditions fallback (top-level trait_set)
    if not result["condition"]:
        traits = doc.findall(".//trait_set/trait")
        conditions = []
        for trait in traits:
            trait_name = trait.find("trait_name")
            if trait_name is not None and trait_name.text:
                conditions.append(trait_name.text)
        if conditions:
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for c in conditions:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            result["condition"] = "; ".join(unique)

    # Submitter count from supporting_submissions > scv > string elements
    supporting = doc.find("supporting_submissions")
    if supporting is not None:
        scv_elem = supporting.find("scv")
        if scv_elem is not None:
            scv_strings = scv_elem.findall("string")
            result["submitter_count"] = len(scv_strings)

    return result


def _validate_result(result: dict[str, Any], variant: str) -> bool:
    """Check if the result plausibly matches the queried variant."""
    hgvs = (result.get("hgvs") or "").lower()
    title = hgvs
    variant_lower = variant.lower().strip()

    # Extract gene and variant description from user input
    match = re.match(r"^([A-Za-z0-9_-]+)\s+(.+)$", variant_lower)
    if match:
        gene = match.group(1)
        desc = match.group(2).strip()
        # Check gene matches
        if gene not in title:
            return False
        # Check variant description partially matches
        # Strip "c." prefix for matching
        desc_core = desc.lstrip("c.")
        # Extract digits from the description for position matching
        desc_digits = re.findall(r"\d+", desc_core)
        if desc_digits:
            # At least the main position number should appear in the title
            if desc_digits[0] in title:
                return True
    return True  # Default to accepting if we can't parse


def query_clinvar(variant: str) -> dict[str, Any]:
    """Query ClinVar for a variant and return structured data.

    Uses structured field-tagged queries for precision, with fallbacks
    to progressively looser searches.

    Args:
        variant: Variant string, e.g. "BRCA1 c.5266dupC"

    Returns:
        Dict with variant information or error message.
    """
    _configure_entrez()

    empty_result = {
        "error": f"No ClinVar records found for '{variant}'",
        "variant_id": None,
        "gene": None,
        "hgvs": None,
        "clinical_significance": None,
        "review_status": None,
        "submitter_count": 0,
        "conflicting_interpretations": False,
        "condition": None,
        "last_evaluated": None,
        "raw_submissions": [],
    }

    try:
        queries = _build_search_queries(variant)
        id_list = []

        # Try each query until we get results
        for query in queries:
            search_handle = Entrez.esearch(
                db="clinvar", term=query, retmax=5
            )
            search_results = Entrez.read(search_handle)
            search_handle.close()

            id_list = search_results.get("IdList", [])
            if id_list:
                break

        if not id_list:
            return empty_result

        # Fetch esummary for the top result(s) and pick the best match
        result = None
        for clinvar_id in id_list[:3]:
            fetch_handle = Entrez.esummary(db="clinvar", id=clinvar_id)
            xml_text = fetch_handle.read()
            fetch_handle.close()

            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode("utf-8")

            candidate = _parse_esummary_xml(xml_text)

            if _validate_result(candidate, variant):
                result = candidate
                break

        if result is None:
            # Use first result as fallback
            clinvar_id = id_list[0]
            fetch_handle = Entrez.esummary(db="clinvar", id=clinvar_id)
            xml_text = fetch_handle.read()
            fetch_handle.close()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode("utf-8")
            result = _parse_esummary_xml(xml_text)

        if not result["variant_id"]:
            result["variant_id"] = id_list[0]

        return result

    except Exception as e:
        return {
            "error": f"ClinVar query failed: {str(e)}",
            "variant_id": None,
            "gene": None,
            "hgvs": None,
            "clinical_significance": None,
            "review_status": None,
            "submitter_count": 0,
            "conflicting_interpretations": False,
            "condition": None,
            "last_evaluated": None,
            "raw_submissions": [],
        }
