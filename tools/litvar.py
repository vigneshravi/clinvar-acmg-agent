"""LitVar2 API client — literature evidence for variants.

Queries NCBI LitVar to retrieve:
- Publication count and disease associations
- PMIDs with PubMed metadata (title, journal, year, publication type)
- Related entities (genes, diseases, chemicals)
- Clinical significance from literature mining

Used for ACMG criteria PS3 (functional studies) and PS4 (case reports).
"""

import json as _json
import logging
import os
import time
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from Bio import Entrez

logger = logging.getLogger(__name__)

LITVAR_API = "https://www.ncbi.nlm.nih.gov/research/bionlp/litvar/api/v1"


def _litvar_get(url: str, retries: int = 2) -> Optional[Any]:
    """GET from LitVar API with retry."""
    for attempt in range(retries + 1):
        try:
            req = Request(url)
            req.add_header("Accept", "application/json")
            with urlopen(req, timeout=15) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(2)
                continue
            if e.code == 404:
                return None
            logger.error("LitVar HTTP %d for %s", e.code, url)
            return None
        except Exception as exc:
            logger.error("LitVar request failed: %s", exc)
            return None
    return None


def _litvar_post(url: str, body: dict) -> Optional[Any]:
    """POST to LitVar API."""
    data = _json.dumps(body).encode("utf-8")
    try:
        req = Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        with urlopen(req, timeout=15) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        logger.error("LitVar POST failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Core queries
# ---------------------------------------------------------------------------

def search_variant(query: str) -> Optional[dict[str, Any]]:
    """Search LitVar for a variant by rsID, HGVS, or gene+variant.

    Returns the best-matching variant entity with metadata.
    """
    encoded = quote(query, safe="")
    url = f"{LITVAR_API}/entity/search/{encoded}"
    data = _litvar_get(url)
    if not data or not isinstance(data, list) or len(data) == 0:
        return None

    # Return the top match
    entity = data[0]
    return _parse_entity(entity)


def get_variant_by_rsid(rsid: str) -> Optional[dict[str, Any]]:
    """Get LitVar entity for a specific rsID."""
    return search_variant(rsid)


def get_pmids(rsid: str) -> list[int]:
    """Get all PMIDs mentioning a variant."""
    url = f"{LITVAR_API}/public/rsids2pmids?rsids={rsid}"
    data = _litvar_get(url)
    if not data or not isinstance(data, list):
        return []
    for entry in data:
        if entry.get("rsid") == rsid:
            return entry.get("pmids", [])
    return []


def get_related_entities(rsid: str) -> dict[str, list[dict]]:
    """Get entities co-occurring with this variant in literature.

    Returns dict with keys: disease, gene, chemical, variant
    Each value is a list of {name, count} dicts.
    """
    url = f"{LITVAR_API}/public/relations"
    data = _litvar_post(url, {
        "accessions": [f"litvar@{rsid}##"],
        "unlimited": False,
    })
    if not data or not isinstance(data, list):
        return {}

    result = {}
    for group in data:
        concept = group.get("concept", "")
        relations = []
        for rel in group.get("relations", []):
            name_raw = rel.get("name", "")
            # Parse "Breast Neoplasms@ncbi_mesh@D001943" format
            name = name_raw.split("@")[0] if "@" in name_raw else name_raw
            count = rel.get("count", 0)
            relations.append({"name": name, "count": count})
        if relations:
            result[concept] = relations

    return result


# ---------------------------------------------------------------------------
# PubMed enrichment
# ---------------------------------------------------------------------------

def enrich_pmids_with_pubmed(
    pmids: list[int], max_fetch: int = 20
) -> list[dict[str, Any]]:
    """Fetch PubMed metadata for a list of PMIDs.

    Returns list of dicts with: pmid, title, journal, year, pub_types, doi.
    """
    Entrez.email = os.getenv("NCBI_EMAIL", "user@example.com")
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key

    # Limit to max_fetch
    fetch_ids = [str(p) for p in pmids[:max_fetch]]
    if not fetch_ids:
        return []

    try:
        handle = Entrez.efetch(
            db="pubmed", id=",".join(fetch_ids),
            rettype="medline", retmode="text",
        )
        text = handle.read()
        handle.close()
        if isinstance(text, bytes):
            text = text.decode("utf-8")
    except Exception as e:
        logger.warning("PubMed fetch failed: %s", e)
        return []

    return _parse_medline(text)


def _parse_medline(text: str) -> list[dict[str, Any]]:
    """Parse MEDLINE format text into structured records."""
    records = []
    current: dict[str, Any] = {}
    current_field = ""

    for line in text.split("\n"):
        if line.startswith("PMID- "):
            if current.get("pmid"):
                records.append(current)
            current = {"pmid": line[6:].strip(), "title": "", "journal": "",
                       "year": "", "pub_types": [], "doi": "", "authors": []}
            current_field = ""
        elif line.startswith("TI  - "):
            current["title"] = line[6:]
            current_field = "title"
        elif line.startswith("      ") and current_field == "title":
            current["title"] += " " + line.strip()
        elif line.startswith("TA  - "):
            current["journal"] = line[6:].strip()
            current_field = ""
        elif line.startswith("DP  - "):
            current["year"] = line[6:].strip()[:4]
            current_field = ""
        elif line.startswith("PT  - "):
            current["pub_types"].append(line[6:].strip())
            current_field = ""
        elif line.startswith("AID - ") and "[doi]" in line:
            current["doi"] = line[6:].replace("[doi]", "").strip()
            current_field = ""
        elif line.startswith("AU  - "):
            current["authors"].append(line[6:].strip())
            current_field = ""
        else:
            current_field = ""

    if current.get("pmid"):
        records.append(current)

    return records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_entity(entity: dict) -> dict[str, Any]:
    """Parse a LitVar entity into a clean dict."""
    data = entity.get("data", {})
    diseases = entity.get("diseases", {})
    years = entity.get("years", [])

    # Sort diseases by count
    disease_list = sorted(diseases.items(), key=lambda x: x[1], reverse=True)

    return {
        "rsid": entity.get("rsid", ""),
        "name": entity.get("name", ""),
        "hgvs": entity.get("hgvs", ""),
        "hgvs_prot": entity.get("hgvs_prot", ""),
        "pmids_count": entity.get("pmids_count", 0),
        "clinical_significance": data.get("clinical_significance", ""),
        "gene": (data.get("genes", [{}])[0].get("name", "")
                 if data.get("genes") else ""),
        "diseases": disease_list,
        "top_diseases": disease_list[:5],
        "years": sorted(years) if years else [],
        "first_published": min(years) if years else None,
        "all_hgvs": entity.get("all_hgvs", []),
    }


# ---------------------------------------------------------------------------
# High-level query — single call for everything
# ---------------------------------------------------------------------------

def query_litvar(rsid: str, max_publications: int = 10) -> dict[str, Any]:
    """Full LitVar query: variant info + publications + related entities.

    Args:
        rsid: dbSNP rsID (e.g. "rs80357906")
        max_publications: Max number of publications to fetch details for

    Returns:
        Structured dict with all LitVar evidence.
    """
    result: dict[str, Any] = {
        "rsid": rsid,
        "available": False,
        "pmids_count": 0,
        "publications": [],
        "diseases": [],
        "related_genes": [],
        "related_chemicals": [],
        "clinical_significance": "",
        "first_published": None,
        "publication_types": {},
        "functional_study_count": 0,
        "case_report_count": 0,
        "review_count": 0,
    }

    if not rsid:
        return result

    # Step 1: Get variant entity
    entity = get_variant_by_rsid(rsid)
    if not entity:
        logger.info("LitVar: no data for %s", rsid)
        return result

    result["available"] = True
    result["pmids_count"] = entity.get("pmids_count", 0)
    result["diseases"] = entity.get("top_diseases", [])
    result["clinical_significance"] = entity.get("clinical_significance", "")
    result["first_published"] = entity.get("first_published")

    # Step 2: Get PMIDs and enrich with PubMed metadata
    pmids = get_pmids(rsid)
    if pmids:
        publications = enrich_pmids_with_pubmed(pmids, max_fetch=max_publications)
        result["publications"] = publications

        # Classify publication types
        type_counts: dict[str, int] = {}
        func_count = 0
        case_count = 0
        review_count = 0

        for pub in publications:
            for pt in pub.get("pub_types", []):
                type_counts[pt] = type_counts.get(pt, 0) + 1
                pt_lower = pt.lower()
                if "case report" in pt_lower:
                    case_count += 1
                if "review" in pt_lower:
                    review_count += 1
                if any(kw in pt_lower for kw in [
                    "research support", "comparative study",
                    "evaluation study", "validation study",
                ]):
                    func_count += 1

        result["publication_types"] = type_counts
        result["functional_study_count"] = func_count
        result["case_report_count"] = case_count
        result["review_count"] = review_count

    # Step 3: Get related entities
    related = get_related_entities(rsid)
    result["related_genes"] = related.get("gene", [])
    result["related_chemicals"] = related.get("chemical", [])[:5]

    # Merge disease info from related entities (higher quality)
    if related.get("disease") and not result["diseases"]:
        result["diseases"] = [
            (d["name"], d["count"]) for d in related["disease"][:5]
        ]

    logger.info(
        "LitVar: %s — %d publications, %d diseases, %d case reports",
        rsid, result["pmids_count"], len(result["diseases"]),
        result["case_report_count"],
    )

    return result
