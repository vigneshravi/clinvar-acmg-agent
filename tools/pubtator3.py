"""PubTator3 API client — NLP-annotated biomedical literature search.

Queries the NCBI PubTator3 API for:
- Article search by gene, variant, disease keywords
- NLP entity annotations (genes, variants, diseases, chemicals)
- Full-text passage annotations for specific PMIDs

PubTator3 provides NLP-mined entity annotations on top of PubMed,
making it superior to plain PubMed for finding variant-specific literature.
"""

import json as _json
import logging
import time
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

PUBTATOR3_API = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api"


def _pubtator_get(url: str, retries: int = 2, timeout: int = 15) -> Optional[Any]:
    """GET from PubTator3 API with retry on 429."""
    for attempt in range(retries + 1):
        try:
            req = Request(url)
            req.add_header("Accept", "application/json")
            with urlopen(req, timeout=timeout) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429 and attempt < retries:
                time.sleep(2 * (attempt + 1))
                continue
            if e.code == 404:
                return None
            logger.warning("PubTator3 HTTP %d for %s", e.code, url)
            return None
        except Exception as exc:
            logger.warning("PubTator3 request failed: %s", exc)
            return None
    return None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_pubtator3(
    query: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search PubTator3 for articles matching a query.

    Args:
        query: Free-text search (gene names, variant notation, disease, etc.)
        max_results: Maximum number of articles to return.

    Returns:
        List of article dicts with: pmid, title, journal, authors, date, doi,
        score, pmcid.
    """
    encoded = quote(query, safe="")
    page_size = min(max_results, 50)
    url = f"{PUBTATOR3_API}/search/?text={encoded}&page=1&page_size={page_size}"

    data = _pubtator_get(url)
    if not data:
        return []

    results = data.get("results", [])
    articles = []
    for r in results[:max_results]:
        articles.append({
            "pmid": str(r.get("pmid", "")),
            "pmcid": r.get("pmcid", ""),
            "title": r.get("title", ""),
            "journal": r.get("journal", ""),
            "authors": r.get("authors", []),
            "date": r.get("date", ""),
            "doi": r.get("doi", ""),
            "score": r.get("score", 0),
            "source": "pubtator3",
        })

    return articles


def search_variant_articles(
    gene: str,
    variant_notation: str,
    max_results: int = 10,
) -> list[dict[str, Any]]:
    """Search PubTator3 for articles mentioning a specific gene variant.

    Builds a targeted query combining gene name and variant notation.
    """
    query = f"{gene} {variant_notation}"
    return search_pubtator3(query, max_results=max_results)


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def get_article_annotations(
    pmids: list[str | int],
    max_pmids: int = 10,
) -> list[dict[str, Any]]:
    """Get NLP entity annotations for articles by PMID.

    Returns list of annotated articles, each with:
    - pmid, title, annotations (list of entities)
    - Each annotation has: text, type (Gene/Variant/Disease/Chemical/Species),
      identifier, locations
    """
    ids = [str(p) for p in pmids[:max_pmids]]
    if not ids:
        return []

    pmid_str = ",".join(ids)
    url = f"{PUBTATOR3_API}/publications/export/biocjson?pmids={pmid_str}"

    data = _pubtator_get(url, timeout=20)
    if not data:
        return []

    # Response can be a dict with PubTator3 key or a list
    articles_raw = []
    if isinstance(data, dict):
        articles_raw = data.get("PubTator3", [])
    elif isinstance(data, list):
        articles_raw = data

    annotated = []
    for article in articles_raw:
        pmid = article.get("id", "")
        passages = article.get("passages", [])

        # Extract title from first passage
        title = ""
        all_annotations = []

        for passage in passages:
            if passage.get("infons", {}).get("type") == "title":
                title = passage.get("text", "")

            for ann in passage.get("annotations", []):
                infons = ann.get("infons", {})
                entity = {
                    "text": ann.get("text", ""),
                    "type": infons.get("type", ""),
                    "identifier": infons.get("identifier", ""),
                    "database": infons.get("database", ""),
                }
                # Add HGVS for variants
                if infons.get("hgvs"):
                    entity["hgvs"] = infons["hgvs"]
                if infons.get("name"):
                    entity["name"] = infons["name"]

                all_annotations.append(entity)

        # Deduplicate by text+type
        seen = set()
        unique_annotations = []
        for a in all_annotations:
            key = (a["text"], a["type"])
            if key not in seen:
                seen.add(key)
                unique_annotations.append(a)

        annotated.append({
            "pmid": pmid,
            "title": title,
            "annotations": unique_annotations,
            "annotation_count": len(unique_annotations),
            "entity_types": list(set(a["type"] for a in unique_annotations)),
        })

    return annotated


def extract_variant_mentions(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract variant-specific annotations from article annotations."""
    variants = []
    for ann in annotations:
        if ann.get("type") == "Variant":
            variants.append(ann)
    return variants


def extract_disease_mentions(
    annotations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract disease annotations from article annotations."""
    diseases = []
    for ann in annotations:
        if ann.get("type") == "Disease":
            diseases.append(ann)
    return diseases
