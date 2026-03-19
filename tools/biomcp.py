"""BioMCP CLI wrapper — subprocess calls to biomcp with JSON output.

Provides structured access to 15+ biomedical APIs through the biomcp CLI:
- CIViC (clinical interpretations of variants in cancer)
- ClinGen (gene-disease validity, dosage sensitivity)
- AlphaFold/PDB (protein structures)
- InterPro (protein domains)
- Reactome (pathways)
- DGIdb (druggability)
- PubTator3 + Europe PMC (federated article search)
- GWAS Catalog (trait associations)
"""

import json
import logging
import subprocess
from typing import Any, Optional

logger = logging.getLogger(__name__)

BIOMCP_TIMEOUT = 30  # seconds per call


def _run_biomcp(*args: str, timeout: int = BIOMCP_TIMEOUT) -> Optional[dict]:
    """Run a biomcp CLI command with JSON output and return parsed dict.

    Returns None if the command fails or returns no data.
    """
    cmd = ["biomcp", "-j"] + list(args)
    logger.info("biomcp: running %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            if stderr:
                logger.warning("biomcp stderr: %s", stderr[:300])
            # Some commands output Markdown to stdout on error
            if result.stdout.strip().startswith("{"):
                return json.loads(result.stdout)
            return None
        output = result.stdout.strip()
        if not output:
            return None
        return json.loads(output)
    except subprocess.TimeoutExpired:
        logger.warning("biomcp: timeout after %ds for %s", timeout, " ".join(cmd))
        return None
    except json.JSONDecodeError as e:
        logger.warning("biomcp: JSON parse error: %s", e)
        return None
    except FileNotFoundError:
        logger.error("biomcp: binary not found — install with 'pip install biomcp-cli'")
        return None
    except Exception as e:
        logger.warning("biomcp: unexpected error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Gene queries
# ---------------------------------------------------------------------------

def get_gene_clingen(gene_symbol: str) -> Optional[dict[str, Any]]:
    """Get ClinGen gene-disease validity and dosage sensitivity."""
    data = _run_biomcp("get", "gene", gene_symbol, "clingen")
    if not data:
        return None
    return data.get("clingen")


def get_gene_pathways(gene_symbol: str) -> Optional[list[dict[str, Any]]]:
    """Get Reactome pathways for a gene."""
    data = _run_biomcp("get", "gene", gene_symbol, "pathways")
    if not data:
        return None
    return data.get("pathways")


def get_gene_druggability(gene_symbol: str) -> Optional[dict[str, Any]]:
    """Get DGIdb druggability information for a gene."""
    data = _run_biomcp("get", "gene", gene_symbol, "druggability")
    if not data:
        return None
    drug_data = data.get("druggability", {})
    if isinstance(drug_data, dict):
        return {
            "categories": drug_data.get("categories", []),
            "interactions": drug_data.get("interactions", []),
        }
    return {"categories": [], "interactions": []}


def get_gene_diseases(gene_symbol: str) -> Optional[list[dict[str, Any]]]:
    """Get disease associations for a gene."""
    data = _run_biomcp("get", "gene", gene_symbol, "diseases")
    if not data:
        return None
    return data.get("clinical_diseases", [])


# ---------------------------------------------------------------------------
# Variant queries
# ---------------------------------------------------------------------------

def get_variant_civic(variant_query: str) -> Optional[dict[str, Any]]:
    """Get CIViC clinical evidence for a variant.

    Args:
        variant_query: rsID (rs113488022), HGVS (chr7:g.140453136A>T),
                       or gene+protein (BRAF V600E).

    Returns normalized dict with:
        cached_evidence: list of evidence items from MyVariant.info cache
        graphql_assertions: list of CIViC assertions
        graphql_evidence_count: total evidence items via GraphQL
    """
    data = _run_biomcp("get", "variant", variant_query, "civic")
    if not data:
        return None
    civic_raw = data.get("civic")
    if not civic_raw:
        return None

    # Normalize the nested structure
    result = {
        "cached_evidence": civic_raw.get("cached_evidence", []),
    }

    # GraphQL data is nested under civic.graphql
    graphql = civic_raw.get("graphql", {})
    if graphql:
        result["graphql_assertions"] = graphql.get("assertions", [])
        result["graphql_evidence_count"] = graphql.get("evidence_total_count")
        result["graphql_evidence_items"] = graphql.get("evidence_items", [])
    else:
        result["graphql_assertions"] = []
        result["graphql_evidence_count"] = None

    return result


def get_variant_gwas(variant_query: str) -> Optional[list[dict[str, Any]]]:
    """Get GWAS Catalog associations for a variant."""
    data = _run_biomcp("get", "variant", variant_query, "gwas")
    if not data:
        return None
    return data.get("gwas")


def get_variant_predictions(variant_query: str) -> Optional[dict[str, Any]]:
    """Get computational predictions for a variant (from MyVariant.info via BioMCP)."""
    data = _run_biomcp("get", "variant", variant_query, "predictions")
    if not data:
        return None
    return data


# ---------------------------------------------------------------------------
# Protein queries
# ---------------------------------------------------------------------------

def get_protein_info(gene_or_accession: str) -> Optional[dict[str, Any]]:
    """Get protein info including function, domains, structures."""
    data = _run_biomcp("get", "protein", gene_or_accession, "domains", "structures")
    if not data:
        return None
    return data


# ---------------------------------------------------------------------------
# Article queries
# ---------------------------------------------------------------------------

def search_articles(
    gene: Optional[str] = None,
    query: Optional[str] = None,
    disease: Optional[str] = None,
    max_results: int = 10,
) -> Optional[list[dict[str, Any]]]:
    """Search PubTator3 + Europe PMC for articles.

    Returns list of article dicts with pmid, title, journal, date, citation_count.
    """
    args = ["search", "article"]
    if gene:
        args.extend(["-g", gene])
    if query:
        args.extend(["--query", query])
    if disease:
        args.extend(["-d", disease])
    args.extend(["--sort", "relevance"])

    data = _run_biomcp(*args)
    if not data:
        return None
    return data.get("results", [])[:max_results]


def get_article_details(pmid: str) -> Optional[dict[str, Any]]:
    """Get full article details by PMID."""
    data = _run_biomcp("get", "article", pmid)
    if not data:
        return None
    return data
