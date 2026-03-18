"""Ensembl REST API calls for transcript and gene annotation."""

import logging
import time
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import json as _json

logger = logging.getLogger(__name__)

ENSEMBL_REST_GRCH38 = "https://rest.ensembl.org"
ENSEMBL_REST_GRCH37 = "https://grch37.rest.ensembl.org"


def _base_url(genome_build: str) -> str:
    if genome_build == "GRCh37":
        return ENSEMBL_REST_GRCH37
    return ENSEMBL_REST_GRCH38


def _ensembl_get(url: str, retries: int = 2) -> Optional[dict]:
    """GET from Ensembl REST with retry on 429."""
    for attempt in range(retries + 1):
        try:
            req = Request(url)
            req.add_header("Content-Type", "application/json")
            with urlopen(req, timeout=15) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = float(e.headers.get("Retry-After", "1"))
                logger.warning("Ensembl 429 — retrying in %.1fs", wait)
                time.sleep(wait)
                continue
            if e.code == 400:
                logger.warning("Ensembl 400 for %s: %s", url, e.reason)
                return None
            logger.error("Ensembl HTTP %d for %s", e.code, url)
            return None
        except Exception as exc:
            logger.error("Ensembl request failed: %s", exc)
            return None
    return None


def _ensembl_post(url: str, body: dict, retries: int = 2) -> Optional[dict | list]:
    """POST to Ensembl REST with retry on 429."""
    data = _json.dumps(body).encode("utf-8")
    for attempt in range(retries + 1):
        try:
            req = Request(url, data=data, method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Accept", "application/json")
            with urlopen(req, timeout=20) as resp:
                return _json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 429 and attempt < retries:
                wait = float(e.headers.get("Retry-After", "1"))
                logger.warning("Ensembl POST 429 — retrying in %.1fs", wait)
                time.sleep(wait)
                continue
            logger.error("Ensembl POST HTTP %d for %s", e.code, url)
            return None
        except Exception as exc:
            logger.error("Ensembl POST failed: %s", exc)
            return None
    return None


def get_transcripts_for_gene(
    gene_symbol: str, genome_build: str = "GRCh38"
) -> list[dict[str, Any]]:
    """Fetch all protein-coding transcripts for a gene from Ensembl REST API.

    Returns list of dicts matching TranscriptRecord schema.
    """
    base = _base_url(genome_build)
    url = f"{base}/lookup/symbol/homo_sapiens/{gene_symbol}?expand=1"
    data = _ensembl_get(url)
    if not data:
        logger.warning("No Ensembl data for gene %s", gene_symbol)
        return []

    gene_full_name = data.get("description", "")
    transcripts_raw = data.get("Transcript", [])

    results = []
    for tx in transcripts_raw:
        biotype = tx.get("biotype", "")
        if biotype != "protein_coding":
            continue

        enst = tx.get("id", "")

        # Extract RefSeq NM_ accession from cross-references
        nm_acc = ""
        for xref_block in tx.get("Translation", {}).get("db_links", []):
            if xref_block.get("dbname") == "RefSeq_mRNA":
                nm_acc = xref_block.get("primary_id", "")
                break
        if not nm_acc:
            for xref_block in tx.get("db_links", []) if isinstance(tx.get("db_links"), list) else []:
                if xref_block.get("dbname") == "RefSeq_mRNA":
                    nm_acc = xref_block.get("primary_id", "")
                    break

        is_mane_select = bool(tx.get("is_mane_select"))
        is_mane_plus = bool(tx.get("is_mane_plus_clinical"))

        results.append({
            "nm_accession": nm_acc,
            "enst_accession": enst,
            "gene_symbol": gene_symbol.upper(),
            "gene_aliases": [],
            "gene_full_name": gene_full_name,
            "is_mane_select": is_mane_select,
            "is_mane_plus_clinical": is_mane_plus,
            "is_most_reported_pathogenic": False,
            "annotation_score": 0,
            "biotype": biotype,
            "equivalent_hgvs": "",
        })

    logger.info(
        "Ensembl: found %d protein-coding transcripts for %s",
        len(results), gene_symbol,
    )
    return results


def recode_variant(
    hgvs_string: str, genome_build: str = "GRCh38"
) -> dict[str, str]:
    """Use Ensembl Variant Recoder to map HGVS to all transcript notations.

    Returns dict keyed by transcript accession with equivalent c. notation.
    """
    base = _base_url(genome_build)
    url = f"{base}/variant_recoder/homo_sapiens"
    resp = _ensembl_post(url, {"ids": [hgvs_string]})
    if not resp or not isinstance(resp, list):
        logger.warning("Variant recoder returned no data for %s", hgvs_string)
        return {}

    mapping: dict[str, str] = {}
    for entry in resp:
        if isinstance(entry, dict):
            for key, val in entry.items():
                if key == "warnings" or not isinstance(val, dict):
                    continue
                hgvsc_list = val.get("hgvsc", [])
                for hgvsc in hgvsc_list:
                    if ":" in hgvsc:
                        tx_id = hgvsc.split(":")[0]
                        mapping[tx_id] = hgvsc

    logger.info("Variant recoder: mapped %s to %d transcripts", hgvs_string, len(mapping))
    return mapping


def get_gene_info(
    gene_symbol: str, genome_build: str = "GRCh38"
) -> dict[str, Any]:
    """Fetch gene metadata from Ensembl REST API."""
    base = _base_url(genome_build)
    url = f"{base}/lookup/symbol/homo_sapiens/{gene_symbol}"
    data = _ensembl_get(url)
    if not data:
        return {}

    return {
        "gene_full_name": data.get("description", ""),
        "strand": data.get("strand"),
        "chromosome": data.get("seq_region_name"),
        "start": data.get("start"),
        "end": data.get("end"),
        "biotype": data.get("biotype", ""),
    }
