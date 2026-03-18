"""gnomAD GraphQL API client for multi-dataset allele frequency queries.

Supports gnomAD v4 (GRCh38), v3 (GRCh38 genomes), and v2.1 (GRCh37)
with all subsets (non-cancer, non-neuro, non-UKB, controls, etc.).
"""

import json as _json
import logging
import time
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

GNOMAD_API = "https://gnomad.broadinstitute.org/api"

# Available datasets with metadata
GNOMAD_DATASETS = {
    # GRCh38
    "gnomad_r4": {
        "label": "gnomAD v4.1.0",
        "build": "GRCh38",
        "samples": 807162,
        "description": "Full dataset",
    },
    "gnomad_r4_non_ukb": {
        "label": "gnomAD v4.1.0 (non-UKB)",
        "build": "GRCh38",
        "samples": 314392,
        "description": "Excluding UK Biobank",
    },
    "gnomad_r3": {
        "label": "gnomAD v3.1.2",
        "build": "GRCh38",
        "samples": 76156,
        "description": "Genomes only",
    },
    "gnomad_r3_non_cancer": {
        "label": "gnomAD v3.1.2 (non-cancer)",
        "build": "GRCh38",
        "samples": 74023,
        "description": "Excluding cancer cohorts",
    },
    "gnomad_r3_non_neuro": {
        "label": "gnomAD v3.1.2 (non-neuro)",
        "build": "GRCh38",
        "samples": 67442,
        "description": "Excluding neurological cohorts",
    },
    "gnomad_r3_non_topmed": {
        "label": "gnomAD v3.1.2 (non-TOPMed)",
        "build": "GRCh38",
        "samples": 40433,
        "description": "Excluding TOPMed samples",
    },
    "gnomad_r3_non_v2": {
        "label": "gnomAD v3.1.2 (non-v2)",
        "build": "GRCh38",
        "samples": 57344,
        "description": "Excluding v2 samples",
    },
    "gnomad_r3_controls_and_biobanks": {
        "label": "gnomAD v3.1.2 (controls/biobanks)",
        "build": "GRCh38",
        "samples": 16465,
        "description": "Controls and biobank samples only",
    },
    # GRCh37
    "gnomad_r2_1": {
        "label": "gnomAD v2.1.1",
        "build": "GRCh37",
        "samples": 141456,
        "description": "Full dataset",
    },
    "gnomad_r2_1_non_cancer": {
        "label": "gnomAD v2.1.1 (non-cancer)",
        "build": "GRCh37",
        "samples": 134187,
        "description": "Excluding cancer cohorts",
    },
    "gnomad_r2_1_non_neuro": {
        "label": "gnomAD v2.1.1 (non-neuro)",
        "build": "GRCh37",
        "samples": 114704,
        "description": "Excluding neurological cohorts",
    },
    "gnomad_r2_1_non_topmed": {
        "label": "gnomAD v2.1.1 (non-TOPMed)",
        "build": "GRCh37",
        "samples": 135743,
        "description": "Excluding TOPMed samples",
    },
    "gnomad_r2_1_controls": {
        "label": "gnomAD v2.1.1 (controls)",
        "build": "GRCh37",
        "samples": 60146,
        "description": "Controls only",
    },
}

# Population display names
POP_NAMES = {
    "afr": "African/African American",
    "amr": "Latino/Admixed American",
    "asj": "Ashkenazi Jewish",
    "eas": "East Asian",
    "fin": "Finnish",
    "mid": "Middle Eastern",
    "nfe": "Non-Finnish European",
    "sas": "South Asian",
    "remaining": "Remaining",
    "ami": "Amish",
}

# Population IDs to include (exclude sex-stratified and aggregated)
POP_IDS = set(POP_NAMES.keys())


def _gnomad_post(query: str) -> Optional[dict]:
    """POST to gnomAD GraphQL API."""
    payload = _json.dumps({"query": query}).encode("utf-8")
    try:
        req = Request(GNOMAD_API, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=25) as resp:
            return _json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:200]
        except Exception:
            pass
        logger.error("gnomAD HTTP %d: %s", e.code, body)
        return None
    except Exception as exc:
        logger.error("gnomAD request failed: %s", exc)
        return None


def _parse_populations(pop_list: list) -> dict[str, dict]:
    """Parse population list, computing AF from ac/an."""
    pops = {}
    for p in pop_list:
        pid = p.get("id", "").lower()
        if pid not in POP_IDS:
            continue
        ac = p.get("ac", 0) or 0
        an = p.get("an", 0) or 0
        af = ac / an if an > 0 else 0.0
        pops[pid] = {
            "af": af,
            "ac": ac,
            "an": an,
            "hom": p.get("homozygote_count", 0),
            "name": POP_NAMES.get(pid, pid),
        }
    return pops


def query_gnomad_variant(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    genome_build: str = "GRCh38",
    dataset: str = "gnomad_r4",
) -> Optional[dict[str, Any]]:
    """Query gnomAD for a variant by coordinates."""
    chrom_clean = str(chrom).lstrip("chr")
    variant_id = f"{chrom_clean}-{pos}-{ref}-{alt}"

    query = (
        f'{{variant(variantId: "{variant_id}", dataset: {dataset}) {{'
        f'variant_id rsid '
        f'exome {{ac an af homozygote_count populations {{id ac an homozygote_count}}}} '
        f'genome {{ac an af homozygote_count populations {{id ac an homozygote_count}}}}'
        f'}}}}'
    )

    resp = _gnomad_post(query)
    if not resp:
        return None

    variant_data = (resp.get("data") or {}).get("variant")
    if not variant_data:
        return None

    return _parse_variant_response(variant_data, dataset)


def query_gnomad_by_rsid(
    rsid: str,
    genome_build: str = "GRCh38",
    dataset: str = "gnomad_r4",
) -> Optional[dict[str, Any]]:
    """Query gnomAD by rsID.

    Falls back to variant_search if rsID maps to multiple variants.
    """
    query = (
        f'{{variant(rsid: "{rsid}", dataset: {dataset}) {{'
        f'variant_id rsid '
        f'exome {{ac an af homozygote_count populations {{id ac an homozygote_count}}}} '
        f'genome {{ac an af homozygote_count populations {{id ac an homozygote_count}}}}'
        f'}}}}'
    )
    resp = _gnomad_post(query)
    if not resp:
        return None

    # Check for "Multiple variants found" error
    errors = resp.get("errors", [])
    if errors:
        for err in errors:
            if "multiple variants" in (err.get("message", "")).lower():
                logger.info("gnomAD: rsID %s has multiple variants, trying variant_search", rsid)
                return _query_gnomad_rsid_via_search(rsid, dataset)

    variant_data = (resp.get("data") or {}).get("variant")
    if not variant_data:
        return None

    return _parse_variant_response(variant_data, dataset)


def _query_gnomad_rsid_via_search(
    rsid: str, dataset: str
) -> Optional[dict[str, Any]]:
    """When rsID maps to multiple variants, use variant_search to find the ID
    then query by variant_id."""
    search_query = (
        f'{{variant_search(query: "{rsid}", dataset: {dataset}) '
        f'{{ variant_id }}}}'
    )
    resp = _gnomad_post(search_query)
    if resp:
        search_results = (resp.get("data") or {}).get("variant_search", [])
        if search_results:
            # Take the first (most common) variant
            vid = search_results[0].get("variant_id")
            if vid:
                parts = vid.split("-")
                if len(parts) == 4:
                    return query_gnomad_variant(
                        parts[0], int(parts[1]), parts[2], parts[3],
                        dataset=dataset,
                    )

    return None


def query_gnomad_multi_dataset(
    rsid: str,
    datasets: Optional[list[str]] = None,
    genome_build: str = "GRCh38",
) -> dict[str, Optional[dict[str, Any]]]:
    """Query gnomAD across multiple datasets for comparison.

    Returns dict mapping dataset_id → variant data (or None if not found).
    """
    if datasets is None:
        datasets = [ds for ds, info in GNOMAD_DATASETS.items()
                    if info["build"] == genome_build]

    results = {}
    for ds in datasets:
        try:
            result = query_gnomad_by_rsid(rsid, genome_build, ds)
            results[ds] = result
            time.sleep(0.2)  # rate limiting
        except Exception as e:
            logger.warning("gnomAD query failed for %s/%s: %s", rsid, ds, e)
            results[ds] = None

    return results


def _parse_variant_response(data: dict, dataset: str) -> dict[str, Any]:
    """Parse gnomAD variant response into structured dict."""
    result: dict[str, Any] = {
        "variant_id": data.get("variant_id"),
        "rsid": data.get("rsid"),
        "dataset": dataset,
        "dataset_label": GNOMAD_DATASETS.get(dataset, {}).get("label", dataset),
        "global_af": None,
        "exome": None,
        "genome": None,
        "populations": {},
        "ac": 0,
        "an": 0,
        "hom": 0,
        "max_pop_af": 0.0,
        "max_pop_name": "",
    }

    exome = data.get("exome")
    if exome:
        result["exome"] = {
            "ac": exome.get("ac", 0),
            "an": exome.get("an", 0),
            "af": exome.get("af", 0),
            "hom": exome.get("homozygote_count", 0),
        }
        result["global_af"] = exome.get("af")
        result["ac"] = exome.get("ac", 0)
        result["an"] = exome.get("an", 0)
        result["hom"] = exome.get("homozygote_count", 0)
        result["populations"].update(_parse_populations(exome.get("populations", [])))

    genome = data.get("genome")
    if genome:
        result["genome"] = {
            "ac": genome.get("ac", 0),
            "an": genome.get("an", 0),
            "af": genome.get("af", 0),
            "hom": genome.get("homozygote_count", 0),
        }
        if result["global_af"] is None:
            result["global_af"] = genome.get("af")
        # Merge genome populations (don't overwrite exome)
        for pid, pdata in _parse_populations(genome.get("populations", [])).items():
            if pid not in result["populations"]:
                result["populations"][pid] = pdata

    # Compute max population AF
    for pid, pdata in result["populations"].items():
        if pdata["af"] > result["max_pop_af"]:
            result["max_pop_af"] = pdata["af"]
            result["max_pop_name"] = pid

    return result


def get_available_datasets(genome_build: str = "GRCh38") -> list[dict[str, str]]:
    """Get list of available gnomAD datasets for a genome build."""
    return [
        {"id": ds_id, **info}
        for ds_id, info in GNOMAD_DATASETS.items()
        if info["build"] == genome_build
    ]
