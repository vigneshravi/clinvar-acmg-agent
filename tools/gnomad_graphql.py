"""gnomAD GraphQL API client for GRCh38-native allele frequency queries.

Queries the gnomAD v4 dataset directly without coordinate liftover.
Provides population-level allele frequency breakdown and homozygote counts.
"""

import json as _json
import logging
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

GNOMAD_API = "https://gnomad.broadinstitute.org/api"

def _build_variant_query(variant_id: str, dataset: str) -> str:
    """Build an inline GraphQL query for gnomAD variant lookup."""
    return f"""
{{
  variant(variantId: "{variant_id}", dataset: {dataset}) {{
    variant_id
    rsid
    exome {{
      ac
      an
      af
      homozygote_count
      populations {{
        id
        ac
        an
        homozygote_count
      }}
    }}
    genome {{
      ac
      an
      af
      homozygote_count
      populations {{
        id
        ac
        an
        homozygote_count
      }}
    }}
  }}
}}
"""

# Map genome builds to gnomAD datasets
DATASET_MAP = {
    "GRCh38": "gnomad_r4",
    "GRCh37": "gnomad_r2_1",
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


def query_gnomad_variant(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    genome_build: str = "GRCh38",
) -> Optional[dict[str, Any]]:
    """Query gnomAD GraphQL API for variant allele frequencies.

    Args:
        chrom: Chromosome (e.g. "17" or "chr17")
        pos: Genomic position
        ref: Reference allele
        alt: Alternate allele
        genome_build: "GRCh38" or "GRCh37"

    Returns:
        Structured dict with allele frequencies, or None if not found.
    """
    chrom_clean = str(chrom).lstrip("chr")
    variant_id = f"{chrom_clean}-{pos}-{ref}-{alt}"
    dataset = DATASET_MAP.get(genome_build, "gnomad_r4")

    query = _build_variant_query(variant_id, dataset)
    payload = _json.dumps({"query": query}).encode("utf-8")

    try:
        req = Request(GNOMAD_API, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8")[:200]
        except Exception:
            pass
        logger.error("gnomAD GraphQL HTTP %d: %s", e.code, body)
        return None
    except Exception as exc:
        logger.error("gnomAD GraphQL failed: %s", exc)
        return None

    variant_data = (data.get("data") or {}).get("variant")
    if not variant_data:
        logger.info("gnomAD: variant %s not found in %s", variant_id, dataset)
        return None

    return _parse_gnomad_response(variant_data)


def _parse_gnomad_response(data: dict) -> dict[str, Any]:
    """Parse gnomAD GraphQL response into structured dict."""
    result: dict[str, Any] = {
        "variant_id": data.get("variant_id"),
        "rsid": data.get("rsid"),
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

    # Parse exome data
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

        # Population breakdown (compute AF from ac/an since v4 API
        # doesn't return af at the population level)
        for pop in exome.get("populations", []):
            pop_id = pop.get("id", "").lower()
            if pop_id in POP_NAMES:
                ac = pop.get("ac", 0) or 0
                an = pop.get("an", 0) or 0
                af = ac / an if an > 0 else 0
                result["populations"][pop_id] = {
                    "af": af,
                    "ac": ac,
                    "an": an,
                    "hom": pop.get("homozygote_count", 0),
                    "name": POP_NAMES.get(pop_id, pop_id),
                }
                if af > result["max_pop_af"]:
                    result["max_pop_af"] = af
                    result["max_pop_name"] = pop_id

    # Parse genome data
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

        # Add genome populations if exome didn't have them
        for pop in genome.get("populations", []):
            pop_id = pop.get("id", "").lower()
            if pop_id in POP_NAMES and pop_id not in result["populations"]:
                ac = pop.get("ac", 0) or 0
                an = pop.get("an", 0) or 0
                af = ac / an if an > 0 else 0
                result["populations"][pop_id] = {
                    "af": af,
                    "ac": ac,
                    "an": an,
                    "hom": pop.get("homozygote_count", 0),
                    "name": POP_NAMES.get(pop_id, pop_id),
                }
                if af > result["max_pop_af"]:
                    result["max_pop_af"] = af
                    result["max_pop_name"] = pop_id

    return result
