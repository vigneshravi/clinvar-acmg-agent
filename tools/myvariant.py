"""MyVariant.info API client for gnomAD frequencies, dbNSFP scores, and CADD.

MyVariant.info uses hg19/GRCh37 variant IDs by default. For GRCh38 input,
we use Ensembl liftover or fall back to rsID-based queries.
"""

import json as _json
import logging
import re
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

MYVARIANT_BASE = "https://myvariant.info/v1"

# Fields to request from MyVariant.info
MYVARIANT_FIELDS = ",".join([
    # gnomAD exome
    "gnomad_exome.af", "gnomad_exome.ac", "gnomad_exome.an", "gnomad_exome.hom",
    # gnomAD genome
    "gnomad_genome.af", "gnomad_genome.ac", "gnomad_genome.an", "gnomad_genome.hom",
    # dbNSFP predictors
    "dbnsfp.revel", "dbnsfp.metarnn", "dbnsfp.bayesdel",
    "dbnsfp.alphamissense",
    # dbNSFP conservation
    "dbnsfp.phylo", "dbnsfp.phastcons",
    # CADD (top-level, includes GERP, phyloP, SIFT, PolyPhen)
    "cadd.phred", "cadd.rawscore", "cadd.sift", "cadd.polyphen",
    "cadd.gerp", "cadd.phylop",
    # dbSNP rsID
    "dbsnp.rsid",
])


def _myvariant_get(url: str, retries: int = 2) -> Optional[dict]:
    """GET from MyVariant.info with retry."""
    for attempt in range(retries + 1):
        try:
            req = Request(url)
            req.add_header("Accept", "application/json")
            with urlopen(req, timeout=15) as resp:
                data = _json.loads(resp.read().decode("utf-8"))
                if isinstance(data, dict) and data.get("notfound"):
                    return None
                return data
        except HTTPError as e:
            if e.code == 404:
                return None
            if e.code == 429 and attempt < retries:
                import time
                time.sleep(1)
                continue
            logger.error("MyVariant HTTP %d for %s", e.code, url)
            return None
        except Exception as exc:
            logger.error("MyVariant request failed: %s", exc)
            return None
    return None


def _extract_score(value: Any) -> Optional[float]:
    """Extract a single float score from a value that may be a list."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        nums = [float(v) for v in value if v is not None]
        return max(nums) if nums else None
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _extract_pred(value: Any) -> Optional[str]:
    """Extract a prediction string from a value that may be a list."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Return most damaging prediction
        preds = [str(v) for v in value if v]
        for p in ["D", "P", "A"]:  # Damaging, Pathogenic, Ambiguous
            if p in preds:
                return p
        return preds[0] if preds else None
    return str(value)


def _build_hg19_variant_id(chrom: str, pos: int, ref: str, alt: str) -> str:
    """Build a MyVariant.info hg19-style variant ID."""
    chrom = str(chrom).lstrip("chr")
    if len(ref) == 1 and len(alt) == 1:
        # SNV
        return f"chr{chrom}:g.{pos}{ref}>{alt}"
    elif len(ref) == 1 and len(alt) > 1:
        # Insertion
        inserted = alt[1:]
        return f"chr{chrom}:g.{pos}_{pos + 1}ins{inserted}"
    elif len(ref) > 1 and len(alt) == 1:
        # Deletion
        if len(ref) == 2:
            return f"chr{chrom}:g.{pos + 1}del"
        else:
            return f"chr{chrom}:g.{pos + 1}_{pos + len(ref) - 1}del"
    else:
        # Complex
        return f"chr{chrom}:g.{pos}_{pos + len(ref) - 1}delins{alt}"


def liftover_grch38_to_grch37(
    chrom: str, pos: int, genome_build: str = "GRCh38"
) -> Optional[tuple[str, int]]:
    """Use Ensembl REST API to liftover a position from GRCh38 to GRCh37."""
    if genome_build == "GRCh37":
        return (str(chrom).lstrip("chr"), pos)

    chrom_clean = str(chrom).lstrip("chr")
    url = (
        f"https://rest.ensembl.org/map/human/GRCh38/"
        f"{chrom_clean}:{pos}..{pos}/GRCh37"
        f"?content-type=application/json"
    )
    try:
        req = Request(url)
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        mappings = data.get("mappings", [])
        if mappings:
            mapped = mappings[0].get("mapped", {})
            return (str(mapped.get("seq_region_name", chrom_clean)),
                    int(mapped.get("start", pos)))
    except Exception as e:
        logger.warning("Liftover failed: %s", e)
    return None


def query_myvariant(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    genome_build: str = "GRCh38",
    rsid: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """Query MyVariant.info for gnomAD frequencies and dbNSFP scores.

    Returns parsed dict with gnomad, insilico, conservation sections,
    or None if variant not found.
    """
    data = None

    # Strategy 1: If we have an rsID, use that (works regardless of build)
    if rsid:
        url = f"{MYVARIANT_BASE}/query?q=dbsnp.rsid:{rsid}&fields={MYVARIANT_FIELDS}"
        resp = _myvariant_get(url)
        if resp and isinstance(resp, dict):
            hits = resp.get("hits", [])
            if hits:
                data = hits[0]

    # Strategy 2: Convert coordinates and query directly
    if not data:
        # Liftover to hg19 if needed
        if genome_build == "GRCh38":
            lifted = liftover_grch38_to_grch37(chrom, pos, genome_build)
            if lifted:
                hg19_chrom, hg19_pos = lifted
                vid = _build_hg19_variant_id(hg19_chrom, hg19_pos, ref, alt)
            else:
                logger.warning("Could not liftover %s:%d to hg19", chrom, pos)
                return None
        else:
            vid = _build_hg19_variant_id(chrom, pos, ref, alt)

        encoded_vid = quote(vid, safe="")
        url = f"{MYVARIANT_BASE}/variant/{encoded_vid}?fields={MYVARIANT_FIELDS}"
        data = _myvariant_get(url)

    if not data:
        return None

    return _parse_myvariant_response(data)


def _parse_myvariant_response(data: dict) -> dict[str, Any]:
    """Parse MyVariant.info response into structured annotation dict."""
    result: dict[str, Any] = {
        "rsid": None,
        "allele_frequency": {},
        "insilico_predictors": {},
        "conservation": {},
    }

    # rsID
    dbsnp = data.get("dbsnp", {})
    if isinstance(dbsnp, dict):
        result["rsid"] = dbsnp.get("rsid")

    # --- gnomAD allele frequencies ---
    af_data: dict[str, Any] = {
        "global_af": None,
        "exome_af": None,
        "genome_af": None,
        "populations": {},
        "ac": None, "an": None, "hom": 0,
    }

    gnomad_exome = data.get("gnomad_exome", {})
    gnomad_genome = data.get("gnomad_genome", {})

    if isinstance(gnomad_exome, dict):
        af_block = gnomad_exome.get("af", {})
        if isinstance(af_block, dict):
            af_data["exome_af"] = af_block.get("af")
            af_data["global_af"] = af_block.get("af")
            # Population AFs
            pop_map = {
                "afr": "af_afr", "amr": "af_amr", "asj": "af_asj",
                "eas": "af_eas", "fin": "af_fin", "nfe": "af_nfe",
                "sas": "af_sas", "oth": "af_oth",
            }
            for pop_name, field in pop_map.items():
                val = af_block.get(field)
                if val is not None:
                    af_data["populations"][pop_name] = val

        ac_block = gnomad_exome.get("ac", {})
        if isinstance(ac_block, dict):
            af_data["ac"] = ac_block.get("ac")
        an_block = gnomad_exome.get("an", {})
        if isinstance(an_block, dict):
            af_data["an"] = an_block.get("an")
        hom_block = gnomad_exome.get("hom", {})
        if isinstance(hom_block, dict):
            af_data["hom"] = hom_block.get("hom", 0)

    if isinstance(gnomad_genome, dict):
        af_block = gnomad_genome.get("af", {})
        if isinstance(af_block, dict):
            af_data["genome_af"] = af_block.get("af")
            if af_data["global_af"] is None:
                af_data["global_af"] = af_block.get("af")

    result["allele_frequency"] = af_data

    # --- In silico predictors (from dbNSFP + CADD) ---
    dbnsfp = data.get("dbnsfp", {})
    cadd = data.get("cadd", {})
    predictors: dict[str, Any] = {}

    if isinstance(dbnsfp, dict):
        # REVEL
        revel = dbnsfp.get("revel", {})
        if isinstance(revel, dict):
            score = _extract_score(revel.get("score"))
            if score is not None:
                interp = "Likely pathogenic" if score > 0.75 else "Uncertain" if score > 0.15 else "Likely benign"
                predictors["revel"] = {"score": round(score, 4), "interpretation": interp}

        # MetaRNN
        metarnn = dbnsfp.get("metarnn", {})
        if isinstance(metarnn, dict):
            score = _extract_score(metarnn.get("score"))
            pred = _extract_pred(metarnn.get("pred"))
            if score is not None:
                predictors["metarnn"] = {
                    "score": round(score, 4),
                    "pred": "Damaging" if pred == "D" else "Tolerated",
                }

        # BayesDel
        bayesdel = dbnsfp.get("bayesdel", {})
        if isinstance(bayesdel, dict):
            for variant_key, label in [("add_af", "bayesdel_af"), ("no_af", "bayesdel_noaf")]:
                sub = bayesdel.get(variant_key, {})
                if isinstance(sub, dict):
                    score = _extract_score(sub.get("score"))
                    pred = _extract_pred(sub.get("pred"))
                    if score is not None:
                        predictors[label] = {
                            "score": round(score, 4),
                            "pred": "Damaging" if pred == "D" else "Tolerated",
                        }

        # AlphaMissense
        alpham = dbnsfp.get("alphamissense", {})
        if isinstance(alpham, dict):
            score = _extract_score(alpham.get("score"))
            pred = _extract_pred(alpham.get("pred"))
            if score is not None:
                interp = {"P": "Pathogenic", "B": "Benign", "A": "Ambiguous"}.get(pred, "Unknown")
                predictors["alphamissense"] = {"score": round(score, 4), "pred": interp}

    # CADD (top-level)
    if isinstance(cadd, dict):
        phred = cadd.get("phred")
        if phred is not None:
            predictors["cadd_phred"] = float(phred)

        sift = cadd.get("sift", {})
        if isinstance(sift, dict) and sift.get("val") is not None:
            predictors["sift"] = {"score": float(sift["val"]), "pred": sift.get("cat", "")}

        polyphen = cadd.get("polyphen", {})
        if isinstance(polyphen, dict) and polyphen.get("val") is not None:
            predictors["polyphen2"] = {"score": float(polyphen["val"]), "pred": polyphen.get("cat", "")}

    result["insilico_predictors"] = predictors

    # --- Conservation scores ---
    conservation: dict[str, Any] = {}
    if isinstance(dbnsfp, dict):
        phylo = dbnsfp.get("phylo", {})
        if isinstance(phylo, dict):
            for key, label in [
                ("100way_vertebrate", "phylop_100v"),
                ("470way_mammalian", "phylop_470m"),
            ]:
                sub = phylo.get(key, {})
                if isinstance(sub, dict):
                    score = _extract_score(sub.get("score"))
                    if score is not None:
                        conservation[label] = round(score, 3)

        phastcons = dbnsfp.get("phastcons", {})
        if isinstance(phastcons, dict):
            for key, label in [
                ("100way_vertebrate", "phastcons_100v"),
                ("470way_mammalian", "phastcons_470m"),
            ]:
                sub = phastcons.get(key, {})
                if isinstance(sub, dict):
                    score = _extract_score(sub.get("score"))
                    if score is not None:
                        conservation[label] = round(score, 3)

    if isinstance(cadd, dict):
        gerp = cadd.get("gerp", {})
        if isinstance(gerp, dict):
            rs = gerp.get("rs")
            if rs is not None:
                conservation["gerp_rs"] = round(float(rs), 3)

    result["conservation"] = conservation

    return result
