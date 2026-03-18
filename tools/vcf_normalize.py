"""VCF variant normalization — left-align indels against the reference genome.

gnomAD and most VCF-based databases use left-aligned representation,
while Ensembl VEP returns right-aligned. This module converts between them.
"""

import json as _json
import logging
from typing import Any, Optional
from urllib.error import HTTPError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _fetch_reference_seq(
    chrom: str, start: int, end: int, genome_build: str = "GRCh38"
) -> str:
    """Fetch reference sequence from Ensembl REST API."""
    base = ("https://grch37.rest.ensembl.org"
            if genome_build == "GRCh37"
            else "https://rest.ensembl.org")
    chrom = str(chrom).lstrip("chr")
    url = f"{base}/sequence/region/human/{chrom}:{start}..{end}?content-type=application/json"
    try:
        req = Request(url)
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=10) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return data.get("seq", "")
    except Exception as e:
        logger.warning("Failed to fetch reference at %s:%d-%d: %s", chrom, start, end, e)
        return ""


def left_align_variant(
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    genome_build: str = "GRCh38",
) -> tuple[str, int, str, str]:
    """Left-align an indel variant against the reference genome.

    Implements the bcftools norm algorithm:
    1. Right-trim: remove matching suffix bases from ref and alt
    2. If ref or alt is empty after trimming, prepend anchor base from left
    3. Left-shift: while last bases match, rotate and shift left
    4. Re-anchor: ensure ref and alt start with the same anchor base

    Returns (chrom, pos, ref, alt) left-aligned.
    """
    chrom = str(chrom).lstrip("chr")

    # SNVs don't need normalization
    if len(ref) == 1 and len(alt) == 1:
        return (chrom, pos, ref, alt)

    # Fetch reference context (100bp upstream for left-shifting)
    context_start = max(1, pos - 100)
    context_end = pos + max(len(ref), len(alt)) + 10
    ref_seq = _fetch_reference_seq(chrom, context_start, context_end, genome_build)
    if not ref_seq:
        logger.warning("Cannot fetch reference for normalization at %s:%d", chrom, pos)
        return (chrom, pos, ref, alt)

    def _ref_base_at(p: int) -> str:
        """Get reference base at genomic position p."""
        idx = p - context_start
        if 0 <= idx < len(ref_seq):
            return ref_seq[idx]
        return ""

    # Step 1: Right-trim common suffix
    while len(ref) > 0 and len(alt) > 0 and ref[-1] == alt[-1]:
        ref = ref[:-1]
        alt = alt[:-1]

    # Step 2: If ref or alt is now empty, we have a pure insertion or deletion
    # Need to add an anchor base from the left
    if len(ref) == 0 or len(alt) == 0:
        pos -= 1
        anchor = _ref_base_at(pos)
        ref = anchor + ref
        alt = anchor + alt

    # Step 3: Left-shift the variant
    # While the rightmost base of ref and alt are equal AND we can shift left
    while pos > 1:
        if len(ref) > 0 and len(alt) > 0 and ref[-1] == alt[-1]:
            # Remove last base from both
            ref = ref[:-1]
            alt = alt[:-1]
            # Prepend the base to the left
            pos -= 1
            left_base = _ref_base_at(pos)
            ref = left_base + ref
            alt = left_base + alt
        else:
            break

    # Step 4: Left-trim common prefix (keep 1 anchor base for VCF format)
    while len(ref) > 1 and len(alt) > 1 and ref[0] == alt[0]:
        ref = ref[1:]
        alt = alt[1:]
        pos += 1

    return (chrom, pos, ref, alt)


def normalize_vep_vcf_string(
    vcf_string: str, genome_build: str = "GRCh38"
) -> tuple[str, int, str, str]:
    """Parse a VEP vcf_string and left-align it.

    VEP vcf_string format: "17-43057063-G-GG"

    Returns (chrom, pos, ref, alt) left-aligned.
    """
    parts = vcf_string.split("-")
    if len(parts) != 4:
        raise ValueError(f"Invalid vcf_string format: {vcf_string}")

    chrom = parts[0]
    pos = int(parts[1])
    ref = parts[2]
    alt = parts[3]

    return left_align_variant(chrom, pos, ref, alt, genome_build)
