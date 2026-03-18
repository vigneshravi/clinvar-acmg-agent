"""HGVS parsing utilities for variant input normalization."""

import re
from typing import Optional


def parse_variant_input(raw_input: str) -> dict:
    """Parse a user-provided variant string into structured components.

    Supports formats:
        "BRCA1 c.5266dupC"          → gene + cDNA
        "NM_007294.4:c.5266dupC"    → transcript:cDNA
        "chr17:43057062:G:GG"       → coordinates

    Returns dict with keys: gene, transcript, cdna, chrom, pos, ref, alt,
    input_mode, error.
    """
    raw_input = raw_input.strip()
    result = {
        "gene": None,
        "transcript": None,
        "cdna": None,
        "chrom": None,
        "pos": None,
        "ref": None,
        "alt": None,
        "input_mode": None,
        "error": None,
    }

    if not raw_input:
        result["error"] = "Empty variant input"
        return result

    # Pattern 1: Transcript:cDNA (e.g. NM_007294.4:c.5266dupC)
    transcript_match = re.match(
        r"^(NM_\d+\.\d+):(.+)$", raw_input, re.IGNORECASE
    )
    if transcript_match:
        result["transcript"] = transcript_match.group(1)
        result["cdna"] = transcript_match.group(2).strip()
        result["input_mode"] = "hgvs"
        return result

    # Pattern 2: Coordinates (e.g. chr17:43057062:G:GG)
    coord_match = re.match(
        r"^chr(\w+):(\d+):([ACGT]+):([ACGT]+)$", raw_input, re.IGNORECASE
    )
    if coord_match:
        result["chrom"] = coord_match.group(1)
        result["pos"] = int(coord_match.group(2))
        result["ref"] = coord_match.group(3).upper()
        result["alt"] = coord_match.group(4).upper()
        result["input_mode"] = "coordinates"
        return result

    # Pattern 3: Gene + variant (e.g. BRCA1 c.5266dupC)
    gene_match = re.match(r"^([A-Za-z0-9_-]+)\s+(.+)$", raw_input)
    if gene_match:
        result["gene"] = gene_match.group(1).upper()
        result["cdna"] = gene_match.group(2).strip()
        result["input_mode"] = "hgvs"
        return result

    # Fallback: treat entire input as a search term
    result["gene"] = raw_input
    result["input_mode"] = "hgvs"
    result["error"] = (
        f"Could not parse '{raw_input}' into a recognized variant format. "
        f"Treating as a free-text search term."
    )
    return result


def build_hgvs_string(
    transcript: Optional[str],
    gene: Optional[str],
    cdna: Optional[str],
) -> Optional[str]:
    """Build an HGVS string from components."""
    if transcript and cdna:
        return f"{transcript}:{cdna}"
    if gene and cdna:
        return f"{gene} {cdna}"
    return None
