"""Genome build / notation consistency validator.

Validates that an input variant notation is consistent with the user's declared
genome build (GRCh37 or GRCh38). Used by the UI to warn the user when, for
example, they paste a GRCh38 NC_ accession but selected GRCh37 from the
dropdown.

Build assignment for NC_ (RefSeq chromosome) accessions follows the canonical
NCBI map for chromosomes 1..22, X, Y. Source: NCBI RefSeq genome assemblies,
https://www.ncbi.nlm.nih.gov/datasets/genome/ (verified 2026-04-30).

Notation classes recognised:
    cdna_hgvs        — NM_/NR_/NP_/ENST/ENSP transcript-level HGVS (build-agnostic)
    genomic_nc       — NC_000XXX.YY:g.... genomic HGVS (build-resolvable)
    bare_coordinates — chrN-pos-ref-alt or chrN:pos:ref:alt (build-ambiguous)
    rsid             — rsNNNNNNN dbSNP IDs (build-agnostic; map is build-aware
                       only when materialised to coordinates)
    unknown          — anything else
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# NC_ accession -> build map
# ---------------------------------------------------------------------------
# Source: NCBI RefSeq genome assemblies, https://www.ncbi.nlm.nih.gov/datasets/genome/
# GRCh38.p14 (GCF_000001405.40) and GRCh37.p13 (GCF_000001405.25). Covers the
# 24 nuclear chromosomes (1..22, X, Y); mitochondrial NC_012920 is build-agnostic
# (identical sequence shared between assemblies) so it is intentionally omitted
# from this map and would route to 'unknown'.

_NC_GRCH38: set[str] = {
    "NC_000001.11", "NC_000002.12", "NC_000003.12", "NC_000004.12",
    "NC_000005.10", "NC_000006.12", "NC_000007.14", "NC_000008.11",
    "NC_000009.12", "NC_000010.11", "NC_000011.10", "NC_000012.12",
    "NC_000013.11", "NC_000014.9",  "NC_000015.10", "NC_000016.10",
    "NC_000017.11", "NC_000018.10", "NC_000019.10", "NC_000020.11",
    "NC_000021.9",  "NC_000022.11", "NC_000023.11", "NC_000024.10",
}

_NC_GRCH37: set[str] = {
    "NC_000001.10", "NC_000002.11", "NC_000003.11", "NC_000004.11",
    "NC_000005.9",  "NC_000006.11", "NC_000007.13", "NC_000008.10",
    "NC_000009.11", "NC_000010.10", "NC_000011.9",  "NC_000012.11",
    "NC_000013.10", "NC_000014.8",  "NC_000015.9",  "NC_000016.9",
    "NC_000017.10", "NC_000018.9",  "NC_000019.9",  "NC_000020.10",
    "NC_000021.8",  "NC_000022.10", "NC_000023.10", "NC_000024.9",
}


# ---------------------------------------------------------------------------
# Notation regexes
# ---------------------------------------------------------------------------

_RX_CDNA_HGVS = re.compile(r"^(NM_|NR_|NP_|ENST|ENSP)", re.IGNORECASE)
_RX_NC = re.compile(r"\b(NC_\d{6}\.\d+)", re.IGNORECASE)
_RX_BARE_COORD = re.compile(
    r"^(?:chr)?([0-9]{1,2}|X|Y|MT)[\-:]\d+[\-:][ACGT\-]+[\-:][ACGT\-]+$",
    re.IGNORECASE,
)
_RX_RSID = re.compile(r"^rs\d+$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Helper — notation type detection
# ---------------------------------------------------------------------------

def detect_notation_type(s: str) -> str:
    """Return notation_type from the parser logic above."""
    if not s or not isinstance(s, str):
        return "unknown"
    s_stripped = s.strip()
    if not s_stripped:
        return "unknown"

    # Order matters: cdna_hgvs prefix check first because some strings could
    # accidentally satisfy a coord regex if malformed.
    if _RX_CDNA_HGVS.match(s_stripped):
        return "cdna_hgvs"
    if _RX_NC.search(s_stripped):
        return "genomic_nc"
    if _RX_RSID.match(s_stripped):
        return "rsid"
    if _RX_BARE_COORD.match(s_stripped):
        return "bare_coordinates"
    return "unknown"


def _implied_build_from_nc(s: str) -> Optional[str]:
    """Look up the NC_ accession and return implied build, or None if unknown."""
    m = _RX_NC.search(s)
    if not m:
        return None
    nc = m.group(1).upper()
    if nc in _NC_GRCH38:
        return "GRCh38"
    if nc in _NC_GRCH37:
        return "GRCh37"
    return None  # unknown / non-canonical (e.g. NC_012920 mitochondrial)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def validate_build(input_str: str, declared_build: str) -> Dict[str, Any]:
    """Validate that an input notation is consistent with the declared genome build.

    Returns dict:
        consistent: bool
        notation_type: 'cdna_hgvs' | 'genomic_nc' | 'bare_coordinates' | 'rsid' | 'unknown'
        notation_implied_build: 'GRCh37' | 'GRCh38' | None  (None for build-agnostic)
        message: str (human-readable)
        recommended_action: 'proceed' | 'use_implied_build' | 'manually_specify' | 'block'
    """
    # Normalise declared build label (accept "GRCh38 / hg38" forms too).
    declared_norm = "GRCh37" if "37" in (declared_build or "") else "GRCh38"

    notation_type = detect_notation_type(input_str)

    if notation_type == "cdna_hgvs":
        return {
            "consistent": True,
            "notation_type": notation_type,
            "notation_implied_build": None,
            "message": (
                f"Transcript-level HGVS is build-agnostic; will be resolved to "
                f"genomic coordinates against {declared_norm}."
            ),
            "recommended_action": "proceed",
        }

    if notation_type == "rsid":
        return {
            "consistent": True,
            "notation_type": notation_type,
            "notation_implied_build": None,
            "message": (
                f"dbSNP rsID is build-agnostic; coordinates will be resolved "
                f"against {declared_norm}."
            ),
            "recommended_action": "proceed",
        }

    if notation_type == "genomic_nc":
        implied = _implied_build_from_nc(input_str)
        if implied is None:
            # Non-canonical NC_ accession (mitochondrial, alt contig, etc.)
            return {
                "consistent": True,
                "notation_type": notation_type,
                "notation_implied_build": None,
                "message": (
                    "NC_ accession is not in the canonical chr1-22/X/Y map "
                    "(mitochondrial or alt contig); cannot determine implied build."
                ),
                "recommended_action": "proceed",
            }
        if implied == declared_norm:
            return {
                "consistent": True,
                "notation_type": notation_type,
                "notation_implied_build": implied,
                "message": f"NC_ accession matches declared build {declared_norm}.",
                "recommended_action": "proceed",
            }
        # Mismatch
        return {
            "consistent": False,
            "notation_type": notation_type,
            "notation_implied_build": implied,
            "message": (
                f"NC_ accession implies {implied} but declared build is "
                f"{declared_norm}. The accession version uniquely identifies the "
                f"assembly; trusting the accession is recommended."
            ),
            "recommended_action": "use_implied_build",
        }

    if notation_type == "bare_coordinates":
        # Bare coordinates carry no build information — the build validator
        # cannot tell which build a notation belongs to from coordinates alone.
        return {
            "consistent": True,
            "notation_type": notation_type,
            "notation_implied_build": None,
            "message": (
                "Bare coordinates are build-ambiguous: the same chr-pos-ref-alt "
                "string can be valid in both GRCh37 and GRCh38 but refer to "
                f"different loci. Trusting your declared build of {declared_norm}, "
                "but please double-check upstream."
            ),
            "recommended_action": "manually_specify",
        }

    # Unknown notation
    return {
        "consistent": False,
        "notation_type": "unknown",
        "notation_implied_build": None,
        "message": (
            "Could not classify the input notation as transcript HGVS, NC_ "
            "genomic HGVS, bare coordinates, or rsID. Please re-check the format."
        ),
        "recommended_action": "block",
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cases = [
        # (label, input_str, declared_build, expected_consistent)
        ("cdna NM + GRCh37 (consistent)",
         "NM_000059.4:c.5946del", "GRCh37", True),
        ("NC GRCh38 + declared GRCh37 (mismatch -> implies GRCh38)",
         "NC_000013.11:g.32340301T>A", "GRCh37", False),
        ("NC GRCh37 + declared GRCh38 (mismatch -> implies GRCh37)",
         "NC_000013.10:g.32915001T>A", "GRCh38", False),
        ("bare coords + GRCh38 (build-ambiguous warning)",
         "13-32340301-T-A", "GRCh38", True),
        ("rsID + GRCh38 (consistent)",
         "rs80359550", "GRCh38", True),
    ]
    print("=" * 70)
    print("build_validator smoke test")
    print("=" * 70)
    for label, inp, blt, expected in cases:
        result = validate_build(inp, blt)
        ok = result["consistent"] == expected
        flag = "PASS" if ok else "FAIL"
        print(f"\n[{flag}] {label}")
        print(f"  input         : {inp}")
        print(f"  declared      : {blt}")
        print(f"  notation_type : {result['notation_type']}")
        print(f"  implied_build : {result['notation_implied_build']}")
        print(f"  consistent    : {result['consistent']}")
        print(f"  action        : {result['recommended_action']}")
        print(f"  message       : {result['message']}")
