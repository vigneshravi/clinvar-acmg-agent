"""Shared VariantState TypedDict flowing through every graph node."""

from typing import Any, Dict, List, Optional, TypedDict


class TranscriptRecord(TypedDict):
    nm_accession: str
    enst_accession: str
    gene_symbol: str
    gene_aliases: List[str]
    gene_full_name: str
    is_mane_select: bool
    is_mane_plus_clinical: bool
    is_most_reported_pathogenic: bool
    annotation_score: int
    biotype: str
    equivalent_hgvs: str


class CriterionResult(TypedDict):
    code: str
    name: str
    direction: str          # "pathogenic" or "benign"
    strength: str           # "Very Strong" / "Strong" / "Moderate" / "Supporting"
    met: bool
    justification: str
    evidence_source: str


class VariantState(TypedDict):
    # --- Input ---
    raw_input: str
    input_mode: str                                 # "hgvs" or "coordinates"
    genome_build: str                               # "GRCh37" or "GRCh38"

    # --- Resolved variant ---
    gene_symbol: Optional[str]
    gene_aliases: Optional[List[str]]
    gene_full_name: Optional[str]
    chrom: Optional[str]
    pos: Optional[int]
    ref: Optional[str]
    alt: Optional[str]
    all_transcripts: Optional[List[TranscriptRecord]]
    selected_transcript: Optional[str]
    hgvs_on_transcript: Optional[str]
    input_parse_error: Optional[str]

    # --- ClinVar evidence ---
    clinvar: Optional[Dict[str, Any]]
    clinvar_error: Optional[str]

    # --- gnomAD evidence (Phase 2) ---
    gnomad: Optional[Dict[str, Any]]
    gnomad_error: Optional[str]

    # --- PubMed evidence (Phase 3) ---
    pubmed: Optional[Dict[str, Any]]
    pubmed_error: Optional[str]

    # --- Structural evidence (AlphaFold + PDB + InterPro via BioMCP) ---
    alphafold: Optional[Dict[str, Any]]
    pdb: Optional[Dict[str, Any]]
    protein_info: Optional[Dict[str, Any]]
    structural_error: Optional[str]

    # --- Clinical evidence (CIViC + ClinGen + GWAS via BioMCP) ---
    civic: Optional[Dict[str, Any]]
    clingen: Optional[Dict[str, Any]]
    gwas: Optional[List[Dict[str, Any]]]
    clinical_evidence_error: Optional[str]

    # --- Pathway + Druggability (via BioMCP) ---
    pathways: Optional[List[Dict[str, Any]]]
    druggability: Optional[Dict[str, Any]]
    pathway_error: Optional[str]

    # --- Legacy TCGA fields (kept for compat) ---
    tcga_somatic: Optional[Dict[str, Any]]
    tcga_expression: Optional[Dict[str, Any]]
    tcga_error: Optional[str]

    # --- ACMG Classification ---
    criteria_triggered: Optional[List[CriterionResult]]
    classification: Optional[str]
    confidence: Optional[str]
    reasoning: Optional[str]
    disclaimer: Optional[str]

    # --- Graph control ---
    current_node: Optional[str]
    errors: List[str]
    warnings: List[str]


def make_initial_state(raw_input: str) -> VariantState:
    """Create an initial VariantState from a raw user input string."""
    return VariantState(
        raw_input=raw_input,
        input_mode="hgvs",
        genome_build="GRCh38",
        gene_symbol=None,
        gene_aliases=None,
        gene_full_name=None,
        chrom=None,
        pos=None,
        ref=None,
        alt=None,
        all_transcripts=None,
        selected_transcript=None,
        hgvs_on_transcript=None,
        input_parse_error=None,
        clinvar=None,
        clinvar_error=None,
        gnomad=None,
        gnomad_error=None,
        pubmed=None,
        pubmed_error=None,
        alphafold=None,
        pdb=None,
        protein_info=None,
        structural_error=None,
        civic=None,
        clingen=None,
        gwas=None,
        clinical_evidence_error=None,
        pathways=None,
        druggability=None,
        pathway_error=None,
        tcga_somatic=None,
        tcga_expression=None,
        tcga_error=None,
        criteria_triggered=None,
        classification=None,
        confidence=None,
        reasoning=None,
        disclaimer=None,
        current_node=None,
        errors=[],
        warnings=[],
    )
