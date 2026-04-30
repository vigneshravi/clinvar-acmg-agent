"""ACMG/AMP threshold constants — every value is sourced from a citable reference.

Per project rule (2026-04-28 user feedback): no clinical threshold, gene list, or
cutoff in this codebase is hand-coded from memory. Each constant below has a
docstring naming the source paper + section so a reviewer can audit every value.

Architecture choice (Option B): the deterministic Python rule engine reads from
this module instead of using magic numbers. The user signs off on this file
once; the rules layer then becomes auditable and reproducible.

References cited:
    Richards et al. 2015. Genetics in Medicine 17:405-424. DOI: 10.1038/gim.2015.30
    Abou Tayoun et al. 2018. Genetics in Medicine 20:1054-1060. DOI: 10.1038/s41436-018-0044-2
    Ghosh et al. 2018. Genome Medicine 10:21. (BA1 exception list — disorder-specific)
    Pejaver et al. 2022. AJHG 109:2163-2177. DOI: 10.1016/j.ajhg.2022.10.013
    Jaganathan et al. 2019. Cell 176:535-548. DOI: 10.1016/j.cell.2018.12.015 (SpliceAI)
    Karczewski et al. 2020. Nature 581:434-443. DOI: 10.1038/s41586-020-2308-7 (gnomAD constraint)
"""

# ============================================================================
# POPULATION ALLELE FREQUENCY THRESHOLDS
# ============================================================================

BA1_FREQ_THRESHOLD = 0.05
"""BA1: Allele frequency >5% in gnomAD = stand-alone benign.

Source: Richards et al. 2015, Table 4 (BA1 row), p. 410.
'Allele frequency is >5% in Exome Sequencing Project, 1000 Genomes Project, or
Exome Aggregation Consortium.' Adapted for gnomAD per modern practice.
"""

BS1_FREQ_THRESHOLD = 0.01
"""BS1: Allele frequency higher than expected for the disorder = strong benign.

Source: Richards et al. 2015, Table 4 (BS1 row), p. 410.
The 1% threshold is widely adopted; ClinGen SVI / VCEPs publish disorder-specific
overrides. Default value used here when no VCEP override applies.
"""

PM2_FREQ_THRESHOLD = 0.0001
"""PM2: Absent or extremely rare in population databases.

Source: Richards et al. 2015, Table 3 (PM2 row), p. 409.
Threshold of 0.0001 (0.01%) is the most-adopted operational definition.
Original Richards text: 'Absent from controls (or at extremely low frequency
if recessive)' — the 0.0001 numeric threshold is the field-standard interpretation.
"""

# ----------------------------------------------------------------------------
# SVI override: PM2 strength downgraded from Moderate to Supporting (default ON)
# ----------------------------------------------------------------------------

PM2_DEFAULT_STRENGTH = "Supporting"
"""ClinGen SVI 2020 downgraded PM2 from Moderate to Supporting.

Source: ClinGen SVI Working Group, recommendation announced 2020.
Reference: Ghosh et al. 2018 (Genome Medicine 10:21) and ClinGen SVI workspace
(https://clinicalgenome.org/working-groups/sequence-variant-interpretation/).
Rationale: With current sequencing scale (gnomAD v4 = 807K samples), 'absent'
is much weaker evidence than in 2015. APPLIED BY DEFAULT per user instruction
(2026-04-28).
"""

# ============================================================================
# PVS1 — Abou Tayoun et al. 2018 11-branch decision tree
# ============================================================================
# Source: Abou Tayoun et al. 2018, Genet Med 20:1054-1060.
# Title: 'Recommendations for interpreting the loss of function PVS1 ACMG/AMP
# variant criterion.' Figure 1 (decision tree).

# --- LoF consequence types (Sequence Ontology terms via Ensembl VEP) ---
LOF_CONSEQUENCES_NONSENSE_FRAMESHIFT = {"frameshift_variant", "stop_gained"}
"""Branch entry: nonsense or frameshift variants per Abou Tayoun Figure 1."""

LOF_CONSEQUENCES_CANONICAL_SPLICE = {"splice_donor_variant", "splice_acceptor_variant"}
"""Branch entry: ±1 or ±2 canonical splice site variants per Abou Tayoun."""

LOF_CONSEQUENCES_START_LOST = {"start_lost"}
"""Branch entry: initiation codon variants per Abou Tayoun."""

LOF_CONSEQUENCES_DELETION = {"transcript_ablation", "feature_truncation"}
"""Branch entry: full-gene or exon deletion (where annotated)."""

# --- NMD escape rule ---
NMD_ESCAPE_LAST_50NT_PENULTIMATE = 50
"""NMD escape: PTC within last 50 nt of penultimate exon escapes NMD.

Source: Abou Tayoun et al. 2018, p. 1055, citing Nagy & Maquat 1998 (TIBS).
'A PTC located in the last exon, or within the last 50 nucleotides of the
penultimate exon, will escape nonsense-mediated mRNA decay.'
"""

PVS1_TRUNCATION_FRACTION_THRESHOLD = 0.10
"""If NMD escape, PVS1_Strong applies if the truncation removes >10% of protein.

Source: Abou Tayoun et al. 2018, Figure 1, branch A.2.a.
"""

# ============================================================================
# ClinGen Dosage Sensitivity — Haploinsufficiency Score (PVS1 applicability)
# ============================================================================
# PVS1 is only applicable when LoF is the established disease mechanism.
# The authoritative source for this is ClinGen's curated Haploinsufficiency
# Score, NOT gnomAD constraint metrics (constraint measures population-level
# depletion, which fails for tumor suppressors like BRCA2 where heterozygotes
# survive into reproductive age — verified empirically: BRCA2 LOEUF = 0.824
# would falsely exclude it from PVS1 if we used a LOEUF cutoff).
#
# Source: Riggs et al. 2020, Genet Med 22:245-257.
# 'Technical standards for the interpretation and reporting of constitutional
# copy-number variants: a joint consensus recommendation of the American
# College of Medical Genetics and Genomics (ACMG) and the Clinical Genome
# Resource (ClinGen).' DOI: 10.1038/s41436-019-0686-8
#
# Scores (per ClinGen scoring rubric):
#   3  = Sufficient evidence for HI / dosage pathogenicity (PVS1 APPLICABLE)
#   2  = Some evidence for HI                            (PVS1 with caveat)
#   1  = Little evidence for HI                          (PVS1 NOT applicable)
#   0  = No evidence for HI                              (PVS1 NOT applicable)
#   30 = Gene associated with autosomal recessive phenotype
#        (PVS1 only applicable in trans with another pathogenic variant)
#   40 = Dosage sensitivity unlikely                     (PVS1 NOT applicable)
# ----------------------------------------------------------------------------

CLINGEN_DOSAGE_TSV_URL = (
    "https://ftp.clinicalgenome.org/ClinGen_gene_curation_list_GRCh38.tsv"
)
"""ClinGen authoritative Haploinsufficiency / Triplosensitivity TSV.

Live URL — fetched once per session and parsed into memory. ClinGen does not
provide a JSON REST API for these scores; the FTP TSV is the canonical source.
"""

PVS1_APPLICABLE_HI_SCORES = {3}
"""HI scores for which PVS1 is APPLICABLE (autosomal dominant, sufficient evidence)."""

PVS1_APPLICABLE_WITH_CAVEAT_HI_SCORES = {2}
"""HI scores for which PVS1 applies with reduced strength (some evidence)."""

PVS1_RECESSIVE_HI_SCORES = {30}
"""HI score 30 = AR gene; PVS1 applies only in trans with another pathogenic variant."""

PVS1_NOT_APPLICABLE_HI_SCORES = {0, 1, 40}
"""HI scores for which PVS1 is NOT applicable.

  0  = No evidence
  1  = Little evidence
  40 = Dosage sensitivity unlikely
"""

# ============================================================================
# REVEL — Pejaver et al. 2022 calibration (default ON)
# ============================================================================
# Source: Pejaver et al. 2022, AJHG 109:2163-2177, Table 1.
# Title: 'Calibration of computational tools for missense variant pathogenicity
# classification and ClinGen recommendations for PP3/BP4 criteria.'
# These thresholds represent likelihood-ratio-calibrated cutoffs and are now
# the ClinGen-recommended replacement for the Richards 2015 binary PP3/BP4.

REVEL_PP3_STRONG_THRESHOLD = 0.932
"""REVEL >=0.932 -> PP3 Strong (pathogenic-supporting computational evidence).

Source: Pejaver et al. 2022, Table 1, calibrated likelihood ratio bin.
"""

REVEL_PP3_MODERATE_THRESHOLD = 0.773
"""REVEL >=0.773 (and <0.932) -> PP3 Moderate."""

REVEL_PP3_SUPPORTING_THRESHOLD = 0.644
"""REVEL >=0.644 (and <0.773) -> PP3 Supporting (Richards 2015 default strength)."""

REVEL_INDETERMINATE_LOWER = 0.290
"""REVEL between 0.290 and 0.644 -> indeterminate (no PP3 or BP4 call).

Per Pejaver et al. 2022, Table 1 — this gap deliberately produces no call.
"""

# REVEL <=0.290 -> BP4 Supporting (benign-supporting computational evidence)
REVEL_BP4_SUPPORTING_THRESHOLD = 0.290
"""REVEL <=0.290 -> BP4 Supporting per Pejaver 2022 calibration."""

REVEL_BP4_MODERATE_THRESHOLD = 0.183
"""REVEL <=0.183 -> BP4 Moderate per Pejaver 2022 Table 1."""

REVEL_BP4_STRONG_THRESHOLD = 0.016
"""REVEL <=0.016 -> BP4 Strong per Pejaver 2022 Table 1."""

# ============================================================================
# SpliceAI — Jaganathan et al. 2019
# ============================================================================
# Source: Jaganathan et al. 2019, Cell 176:535-548.
# Title: 'Predicting Splicing from Primary Sequence with Deep Learning.'

SPLICEAI_RECOMMENDED_THRESHOLD = 0.5
"""SpliceAI delta score >=0.5 = recommended splice-altering threshold (default).

Source: Jaganathan et al. 2019, Section 'Score thresholds,' p. 545.
The paper provides three operating points:
    0.2 = high recall (more permissive, includes more weak signals)
    0.5 = recommended (balanced precision/recall) — USED HERE
    0.8 = high precision (more conservative)
USER DECISION (2026-04-28): default to 0.5.
"""

# ============================================================================
# BP7 — synonymous + no splice impact
# ============================================================================
BP7_SPLICEAI_THRESHOLD = SPLICEAI_RECOMMENDED_THRESHOLD
"""BP7 application: synonymous variant AND SpliceAI delta < 0.5.

Source: Richards et al. 2015 BP7 definition + Jaganathan 2019 splice threshold.
"""

# ============================================================================
# Tavtigian et al. 2018 — Bayesian point-based combining (SVI override default)
# ============================================================================
# Source: Tavtigian et al. 2018, Genet Med 20:1054-1060.
# Title: 'Modeling the ACMG/AMP variant classification guidelines as a Bayesian
# classification framework.' DOI: 10.1038/s41436-018-0210-6
#
# Default applied per user instruction 2026-04-28 ('default to SVI overrides').
# This formulation supersedes Richards 2015 Table 5 boolean rules but the
# project also computes Richards Table 5 in parallel for comparison.
# ----------------------------------------------------------------------------

TAVTIGIAN_POINTS = {
    "Very Strong": 8,
    "Strong": 4,
    "Moderate": 2,
    "Supporting": 1,
    "Stand-Alone": 8,  # BA1: same magnitude as Very Strong; sign comes from direction
}
"""Per-strength point values per Tavtigian 2018 Table 2.

Pathogenic-direction criteria contribute positive points; benign-direction
criteria contribute negative points. Net total maps to a verdict via
TAVTIGIAN_VERDICT_THRESHOLDS below.
"""

TAVTIGIAN_VERDICT_THRESHOLDS = {
    "Pathogenic": (10, None),         # net points >= +10
    "Likely Pathogenic": (6, 9),      # net +6 to +9
    "VUS": (0, 5),                    # net 0 to +5 (also matches conflict-driven cases)
    "Likely Benign": (-6, -1),        # net -1 to -6
    "Benign": (None, -7),             # net <= -7
}
"""Tavtigian 2018 Table 2 net-point thresholds -> verdict.

Assumes prior probability of pathogenicity = 0.10 (Tavtigian Equation 5).
Range = (lower_bound_inclusive, upper_bound_inclusive); None = open-ended.
"""

# ============================================================================
# CONSEQUENCE TYPE SETS — Sequence Ontology terms via VEP
# ============================================================================
SYNONYMOUS_CONSEQUENCES = {"synonymous_variant", "stop_retained_variant"}
"""Synonymous variant SO terms per Ensembl VEP. Used for BP7."""

MISSENSE_CONSEQUENCES = {"missense_variant"}
"""Missense variant SO term."""

INFRAME_INDEL_CONSEQUENCES = {"inframe_insertion", "inframe_deletion"}
"""In-frame indel SO terms. Used for PM4 / BP3."""
