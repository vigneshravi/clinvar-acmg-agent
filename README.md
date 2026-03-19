# ClinVar ACMG Variant Classifier

An AI-powered genomic variant annotation and classification tool built with LangGraph multi-agent architecture, Streamlit, and Claude (claude-sonnet-4-5). It integrates evidence from ClinVar, gnomAD, dbNSFP, UniProt, LitVar, and Ensembl VEP to produce ACMG/AMP 2015-compliant variant classifications with full transparency into every decision.

> **DISCLAIMER:** This is a research prototype. It has **not** been validated for clinical use. Variant classifications should be reviewed by a certified clinical molecular geneticist and confirmed through validated, CLIA-certified processes.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Data Sources & Evidence Flow](#data-sources--evidence-flow)
- [ACMG Criteria Implementation](#acmg-criteria-implementation)
- [Step-by-Step Classification Walkthrough](#step-by-step-classification-walkthrough)
- [Assumptions & Design Decisions](#assumptions--design-decisions)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Limitations](#limitations)

---

## Architecture Overview

The application uses a **LangGraph StateGraph** with 8 sequential nodes. Each node is an independent agent that reads from and writes to a shared `VariantState` dictionary.

### System Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app.py)                        │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────────┐  │
│  │ Gene + HGVS     │  │ Genomic          │  │ Look Up           │  │
│  │ Input           │  │ Coordinates      │  │ Transcripts       │  │
│  └────────┬────────┘  └────────┬─────────┘  └────────┬──────────┘  │
│           └────────────────────┴─────────────────────┘             │
│                                │                                    │
│                    ┌───────────▼───────────┐                       │
│                    │   Classify Variant    │                       │
│                    └───────────┬───────────┘                       │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │     LangGraph Pipeline   │
                    └────────────┬────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────┐
│                         GRAPH NODES                                 │
│                                                                     │
│  ┌──────────────┐    ┌────────────┐    ┌──────────────────────┐    │
│  │ 1. INPUT     │───▶│ 2. SUPER-  │───▶│ 3. CLINVAR AGENT     │    │
│  │    PARSER    │    │    VISOR   │    │    Entrez esearch/    │    │
│  │              │    │ (router)   │    │    esummary + XML     │    │
│  │ • Parse HGVS │    │            │    │    parse + HGVS       │    │
│  │ • VEP annot. │    │ • Check    │    │    validation         │    │
│  │ • NM↔ENST    │    │   fields   │    │                      │    │
│  │ • Transcript │    │ • Route or │    │ Outputs:             │    │
│  │   ranking    │    │   END      │    │ • variant_id, HGVS   │    │
│  │ • Left-align │    │            │    │ • significance       │    │
│  │   coords     │    │            │    │ • star_rating, rsID  │    │
│  └──────────────┘    └────────────┘    └──────────┬───────────┘    │
│                                                    │                │
│  ┌──────────────────────────────────────────────────▼──────────┐   │
│  │ 4. GNOMAD + IN SILICO AGENT                                 │   │
│  │                                                              │   │
│  │ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ ┌─────────┐ │   │
│  │ │ gnomAD      │ │ MyVariant   │ │ gnomAD     │ │ UniProt │ │   │
│  │ │ GraphQL     │ │ .info       │ │ Gene       │ │ REST    │ │   │
│  │ │             │ │             │ │ Constraint │ │         │ │   │
│  │ │ • 3 cohorts │ │ • dbNSFP    │ │            │ │ Domains │ │   │
│  │ │   overall   │ │   REVEL     │ │ • mis_z    │ │ Repeats │ │   │
│  │ │   non-cancer│ │   CADD      │ │ • LOEUF    │ │ Motifs  │ │   │
│  │ │   controls  │ │   MetaRNN   │ │ • pLI      │ │         │ │   │
│  │ │ • Per-pop   │ │   BayesDel  │ │            │ │         │ │   │
│  │ │   AFs       │ │   AlphaM.   │ │            │ │         │ │   │
│  │ │ • rsID      │ │   SIFT      │ │            │ │         │ │   │
│  │ │   lookup    │ │   PolyPhen  │ │            │ │         │ │   │
│  │ │             │ │ • Conserv.  │ │            │ │         │ │   │
│  │ │             │ │   phyloP    │ │            │ │         │ │   │
│  │ │             │ │   phastCons │ │            │ │         │ │   │
│  │ │             │ │   GERP++    │ │            │ │         │ │   │
│  │ └─────────────┘ └─────────────┘ └────────────┘ └─────────┘ │   │
│  │                                                              │   │
│  │ Pre-computes: BA1, BS1, PM2, PP3, BP4, PM1, PM4, BP3        │   │
│  └──────────────────────────────────────────┬───────────────────┘   │
│                                              │                      │
│  ┌───────────────────────────────────────────▼──────────────────┐   │
│  │ 5. LITVAR / PUBMED AGENT                                     │   │
│  │                                                               │   │
│  │ • LitVar entity search by rsID                                │   │
│  │ • Publication count + disease associations                    │   │
│  │ • PubMed MEDLINE enrichment (title, journal, pub type)        │   │
│  │ • Classify: case reports / functional studies / reviews        │   │
│  │ • Related genes + chemicals from co-occurrence                │   │
│  │                                                               │   │
│  │ Supports: PS3 (functional studies), PS4 (case reports)        │   │
│  └───────────────────────────────────┬───────────────────────────┘   │
│                                       │                              │
│  ┌────────────────────────────────────▼──────────────────────────┐   │
│  │ 6-7. ALPHAFOLD + TCGA AGENTS (stubs — future phases)         │   │
│  └────────────────────────────────────┬──────────────────────────┘   │
│                                       │                              │
│  ┌────────────────────────────────────▼──────────────────────────┐   │
│  │ 8. ACMG CLASSIFIER (Claude claude-sonnet-4-5)                │   │
│  │                                                               │   │
│  │ • Receives ALL evidence from prior nodes                      │   │
│  │ • Structured prompt with pre-computed criteria flags           │   │
│  │ • PVS1 assessment with last-exon / penultimate caveats        │   │
│  │ • Applies ACMG combining rules                                │   │
│  │ • Returns: criteria_triggered, classification, reasoning       │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Graph Edge Flow

```
START → input_parser → supervisor ─┬─→ clinvar_agent → gnomad_agent
                                    │    → pubmed_agent → alphafold_agent (stub)
                                    │    → tcga_agent (stub) → acmg_classifier → END
                                    │
                                    └─→ END  (on parse error)
```

---

## Data Sources & Evidence Flow

| Source | API | Data Retrieved | ACMG Criteria |
|--------|-----|---------------|---------------|
| **Ensembl VEP** | REST `/vep/homo_sapiens/hgvs/` | Transcript consequences, exon position, amino acid change, impact, protein position | PVS1 (exon assessment) |
| **Ensembl Xrefs** | REST `/xrefs/id/`, `/xrefs/symbol/` | NM_ ↔ ENST transcript mapping | Transcript resolution |
| **NCBI ClinVar** | Entrez esearch + esummary | Clinical significance, star rating, submitters, rsID, condition | PS1, PP5, BP6 |
| **NCBI Gene** | Entrez esearch + esummary | Gene aliases, full name, pathogenic submission counts per NM_ | Transcript ranking |
| **gnomAD v3.1.2 / v2.1.1** | GraphQL API | Allele frequencies (3 cohorts: overall, non-cancer, controls), per-population breakdown, homozygotes | BA1, BS1, PM2 |
| **MyVariant.info** | REST API | dbNSFP scores (REVEL, CADD, MetaRNN, BayesDel, AlphaMissense, SIFT, PolyPhen2), conservation (phyloP, phastCons, GERP++) | PP3, BP4 |
| **gnomAD Gene Constraint** | GraphQL API | Missense Z-score, o/e ratio, pLI, LOEUF | PM1 |
| **UniProt** | REST API | Functional domains (Domain, Zinc finger, DNA binding, Motif), repeat regions, protein length | PM1, PM4, BP3 |
| **NCBI LitVar** | REST API | Publication count, disease associations, related entities, PubMed metadata enrichment | PS3, PS4 |
| **Claude claude-sonnet-4-5** | LangChain / Anthropic API | ACMG criteria synthesis, combining rules, clinical reasoning | Final classification |

---

## ACMG Criteria Implementation

### Criteria Evaluated (14 of 28)

#### Pathogenic Criteria

| Criterion | Strength | Source | Logic | Assumptions |
|-----------|----------|--------|-------|-------------|
| **PVS1** | Very Strong / Strong / Moderate | VEP | Null variant (frameshift, stop_gained, splice_donor/acceptor, start_lost) in a gene where LOF is a known disease mechanism. **Caveats:** Last exon → Moderate (may escape NMD); Penultimate exon → Strong; Single-exon gene → Moderate; Not null → does not apply. | Assumes `is_canonical` from Ensembl equals the clinically relevant transcript. Uses VEP `exon` field (e.g., "19/23") to determine exon position. |
| **PS1** | Strong | ClinVar | Same amino acid change classified as pathogenic in ClinVar with ≥2-star review. | Evaluated by Claude based on ClinVar classification and review status. |
| **PS3** | Strong | LitVar | Well-established functional studies. Assessed from LitVar publication types — looks for "Research Support", "Comparative Study", "Evaluation Study" in PubMed pub_types. | Publication type classification is heuristic. LitVar may not capture all functional studies. Count of "functional studies" is approximate. |
| **PS4** | Strong | LitVar | Prevalence significantly increased in affected vs controls. Assessed from case reports and disease association counts in LitVar. | Relies on LitVar's automated literature mining. Does not perform independent statistical analysis of case prevalence. |
| **PM1** | Moderate / Supporting | gnomAD Constraint + UniProt | Missense variant in a well-established functional domain (UniProt Domain, Zinc finger, DNA binding, Motif) in a missense-constrained gene (gnomAD mis_z > 2.0). Domain + constraint → Moderate; domain only → Supporting. | Only applies to missense variants. UniProt "Region" annotations are excluded (too noisy). Motifs are included as functional. |
| **PM2** | Supporting | gnomAD Controls | Global AF < 0.01% in controls cohort. Uses controls global AF only — not per-population max. | Downgraded to Supporting per ClinGen SVI 2020 recommendation (was originally Moderate). Threshold of 0.0001 may be too strict for some disorders. |
| **PM4** | Moderate | VEP + UniProt | In-frame deletion/insertion in a non-repeat region causing protein length change. | Checks VEP `consequence_terms` for `inframe_deletion` / `inframe_insertion`. Repeat region check uses UniProt "Repeat" features only. |
| **PP3** | Supporting | dbNSFP + CADD | Multiple computational tools predict damaging effect. Requires ≥3 tools predicting damaging with majority consensus. Thresholds: REVEL > 0.75, CADD phred > 25, AlphaMissense > 0.564, BayesDel > 0.0692, MetaRNN = "D", SIFT < 0.05, PolyPhen2 > 0.85. | Uses ClinGen SVI-recommended REVEL thresholds. Consensus requires at least 3 tools agreeing. Only available for SNVs with dbNSFP annotations (not indels). |
| **PP5** | Supporting / Moderate / Strong | ClinVar | Reputable source classifies as pathogenic. Strength scaled by ClinVar star rating: 3+ stars (expert panel) → Strong; 2 stars → Moderate; 1 star → Supporting. | Per ClinGen recommendation for evidence strength scaling based on review status. |

#### Benign Criteria

| Criterion | Strength | Source | Logic | Assumptions |
|-----------|----------|--------|-------|-------------|
| **BA1** | Stand-alone | gnomAD Controls | Global AF > 5% in controls cohort. Stand-alone sufficient for Benign. | Uses controls global AF only. 5% is the standard ACMG threshold. |
| **BS1** | Strong | gnomAD Controls | Global AF > 1% in controls cohort, greater than expected for rare disease. | 1% threshold assumes rare disease gene. May need adjustment for common disease genes. |
| **BP3** | Supporting | VEP + UniProt | In-frame deletion/insertion within a repetitive region without known function. | Only triggers for in-frame variants in UniProt-annotated repeat regions. |
| **BP4** | Supporting | dbNSFP + CADD | Multiple computational tools predict no impact. Requires ≥3 tools predicting benign. Thresholds: REVEL < 0.15, CADD phred < 15. | Mirror image of PP3 logic. |
| **BP6** | Supporting / Moderate / Strong | ClinVar | Reputable source classifies as benign. Same star-based strength scaling as PP5. | Assumes ClinVar benign classifications with expert panel review are reliable. |

### Frequency Criteria: Cohort Selection Logic

```
Priority for BA1 / BS1 / PM2 evaluation:

1. gnomAD controls/biobanks (GRCh38: v3.1.2, GRCh37: v2.1.1 controls)
2. gnomAD non-cancer (if controls unavailable)
3. gnomAD overall (if neither available)

Within each cohort:
- Use GLOBAL allele frequency only
- Do NOT use per-population max AF (avoids founder effects, e.g., Finnish)

The UI explicitly states which cohort and AF value was used:
  "evaluated on controls global AF = 0.002368"
```

### ACMG Combining Rules

The classifier applies standard ACMG/AMP 2015 combining rules:

```
Pathogenic:
  • 1 Very Strong + ≥1 Strong
  • ≥2 Strong
  • 1 Strong + ≥3 Moderate/Supporting
  • 1 Very Strong + ≥2 Moderate

Likely Pathogenic:
  • 1 Strong + 1-2 Moderate
  • 1 Strong + ≥2 Supporting
  • ≥3 Moderate
  • 2 Moderate + ≥2 Supporting

Benign:
  • 1 Stand-alone (BA1)
  • ≥2 Strong benign

Likely Benign:
  • 1 Strong benign + 1 Supporting benign

VUS:
  • Does not meet any of the above
```

---

## Step-by-Step Classification Walkthrough

### Example: BRCA1 c.5266dupC (GRCh38)

This traces every decision point in the pipeline for the well-known BRCA1 5382insC founder mutation.

#### Step 1: Input Parsing

```
Input: "BRCA1 c.5266dupC"
├── Parse: gene=BRCA1, cdna=c.5266dupC, mode=hgvs
├── ClinVar pathogenic count lookup:
│   └── NM_007294.4: 196 submissions → primary transcript
├── VEP query: NM_007294.4:c.5266dupC
│   ├── VEP accepted ✓
│   ├── vcf_string: 17-43057063-G-GG (right-aligned)
│   └── Left-aligned: 17-43057062-T-TG (matches gnomAD)
├── Ensembl NM↔ENST mapping: 32 protein-coding transcripts
│   └── ENST00000357654 → NM_007294.4 (via xrefs/id)
├── Gene aliases: BRCAI, BRCC1, BROVCA1, FANCS, IRIS, ...
├── Transcript ranking:
│   └── NM_007294.4 / ENST00000357654  score=9
│       [MANE Select + Most Reported + Canonical + has NM_]
├── VEP annotation for selected transcript:
│   ├── Consequence: frameshift_variant
│   ├── Impact: HIGH
│   ├── Exon: 19/23
│   ├── Protein: position 1755-1756, AA: -/X
│   └── Strand: -1 (minus strand)
└── Output:
    ├── selected_transcript: NM_007294.4
    ├── hgvs_on_transcript: NM_007294.4:c.5266dupC
    └── coordinates: chr17:43057062 T>TG
```

#### Step 2: Supervisor Routing

```
Check: gene_symbol=BRCA1 ✓, hgvs_on_transcript set ✓
Decision: Route to clinvar_agent ✓
```

#### Step 3: ClinVar Agent

```
Query strategy (ranked):
├── Try 1: "NM_007294.4:c.5266dupC[HGVS]"
│   └── Result: variant_id=3336480, HGVS=c.206_207delinsTG
│       └── HGVS mismatch! Position c.5266 ≠ c.206 → REJECTED
├── Try 2: "NM_007294.4:c.5266dupC[Variant name]"
│   └── Same wrong result → REJECTED
├── Try 3: "BRCA1 c.5266dupC" (raw input fallback)
│   └── Search: BRCA1[gene] AND c.5266dupC[Variant name]
│       └── Result: variant_id=17677 ✓
│           HGVS: NM_007294.4(BRCA1):c.5266dup (p.Gln1756fs)
│           Position match: c.5266 = c.5266 ✓
│           Type match: dup = dup ✓ → ACCEPTED
├── Dup notation expansion:
│   └── Also tried: c.5266dupA, c.5266dupC, c.5266dupG, c.5266dupT
│       (ClinVar stores "c.5266dupC" but some entries use "c.5266dup")
└── Final ClinVar record:
    ├── Variant ID: 17677
    ├── Clinical Significance: Pathogenic
    ├── Review Status: reviewed by expert panel
    ├── Star Rating: 3/4
    ├── Submitters: 91
    ├── Condition: Breast-ovarian cancer, familial, susceptibility to, 1
    ├── rsID: rs80357906 (from dbSNP xref in esummary XML)
    └── Conflicting interpretations: No
```

#### Step 4: gnomAD + In Silico Agent

```
Coordinate resolution:
├── Coordinates from input_parser: chr17:43057062 T>TG ✓
├── gnomAD variant ID: 17-43057062-T-TG
└── rsID from ClinVar: rs80357906

gnomAD queries (3 cohorts via rsID):
├── gnomAD v3.1.2 (overall): AF = 6.91e-05
├── gnomAD v3.1.2 (non-cancer): AF = 6.76e-06
├── gnomAD v3.1.2 (controls/biobanks): not found
│   └── Fallback: use non-cancer for ACMG criteria
└── Per-population breakdown available for overall + non-cancer

Frequency criteria (evaluated on non-cancer global AF = 6.76e-06):
├── BA1 (AF > 5%): NOT MET — AF far below 5%
├── BS1 (AF > 1%): NOT MET — AF far below 1%
└── PM2 (AF < 0.01%): MET ✓ — AF 0.000007 < 0.0001

In silico predictors (MyVariant.info):
├── Liftover: GRCh38 chr17:43057062 → GRCh37 (via Ensembl /map/)
├── Result: No dbNSFP data (insertion/frameshift — not a SNV)
└── Consensus: Uncertain (no predictors available for indels)

Gene constraint (gnomAD GraphQL):
├── Missense Z-score: 2.34 (moderately constrained)
├── LOEUF: 0.885 (LOF tolerant — surprising for BRCA1)
├── pLI: ~0 (low LOF intolerance by pLI, but LOEUF is the modern metric)
└── Interpretation: Moderately missense-constrained, LOF tolerant

UniProt domains (P38398):
├── BRCT 1: aa 1642-1736
├── BRCT 2: aa 1756-1855
├── RING-type zinc finger: aa 24-65
├── Protein length: 1863 aa
├── Variant at protein position 1755-1756
│   └── Domain overlap check: position 1755 NOT in BRCT 1 (1642-1736)
│       and NOT in BRCT 2 (1756-1855) — just at the boundary
└── PM1: NOT MET — not a missense variant (PM1 is missense-only)

PM4/BP3:
└── NOT MET — not an in-frame insertion/deletion (it's a frameshift)
```

#### Step 5: LitVar / PubMed Agent

```
LitVar query: rs80357906
├── Entity search: found ✓
│   ├── Name: c.5266dupC
│   ├── Total publications: 266
│   ├── Clinical significance (literature): risk-factor, pathogenic
│   └── First published: 2005
├── Disease associations:
│   ├── Breast Neoplasms: 218 publications
│   ├── Neoplasms: 199
│   ├── Hereditary Breast and Ovarian Cancer Syndrome: 112
│   ├── Ovarian Neoplasms: 80
│   └── Melanoma: 63
├── PubMed enrichment (top 15 PMIDs):
│   ├── Publication types classified:
│   │   ├── Case Reports: 1
│   │   ├── Functional studies (Research Support): 9
│   │   └── Reviews: 1
│   └── Sample publications:
│       ├── [2020] Case Reports: Recurrent Germline BRCA2 Gene Mutation...
│       ├── [2021] Review: Pathogenic BRCA Variants as Biomarkers...
│       └── [2018] Journal Article: Ethnic-Specific Spectrum...
└── Related entities:
    ├── Genes: BRCA1, BRCA2
    └── Chemicals: imidazole mustard (17 co-occurrences)
```

#### Step 6-7: AlphaFold + TCGA (Stubs)

```
Both return: None (not yet implemented)
Warnings added:
├── "Structural data not available — PM1 cannot be assessed from structural data"
└── "TCGA data not available — somatic evidence cannot be assessed"
```

#### Step 8: ACMG Classifier (Claude claude-sonnet-4-5)

```
Evidence prompt sent to Claude includes:
├── Variant: NM_007294.4:c.5266dupC on BRCA1 (GRCh38)
├── VEP annotation: frameshift variant, Exon 19/23, HIGH impact
├── PVS1 assessment:
│   ├── Is null variant: YES (frameshift)
│   ├── Exon: 19/23 — NOT last exon, NOT penultimate
│   ├── PVS1 applicable: YES
│   └── PVS1 strength: Very Strong (no caveats apply)
├── ClinVar: Pathogenic, 3-star expert panel, 91 submitters
├── gnomAD: non-cancer global AF = 6.76e-06, PM2 = MET
├── In silico: no predictors (indel)
├── Gene constraint: mis_z=2.34, LOEUF=0.885
├── UniProt: 3 domains, variant NOT in domain
├── LitVar: 266 publications, 9 functional studies, 218 breast neoplasm
└── Instructions: evaluate PVS1, PS1, PS3, PS4, PP5, PM1, PM2, PM4,
    PP3, BA1, BS1, BP3, BP4, BP6 with pre-computed flags

Claude's response (JSON):
├── PVS1: Very Strong ✓ — frameshift in BRCA1 exon 19/23
├── PS3: Strong ✓ — 9 functional studies, 266 publications
├── PS4: Strong ✓ — extensive case reports across populations
├── PP5: Strong ✓ — ClinVar Pathogenic, expert panel, 3-star
├── PM2: Supporting ✓ — global AF < 0.01% in non-cancer
├── PM1: not met — not a missense variant
├── PP3/BP4: not evaluable — no in silico scores for indels
├── BA1/BS1: not met — AF far below thresholds
└── BP6/BP3: not met — not benign

ACMG combining:
├── PVS1 (Very Strong) + PS3 (Strong) + PS4 (Strong) + PP5 (Strong)
├── Rule: 1 Very Strong + ≥1 Strong = Pathogenic ✓
└── Classification: PATHOGENIC (High confidence)
```

#### Summary

```
┌─────────────────────────────────────────────────────────┐
│                    PATHOGENIC                            │
│                                                          │
│  PVS1 (Very Strong) — frameshift, exon 19/23            │
│  PS3  (Strong)      — 9 functional studies, 266 pubs    │
│  PS4  (Strong)      — 218 breast neoplasm associations  │
│  PP5  (Strong)      — ClinVar expert panel, 3 stars     │
│  PM2  (Supporting)  — gnomAD non-cancer AF < 0.01%      │
│                                                          │
│  Combining: 1 Very Strong + 3 Strong = PATHOGENIC        │
│  Confidence: HIGH                                        │
└─────────────────────────────────────────────────────────┘
```

---

## Assumptions & Design Decisions

### Input Processing

1. **VEP transcript trial**: When user enters "GENE c.XXX", we don't know which NM_ transcript the notation refers to. We try ClinVar pathogenic counts (top 5 NM_), then Ensembl canonical, testing each against VEP until one accepts the notation. This handles genes where the cDNA position differs between isoforms (e.g., MUTYH).

2. **VCF left-alignment**: VEP returns right-aligned alleles (e.g., G>GG). We left-align against the reference genome (bcftools-style) to match gnomAD's normalization (T>TG). For coordinate input, we trust the user's values as-is (assumed VCF-normalized from a VCF file).

3. **Transcript ranking**: Scored by MANE Select (4pts) + Most Reported Pathogenic (3pts) + Canonical (1pt) + has NM_ (1pt). MANE Select label is removed on GRCh37 (MANE is GRCh38-only). Version-agnostic NM_ matching (NM_007194.4 matches NM_007194.3) for "Most Reported" flag.

4. **HGVS preference**: `hgvs_on_transcript` uses NM_-based notation (NM_007294.4:c.5266dupC) over ENST-based (ENST00000357654.9:c.5266dup) for better ClinVar search compatibility.

### ClinVar Matching

5. **HGVS validation**: Every ClinVar result is validated against the query by checking cDNA position number AND variant type (dup/del/>/delins). Mismatches are rejected and the next search query is tried.

6. **Dup notation expansion**: ClinVar may store `c.5266dupC` while VEP normalizes to `c.5266dup`. The search generator expands bare `dup` to `dupA`, `dupC`, `dupG`, `dupT` to find the right ClinVar entry.

7. **Multi-query fallback**: Tries 4-8 queries in ranked order: HGVS-tagged → field-tagged (gene + variant name) → expanded dup → free text. Picks the best validated match by star rating.

### Frequency Evaluation

8. **Global AF only**: BA1, BS1, PM2 all evaluate against the **global allele frequency** of the controls cohort. Per-population max AF is NOT used, to avoid false positives from founder effects (e.g., Finnish CHEK2 1100delC at 1.0% would falsely trigger BS1 if population-level AF was used).

9. **Controls cohort priority**: GRCh38 uses v3.1.2 controls/biobanks; GRCh37 uses v2.1.1 controls. Falls back to non-cancer → overall if controls unavailable. The specific cohort and AF value are explicitly displayed.

10. **gnomAD rsID-based lookup**: Uses ClinVar's rsID for gnomAD queries to avoid VCF normalization mismatches. Falls back to VEP vcf_string → coordinate-based queries.

### In Silico Predictors

11. **Score extraction**: Many dbNSFP fields return lists (one per transcript). We take the **max** score for pathogenicity predictors.

12. **Consensus threshold**: PP3/BP4 require ≥3 tools agreeing (majority). REVEL is weighted most heavily per ClinGen SVI recommendation.

13. **Indel limitation**: dbNSFP scores are only available for SNVs. Insertions, deletions, and frameshifts have no in silico predictions — PP3/BP4 are marked as "not evaluable."

### PVS1 Caveats

14. **Last exon rule**: Truncating variants in the last exon get PVS1 downgraded to Moderate (may produce stable truncated protein that escapes NMD).

15. **Penultimate exon**: PVS1 downgraded to Strong (last 50bp may escape NMD).

16. **Single-exon genes**: PVS1 downgraded to Moderate (NMD not applicable).

17. **Non-null variants**: Missense, in-frame, synonymous variants do NOT get PVS1. Only frameshift, stop_gained, splice_donor/acceptor, start_lost.

### Domain Assessment

18. **PM1 missense-only**: PM1 only applies to missense variants. Frameshift/nonsense in a functional domain does NOT trigger PM1 (those are covered by PVS1).

19. **Domain + constraint**: Full PM1 Moderate requires both UniProt domain overlap AND gnomAD missense Z > 2.0. Domain overlap alone → PM1 Supporting.

### Literature Evidence

20. **Publication type heuristic**: "Case Reports", "Research Support" etc. are classified from PubMed's `PT` field. This is approximate — not all functional studies are labeled as "Research Support."

21. **Top 15 publications**: Only the first 15 PMIDs are enriched with PubMed metadata. Total count comes from LitVar.

---

## Project Structure

```
clinvar-acmg-agent/
├── app.py                          Streamlit UI
├── graph/
│   ├── graph.py                    LangGraph StateGraph definition
│   ├── state.py                    VariantState TypedDict
│   └── supervisor.py               Router (pure Python, no LLM)
├── agents/
│   ├── input_parser.py             Node 1: VEP + transcript resolution
│   ├── clinvar_agent.py            Node 2: ClinVar query + HGVS validation
│   ├── gnomad_agent.py             Node 3: gnomAD + dbNSFP + constraint + domains
│   ├── pubmed_agent.py             Node 4: LitVar + PubMed enrichment
│   ├── alphafold_agent.py          Node 5: stub
│   ├── tcga_agent.py               Node 6: stub
│   └── acmg_classifier.py         Node 7: Claude LLM classification
├── tools/
│   ├── ensembl.py                  Ensembl VEP, xrefs, gene lookup
│   ├── entrez.py                   NCBI ClinVar, Gene
│   ├── gnomad_graphql.py           gnomAD v4/v3/v2 GraphQL + multi-dataset
│   ├── myvariant.py                MyVariant.info (dbNSFP, CADD, conservation)
│   ├── litvar.py                   NCBI LitVar + PubMed MEDLINE enrichment
│   ├── gene_constraint.py          gnomAD constraint + UniProt domains + PM1/PM4/BP3
│   ├── vcf_normalize.py            VCF left-alignment against reference
│   ├── variant_utils.py            HGVS parsing
│   └── case_control.py             Fisher's exact + weighted GLM
├── cache/
│   └── cache_manager.py            SQLite cache (transcripts, variants)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Installation & Setup

### Prerequisites

- Python 3.10+
- Anthropic API key ([console.anthropic.com](https://console.anthropic.com/))
- NCBI API key (optional, recommended) ([ncbi.nlm.nih.gov/account/settings](https://www.ncbi.nlm.nih.gov/account/settings/))

### Install

```bash
git clone https://github.com/vigneshravi/clinvar-acmg-agent.git
cd clinvar-acmg-agent
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
streamlit run app.py
```

---

## Usage

### Input Methods

| Method | Example | Description |
|--------|---------|-------------|
| Gene + HGVS | `BRCA1 c.5266dupC` | Gene symbol + cDNA notation |
| Transcript + HGVS | `NM_007294.4:c.5266dupC` | Explicit RefSeq transcript |
| Coordinates | chr17 / 43057062 / T / TG | VCF-style (left-aligned) |

### Workflow

1. Enter variant → select genome build (GRCh38 or GRCh37)
2. Click **Look Up Transcripts** → see ranked transcripts with annotations
3. Select transcript (or use pre-selected) → review coordinates, consequence, exon
4. Click **Classify Variant** → pipeline runs all 8 nodes
5. Review: ClinVar record → gnomAD frequencies → in silico scores → gene constraint/domains → literature evidence → ACMG criteria → final classification

### Example Variants

| Variant | Build | Expected | Key Evidence |
|---------|-------|----------|-------------|
| BRCA1 c.5266dupC | GRCh38 | Pathogenic | PVS1 + PS3 + PS4 + PP5 |
| TP53 c.817C>T | GRCh38 | Pathogenic | PS1 + PS3 + PS4 + PM1 + PM2 + PP3 + PP5 |
| BRCA2 c.1114A>C | GRCh38 | Benign | BA1 (AF=28%) + BP4 + BP6 |
| CHEK2 c.1100delC | GRCh37 | Pathogenic | PVS1 + PP5 + PM2 |
| APC I1307K | GRCh37 | VUS / Likely Benign | BS1 (controls AF) |

---

## Limitations

1. **14 of 28 criteria**: PS2 (de novo), PM3 (trans with pathogenic), PM6 (de novo unconfirmed), PP1 (co-segregation), PP2 (missense in gene with low benign rate), PP4 (phenotype specific), BS2 (observed in healthy adult), BS3 (functional benign), BS4 (non-segregation), BP1 (missense in truncating gene), BP2 (observed in trans/cis), BP5 (alternate mechanism), BP7 (synonymous) are NOT evaluated.

2. **No structural data**: AlphaFold/PDB integration is stubbed out. PM1 domain assessment relies on UniProt annotations only.

3. **No somatic data**: TCGA integration is stubbed out.

4. **Literature classification is heuristic**: LitVar + PubMed pub_type classification of "functional study" vs "case report" is approximate.

5. **LLM variability**: Final classification relies on Claude which may vary between runs (mitigated by temperature=0).

6. **Ensembl API latency**: NM_↔ENST xrefs mapping requires per-transcript API calls (~0.5s each), making transcript resolution slow for genes with many isoforms.

7. **Not validated for clinical use**: This tool has not undergone analytical or clinical validation.
