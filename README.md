# ClinVar ACMG Variant Classifier

An AI-powered Streamlit web application that queries genomic variants against the NCBI ClinVar database and classifies them according to the ACMG/AMP 2015 variant interpretation guidelines. The application combines a deterministic, rule-based ACMG criteria engine with a LangChain ReAct agent backed by Anthropic's Claude (claude-sonnet-4-5) to provide both programmatic classification and natural-language clinical reasoning. The tool is designed for educational and research use in clinical genomics, enabling rapid variant lookup, automated evidence assessment, and transparent classification logic.

> **DISCLAIMER:** This is a research prototype developed as a course assignment. It has **not** been validated for clinical use. Variant classifications produced by this tool should **never** be used for clinical decision-making. All results must be reviewed by a board-certified clinical molecular geneticist and confirmed through validated, CLIA-certified laboratory processes before any clinical action is taken.

---

## Table of Contents

- [Background](#background)
- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Usage Guide](#usage-guide)
- [Example Variants](#example-variants)
- [Technical Deep Dive](#technical-deep-dive)
  - [ClinVar Query Module](#clinvar-query-module-agentclinvarpy)
  - [ACMG Criteria Engine](#acmg-criteria-engine-agentacmgpy)
  - [LangChain Agent](#langchain-agent-agentagentpy)
  - [Streamlit UI](#streamlit-ui-apppy)
- [ACMG/AMP 2015 Criteria Reference](#acmgamp-2015-criteria-reference)
- [ClinVar Star Rating System](#clinvar-star-rating-system)
- [Limitations and Scope](#limitations-and-scope)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Background

### What is ClinVar?

[ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) is a freely accessible, public archive hosted by the National Center for Biotechnology Information (NCBI) that aggregates information about genomic variation and its relationship to human health. Laboratories, clinical testing providers, research groups, and expert panels submit interpretations of variant pathogenicity to ClinVar, making it the central repository for variant-level clinical evidence in human genetics.

Each ClinVar record includes:
- The variant's genomic location and HGVS nomenclature
- Clinical significance classifications submitted by one or more laboratories
- A review status (star rating) reflecting the level of evidence and consensus
- Associated diseases or conditions
- Supporting evidence and submitter details

### What are the ACMG/AMP 2015 Guidelines?

The American College of Medical Genetics and Genomics (ACMG) and the Association for Molecular Pathology (AMP) published joint guidelines in 2015 (Richards et al., *Genetics in Medicine*, 2015) that established a standardized framework for classifying germline sequence variants. The framework defines 28 evidence criteria organized into categories:

- **Pathogenic criteria** (PVS1, PS1-PS4, PM1-PM6, PP1-PP5): Evidence supporting a variant's role in disease
- **Benign criteria** (BA1, BS1-BS4, BP1-BP7): Evidence supporting a variant as non-disease-causing

Each criterion has an assigned strength level (Stand-alone, Very Strong, Strong, Moderate, or Supporting). Variants are classified into one of five tiers by combining triggered criteria according to specific rules:

| Classification | Meaning |
|---------------|---------|
| **Pathogenic** | The variant is disease-causing with high confidence |
| **Likely Pathogenic** | There is strong evidence the variant is disease-causing (>90% certainty) |
| **VUS** (Variant of Uncertain Significance) | Insufficient evidence to classify as pathogenic or benign |
| **Likely Benign** | There is strong evidence the variant is not disease-causing |
| **Benign** | The variant is not disease-causing with high confidence |

### What does this tool do?

This application automates a subset of the ACMG/AMP classification workflow using ClinVar as the primary evidence source. Given a variant identifier (e.g., "BRCA1 c.5266dupC"), it:

1. Queries the NCBI ClinVar database via the Entrez E-utilities API
2. Parses the returned XML to extract structured clinical data
3. Evaluates six ACMG criteria that can be assessed from ClinVar data alone
4. Applies the ACMG combining rules to produce a five-tier classification
5. Uses a Claude-powered LangChain agent to provide additional clinical reasoning
6. Displays all results in an interactive Streamlit web interface

---

## Architecture Overview

The application follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (app.py)                 │
│         User input → Results display → Disclaimer       │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│              LangChain Agent (agent/agent.py)            │
│    Orchestrates ClinVar lookup + ACMG evaluation         │
│    Claude claude-sonnet-4-5 provides enhanced reasoning  │
│    LangGraph ReAct agent with tool calling               │
└───────┬─────────────────────────────┬───────────────────┘
        │                             │
┌───────▼───────────┐   ┌────────────▼────────────────────┐
│  ClinVar Module   │   │     ACMG Criteria Engine        │
│ (agent/clinvar.py)│   │     (agent/acmg.py)             │
│                   │   │                                  │
│ • Entrez esearch  │   │ • PS1, PP5, BP6, BA1, BS1, PM5  │
│ • Entrez esummary │   │ • Star-based strength scaling    │
│ • XML parsing     │   │ • ACMG combining rules           │
│ • Result validate │   │ • Five-tier classification        │
└───────┬───────────┘   └──────────────────────────────────┘
        │
┌───────▼───────────┐
│   NCBI Entrez API │
│   (ClinVar DB)    │
└───────────────────┘
```

**Data flow:**

1. The user enters a variant string in the Streamlit interface
2. `classify_variant_with_agent()` in `agent/agent.py` orchestrates the pipeline
3. `query_clinvar()` searches ClinVar using field-tagged Entrez queries and parses the eSummary XML
4. `evaluate_acmg_criteria()` assesses six ACMG criteria from the ClinVar record
5. `classify_variant()` applies ACMG combining rules to produce a classification
6. The LangGraph ReAct agent (Claude claude-sonnet-4-5) independently queries ClinVar via tool calling and produces natural-language clinical reasoning
7. Results are combined and displayed in three expandable sections in the UI

---

## Project Structure

```
clinvar-acmg-agent/
│
├── app.py                  # Streamlit web UI entry point
│                           #   - Page configuration and layout
│                           #   - Variant input form
│                           #   - Three expandable result sections
│                           #   - Color-coded classification display
│                           #   - Star rating visualization
│                           #   - Clinical disclaimer
│
├── agent/
│   ├── __init__.py         # Package initializer
│   │
│   ├── clinvar.py          # ClinVar Entrez API integration
│   │                       #   - _configure_entrez(): API key setup
│   │                       #   - _build_search_queries(): Smart query builder
│   │                       #   - _parse_esummary_xml(): XML → structured dict
│   │                       #   - _validate_result(): Result relevance check
│   │                       #   - query_clinvar(): Main entry point
│   │
│   ├── acmg.py             # ACMG criteria evaluation engine
│   │                       #   - _get_star_rating(): Review status → stars
│   │                       #   - _normalize_significance(): Handle compound
│   │                       #     classifications (e.g., "Pathogenic/Likely
│   │                       #     pathogenic")
│   │                       #   - evaluate_acmg_criteria(): Assess 6 criteria
│   │                       #   - classify_variant(): Apply combining rules
│   │
│   └── agent.py            # LangChain/LangGraph agent orchestration
│                           #   - clinvar_lookup tool: LangChain tool wrapper
│                           #   - build_agent(): LangGraph ReAct agent factory
│                           #   - classify_variant_with_agent(): Full pipeline
│
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Prerequisites

### Software Requirements

- **Python 3.10 or higher** — Required for type hint syntax (`dict[str, Any]`, `str | None`)
- **pip** — Python package manager
- **Git** — For cloning the repository

### API Keys Required

This application requires two API keys (one mandatory, one recommended):

#### 1. Anthropic API Key (Required)

The Anthropic API key is used to access Claude claude-sonnet-4-5, which powers the LangChain agent for clinical reasoning.

- **How to obtain:** Create an account at [console.anthropic.com](https://console.anthropic.com/), navigate to API Keys, and generate a new key
- **Format:** Starts with `sk-ant-api03-...`
- **Cost:** Claude API usage is billed per token. A typical variant classification uses approximately 2,000–4,000 tokens (~$0.01–0.03 per query at current Sonnet pricing)
- **Environment variable:** `ANTHROPIC_API_KEY`

#### 2. NCBI API Key (Recommended)

The NCBI API key increases your Entrez API rate limit from 3 requests/second to 10 requests/second. The application will work without it, but may be rate-limited during heavy use.

- **How to obtain:** Create an NCBI account at [ncbi.nlm.nih.gov](https://www.ncbi.nlm.nih.gov/), go to Account Settings, and find or generate your API key under "API Key Management"
- **Format:** 36-character hexadecimal string
- **Cost:** Free
- **Environment variable:** `NCBI_API_KEY`

#### 3. NCBI Email (Required by NCBI Policy)

NCBI requires an email address for all Entrez API usage so they can contact users if there are problems with their queries.

- **Environment variable:** `NCBI_EMAIL`

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd clinvar-acmg-agent
```

### Step 2: Create a Virtual Environment

It is strongly recommended to use a virtual environment to avoid dependency conflicts:

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# OR
venv\Scripts\activate           # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the following packages (with their key roles):

| Package | Version | Role |
|---------|---------|------|
| `langchain` | >=0.3.0 | Core LangChain framework for agent orchestration |
| `langchain-anthropic` | >=0.3.0 | ChatAnthropic LLM integration for Claude models |
| `langchain-core` | >=0.3.0 | Core abstractions (tools, messages, prompts) |
| `langgraph` | (auto-installed) | LangGraph framework for ReAct agent execution |
| `streamlit` | >=1.38.0 | Web application framework for the UI |
| `biopython` | >=1.84 | Biopython Entrez module for NCBI API access |
| `python-dotenv` | >=1.0.0 | Loads `.env` files into environment variables |
| `anthropic` | >=0.40.0 | Anthropic Python SDK (dependency of langchain-anthropic) |

### Step 4: Configure Environment Variables

```bash
cp .env.example .env
```

Edit the `.env` file with your actual credentials:

```env
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
NCBI_API_KEY=your_36_character_ncbi_api_key
NCBI_EMAIL=your.email@institution.edu
```

**Security note:** The `.env` file contains sensitive API keys. Never commit it to version control. The `.gitignore` file should exclude it.

---

## Running the Application

```bash
streamlit run app.py
```

The application will start and display:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://<your-ip>:8501
```

Open `http://localhost:8501` in your web browser.

To stop the application, press `Ctrl+C` in the terminal.

### Running Options

```bash
# Run on a specific port
streamlit run app.py --server.port 8080

# Run in headless mode (no browser auto-open)
streamlit run app.py --server.headless true

# Run with a specific theme
streamlit run app.py --theme.base dark
```

---

## Usage Guide

### Step 1: Enter a Variant

Type a variant identifier into the text input field. The application accepts several input formats:

| Format | Example | Description |
|--------|---------|-------------|
| Gene + cDNA | `BRCA1 c.5266dupC` | Gene symbol followed by HGVS cDNA notation |
| Gene + cDNA (substitution) | `TP53 c.817C>T` | Gene symbol with single nucleotide change |
| Transcript + cDNA | `NM_007294.4:c.5266dupC` | RefSeq transcript with HGVS notation |
| Gene + legacy name | `BRCA1 5382insC` | Gene symbol with legacy mutation name |

### Step 2: Click "Classify Variant"

A spinner will appear while the application:
1. Queries ClinVar (1–3 seconds depending on network and API rate limits)
2. Evaluates ACMG criteria (instant, programmatic)
3. Runs the Claude agent for enhanced reasoning (3–10 seconds)

### Step 3: Review Results

Results are displayed in three expandable sections:

#### Section 1: ClinVar Record

Displays the raw data retrieved from ClinVar in a two-column layout:

- **Gene** — The HGNC gene symbol (e.g., BRCA1)
- **HGVS** — The standardized HGVS nomenclature from the ClinVar record title
- **Clinical Significance** — ClinVar's aggregate classification (Pathogenic, Likely Pathogenic, VUS, etc.)
- **Condition** — The associated disease or phenotype (e.g., "Breast-ovarian cancer, familial, susceptibility to, 1")
- **Review Status** — Displayed as a star rating (see [ClinVar Star Rating System](#clinvar-star-rating-system)) with the full status string
- **Submitter Count** — Number of laboratories/groups that submitted classifications
- **Last Evaluated** — Date of the most recent classification review
- **Variant ID** — ClinVar's internal variation identifier (VCV number)
- **Conflicting Interpretations Warning** — An orange warning badge appears if submitters disagree on the classification

#### Section 2: ACMG Criteria Triggered

Lists each ACMG criterion that was triggered, with:
- A colored indicator: red circle for pathogenic-direction criteria, green circle for benign-direction criteria
- The criterion code (e.g., PS1, PP5)
- The assigned evidence strength (Strong, Moderate, Supporting)
- A plain-English justification explaining why the criterion was triggered

#### Section 3: Final Classification

Displays the overall result:
- A large color-coded classification box:
  - **Red** — Pathogenic
  - **Orange** — Likely Pathogenic
  - **Yellow** — VUS (Variant of Uncertain Significance)
  - **Light Green** — Likely Benign
  - **Green** — Benign
- **Confidence level** — High, Moderate, or Low
- **Reasoning** — Explanation of the ACMG combining rules applied, followed by the Claude agent's additional clinical analysis

#### Disclaimer

A clinical disclaimer is displayed at the bottom in small gray text, reminding users that this tool is not validated for clinical use.

---

## Example Variants

### Pathogenic Variants

| Variant | Gene | Condition | Expected ClinVar Classification | Expected ACMG Result |
|---------|------|-----------|---------------------------------|---------------------|
| `BRCA1 c.5266dupC` | BRCA1 | Hereditary breast-ovarian cancer | Pathogenic (3-star, expert panel) | **Pathogenic** (High confidence) |
| `PALB2 c.3113G>A` | PALB2 | PALB2-related cancer predisposition | Pathogenic (3-star, expert panel) | **Pathogenic** (High confidence) |
| `TP53 c.817C>T` | TP53 | Li-Fraumeni syndrome | Pathogenic/Likely pathogenic (2-star) | **Likely Pathogenic** (Moderate confidence) |

### About These Variants

**BRCA1 c.5266dupC (p.Gln1756fs)** — Also known as 5382insC, this is one of the three BRCA1/BRCA2 Ashkenazi Jewish founder mutations. It is a frameshift duplication in exon 20 of BRCA1 that introduces a premature stop codon, leading to loss of function of the BRCA1 tumor suppressor protein. It confers a significantly elevated lifetime risk of breast cancer (60–85%) and ovarian cancer (20–40%). This variant has been reviewed by the ClinGen BRCA1/2 Expert Panel and classified as Pathogenic with the highest review status (3 stars, reviewed by expert panel).

**TP53 c.817C>T (p.Arg273Cys)** — A missense variant in the DNA-binding domain of the TP53 tumor suppressor gene. Arginine at position 273 is one of the most frequently mutated "hotspot" residues in TP53 across all human cancers. Germline mutations at this position cause Li-Fraumeni syndrome, a hereditary cancer predisposition syndrome with elevated risk for sarcomas, breast cancer, brain tumors, adrenocortical carcinoma, and leukemia. This variant is classified as Pathogenic/Likely pathogenic in ClinVar with 2-star review status (criteria provided, multiple submitters, no conflicts).

**PALB2 c.3113G>A (p.Trp1038Ter)** — A nonsense variant in the PALB2 gene that introduces a premature stop codon, leading to a truncated, non-functional protein. PALB2 (Partner and Localizer of BRCA2) is essential for DNA double-strand break repair via homologous recombination. Loss-of-function variants in PALB2 are associated with a moderately increased lifetime risk of breast cancer (33–58%) and are now included in clinical genetic testing panels for hereditary breast cancer. This variant has been reviewed by the ClinGen PALB2 Expert Panel and classified as Pathogenic (3 stars).

---

## Technical Deep Dive

### ClinVar Query Module (`agent/clinvar.py`)

#### Search Strategy

The ClinVar search implementation uses a ranked query strategy to maximize precision. ClinVar's Entrez search supports field-tagged queries that constrain searches to specific database fields, dramatically improving result accuracy over free-text search.

For a user input like `"BRCA1 c.5266dupC"`, the module generates these queries in order:

1. `BRCA1[gene] AND c.5266dupC[Variant name]` — Most precise, restricts to gene field and variant name field
2. `BRCA1[gene] AND 5266dupC[Variant name]` — Strips `c.` prefix in case ClinVar stores it differently
3. `BRCA1[gene] AND c.5266dupC` — Gene field + free text
4. `BRCA1 c.5266dupC` — Unstructured fallback

The module iterates through these queries and uses the first one that returns results. This approach solves a critical problem: ClinVar's free-text search often returns unrelated variants first (e.g., searching "BRCA1 c.5266dupC" without field tags returns variant ID 3336480, which is an unrelated variant, instead of the correct ID 17677).

#### XML Parsing

The module uses the Entrez `esummary` endpoint (not `efetch`) because:
- `esummary` returns a compact XML document with all needed fields
- `efetch` with `rettype="vcv"` returns empty results for many ClinVar record UIDs
- `esummary` is faster and uses less bandwidth

The XML is parsed using Python's built-in `xml.etree.ElementTree` module. The parser handles two ClinVar XML schemas:
- **Current format**: Uses `<germline_classification>` for classification data
- **Legacy format**: Uses `<clinical_significance>` (fallback)

Key fields are extracted from nested elements:
- **Gene symbol**: Parsed from the title using regex pattern `\(([A-Za-z0-9_-]+)\)` (e.g., extracting "BRCA1" from "NM_007294.4(BRCA1):c.5266dup")
- **Submitter count**: Counted from `<supporting_submissions><scv><string>` elements
- **Conditions**: Extracted from `<germline_classification><trait_set><trait><trait_name>` elements

#### Result Validation

After fetching results, `_validate_result()` performs a sanity check to ensure the returned variant plausibly matches the user's query. It verifies that:
- The gene name from the input appears in the result's HGVS title
- Key position numbers from the variant description appear in the result

If validation fails for the first result, the module tries the next results (up to 3) before falling back.

### ACMG Criteria Engine (`agent/acmg.py`)

#### Criteria Evaluated

The engine evaluates six ACMG/AMP criteria that can be assessed from ClinVar data alone:

| Criterion | Category | Default Strength | What it Assesses |
|-----------|----------|-----------------|------------------|
| **PS1** | Pathogenic Strong | Strong | Same amino acid change as an established pathogenic variant. Triggered when ClinVar classifies the variant as pathogenic with >=2-star review and >=2 submitters. |
| **PP5** | Pathogenic Supporting | Supporting* | Reputable source reports variant as pathogenic. *Upgraded to **Strong** for 3+ star (expert panel/practice guideline) review, **Moderate** for 2-star review, per ClinGen recommendations. |
| **BP6** | Benign Supporting | Supporting* | Reputable source reports variant as benign. *Same upgrade logic as PP5: **Strong** for 3+ star, **Moderate** for 2-star. |
| **BA1** | Benign Stand-alone | Stand-alone | Allele frequency >5% in population databases. Inferred from ClinVar annotations mentioning "common" or "polymorphism" in benign-classified variants. |
| **BS1** | Benign Strong | Strong | Allele frequency greater than expected for disorder. Inferred when multiple submitters classify a variant as benign. |
| **PM5** | Pathogenic Moderate | Moderate | Novel missense at same position as known pathogenic missense. Detected by regex-matching HGVS protein notation for missense patterns (e.g., `p.Arg273Cys`). Only triggered when PS1 is not already applied. |

#### Evidence Strength Scaling

A key design decision is the dynamic scaling of PP5 and BP6 evidence strength based on ClinVar review status. The standard ACMG guidelines assign PP5/BP6 as "Supporting" level evidence, but ClinGen's SVI (Sequence Variant Interpretation) working group has recommended that evidence from expert panels and practice guidelines can support stronger classifications:

| ClinVar Review Status | Star Rating | PP5/BP6 Strength |
|-----------------------|-------------|------------------|
| Practice guideline | 4 stars | Strong |
| Reviewed by expert panel | 3 stars | Strong |
| Criteria provided, multiple submitters, no conflicts | 2 stars | Moderate |
| Criteria provided, single submitter | 1 star | Supporting |
| No assertion criteria provided | 0 stars | Not triggered |

#### Compound Classification Handling

ClinVar sometimes reports compound classifications like "Pathogenic/Likely pathogenic". The `_normalize_significance()` function handles this by splitting on "/" and selecting the strongest applicable term (e.g., "Pathogenic/Likely pathogenic" → "pathogenic"). This ensures that well-established pathogenic variants are not inadvertently downgraded.

#### ACMG Combining Rules

The `classify_variant()` function implements the ACMG/AMP 2015 combining rules:

**Pathogenic** (any one of these combinations):
- 1 Very Strong AND 1+ Strong
- 2+ Strong
- 1 Strong AND 3+ Moderate/Supporting
- 1 Very Strong AND 2+ Moderate

**Likely Pathogenic** (any one of these combinations):
- 1 Strong AND 1+ Moderate
- 1 Strong AND 2+ Supporting
- 3+ Moderate
- 2 Moderate AND 2+ Supporting
- 1 Very Strong AND 1 Moderate

**Benign** (any one of these combinations):
- 1 Stand-alone (BA1)
- 2+ Strong benign

**Likely Benign:**
- 1 Strong benign AND 1+ Supporting benign

**VUS:**
- Does not meet any of the above combinations

### LangChain Agent (`agent/agent.py`)

#### Agent Architecture

The agent uses the **LangGraph ReAct (Reasoning + Acting)** pattern:

1. **LLM**: `ChatAnthropic` with `model="claude-sonnet-4-5"`, `temperature=0` (deterministic), `max_tokens=4096`
2. **Tools**: A single `clinvar_lookup` tool that wraps `query_clinvar()` and returns JSON
3. **System Prompt**: Instructs the agent to act as a clinical molecular geneticist applying ACMG/AMP 2015 criteria
4. **Execution**: `create_react_agent()` from `langgraph.prebuilt` creates the agent, which autonomously decides when to call tools

#### Dual Pipeline Design

The `classify_variant_with_agent()` function runs two parallel analysis paths:

1. **Programmatic path** (deterministic):
   - Calls `query_clinvar()` directly
   - Runs `evaluate_acmg_criteria()` for rule-based assessment
   - Applies `classify_variant()` for ACMG combining rules
   - Produces a reproducible, auditable classification

2. **Agent path** (LLM-powered):
   - The LangGraph agent independently queries ClinVar via the `clinvar_lookup` tool
   - Claude analyzes the data with clinical genomics expertise
   - Produces natural-language reasoning about the variant

The final output combines both: the programmatic classification is authoritative, and the agent's reasoning is appended as supplementary analysis. This ensures the classification is deterministic and reproducible while still providing the clinical context that makes the result interpretable.

#### System Prompt

The agent's system prompt establishes its role and behavior:

```
You are a clinical molecular geneticist specializing in hereditary cancer
risk assessment. Your role is to classify germline variants using the
ACMG/AMP 2015 guidelines.

When a user provides a variant, you should:
1. Query ClinVar using the clinvar_lookup tool to retrieve the variant record.
2. Analyze the ClinVar data and apply ACMG criteria evaluation.
3. Provide a structured classification with reasoning.

Always be precise and evidence-based. Cite the specific ACMG criteria that
apply and explain why. If data is limited, acknowledge the uncertainty.

IMPORTANT: You must always use the clinvar_lookup tool to look up variants.
Do not rely on your training data for variant classifications, as ClinVar
is updated regularly and your knowledge may be outdated.
```

### Streamlit UI (`app.py`)

#### Layout

The UI uses Streamlit's `wide` layout mode and consists of:

1. **Header**: Title with DNA emoji, subtitle with clinical disclaimer
2. **Input area**: Text input field with placeholder text, "Classify Variant" primary button
3. **Results area**: Three `st.expander` sections (ClinVar Record, ACMG Criteria, Classification)
4. **Footer**: Clinical disclaimer in small gray text

#### Star Rating Display

The review status is converted to a visual star rating using filled (⭐) and empty (☆) star characters:

| Review Status | Display |
|---------------|---------|
| Practice guideline | ⭐⭐⭐⭐ |
| Reviewed by expert panel | ⭐⭐⭐☆ |
| Multiple submitters, no conflicts | ⭐⭐☆☆ |
| Single submitter / Conflicting | ⭐☆☆☆ |
| No assertion criteria | ☆☆☆☆ |

#### Classification Color Coding

The final classification is displayed in a large, centered HTML div with color-coded backgrounds:

| Classification | Background Color | Text Color |
|---------------|-----------------|------------|
| Pathogenic | Red (#ff4b4b) | White |
| Likely Pathogenic | Orange (#ff8c00) | White |
| VUS | Yellow (#ffd700) | Dark gray |
| Likely Benign | Light green (#90ee90) | Dark gray |
| Benign | Green (#4caf50) | White |

---

## ClinVar Star Rating System

ClinVar uses a four-star review status system to indicate the level of evidence supporting a variant's classification:

| Stars | Review Status | Meaning |
|-------|--------------|---------|
| ⭐⭐⭐⭐ | Practice guideline | Classification is part of an established practice guideline (e.g., CPIC for pharmacogenomics) |
| ⭐⭐⭐ | Reviewed by expert panel | Classification was reviewed by a ClinGen Expert Panel (e.g., BRCA1/2 VCEP, RASopathy VCEP) |
| ⭐⭐ | Criteria provided, multiple submitters, no conflicts | Multiple laboratories submitted concordant classifications with specified ACMG criteria |
| ⭐ | Criteria provided, single submitter / Conflicting | Either a single lab submitted with criteria, or multiple labs disagree |
| ☆ | No assertion criteria provided | Classification submitted without ACMG criteria or supporting evidence |

Higher star ratings indicate more reliable classifications and are weighted more heavily in this tool's ACMG criteria assessment.

---

## Limitations and Scope

### Criteria Not Evaluated

This tool evaluates only 6 of the 28 ACMG criteria. The remaining 22 criteria require data sources beyond ClinVar:

| Data Source Needed | Criteria | Examples |
|-------------------|----------|----------|
| Population databases (gnomAD, ExAC) | BA1, BS1, PM2 | Allele frequency analysis |
| Functional studies | PS3, BS3 | In vitro / in vivo assays |
| Segregation data | PP1, BS4 | Co-segregation with disease in families |
| Computational predictions | PP3, BP4 | REVEL, CADD, SpliceAI scores |
| De novo status | PS2, PM6 | Confirmed de novo in a proband |
| Protein domain / hotspot | PM1 | Functional domain without benign variation |
| Allelic data | PM3, BP2 | Trans / cis with a known pathogenic variant |
| Gene-level constraints | PVS1 | Null variant in a gene where LOF is a mechanism |
| Case-level data | PS4 | Prevalence in affected vs. controls |

### Known Limitations

1. **ClinVar-only evidence**: The tool cannot assess allele frequency, functional, computational, or segregation evidence. This limits its ability to classify novel or rare variants that lack ClinVar submissions.

2. **Search precision**: While the field-tagged search strategy is significantly more accurate than free-text search, some variant nomenclature formats may not return the expected result. If the wrong variant is returned, try using the transcript-level HGVS notation (e.g., `NM_007294.4:c.5266dupC`).

3. **XML schema changes**: ClinVar periodically updates its XML schema. The parser handles both the current (`germline_classification`) and legacy (`clinical_significance`) formats, but future schema changes may require parser updates.

4. **API rate limits**: Without an NCBI API key, Entrez limits requests to 3/second. With an API key, the limit increases to 10/second.

5. **LLM reasoning variability**: While the programmatic ACMG classification is deterministic, the Claude agent's natural-language reasoning may vary slightly between runs due to the stochastic nature of LLMs (mitigated by `temperature=0`).

6. **Not validated for clinical use**: This tool has not undergone analytical validation, clinical validation, or regulatory review. It should not be used for clinical decision-making.

---

## Troubleshooting

### Common Issues

**"No ClinVar records found"**
- Verify your variant nomenclature is correct
- Try alternative formats: `BRCA1 c.5266dupC` vs `NM_007294.4:c.5266dupC`
- Check that your NCBI API key and email are correctly set in `.env`

**"Agent reasoning unavailable"**
- Verify your `ANTHROPIC_API_KEY` is valid and has available credits
- Check your network connection
- The programmatic ACMG classification will still work even if the agent fails

**Import errors on startup**
- Ensure you're using Python 3.10+ (`python --version`)
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Ensure your virtual environment is activated

**Rate limit errors from NCBI**
- Add an NCBI API key to your `.env` file to increase rate limits
- Wait a few seconds between queries

**Streamlit port already in use**
- Use an alternative port: `streamlit run app.py --server.port 8080`
- Or kill the existing process: `lsof -ti:8501 | xargs kill`

---

## References

1. **Richards, S., et al.** (2015). Standards and guidelines for the interpretation of sequence variants: a joint consensus recommendation of the American College of Medical Genetics and Genomics and the Association for Molecular Pathology. *Genetics in Medicine*, 17(5), 405–424. https://doi.org/10.1038/gim.2015.30

2. **Landrum, M.J., et al.** (2018). ClinVar: improving access to variant interpretations and supporting evidence. *Nucleic Acids Research*, 46(D1), D1062–D1067. https://doi.org/10.1093/nar/gkx1153

3. **Tavtigian, S.V., et al.** (2018). Modeling the ACMG/AMP variant classification guidelines as a Bayesian classification framework. *Genetics in Medicine*, 20(9), 1054–1060. https://doi.org/10.1038/gim.2017.210

4. **ClinGen Sequence Variant Interpretation (SVI) Working Group.** Recommendations for applying ACMG/AMP criteria. https://clinicalgenome.org/working-groups/sequence-variant-interpretation/

5. **NCBI Entrez Programming Utilities (E-utilities).** https://www.ncbi.nlm.nih.gov/books/NBK25501/

6. **LangChain Documentation.** https://python.langchain.com/docs/

7. **Anthropic Claude API Documentation.** https://docs.anthropic.com/
