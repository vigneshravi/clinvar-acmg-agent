#!/usr/bin/env python3
"""
ClinVar ACMG Variant Classifier — Single-file standalone script.

Queries a genomic variant against NCBI ClinVar and classifies it using
ACMG/AMP 2015 criteria via a LangChain ReAct agent powered by Claude.

Usage:
    python classify_variant.py "BRCA1 c.5266dupC"
    python classify_variant.py "TP53 c.817C>T"
    python classify_variant.py "PALB2 c.3113G>A"

Requirements:
    pip install langchain langchain-anthropic langchain-core biopython python-dotenv anthropic

Environment variables (set in .env or export):
    ANTHROPIC_API_KEY  - Anthropic API key (required)
    NCBI_API_KEY       - NCBI API key (recommended, increases rate limit)
    NCBI_EMAIL         - Email for NCBI Entrez (required by NCBI policy)
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from typing import Any

from Bio import Entrez
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

load_dotenv()

# ============================================================================
# SECTION 1: ClinVar Entrez API Queries and XML Parsing
# ============================================================================

REVIEW_STATUS_STARS: dict[str, int] = {
    "practice guideline": 4,
    "reviewed by expert panel": 3,
    "criteria provided, multiple submitters, no conflicts": 2,
    "criteria provided, conflicting classifications": 1,
    "criteria provided, conflicting interpretations": 1,
    "criteria provided, single submitter": 1,
    "no assertion criteria provided": 0,
    "no assertion provided": 0,
    "no classification provided": 0,
    "no classification for the single variant": 0,
}


def _configure_entrez() -> None:
    """Configure Entrez with credentials from environment."""
    Entrez.email = os.getenv("NCBI_EMAIL", "user@example.com")
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        Entrez.api_key = api_key


def _build_search_queries(variant: str) -> list[str]:
    """Build a ranked list of ClinVar search queries from a variant string.

    Tries structured field-tagged queries first (most precise), then falls
    back to progressively looser searches.
    """
    queries = []
    variant = variant.strip()

    # Try to parse "GENE variant_desc" pattern (e.g. "BRCA1 c.5266dupC")
    match = re.match(r"^([A-Za-z0-9_-]+)\s+(.+)$", variant)
    if match:
        gene = match.group(1)
        desc = match.group(2).strip()
        queries.append(f"{gene}[gene] AND {desc}[Variant name]")
        if desc.startswith("c."):
            queries.append(f"{gene}[gene] AND {desc[2:]}[Variant name]")
        queries.append(f"{gene}[gene] AND {desc}")

    if ":" in variant:
        queries.append(f"{variant}[Variant name]")
        queries.append(variant)

    if variant not in queries:
        queries.append(variant)

    return queries


def _parse_esummary_xml(xml_text: str) -> dict[str, Any]:
    """Parse ClinVar eSummary XML and extract structured data."""
    root = ET.fromstring(xml_text)

    result: dict[str, Any] = {
        "variant_id": None,
        "gene": None,
        "hgvs": None,
        "clinical_significance": None,
        "review_status": None,
        "submitter_count": 0,
        "conflicting_interpretations": False,
        "condition": None,
        "last_evaluated": None,
        "raw_submissions": [],
    }

    doc_summaries = root.findall(".//DocumentSummary")
    if not doc_summaries:
        return result

    doc = doc_summaries[0]

    # Variation ID
    variation_id = doc.get("uid")
    if variation_id:
        result["variant_id"] = variation_id

    # Title contains HGVS notation
    title = doc.find("title")
    if title is not None and title.text:
        result["hgvs"] = title.text

    # Gene: extract from title pattern "NM_...(GENE):..."
    if result["hgvs"]:
        gene_match = re.search(r"\(([A-Za-z0-9_-]+)\)", result["hgvs"])
        if gene_match:
            result["gene"] = gene_match.group(1)

    # Also try the variation element
    variation = doc.find(".//variation_set/variation")
    if variation is not None:
        if not result["gene"]:
            var_name = variation.find("variation_name")
            if var_name is not None and var_name.text:
                gene_match = re.search(r"\(([A-Za-z0-9_-]+)\)", var_name.text)
                if gene_match:
                    result["gene"] = gene_match.group(1)

    # Germline classification (current ClinVar XML format)
    germline = doc.find("germline_classification")
    if germline is not None:
        desc = germline.find("description")
        if desc is not None and desc.text:
            result["clinical_significance"] = desc.text
        review = germline.find("review_status")
        if review is not None and review.text:
            result["review_status"] = review.text
        last_eval = germline.find("last_evaluated")
        if last_eval is not None and last_eval.text:
            result["last_evaluated"] = last_eval.text

        traits = germline.findall(".//trait_set/trait")
        conditions = []
        for trait in traits:
            trait_name = trait.find("trait_name")
            if trait_name is not None and trait_name.text:
                conditions.append(trait_name.text)
        if conditions:
            result["condition"] = "; ".join(conditions)

    # Fallback: older clinical_significance element
    if not result["clinical_significance"]:
        clin_sig = doc.find("clinical_significance")
        if clin_sig is not None:
            desc = clin_sig.find("description")
            if desc is not None and desc.text:
                result["clinical_significance"] = desc.text
            review = clin_sig.find("review_status")
            if review is not None and review.text:
                result["review_status"] = review.text
            last_eval = clin_sig.find("last_evaluated")
            if last_eval is not None and last_eval.text:
                result["last_evaluated"] = last_eval.text

    # Conditions fallback
    if not result["condition"]:
        traits = doc.findall(".//trait_set/trait")
        conditions = []
        for trait in traits:
            trait_name = trait.find("trait_name")
            if trait_name is not None and trait_name.text:
                conditions.append(trait_name.text)
        if conditions:
            seen = set()
            unique = []
            for c in conditions:
                if c not in seen:
                    seen.add(c)
                    unique.append(c)
            result["condition"] = "; ".join(unique)

    # Submitter count
    supporting = doc.find("supporting_submissions")
    if supporting is not None:
        scv_elem = supporting.find("scv")
        if scv_elem is not None:
            scv_strings = scv_elem.findall("string")
            result["submitter_count"] = len(scv_strings)

    return result


def _validate_result(result: dict[str, Any], variant: str) -> bool:
    """Check if the result plausibly matches the queried variant."""
    hgvs = (result.get("hgvs") or "").lower()
    variant_lower = variant.lower().strip()

    match = re.match(r"^([A-Za-z0-9_-]+)\s+(.+)$", variant_lower)
    if match:
        gene = match.group(1)
        desc = match.group(2).strip()
        if gene not in hgvs:
            return False
        desc_core = desc.lstrip("c.")
        desc_digits = re.findall(r"\d+", desc_core)
        if desc_digits:
            if desc_digits[0] in hgvs:
                return True
    return True


def query_clinvar(variant: str) -> dict[str, Any]:
    """Query ClinVar for a variant and return structured data.

    Uses structured field-tagged queries for precision, with fallbacks
    to progressively looser searches.
    """
    _configure_entrez()

    empty_result = {
        "error": f"No ClinVar records found for '{variant}'",
        "variant_id": None,
        "gene": None,
        "hgvs": None,
        "clinical_significance": None,
        "review_status": None,
        "submitter_count": 0,
        "conflicting_interpretations": False,
        "condition": None,
        "last_evaluated": None,
        "raw_submissions": [],
    }

    try:
        queries = _build_search_queries(variant)
        id_list = []

        for query in queries:
            search_handle = Entrez.esearch(db="clinvar", term=query, retmax=5)
            search_results = Entrez.read(search_handle)
            search_handle.close()
            id_list = search_results.get("IdList", [])
            if id_list:
                break

        if not id_list:
            return empty_result

        result = None
        for clinvar_id in id_list[:3]:
            fetch_handle = Entrez.esummary(db="clinvar", id=clinvar_id)
            xml_text = fetch_handle.read()
            fetch_handle.close()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode("utf-8")
            candidate = _parse_esummary_xml(xml_text)
            if _validate_result(candidate, variant):
                result = candidate
                break

        if result is None:
            clinvar_id = id_list[0]
            fetch_handle = Entrez.esummary(db="clinvar", id=clinvar_id)
            xml_text = fetch_handle.read()
            fetch_handle.close()
            if isinstance(xml_text, bytes):
                xml_text = xml_text.decode("utf-8")
            result = _parse_esummary_xml(xml_text)

        if not result["variant_id"]:
            result["variant_id"] = id_list[0]

        return result

    except Exception as e:
        return {
            "error": f"ClinVar query failed: {str(e)}",
            "variant_id": None,
            "gene": None,
            "hgvs": None,
            "clinical_significance": None,
            "review_status": None,
            "submitter_count": 0,
            "conflicting_interpretations": False,
            "condition": None,
            "last_evaluated": None,
            "raw_submissions": [],
        }


# ============================================================================
# SECTION 2: ACMG Criteria Evaluation and Classification Rules
# ============================================================================


def _get_star_rating(review_status: str | None) -> int:
    """Convert review status string to star rating (0-4)."""
    if not review_status:
        return 0
    status_lower = review_status.strip().lower()
    for key, stars in REVIEW_STATUS_STARS.items():
        if key in status_lower:
            return stars
    return 0


def _normalize_significance(sig: str | None) -> str:
    """Normalize clinical significance to standard terms.

    Handles compound classifications like "Pathogenic/Likely pathogenic"
    by returning the strongest applicable classification.
    """
    if not sig:
        return "uncertain significance"
    sig_lower = sig.strip().lower()
    if "/" in sig_lower:
        parts = [p.strip() for p in sig_lower.split("/")]
        if "pathogenic" in parts:
            return "pathogenic"
        if "likely pathogenic" in parts:
            return "likely pathogenic"
        if "benign" in parts:
            return "benign"
        if "likely benign" in parts:
            return "likely benign"
    if "pathogenic" in sig_lower and "likely" in sig_lower:
        return "likely pathogenic"
    if "pathogenic" in sig_lower:
        return "pathogenic"
    if "benign" in sig_lower and "likely" in sig_lower:
        return "likely benign"
    if "benign" in sig_lower:
        return "benign"
    return "uncertain significance"


def evaluate_acmg_criteria(
    clinvar_record: dict[str, Any],
) -> list[dict[str, str]]:
    """Evaluate ACMG criteria from ClinVar data.

    Criteria assessed:
        PS1 - Same amino acid change as established pathogenic variant
        PP5 - Reputable source reports variant as pathogenic
        BP6 - Reputable source reports variant as benign
        BA1 - Allele frequency >5% (if noted in ClinVar)
        BS1 - Allele frequency greater than expected for disorder
        PM5 - Novel missense at same position as known pathogenic missense
    """
    triggered: list[dict[str, str]] = []

    significance = _normalize_significance(
        clinvar_record.get("clinical_significance")
    )
    review_status = clinvar_record.get("review_status", "")
    stars = _get_star_rating(review_status)
    submitter_count = clinvar_record.get("submitter_count", 0)
    hgvs = clinvar_record.get("hgvs", "") or ""
    raw_submissions = clinvar_record.get("raw_submissions", [])
    condition = (clinvar_record.get("condition") or "").lower()

    # PS1: Same amino acid change as established pathogenic variant
    if significance == "pathogenic" and stars >= 2 and submitter_count >= 2:
        triggered.append({
            "criterion": "PS1",
            "strength": "Strong",
            "direction": "pathogenic",
            "justification": (
                f"ClinVar classifies this variant as pathogenic with "
                f"{stars}-star review status and {submitter_count} submitters, "
                f"indicating an established pathogenic amino acid change."
            ),
        })

    # PP5: Reputable source reports variant as pathogenic
    # Strength scaled by ClinVar review status per ClinGen recommendations
    if significance in ("pathogenic", "likely pathogenic") and stars >= 3:
        triggered.append({
            "criterion": "PP5",
            "strength": "Strong",
            "direction": "pathogenic",
            "justification": (
                f"ClinVar reports this variant as '{significance}' with "
                f"{stars}-star review status ({review_status}). Upgraded to "
                f"Strong evidence per ClinGen recommendation for expert "
                f"panel / practice guideline level review."
            ),
        })
    elif significance in ("pathogenic", "likely pathogenic") and stars >= 2:
        triggered.append({
            "criterion": "PP5",
            "strength": "Moderate",
            "direction": "pathogenic",
            "justification": (
                f"ClinVar reports this variant as '{significance}' with "
                f"{stars}-star review status ({review_status}), meeting the "
                f"threshold for a reputable source classification."
            ),
        })
    elif significance in ("pathogenic", "likely pathogenic") and stars >= 1:
        triggered.append({
            "criterion": "PP5",
            "strength": "Supporting",
            "direction": "pathogenic",
            "justification": (
                f"ClinVar reports this variant as '{significance}' with "
                f"{stars}-star review status. Lower confidence due to "
                f"limited review status."
            ),
        })

    # BP6: Reputable source reports variant as benign
    # Strength scaled by ClinVar review status per ClinGen recommendations
    if significance in ("benign", "likely benign") and stars >= 3:
        triggered.append({
            "criterion": "BP6",
            "strength": "Strong",
            "direction": "benign",
            "justification": (
                f"ClinVar reports this variant as '{significance}' with "
                f"{stars}-star review status ({review_status}). Upgraded to "
                f"Strong evidence per ClinGen recommendation for expert "
                f"panel / practice guideline level review."
            ),
        })
    elif significance in ("benign", "likely benign") and stars >= 2:
        triggered.append({
            "criterion": "BP6",
            "strength": "Moderate",
            "direction": "benign",
            "justification": (
                f"ClinVar reports this variant as '{significance}' with "
                f"{stars}-star review status ({review_status}), meeting the "
                f"threshold for a reputable source classification."
            ),
        })
    elif significance in ("benign", "likely benign") and stars >= 1:
        triggered.append({
            "criterion": "BP6",
            "strength": "Supporting",
            "direction": "benign",
            "justification": (
                f"ClinVar reports this variant as '{significance}' with "
                f"{stars}-star review status. Lower confidence due to "
                f"limited review status."
            ),
        })

    # BA1: Allele frequency >5% in population databases
    condition_text = condition + " " + " ".join(raw_submissions).lower()
    if any(kw in condition_text for kw in ["common", "polymorphism"]):
        if significance in ("benign", "likely benign"):
            triggered.append({
                "criterion": "BA1",
                "strength": "Stand-alone",
                "direction": "benign",
                "justification": (
                    "ClinVar annotations suggest this variant has high allele "
                    "frequency in population databases (>5%), consistent with "
                    "a benign polymorphism."
                ),
            })

    # BS1: Allele frequency greater than expected for disorder
    if significance in ("benign", "likely benign") and submitter_count >= 2:
        if not any(c["criterion"] == "BA1" for c in triggered):
            triggered.append({
                "criterion": "BS1",
                "strength": "Strong",
                "direction": "benign",
                "justification": (
                    f"Multiple ClinVar submitters ({submitter_count}) classify "
                    f"this variant as '{significance}', suggesting allele "
                    f"frequency greater than expected for the associated "
                    f"disorder."
                ),
            })

    # PM5: Novel missense at same position as known pathogenic missense
    is_missense = False
    if hgvs:
        missense_pattern = re.compile(
            r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}", re.IGNORECASE
        )
        if missense_pattern.search(hgvs):
            is_missense = True

    if is_missense and significance in ("pathogenic", "likely pathogenic"):
        if not any(c["criterion"] == "PS1" for c in triggered):
            triggered.append({
                "criterion": "PM5",
                "strength": "Moderate",
                "direction": "pathogenic",
                "justification": (
                    "This missense variant occurs at a position where other "
                    "pathogenic missense changes have been reported in ClinVar."
                ),
            })

    return triggered


def classify_variant(
    criteria: list[dict[str, str]],
) -> dict[str, str]:
    """Apply ACMG combination rules to produce final classification.

    Implements simplified ACMG/AMP 2015 combining rules:
    - Pathogenic: 1 Very Strong + 1 Strong; OR 2 Strong; OR
                  1 Strong + 3 Moderate/Supporting
    - Likely Pathogenic: 1 Strong + 1-2 Moderate; OR
                         1 Strong + 2 Supporting; OR 3+ Moderate
    - Likely Benign: 1 Strong benign + 1 Supporting benign
    - Benign: 1 Stand-alone; OR 2+ Strong benign
    - VUS: does not meet criteria for other categories
    """
    if not criteria:
        return {
            "classification": "VUS",
            "confidence": "Low",
            "reasoning": (
                "No ACMG criteria could be evaluated from available ClinVar "
                "data. The variant is classified as a Variant of Uncertain "
                "Significance (VUS) by default."
            ),
        }

    path_very_strong = 0
    path_strong = 0
    path_moderate = 0
    path_supporting = 0
    benign_standalone = 0
    benign_strong = 0
    benign_supporting = 0

    for c in criteria:
        direction = c.get("direction", "")
        strength = c.get("strength", "").lower()

        if direction == "pathogenic":
            if "very strong" in strength:
                path_very_strong += 1
            elif "strong" in strength:
                path_strong += 1
            elif "moderate" in strength:
                path_moderate += 1
            elif "supporting" in strength:
                path_supporting += 1
        elif direction == "benign":
            if "stand-alone" in strength or "stand alone" in strength:
                benign_standalone += 1
            elif "strong" in strength:
                benign_strong += 1
            elif "supporting" in strength:
                benign_supporting += 1

    # Benign (check first since BA1 is definitive)
    if benign_standalone >= 1:
        return {
            "classification": "Benign",
            "confidence": "High",
            "reasoning": (
                "Stand-alone benign evidence (BA1) is sufficient for a "
                "Benign classification under ACMG guidelines."
            ),
        }
    if benign_strong >= 2:
        return {
            "classification": "Benign",
            "confidence": "High",
            "reasoning": (
                f"{benign_strong} strong benign criteria met, sufficient for "
                f"Benign classification."
            ),
        }

    # Likely Benign
    if benign_strong >= 1 and benign_supporting >= 1:
        return {
            "classification": "Likely Benign",
            "confidence": "Moderate",
            "reasoning": (
                "Combination of strong and supporting benign evidence meets "
                "ACMG criteria for Likely Benign classification."
            ),
        }

    # Pathogenic
    is_pathogenic = False
    if path_very_strong >= 1 and path_strong >= 1:
        is_pathogenic = True
    elif path_strong >= 2:
        is_pathogenic = True
    elif path_strong >= 1 and path_moderate + path_supporting >= 3:
        is_pathogenic = True
    elif path_very_strong >= 1 and path_moderate >= 2:
        is_pathogenic = True

    if is_pathogenic:
        return {
            "classification": "Pathogenic",
            "confidence": "High",
            "reasoning": (
                f"ACMG criteria combination met for Pathogenic: "
                f"{path_very_strong} Very Strong, {path_strong} Strong, "
                f"{path_moderate} Moderate, {path_supporting} Supporting "
                f"pathogenic criteria triggered."
            ),
        }

    # Likely Pathogenic
    is_likely_path = False
    if path_strong >= 1 and path_moderate >= 1:
        is_likely_path = True
    elif path_strong >= 1 and path_supporting >= 2:
        is_likely_path = True
    elif path_moderate >= 3:
        is_likely_path = True
    elif path_moderate >= 2 and path_supporting >= 2:
        is_likely_path = True
    elif path_very_strong >= 1 and path_moderate >= 1:
        is_likely_path = True

    if is_likely_path:
        return {
            "classification": "Likely Pathogenic",
            "confidence": "Moderate",
            "reasoning": (
                f"ACMG criteria combination met for Likely Pathogenic: "
                f"{path_very_strong} Very Strong, {path_strong} Strong, "
                f"{path_moderate} Moderate, {path_supporting} Supporting "
                f"pathogenic criteria triggered."
            ),
        }

    if path_strong >= 1 or path_moderate >= 1 or path_supporting >= 1:
        return {
            "classification": "VUS",
            "confidence": "Low",
            "reasoning": (
                "Some pathogenic evidence exists but does not meet the "
                "threshold for Likely Pathogenic or Pathogenic under ACMG "
                "combining rules. Classified as VUS."
            ),
        }

    if benign_strong >= 1 or benign_supporting >= 1:
        return {
            "classification": "VUS",
            "confidence": "Low",
            "reasoning": (
                "Some benign evidence exists but does not meet the threshold "
                "for Likely Benign or Benign under ACMG combining rules. "
                "Classified as VUS."
            ),
        }

    return {
        "classification": "VUS",
        "confidence": "Low",
        "reasoning": (
            "Insufficient evidence to classify this variant under ACMG "
            "guidelines. Defaults to Variant of Uncertain Significance (VUS)."
        ),
    }


# ============================================================================
# SECTION 3: LangChain Agent with ClinVar Tool
# ============================================================================

SYSTEM_PROMPT = """You are a clinical molecular geneticist specializing in \
hereditary cancer risk assessment. Your role is to classify germline variants \
using the ACMG/AMP 2015 guidelines.

When a user provides a variant, you should:
1. Query ClinVar using the clinvar_lookup tool to retrieve the variant record.
2. Analyze the ClinVar data and apply ACMG criteria evaluation.
3. Provide a structured classification with reasoning.

Always be precise and evidence-based. Cite the specific ACMG criteria that \
apply and explain why. If data is limited, acknowledge the uncertainty.

IMPORTANT: You must always use the clinvar_lookup tool to look up variants. \
Do not rely on your training data for variant classifications, as ClinVar \
is updated regularly and your knowledge may be outdated.

After retrieving ClinVar data, provide a plain-English summary of your \
classification reasoning, including which ACMG criteria apply and why."""

DISCLAIMER = (
    "This is an AI-assisted research tool and should not be used for clinical "
    "decision-making. Variant classifications should be reviewed by a "
    "certified clinical molecular geneticist and confirmed through validated "
    "clinical-grade processes."
)


@tool
def clinvar_lookup(variant: str) -> str:
    """Query ClinVar for a genomic variant and return structured data.

    Use this tool to look up any genomic variant in the NCBI ClinVar database.
    Input should be a variant description such as 'BRCA1 c.5266dupC' or
    'NM_007294.4:c.5266dupC' or 'TP53 c.817C>T'.

    Returns a JSON string with variant details including clinical significance,
    review status, submitter information, and associated conditions.
    """
    result = query_clinvar(variant)
    return json.dumps(result, indent=2, default=str)


def build_agent():
    """Build and return the LangGraph ReAct agent with ClinVar tools."""
    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        temperature=0,
        max_tokens=4096,
    )
    tools = [clinvar_lookup]
    agent = create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )
    return agent


def classify_variant_with_agent(variant: str) -> dict[str, Any]:
    """Run the full classification pipeline for a variant.

    1. Queries ClinVar directly for programmatic ACMG evaluation
    2. Evaluates ACMG criteria programmatically
    3. Applies ACMG combination rules
    4. Runs the LangChain agent for enhanced clinical reasoning
    5. Combines programmatic classification with agent reasoning
    """
    # Step 1: Query ClinVar
    clinvar_record = query_clinvar(variant)

    if clinvar_record.get("error") and not clinvar_record.get("variant_id"):
        return {
            "variant": variant,
            "clinvar_record": clinvar_record,
            "criteria_triggered": [],
            "classification": "Unable to classify",
            "confidence": "N/A",
            "reasoning": clinvar_record["error"],
            "disclaimer": DISCLAIMER,
        }

    # Step 2: Evaluate ACMG criteria
    criteria = evaluate_acmg_criteria(clinvar_record)

    # Step 3: Apply ACMG combination rules
    classification_result = classify_variant(criteria)

    # Step 4: Run the LangChain agent for enhanced reasoning
    agent = build_agent()
    try:
        response = agent.invoke(
            {"messages": [HumanMessage(content=f"Classify the following variant: {variant}")]}
        )
        messages = response.get("messages", [])
        agent_reasoning = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content:
                agent_reasoning = msg.content
                break
    except Exception as e:
        agent_reasoning = f"Agent reasoning unavailable: {str(e)}"

    # Step 5: Combine results
    enhanced_reasoning = classification_result["reasoning"]
    if agent_reasoning and "reasoning" not in agent_reasoning[:50].lower():
        enhanced_reasoning = (
            f"{classification_result['reasoning']}\n\n"
            f"Additional AI analysis: {agent_reasoning}"
        )

    return {
        "variant": variant,
        "clinvar_record": clinvar_record,
        "criteria_triggered": criteria,
        "classification": classification_result["classification"],
        "confidence": classification_result["confidence"],
        "reasoning": enhanced_reasoning,
        "disclaimer": DISCLAIMER,
    }


# ============================================================================
# SECTION 4: CLI Output Formatting
# ============================================================================


def _stars_display(review_status: str | None) -> str:
    """Convert review status to star display string."""
    count = _get_star_rating(review_status)
    return "\u2605" * count + "\u2606" * (4 - count)


def print_results(result: dict[str, Any]) -> None:
    """Print classification results to stdout in a readable format."""
    clinvar = result.get("clinvar_record", {})
    criteria = result.get("criteria_triggered", [])

    width = 72
    print("\n" + "=" * width)
    print("  ClinVar ACMG Variant Classifier")
    print("=" * width)

    # --- Input ---
    print(f"\n  Variant Query:  {result.get('variant', 'N/A')}")

    # --- ClinVar Record ---
    print("\n" + "-" * width)
    print("  CLINVAR RECORD")
    print("-" * width)

    if clinvar.get("error"):
        print(f"  Error: {clinvar['error']}")
    else:
        review_status = clinvar.get("review_status", "N/A")
        stars = _stars_display(review_status)

        print(f"  Variant ID:              {clinvar.get('variant_id', 'N/A')}")
        print(f"  Gene:                    {clinvar.get('gene', 'N/A')}")
        print(f"  HGVS:                    {clinvar.get('hgvs', 'N/A')}")
        print(f"  Clinical Significance:   {clinvar.get('clinical_significance', 'N/A')}")
        print(f"  Review Status:           {stars}  ({review_status})")
        print(f"  Submitter Count:         {clinvar.get('submitter_count', 0)}")
        print(f"  Condition:               {clinvar.get('condition', 'N/A')}")
        print(f"  Last Evaluated:          {clinvar.get('last_evaluated', 'N/A')}")

        if clinvar.get("conflicting_interpretations"):
            print(f"  WARNING:                 Conflicting interpretations among submitters")

    # --- ACMG Criteria ---
    print("\n" + "-" * width)
    print("  ACMG CRITERIA TRIGGERED")
    print("-" * width)

    if not criteria:
        print("  No ACMG criteria could be evaluated from available data.")
    else:
        for c in criteria:
            direction = c.get("direction", "")
            marker = "[PATH]" if direction == "pathogenic" else "[BNGN]"
            print(
                f"  {marker} {c['criterion']} ({c['strength']})"
            )
            # Wrap justification text
            justification = c.get("justification", "")
            indent = "         "
            words = justification.split()
            line = indent
            for word in words:
                if len(line) + len(word) + 1 > width:
                    print(line)
                    line = indent + word
                else:
                    line = line + " " + word if line.strip() else indent + word
            if line.strip():
                print(line)
            print()

    # --- Final Classification ---
    print("-" * width)
    print("  FINAL CLASSIFICATION")
    print("-" * width)

    classification = result.get("classification", "VUS")
    confidence = result.get("confidence", "Low")
    reasoning = result.get("reasoning", "")

    print(f"\n  >>> {classification} <<<")
    print(f"  Confidence: {confidence}\n")

    # Wrap reasoning text
    print("  Reasoning:")
    indent = "    "
    for paragraph in reasoning.split("\n\n"):
        words = paragraph.split()
        line = indent
        for word in words:
            if len(line) + len(word) + 1 > width:
                print(line)
                line = indent + word
            else:
                line = line + " " + word if line.strip() else indent + word
        if line.strip():
            print(line)
        print()

    # --- Disclaimer ---
    print("-" * width)
    print(f"  DISCLAIMER: {DISCLAIMER}")
    print("=" * width + "\n")


# ============================================================================
# SECTION 5: Main Entry Point
# ============================================================================


def main():
    """Main entry point for CLI usage."""
    if len(sys.argv) < 2:
        print("Usage: python classify_variant.py <variant>")
        print()
        print("Examples:")
        print('  python classify_variant.py "BRCA1 c.5266dupC"')
        print('  python classify_variant.py "TP53 c.817C>T"')
        print('  python classify_variant.py "PALB2 c.3113G>A"')
        sys.exit(1)

    variant = " ".join(sys.argv[1:])

    # Validate environment
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.")
        print("Set it in a .env file or export it in your shell.")
        sys.exit(1)

    print(f"\nClassifying variant: {variant}")
    print("Querying ClinVar and running ACMG classification...\n")

    result = classify_variant_with_agent(variant)
    print_results(result)


if __name__ == "__main__":
    main()
