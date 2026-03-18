"""Node 7: ACMG Classifier — LLM-powered final classification.

Uses Claude (claude-sonnet-4-5) to synthesize all evidence and produce
a five-tier ACMG classification with explicit criteria evaluation.
"""

import json
import logging
import os
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import VariantState

logger = logging.getLogger(__name__)

DISCLAIMER = (
    "This is an AI-assisted research tool and should not be used for clinical "
    "decision-making. Variant classifications should be reviewed by a "
    "certified clinical molecular geneticist and confirmed through validated "
    "clinical-grade processes."
)

SYSTEM_PROMPT = """\
You are a clinical molecular geneticist applying the ACMG/AMP 2015 \
variant classification framework to germline variants for hereditary \
cancer risk assessment. You will receive structured evidence from \
one or more genomic knowledge sources and must apply all applicable \
ACMG criteria, then produce a final five-tier classification.

Be explicit about which criteria are met, which are not met, \
and which cannot be evaluated due to missing evidence. \
Always note when evidence is absent rather than assuming benign.

You MUST respond with ONLY a valid JSON object — no markdown fences, \
no commentary before or after. The JSON must have exactly these keys:

{
  "criteria_triggered": [
    {
      "code": "PS1",
      "name": "Same amino acid change as established pathogenic",
      "direction": "pathogenic",
      "strength": "Strong",
      "met": true,
      "justification": "...",
      "evidence_source": "ClinVar"
    }
  ],
  "classification": "Pathogenic",
  "confidence": "High",
  "reasoning": "2-3 sentence summary"
}

classification must be one of: Pathogenic, Likely Pathogenic, VUS, \
Likely Benign, Benign.
confidence must be one of: High, Moderate, Low.
For criteria_triggered, include ALL criteria you evaluated (both met \
and not met) with met=true or met=false. direction is "pathogenic" \
or "benign". strength is one of: Very Strong, Strong, Moderate, Supporting, \
Stand-alone.
"""


def _build_evidence_prompt(state: VariantState) -> str:
    """Build the user prompt with all available evidence."""
    parts = []

    # Variant identity
    gene = state.get("gene_symbol") or "Unknown"
    hgvs = state.get("hgvs_on_transcript") or state.get("raw_input", "")
    build = state.get("genome_build", "GRCh38")
    parts.append(f"## Variant\n{hgvs} on {gene} ({build})")

    # Gene info
    aliases = state.get("gene_aliases")
    full_name = state.get("gene_full_name")
    if full_name:
        parts.append(f"Gene full name: {full_name}")
    if aliases:
        parts.append(f"Gene aliases: {', '.join(aliases)}")

    # ClinVar evidence
    clinvar = state.get("clinvar")
    if clinvar and clinvar.get("variant_id"):
        cv_lines = [
            "## ClinVar Evidence",
            f"- Variant ID: {clinvar.get('variant_id')}",
            f"- HGVS: {clinvar.get('hgvs', 'N/A')}",
            f"- Clinical Significance: {clinvar.get('clinical_significance', 'N/A')}",
            f"- Review Status: {clinvar.get('review_status', 'N/A')}",
            f"- Star Rating: {clinvar.get('star_rating', 0)}/4",
            f"- Submitter Count: {clinvar.get('submitter_count', 0)}",
            f"- Conflicting Interpretations: {clinvar.get('conflicting_interpretations', False)}",
            f"- Condition: {clinvar.get('condition', 'N/A')}",
            f"- Last Evaluated: {clinvar.get('last_evaluated', 'N/A')}",
        ]
        raw_subs = clinvar.get("raw_submissions", [])
        if raw_subs:
            cv_lines.append(f"- Individual Submissions: {', '.join(str(s) for s in raw_subs[:10])}")
        parts.append("\n".join(cv_lines))
    else:
        parts.append("## ClinVar Evidence\nNot available — no ClinVar record found.")

    # gnomAD
    if state.get("gnomad"):
        parts.append(f"## gnomAD Evidence\n{json.dumps(state['gnomad'], indent=2)}")
    else:
        parts.append(
            "## gnomAD Evidence\n"
            "Not available — BA1 (allele frequency >5%), BS1 (frequency greater "
            "than expected), PM2 (absent from controls) cannot be evaluated."
        )

    # PubMed
    if state.get("pubmed"):
        parts.append(f"## PubMed Evidence\n{json.dumps(state['pubmed'], indent=2)}")
    else:
        parts.append(
            "## PubMed Evidence\n"
            "Not available — PS3/BS3 (functional studies), PP1/BS4 "
            "(co-segregation) cannot be evaluated."
        )

    # AlphaFold / PDB
    if state.get("alphafold") or state.get("pdb"):
        parts.append("## Structural Evidence")
        if state.get("alphafold"):
            parts.append(f"AlphaFold: {json.dumps(state['alphafold'], indent=2)}")
        if state.get("pdb"):
            parts.append(f"PDB: {json.dumps(state['pdb'], indent=2)}")
    else:
        parts.append(
            "## Structural Evidence\n"
            "Not available — PM1 (functional domain) cannot be evaluated "
            "from structural data."
        )

    # TCGA
    if state.get("tcga_somatic"):
        parts.append(f"## TCGA Evidence\n{json.dumps(state['tcga_somatic'], indent=2)}")
    else:
        parts.append(
            "## TCGA Somatic Evidence\n"
            "Not available — somatic hotspot and biallelic loss evidence "
            "cannot be evaluated."
        )

    # Instructions
    parts.append(
        "## Instructions\n"
        "Evaluate these ACMG criteria based on the evidence above:\n"
        "- PS1: Same amino acid change as established pathogenic variant\n"
        "- PP5: Reputable source reports variant as pathogenic "
        "(upgrade to Strong for expert panel 3+ stars, Moderate for 2 stars)\n"
        "- BP6: Reputable source reports variant as benign "
        "(upgrade to Strong for expert panel 3+ stars, Moderate for 2 stars)\n"
        "- BA1: Flag if ClinVar notes high population frequency\n"
        "- PM5: Novel missense at same position as known pathogenic missense\n"
        "- For any criterion that cannot be evaluated due to missing evidence, "
        "set met=false and explain why in justification.\n"
        "- Apply ACMG combining rules to determine final classification.\n"
        "\nRespond with ONLY valid JSON."
    )

    return "\n\n".join(parts)


def _parse_llm_response(text: str) -> dict[str, Any]:
    """Parse the LLM JSON response, handling markdown fences."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return json.loads(text)


def acmg_classifier_node(state: VariantState) -> dict[str, Any]:
    """Classify variant using ACMG criteria with Claude LLM reasoning."""
    logger.info("acmg_classifier_node: starting classification")

    updates: dict[str, Any] = {"current_node": "acmg_classifier"}

    # Build the evidence prompt
    evidence_prompt = _build_evidence_prompt(state)

    # Call Claude
    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        temperature=0,
        max_tokens=4096,
    )

    try:
        response = llm.invoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=evidence_prompt),
        ])

        response_text = response.content if isinstance(response.content, str) else str(response.content)

        try:
            parsed = _parse_llm_response(response_text)
        except json.JSONDecodeError:
            # Retry with explicit JSON instruction
            logger.warning("acmg_classifier: JSON parse failed, retrying")
            retry_response = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(
                    content=evidence_prompt + "\n\nYour previous response was not valid JSON. "
                    "Respond ONLY with valid JSON, no markdown fences or extra text."
                ),
            ])
            retry_text = retry_response.content if isinstance(retry_response.content, str) else str(retry_response.content)
            parsed = _parse_llm_response(retry_text)

        # Extract fields from parsed response
        updates["criteria_triggered"] = parsed.get("criteria_triggered", [])
        updates["classification"] = parsed.get("classification", "VUS")
        updates["confidence"] = parsed.get("confidence", "Low")
        updates["reasoning"] = parsed.get("reasoning", "")
        updates["disclaimer"] = DISCLAIMER

        logger.info(
            "acmg_classifier: classification=%s, confidence=%s, %d criteria",
            updates["classification"],
            updates["confidence"],
            len(updates["criteria_triggered"]),
        )

    except Exception as e:
        logger.exception("acmg_classifier: LLM call failed: %s", e)
        updates["criteria_triggered"] = []
        updates["classification"] = "VUS"
        updates["confidence"] = "Low"
        updates["reasoning"] = f"ACMG classification failed: {str(e)}. Defaulting to VUS."
        updates["disclaimer"] = DISCLAIMER
        updates["errors"] = state.get("errors", []) + [f"ACMG classifier error: {str(e)}"]

    return updates
