"""Audience-specific natural-language explainers for ACMG classifications.

Two functions, each making one Anthropic call (claude-sonnet-4-5-20250929):

    explain_for_patient(state)  - lay explanation, ~150 words, empathetic
    explain_for_curator(state)  - ACMG shorthand, ~100 words, action items

Both consume the full classification state from the LangGraph pipeline and
return:
    {"text": str, "disclaimer": str, "usage_tokens": int, "error": bool}

Errors (network, missing API key, JSON, etc.) are caught and returned as a
graceful placeholder dict with error=True so the UI does not blow up.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

from svi.guardrails import DISCLAIMER


# ---------------------------------------------------------------------------
# Model + prompt constants
# ---------------------------------------------------------------------------

EXPLAINER_MODEL = "claude-sonnet-4-5-20250929"
"""Anthropic model ID used for both explainer prompts."""

PATIENT_SYSTEM_PROMPT = (
    "You are a genetic counselor explaining a variant classification to a patient "
    "or family member who has no genetics background. Define every technical term "
    "you use. Avoid jargon. Use a calm, empathetic, professional tone. Limit to "
    "~150 words. End with: \"Discuss what this means for you with your healthcare "
    "provider or a genetic counselor.\" NEVER include the word \"ACMG\" or any "
    "criterion code (PVS1, PM2, etc.) in the patient-facing text."
)

CURATOR_SYSTEM_PROMPT = (
    "You are briefing a variant curator who knows ACMG/AMP terminology fluently. "
    "Use criterion codes (PVS1, PM2_Supporting, etc.). Note any framework "
    "disagreements between Tavtigian and Richards 2015 and recommend additional "
    "evidence searches that would resolve them. State which SVI overrides "
    "applied. Limit to ~100 words. End with one specific actionable next step."
)


# ---------------------------------------------------------------------------
# State -> evidence summary helpers
# ---------------------------------------------------------------------------

def _summarise_criteria(criteria: List[Dict[str, Any]]) -> str:
    """Render criteria_triggered as a compact string of met codes only."""
    if not criteria:
        return "(none)"
    met = [c for c in criteria if c.get("met")]
    if not met:
        return "(none met)"
    parts = []
    for c in met:
        code = c.get("code", "?")
        strength = c.get("strength", "")
        parts.append(f"{code} ({strength})" if strength else code)
    return ", ".join(parts)


def _build_evidence_block(state: Dict[str, Any]) -> str:
    """Compose the human-message body shared by both explainer prompts."""
    gene = state.get("gene_symbol", "?")
    hgvs = state.get("hgvs_on_transcript") or state.get("hgvs", "?")
    primary = state.get("primary_classification") or state.get("classification", "VUS")
    tav = state.get("tavtigian") or {}
    rich = state.get("richards_2015") or {}
    frameworks_agree = state.get("frameworks_agree")
    disagreement = state.get("disagreement_explanation", "")
    criteria = state.get("criteria_triggered") or []
    dosage = state.get("clingen_dosage") or {}
    overrides = state.get("svi_overrides_applied") or []

    tav_class = tav.get("classification", "?") if isinstance(tav, dict) else "?"
    tav_pts = tav.get("net_points") if isinstance(tav, dict) else None
    rich_class = rich.get("classification", "?") if isinstance(rich, dict) else "?"
    rich_rule = rich.get("rule_fired") if isinstance(rich, dict) else None

    lines = [
        f"Gene: {gene}",
        f"Variant (HGVS): {hgvs}",
        f"Primary classification: {primary}",
        f"Tavtigian 2018 (Bayesian): {tav_class} (net points: {tav_pts})",
        f"Richards 2015 Table 5: {rich_class} (rule: {rich_rule})",
        f"Frameworks agree: {frameworks_agree}",
    ]
    if disagreement:
        lines.append(f"Disagreement: {disagreement}")
    lines.append(f"ACMG criteria met: {_summarise_criteria(criteria)}")
    if dosage:
        hi = dosage.get("haploinsufficiency_score")
        ts = dosage.get("triplosensitivity_score")
        lines.append(f"ClinGen dosage: HI={hi} TS={ts}")
    if overrides:
        lines.append(f"SVI overrides applied: {', '.join(overrides)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Anthropic call wrapper
# ---------------------------------------------------------------------------

def _call_anthropic(system_prompt: str, user_msg: str) -> Dict[str, Any]:
    """Run one Sonnet 4.5 call and return {text, usage_tokens, error}.

    Reads the API key from os.environ['ANTHROPIC_API_KEY']. Never logs or echoes
    the key. On any exception, returns a placeholder text and error=True.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "text": "Explainer unavailable: ANTHROPIC_API_KEY not set in environment.",
            "usage_tokens": 0,
            "error": True,
        }

    try:
        # Local import keeps the module importable even if the SDK is missing
        # in a verifier-only environment.
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=EXPLAINER_MODEL,
            max_tokens=512,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
        )
        # response.content is a list of content blocks; we want text blocks only.
        text_parts: List[str] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        text = "\n".join(t for t in text_parts if t).strip()

        usage = getattr(response, "usage", None)
        in_tok = getattr(usage, "input_tokens", 0) if usage else 0
        out_tok = getattr(usage, "output_tokens", 0) if usage else 0

        return {
            "text": text or "(empty response)",
            "usage_tokens": int(in_tok) + int(out_tok),
            "error": False,
        }
    except Exception as exc:  # noqa: BLE001 - intentional broad catch for UI safety
        # Surface only the exception class + short message, never echo headers
        # or env values. Truncate to keep stack-trace-style prefixes out.
        reason = f"{exc.__class__.__name__}: {str(exc)[:200]}"
        return {
            "text": f"Explainer unavailable: {reason}",
            "usage_tokens": 0,
            "error": True,
        }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def explain_for_patient(state: Dict[str, Any]) -> Dict[str, Any]:
    """Plain-English explanation for a patient or family member.

    Returns:
        text: str  (~150 words, lay vocabulary, empathetic)
        disclaimer: str  (research-only disclaimer constant)
        usage_tokens: int  (Anthropic input + output tokens)
        error: bool  (True if the API call failed)
    """
    evidence = _build_evidence_block(state)
    user_msg = (
        "Please explain this variant classification in plain language for the "
        "patient or family member. Here is the structured ACMG verdict and "
        "supporting evidence:\n\n" + evidence
    )
    result = _call_anthropic(PATIENT_SYSTEM_PROMPT, user_msg)
    result["disclaimer"] = DISCLAIMER
    return result


def explain_for_curator(state: Dict[str, Any]) -> Dict[str, Any]:
    """Brief for a variant curator using ACMG/SVI shorthand.

    Returns:
        text: str  (~100 words, ACMG vocabulary, action items)
        disclaimer: str  (research-only disclaimer constant)
        usage_tokens: int  (Anthropic input + output tokens)
        error: bool
    """
    evidence = _build_evidence_block(state)
    user_msg = (
        "Brief a variant curator on this case in ACMG/AMP shorthand. Highlight "
        "any framework disagreement and recommend the next-most-useful evidence "
        "search. Structured verdict and evidence:\n\n" + evidence
    )
    result = _call_anthropic(CURATOR_SYSTEM_PROMPT, user_msg)
    result["disclaimer"] = DISCLAIMER
    return result
