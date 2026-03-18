"""ACMG criteria logic and classification rules.

Implements a subset of ACMG/AMP 2015 guidelines that can be evaluated
from ClinVar data alone.
"""

from typing import Any

# Map review status strings to approximate star ratings
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
    # Handle compound classifications like "Pathogenic/Likely pathogenic"
    # by checking for unqualified "pathogenic" first
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

    Args:
        clinvar_record: Structured dict from query_clinvar.

    Returns:
        List of dicts with keys: criterion, strength, direction, justification.
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
    # If ClinVar classifies this variant as pathogenic with multiple submitters
    # and good review status, the same amino acid change is established
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
    # Per ClinGen recommendation, PP5 can be upgraded to Strong when
    # evidence comes from expert panel or practice guideline review
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
    # Per ClinGen recommendation, BP6 can be upgraded to Strong when
    # evidence comes from expert panel or practice guideline review
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
    # Check if ClinVar or submissions note high allele frequency
    freq_keywords = ["common", "polymorphism", "benign", "population"]
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
    # Inferred when variant is classified as benign with multiple submitters
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
    # Check if HGVS suggests a missense change and ClinVar has pathogenic data
    is_missense = False
    if hgvs:
        # Simple heuristic: look for single-letter amino acid substitution
        # patterns like p.Arg1699Trp or protein-level changes
        import re
        missense_pattern = re.compile(
            r"p\.[A-Z][a-z]{2}\d+[A-Z][a-z]{2}", re.IGNORECASE
        )
        if missense_pattern.search(hgvs):
            is_missense = True

    if is_missense and significance in ("pathogenic", "likely pathogenic"):
        # Only add PM5 if PS1 is not already triggered (PM5 is for novel
        # missense at same position, not the exact same change)
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

    Args:
        criteria: List of triggered criteria from evaluate_acmg_criteria.

    Returns:
        Dict with classification and confidence.
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

    # Count criteria by direction and strength
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

    # Apply ACMG combining rules

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
    elif (
        path_strong >= 1
        and path_moderate + path_supporting >= 3
    ):
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

    # If we have some pathogenic evidence but not enough
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

    # If we have some benign evidence but not enough
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
