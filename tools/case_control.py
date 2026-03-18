"""Case-control statistical analysis for variant enrichment.

Performs Fisher's exact test (overall and per-ancestry) and weighted
GLM with per-ethnicity case/control counts.
"""

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import numpy as np
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

# gnomAD population IDs and display names
POP_DISPLAY = {
    "afr": "African/African American",
    "amr": "Latino/Admixed American",
    "asj": "Ashkenazi Jewish",
    "eas": "East Asian",
    "fin": "Finnish",
    "mid": "Middle Eastern",
    "nfe": "Non-Finnish European",
    "sas": "South Asian",
    "remaining": "Remaining",
    "ami": "Amish",
}


def fishers_exact_test(
    case_ac: int, case_an: int, control_ac: int, control_an: int,
) -> dict[str, Any]:
    """Run 2x2 Fisher's exact test."""
    case_no = case_an - case_ac
    ctrl_no = control_an - control_ac
    table = [[case_ac, case_no], [control_ac, ctrl_no]]

    result = {
        "case_ac": case_ac, "case_an": case_an,
        "case_af": case_ac / case_an if case_an > 0 else 0,
        "control_ac": control_ac, "control_an": control_an,
        "control_af": control_ac / control_an if control_an > 0 else 0,
        "odds_ratio": None, "p_value": None,
        "ci_lower": None, "ci_upper": None,
        "interpretation": "", "significant": False,
    }

    if not HAS_SCIPY:
        result["interpretation"] = "scipy not installed"
        return result

    try:
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        result["odds_ratio"] = odds_ratio
        result["p_value"] = p_value
        result["significant"] = p_value < 0.05

        # Approximate 95% CI for log(OR)
        if odds_ratio > 0 and odds_ratio != float("inf"):
            try:
                log_or = math.log(odds_ratio)
                se = math.sqrt(
                    1 / max(case_ac, 0.5) + 1 / max(case_no, 0.5)
                    + 1 / max(control_ac, 0.5) + 1 / max(ctrl_no, 0.5)
                )
                result["ci_lower"] = math.exp(log_or - 1.96 * se)
                result["ci_upper"] = math.exp(log_or + 1.96 * se)
            except (ValueError, ZeroDivisionError):
                pass

        sig_str = f"p={p_value:.2e}" if p_value < 0.001 else f"p={p_value:.4f}"
        if odds_ratio == float("inf"):
            result["interpretation"] = f"Variant only in cases (OR=Inf, {sig_str})"
        elif p_value < 0.05:
            direction = "enriched" if odds_ratio > 1 else "depleted"
            result["interpretation"] = f"{direction} in cases (OR={odds_ratio:.2f}, {sig_str})"
        else:
            result["interpretation"] = f"Not significant (OR={odds_ratio:.2f}, {sig_str})"
    except Exception as e:
        result["interpretation"] = f"Test failed: {e}"

    return result


def run_case_control_analysis(
    case_data: dict[str, dict[str, int]],
    gnomad_data: dict[str, Any],
) -> dict[str, Any]:
    """Run full case-control analysis with per-ethnicity case data.

    Args:
        case_data: Dict mapping pop_id → {"carriers": int, "total": int}
            Must include "overall" key for aggregate counts.
            Optional ancestry keys: "afr", "amr", "asj", "eas", "fin", "mid", "nfe", "sas"
        gnomad_data: gnomAD result dict with populations

    Returns:
        Dict with overall Fisher's, per-ancestry Fisher's, weighted GLM results.
    """
    overall_case = case_data.get("overall", {"carriers": 0, "total": 0})
    case_ac = overall_case["carriers"]
    case_an = overall_case["total"] * 2

    control_ac = gnomad_data.get("ac", 0)
    control_an = gnomad_data.get("an", 0)
    populations = gnomad_data.get("populations", {})

    # Overall Fisher's
    overall_fishers = fishers_exact_test(case_ac, case_an, control_ac, control_an)

    # Per-ancestry Fisher's
    ancestry_fishers = []
    for pop_id, pop_data in sorted(populations.items()):
        if not isinstance(pop_data, dict) or pop_data.get("an", 0) == 0:
            continue

        # Use ethnicity-specific case data if provided, otherwise use overall
        pop_case = case_data.get(pop_id, overall_case)
        pop_case_ac = pop_case["carriers"]
        pop_case_an = pop_case["total"] * 2

        test = fishers_exact_test(
            pop_case_ac, pop_case_an,
            pop_data.get("ac", 0), pop_data.get("an", 0),
        )
        test["population"] = pop_data.get("name", POP_DISPLAY.get(pop_id, pop_id))
        test["population_id"] = pop_id
        ancestry_fishers.append(test)

    ancestry_fishers.sort(key=lambda x: x.get("p_value", 1) or 1)

    # Weighted GLM with per-ethnicity data
    glm_result = _weighted_glm(case_data, populations)

    return {
        "case_data": case_data,
        "overall_fishers": overall_fishers,
        "ancestry_fishers": ancestry_fishers,
        "weighted_glm": glm_result,
    }


def _weighted_glm(
    case_data: dict[str, dict[str, int]],
    gnomad_populations: dict[str, dict],
) -> dict[str, Any]:
    """Weighted logistic regression: cases vs gnomAD controls per ancestry."""
    result: dict[str, Any] = {
        "model": "Weighted logistic regression",
        "coefficient": None, "std_error": None,
        "z_score": None, "p_value": None,
        "interpretation": "",
    }

    if not HAS_STATSMODELS:
        result["interpretation"] = "statsmodels not installed"
        return result

    try:
        overall_case = case_data.get("overall", {"carriers": 0, "total": 0})
        rows = []

        for pid, pd in gnomad_populations.items():
            if not isinstance(pd, dict) or pd.get("an", 0) == 0:
                continue
            pop_case = case_data.get(pid, overall_case)
            case_ac = pop_case["carriers"]
            case_an = pop_case["total"] * 2
            if case_an == 0:
                continue
            rows.append({
                "pop": pid,
                "case_ac": case_ac, "case_an": case_an,
                "ctrl_ac": pd.get("ac", 0), "ctrl_an": pd.get("an", 0),
            })

        if len(rows) < 2:
            result["interpretation"] = "Too few populations for GLM"
            return result

        n = len(rows) * 2  # case + control per population
        y = np.zeros(n)
        x = np.zeros(n)
        weights = np.zeros(n)

        for i, r in enumerate(rows):
            # Case row
            case_af = r["case_ac"] / r["case_an"] if r["case_an"] > 0 else 0
            y[2 * i] = np.clip(case_af, 1e-10, 1 - 1e-10)
            x[2 * i] = 1.0
            weights[2 * i] = r["case_an"]

            # Control row
            ctrl_af = r["ctrl_ac"] / r["ctrl_an"] if r["ctrl_an"] > 0 else 0
            y[2 * i + 1] = np.clip(ctrl_af, 1e-10, 1 - 1e-10)
            x[2 * i + 1] = 0.0
            weights[2 * i + 1] = r["ctrl_an"]

        logit_y = np.log(y / (1 - y))
        X = sm.add_constant(x)
        model = sm.WLS(logit_y, X, weights=weights)
        fit = model.fit()

        coef = fit.params[1]
        pval = fit.pvalues[1]
        result["coefficient"] = float(coef)
        result["std_error"] = float(fit.bse[1])
        result["z_score"] = float(fit.tvalues[1])
        result["p_value"] = float(pval)

        if pval < 0.05:
            direction = "enriched" if coef > 0 else "depleted"
            result["interpretation"] = (
                f"Significant: {direction} (coef={coef:.3f}, p={pval:.4f})"
            )
        else:
            result["interpretation"] = f"Not significant (coef={coef:.3f}, p={pval:.4f})"

    except Exception as e:
        result["interpretation"] = f"GLM failed: {e}"

    return result
