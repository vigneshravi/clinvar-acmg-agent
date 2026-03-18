"""Case-control statistical analysis for variant enrichment.

Performs Fisher's exact test (overall and per-ancestry) and weighted
GLM with ancestry proportions as weights.
"""

import logging
import math
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Try importing scipy for Fisher's exact test
try:
    from scipy.stats import fisher_exact
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not installed — Fisher's test unavailable")

# Try importing statsmodels for weighted GLM
try:
    import numpy as np
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    logger.warning("statsmodels not installed — weighted GLM unavailable")


def fishers_exact_test(
    case_ac: int,
    case_an: int,
    control_ac: int,
    control_an: int,
) -> dict[str, Any]:
    """Run Fisher's exact test for variant enrichment.

    Args:
        case_ac: Allele count in cases
        case_an: Total alleles in cases (2 * sample size for diploid)
        control_ac: Allele count in controls (gnomAD)
        control_an: Total alleles in controls (gnomAD)

    Returns:
        Dict with odds_ratio, p_value, contingency_table, interpretation.
    """
    # Build 2x2 contingency table:
    #                 Variant+  Variant-
    # Cases           case_ac   case_an - case_ac
    # Controls        ctrl_ac   ctrl_an - ctrl_ac
    case_no_var = case_an - case_ac
    control_no_var = control_an - control_ac

    table = [[case_ac, case_no_var], [control_ac, control_no_var]]

    result = {
        "case_ac": case_ac,
        "case_an": case_an,
        "case_af": case_ac / case_an if case_an > 0 else 0,
        "control_ac": control_ac,
        "control_an": control_an,
        "control_af": control_ac / control_an if control_an > 0 else 0,
        "contingency_table": table,
        "odds_ratio": None,
        "p_value": None,
        "interpretation": "",
        "significant": False,
    }

    if not HAS_SCIPY:
        result["interpretation"] = "scipy not installed — cannot run Fisher's test"
        return result

    try:
        odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
        result["odds_ratio"] = odds_ratio
        result["p_value"] = p_value
        result["significant"] = p_value < 0.05

        if p_value < 0.001:
            sig_str = f"p={p_value:.2e}"
        else:
            sig_str = f"p={p_value:.4f}"

        if odds_ratio == float("inf"):
            result["interpretation"] = f"Variant only in cases (OR=inf, {sig_str})"
        elif p_value < 0.05:
            direction = "enriched in cases" if odds_ratio > 1 else "depleted in cases"
            result["interpretation"] = f"Significant: {direction} (OR={odds_ratio:.2f}, {sig_str})"
        else:
            result["interpretation"] = f"Not significant (OR={odds_ratio:.2f}, {sig_str})"

    except Exception as e:
        result["interpretation"] = f"Fisher's test failed: {str(e)}"

    return result


def ancestry_specific_fishers(
    case_ac: int,
    case_an: int,
    gnomad_populations: dict[str, dict],
) -> list[dict[str, Any]]:
    """Run Fisher's exact test against each gnomAD ancestry group.

    Args:
        case_ac: Allele count in cases
        case_an: Total alleles in cases
        gnomad_populations: Dict mapping pop_id → {ac, an, af, name}

    Returns:
        List of dicts with population-specific Fisher's test results.
    """
    results = []
    for pop_id, pop_data in sorted(gnomad_populations.items()):
        if not isinstance(pop_data, dict):
            continue
        control_ac = pop_data.get("ac", 0)
        control_an = pop_data.get("an", 0)
        if control_an == 0:
            continue

        test_result = fishers_exact_test(case_ac, case_an, control_ac, control_an)
        test_result["population"] = pop_data.get("name", pop_id)
        test_result["population_id"] = pop_id
        results.append(test_result)

    # Sort by p-value
    results.sort(key=lambda x: x.get("p_value", 1) or 1)
    return results


def weighted_glm_analysis(
    case_ac: int,
    case_an: int,
    gnomad_populations: dict[str, dict],
) -> dict[str, Any]:
    """Run weighted GLM with ancestry allele counts as observations.

    Each ancestry is one observation. The model tests whether variant
    frequency differs between cases and the weighted average of gnomAD
    populations.

    Uses logistic regression with population sample sizes as weights.

    Args:
        case_ac: Allele count in cases
        case_an: Total alleles in cases
        gnomad_populations: Dict mapping pop_id → {ac, an, af, name}

    Returns:
        Dict with model summary, coefficient, p-value, interpretation.
    """
    result: dict[str, Any] = {
        "model": "Weighted logistic regression",
        "coefficient": None,
        "std_error": None,
        "z_score": None,
        "p_value": None,
        "interpretation": "",
        "population_weights": {},
    }

    if not HAS_STATSMODELS:
        result["interpretation"] = "statsmodels not installed — cannot run weighted GLM"
        return result

    try:
        # Build data: each row is (is_case, variant_present, weight)
        # Cases: one group with case_ac successes out of case_an trials
        # Controls: one group per population

        pop_data = []
        for pid, pd in gnomad_populations.items():
            if not isinstance(pd, dict) or pd.get("an", 0) == 0:
                continue
            pop_data.append({
                "pop": pid,
                "name": pd.get("name", pid),
                "ac": pd.get("ac", 0),
                "an": pd.get("an", 0),
                "af": pd.get("af", 0),
                "weight": pd.get("an", 0),
            })

        if len(pop_data) < 2:
            result["interpretation"] = "Too few populations with data for GLM"
            return result

        # Construct arrays for weighted logistic regression
        # Y = variant allele frequency, X = is_case indicator, weights = AN
        n_obs = len(pop_data) + 1  # populations + cases

        y = np.zeros(n_obs)      # number of successes (ac)
        n = np.zeros(n_obs)      # number of trials (an)
        x = np.zeros(n_obs)      # 1 for cases, 0 for controls
        weights = np.zeros(n_obs)

        # Cases row
        y[0] = case_ac
        n[0] = case_an
        x[0] = 1.0
        weights[0] = case_an

        # Population rows
        for i, pd in enumerate(pop_data):
            y[i + 1] = pd["ac"]
            n[i + 1] = pd["an"]
            x[i + 1] = 0.0
            weights[i + 1] = pd["an"]
            result["population_weights"][pd["pop"]] = pd["an"]

        # Proportions
        props = y / n
        props = np.clip(props, 1e-10, 1 - 1e-10)  # avoid log(0)

        # Logit transform
        logit_y = np.log(props / (1 - props))

        # Weighted OLS on logit scale (approximate logistic)
        X = sm.add_constant(x)
        model = sm.WLS(logit_y, X, weights=weights)
        fit = model.fit()

        coef = fit.params[1]
        se = fit.bse[1]
        z = fit.tvalues[1]
        pval = fit.pvalues[1]

        result["coefficient"] = float(coef)
        result["std_error"] = float(se)
        result["z_score"] = float(z)
        result["p_value"] = float(pval)

        if pval < 0.05:
            direction = "enriched" if coef > 0 else "depleted"
            result["interpretation"] = (
                f"Significant: variant {direction} in cases vs gnomAD "
                f"(coef={coef:.3f}, z={z:.2f}, p={pval:.4f})"
            )
        else:
            result["interpretation"] = (
                f"Not significant (coef={coef:.3f}, z={z:.2f}, p={pval:.4f})"
            )

    except Exception as e:
        result["interpretation"] = f"Weighted GLM failed: {str(e)}"
        logger.warning("Weighted GLM failed: %s", e)

    return result


def run_case_control_analysis(
    case_carriers: int,
    case_total: int,
    gnomad_data: dict[str, Any],
    selected_datasets: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Run full case-control analysis.

    Args:
        case_carriers: Number of variant carriers in case cohort
        case_total: Total number of individuals in case cohort
        gnomad_data: gnomAD result dict with populations
        selected_datasets: Optional list of dataset IDs for multi-dataset comparison

    Returns:
        Dict with overall Fisher's, per-ancestry Fisher's, and weighted GLM results.
    """
    case_ac = case_carriers  # assume heterozygous
    case_an = case_total * 2  # diploid

    control_ac = gnomad_data.get("ac", 0)
    control_an = gnomad_data.get("an", 0)
    populations = gnomad_data.get("populations", {})

    result = {
        "case_carriers": case_carriers,
        "case_total": case_total,
        "case_af": case_ac / case_an if case_an > 0 else 0,
        "control_dataset": gnomad_data.get("dataset_label", "gnomAD"),
        "overall_fishers": fishers_exact_test(case_ac, case_an, control_ac, control_an),
        "ancestry_fishers": ancestry_specific_fishers(case_ac, case_an, populations),
        "weighted_glm": weighted_glm_analysis(case_ac, case_an, populations),
    }

    return result
