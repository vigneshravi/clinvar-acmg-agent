"""Streamlit UI for ClinVar ACMG Variant Classifier."""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from agent.agent import classify_variant_with_agent


def _get_star_count(review_status: str) -> int:
    """Convert review status string to star count."""
    if not review_status:
        return 0
    status_lower = review_status.lower()
    if "practice guideline" in status_lower:
        return 4
    if "expert panel" in status_lower:
        return 3
    if "multiple submitters, no conflicts" in status_lower:
        return 2
    if "single submitter" in status_lower or "conflicting" in status_lower:
        return 1
    return 0


# Page config
st.set_page_config(
    page_title="ClinVar ACMG Variant Classifier",
    page_icon="\U0001f9ec",
    layout="wide",
)

# Title and subtitle
st.title("\U0001f9ec ClinVar ACMG Variant Classifier")
st.markdown(
    "*AI-assisted research tool for variant classification using ACMG/AMP 2015 "
    "guidelines. **Not for clinical use.***"
)

st.divider()

# Input
variant_input = st.text_input(
    "Enter variant (e.g. BRCA1 c.5266dupC)",
    placeholder="BRCA1 c.5266dupC",
)

classify_button = st.button("Classify Variant", type="primary")

if classify_button and variant_input:
    with st.spinner("Querying ClinVar and running ACMG classification..."):
        try:
            result = classify_variant_with_agent(variant_input)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.stop()

    # Check for errors
    clinvar = result.get("clinvar_record", {})
    if result["classification"] == "Unable to classify":
        st.error(
            f"Could not classify variant: {result.get('reasoning', 'Unknown error')}"
        )
        st.stop()

    # Section 1: ClinVar Record
    with st.expander("\U0001f4cb ClinVar Record", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Gene:** {clinvar.get('gene', 'N/A')}")
            st.markdown(f"**HGVS:** {clinvar.get('hgvs', 'N/A')}")
            st.markdown(
                f"**Clinical Significance:** "
                f"{clinvar.get('clinical_significance', 'N/A')}"
            )
            st.markdown(
                f"**Condition:** {clinvar.get('condition', 'N/A')}"
            )

        with col2:
            # Star rating
            review_status = clinvar.get("review_status", "N/A")
            stars = _get_star_count(review_status)
            star_display = "\u2B50" * stars + "\u2606" * (4 - stars)
            st.markdown(f"**Review Status:** {star_display} ({review_status})")
            st.markdown(
                f"**Submitter Count:** {clinvar.get('submitter_count', 0)}"
            )
            st.markdown(
                f"**Last Evaluated:** "
                f"{clinvar.get('last_evaluated', 'N/A')}"
            )
            st.markdown(
                f"**Variant ID:** {clinvar.get('variant_id', 'N/A')}"
            )

        # Conflicting interpretations warning
        if clinvar.get("conflicting_interpretations"):
            st.warning(
                "\u26A0\uFE0F **Conflicting Interpretations:** Submitters "
                "disagree on the classification of this variant."
            )

        # Show raw submissions if available
        raw_subs = clinvar.get("raw_submissions", [])
        if raw_subs:
            with st.expander("View individual submitter classifications"):
                for i, sub in enumerate(raw_subs, 1):
                    st.markdown(f"{i}. {sub}")

    # Section 2: ACMG Criteria
    with st.expander("\U0001f3af ACMG Criteria Triggered", expanded=True):
        criteria = result.get("criteria_triggered", [])
        if not criteria:
            st.info("No ACMG criteria could be evaluated from available data.")
        else:
            for c in criteria:
                direction = c.get("direction", "")
                criterion = c.get("criterion", "")
                strength = c.get("strength", "")
                justification = c.get("justification", "")

                if direction == "pathogenic":
                    icon = "\U0001f534"
                else:
                    icon = "\U0001f7e2"

                st.markdown(
                    f"{icon} **{criterion}** "
                    f"({strength}) &mdash; {justification}"
                )

    # Section 3: Classification
    with st.expander("\U0001f3c6 Final Classification", expanded=True):
        classification = result.get("classification", "VUS")
        confidence = result.get("confidence", "Low")
        reasoning = result.get("reasoning", "")

        # Color mapping
        class_colors = {
            "Pathogenic": ("#ff4b4b", "#fff"),
            "Likely Pathogenic": ("#ff8c00", "#fff"),
            "VUS": ("#ffd700", "#333"),
            "Likely Benign": ("#90ee90", "#333"),
            "Benign": ("#4caf50", "#fff"),
        }
        bg_color, text_color = class_colors.get(
            classification, ("#ffd700", "#333")
        )

        st.markdown(
            f"""
            <div style="
                background-color: {bg_color};
                color: {text_color};
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                font-size: 24px;
                font-weight: bold;
                margin: 10px 0;
            ">
                {classification}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(f"**Confidence:** {confidence}")
        st.markdown(f"**Reasoning:** {reasoning}")

    # Disclaimer
    st.divider()
    st.markdown(
        f'<p style="color: gray; font-size: 12px;">{result.get("disclaimer", "")}</p>',
        unsafe_allow_html=True,
    )

elif classify_button and not variant_input:
    st.warning("Please enter a variant to classify.")
