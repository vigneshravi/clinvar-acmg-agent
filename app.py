"""Streamlit UI for ClinVar ACMG Variant Classifier — LangGraph multi-agent."""

import json
import threading

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from cache.cache_manager import HEREDITARY_CANCER_GENES, CacheManager
from graph.graph import run_graph_stream
from graph.state import VariantState, make_initial_state

# ---------------------------------------------------------------------------
# Background cache prewarm
# ---------------------------------------------------------------------------
_cache = CacheManager()
if "cache_started" not in st.session_state:
    st.session_state.cache_started = True
    if _cache.needs_prewarm(HEREDITARY_CANCER_GENES, "GRCh38"):
        thread = threading.Thread(
            target=_cache.prewarm,
            args=(HEREDITARY_CANCER_GENES, "GRCh38"),
            daemon=True,
        )
        thread.start()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_star_count(review_status: str) -> int:
    if not review_status:
        return 0
    s = review_status.lower()
    if "practice guideline" in s:
        return 4
    if "expert panel" in s:
        return 3
    if "multiple submitters, no conflicts" in s:
        return 2
    if "single submitter" in s or "conflicting" in s:
        return 1
    return 0


def _transcript_label(tx: dict, id_format: str = "Both") -> str:
    """Build a display label for a transcript dropdown option."""
    nm = tx.get("nm_accession", "")
    enst = tx.get("enst_accession", "")
    gene = tx.get("gene_symbol", "")
    equiv = tx.get("equivalent_hgvs", "")

    if id_format == "RefSeq (NM_)":
        id_part = nm or enst
    elif id_format == "Ensembl (ENST)":
        id_part = enst or nm
    else:
        id_part = f"{nm} / {enst}" if nm and enst else (nm or enst)

    tags = []
    if tx.get("is_mane_select"):
        tags.append("MANE Select")
    if tx.get("is_mane_plus_clinical"):
        tags.append("MANE Plus Clinical")
    if tx.get("is_most_reported_pathogenic"):
        tags.append("Most Reported Pathogenic")
    tag_str = " \u00b7 ".join(tags)

    parts = [id_part]
    if gene:
        aliases = tx.get("gene_aliases", [])
        alias_str = f" ({', '.join(aliases[:3])})" if aliases else ""
        parts.append(f"{gene}{alias_str}")
    if tag_str:
        parts.append(tag_str)
    if equiv:
        parts.append(equiv)

    return " | ".join(parts)


NODE_LABELS = {
    "input_parser": "Parse Input",
    "supervisor": "Route",
    "clinvar_agent": "ClinVar",
    "gnomad_agent": "gnomAD",
    "pubmed_agent": "PubMed",
    "alphafold_agent": "AlphaFold",
    "tcga_agent": "TCGA",
    "acmg_classifier": "ACMG Classifier",
}


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ClinVar ACMG Variant Classifier",
    page_icon="\U0001f9ec",
    layout="wide",
)

st.title("\U0001f9ec ClinVar ACMG Variant Classifier")
st.markdown(
    "*AI-assisted research tool for variant classification using ACMG/AMP 2015 "
    "guidelines. Multi-agent LangGraph pipeline. **Not for clinical use.***"
)
st.divider()


# ---------------------------------------------------------------------------
# Input section — two-mode input
# ---------------------------------------------------------------------------

input_method = st.radio(
    "Input method",
    ["Gene + HGVS", "Genomic Coordinates"],
    horizontal=True,
)

genome_build_options = ["GRCh38 / hg38", "GRCh37 / hg19"]
raw_input_str = ""

if input_method == "Gene + HGVS":
    col_input, col_build = st.columns([3, 1])
    with col_input:
        variant_text = st.text_input(
            "Gene + HGVS notation",
            placeholder="e.g. BRCA1 c.5266dupC or NM_007294.4:c.5266dupC",
            key="hgvs_input",
        )
    with col_build:
        genome_build_sel = st.selectbox(
            "Genome build", genome_build_options, key="build_hgvs"
        )
    raw_input_str = variant_text.strip() if variant_text else ""
else:
    genome_build_sel = st.selectbox(
        "Genome build", genome_build_options, key="build_coord"
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        chrom = st.text_input("Chromosome", placeholder="17", key="coord_chr")
    with c2:
        pos = st.text_input("Position", placeholder="43057062", key="coord_pos")
    with c3:
        ref = st.text_input("Ref", placeholder="G", key="coord_ref")
    with c4:
        alt = st.text_input("Alt", placeholder="GG", key="coord_alt")
    if chrom and pos and ref and alt:
        raw_input_str = f"chr{chrom.strip().lstrip('chr')}:{pos.strip()}:{ref.strip().upper()}:{alt.strip().upper()}"

genome_build = "GRCh37" if "37" in genome_build_sel else "GRCh38"

# Transcript ID format toggle
id_format = st.radio(
    "Show transcript IDs as",
    ["Both", "RefSeq (NM_)", "Ensembl (ENST)"],
    horizontal=True,
    key="id_format",
)

# ---------------------------------------------------------------------------
# Look Up Transcripts button
# ---------------------------------------------------------------------------

lookup_col, classify_col = st.columns(2)

with lookup_col:
    lookup_button = st.button("Look Up Transcripts", disabled=not raw_input_str)

with classify_col:
    classify_button = st.button(
        "Classify Variant",
        type="primary",
        disabled=not raw_input_str,
    )

# Handle transcript lookup (runs input_parser only)
if lookup_button and raw_input_str:
    from agents.input_parser import input_parser_node

    init = make_initial_state(raw_input_str)
    init["genome_build"] = genome_build
    with st.spinner("Looking up transcripts..."):
        parser_output = input_parser_node(init)

    st.session_state["parser_output"] = parser_output
    st.session_state["raw_input_str"] = raw_input_str
    st.session_state["genome_build"] = genome_build

# Show transcript dropdown if we have results
if "parser_output" in st.session_state:
    po = st.session_state["parser_output"]
    transcripts = po.get("all_transcripts") or []

    if po.get("input_parse_error") and not po.get("gene_symbol"):
        st.error(f"Parse error: {po['input_parse_error']}")
    elif transcripts:
        # Info box
        gene_sym = po.get("gene_symbol", "")
        full_name = po.get("gene_full_name", "")
        aliases = po.get("gene_aliases", [])
        selected_tx = po.get("selected_transcript", "")

        # Find why selected
        top = transcripts[0] if transcripts else {}
        tags = []
        if top.get("is_mane_select"):
            tags.append("MANE Select")
        if top.get("is_mane_plus_clinical"):
            tags.append("MANE Plus Clinical")
        if top.get("is_most_reported_pathogenic"):
            tags.append("Most Reported Pathogenic")

        st.info(
            f"**Gene:** {full_name} ({gene_sym})  \n"
            f"**Aliases:** {', '.join(aliases) if aliases else 'N/A'}  \n"
            f"**Transcripts found:** {len(transcripts)}  \n"
            f"**Pre-selected:** {selected_tx}"
            + (f" \u2014 reason: {', '.join(tags)}" if tags else "")
        )

        # Dropdown
        options = [_transcript_label(tx, id_format) for tx in transcripts]
        selected_idx = st.selectbox(
            "Select transcript",
            range(len(options)),
            format_func=lambda i: options[i],
            key="tx_select",
        )
        if selected_idx is not None:
            chosen = transcripts[selected_idx]
            st.session_state["chosen_transcript"] = chosen
    else:
        st.warning(
            f"No Ensembl transcripts found. Will use: {po.get('hgvs_on_transcript', raw_input_str)}"
        )


# ---------------------------------------------------------------------------
# Classify Variant (full pipeline)
# ---------------------------------------------------------------------------

if classify_button and raw_input_str:
    initial_state = make_initial_state(raw_input_str)
    initial_state["genome_build"] = genome_build

    # If user chose a specific transcript, override
    if "chosen_transcript" in st.session_state:
        ct = st.session_state["chosen_transcript"]
        equiv = ct.get("equivalent_hgvs", "")
        if equiv:
            initial_state["raw_input"] = equiv

    final_state = dict(initial_state)
    completed_nodes = []

    # Progress tracking
    with st.status("Running classification pipeline...", expanded=True) as status:
        try:
            for node_name, node_output in run_graph_stream(initial_state):
                completed_nodes.append(node_name)
                final_state.update(node_output)
                label = NODE_LABELS.get(node_name, node_name)
                st.write(f"\u2705 **{label}** completed")

                if node_name == "supervisor" and final_state.get("input_parse_error"):
                    if not final_state.get("gene_symbol"):
                        status.update(label="Pipeline stopped", state="error")
                        st.error(f"Parse error: {final_state['input_parse_error']}")
                        st.stop()

            status.update(label="Pipeline complete!", state="complete")
        except Exception as e:
            status.update(label="Pipeline error", state="error")
            st.error(f"An error occurred: {str(e)}")
            st.stop()

    if not final_state.get("classification"):
        st.error("No classification produced.")
        st.stop()

    # --- Variant summary bar ---
    gene_sym = final_state.get("gene_symbol", "")
    hgvs_disp = final_state.get("hgvs_on_transcript", "")
    build_disp = final_state.get("genome_build", "")
    sel_tx = final_state.get("selected_transcript", "")

    # Gather annotation tags from selected transcript
    all_tx = final_state.get("all_transcripts") or []
    sel_tags = []
    for t in all_tx:
        if t.get("nm_accession") == sel_tx or t.get("enst_accession") == sel_tx:
            if t.get("is_mane_select"):
                sel_tags.append("MANE Select")
            if t.get("is_mane_plus_clinical"):
                sel_tags.append("MANE+Clinical")
            if t.get("is_most_reported_pathogenic"):
                sel_tags.append("Most Reported")
            break

    summary_parts = [p for p in [gene_sym, hgvs_disp, build_disp, sel_tx] if p]
    if sel_tags:
        summary_parts.append(" \u00b7 ".join(sel_tags))
    st.markdown(f"**{' | '.join(summary_parts)}**")

    # --- Section 1: ClinVar Record ---
    clinvar = final_state.get("clinvar") or {}
    with st.expander("\U0001f4cb ClinVar Record", expanded=True):
        if not clinvar:
            st.warning(final_state.get("clinvar_error", "No ClinVar data"))
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Gene:** {clinvar.get('gene', 'N/A')}")
                st.markdown(f"**HGVS:** {clinvar.get('hgvs', 'N/A')}")
                st.markdown(f"**Clinical Significance:** {clinvar.get('clinical_significance', 'N/A')}")
                st.markdown(f"**Condition:** {clinvar.get('condition', 'N/A')}")
            with col2:
                rs = clinvar.get("review_status", "N/A")
                stars = _get_star_count(rs)
                st.markdown(f"**Review Status:** {'\u2B50' * stars}{'\u2606' * (4 - stars)} ({rs})")
                st.markdown(f"**Submitter Count:** {clinvar.get('submitter_count', 0)}")
                st.markdown(f"**Last Evaluated:** {clinvar.get('last_evaluated', 'N/A')}")
                st.markdown(f"**Variant ID:** {clinvar.get('variant_id', 'N/A')}")

            if clinvar.get("conflicting_interpretations"):
                st.warning("\u26A0\uFE0F **Conflicting Interpretations**")

    # --- Section 2: ACMG Criteria ---
    with st.expander("\U0001f3af ACMG Criteria Triggered", expanded=True):
        criteria = final_state.get("criteria_triggered") or []
        met_criteria = [c for c in criteria if c.get("met", True)]
        unmet = [c for c in criteria if not c.get("met", True)]

        if not met_criteria and not unmet:
            st.info("No ACMG criteria evaluated.")
        else:
            if met_criteria:
                st.markdown("**Criteria Met:**")
                for c in met_criteria:
                    d = c.get("direction", "")
                    icon = "\U0001f534" if d == "pathogenic" else "\U0001f7e2"
                    code = c.get("code", "")
                    strength = c.get("strength", "")
                    just = c.get("justification", "")
                    st.markdown(f"{icon} **{code}** ({strength}) &mdash; {just}")

            if unmet:
                with st.expander(f"Criteria Not Met / Not Evaluable ({len(unmet)})", expanded=False):
                    for c in unmet:
                        code = c.get("code", "")
                        just = c.get("justification", "")
                        st.markdown(f"\u2B1C **{code}** &mdash; {just}")

    # --- Section 3: Classification ---
    with st.expander("\U0001f3c6 Final Classification", expanded=True):
        classification = final_state.get("classification", "VUS")
        confidence = final_state.get("confidence", "Low")
        reasoning = final_state.get("reasoning", "")

        colors = {
            "Pathogenic": ("#ff4b4b", "#fff"),
            "Likely Pathogenic": ("#ff8c00", "#fff"),
            "VUS": ("#ffd700", "#333"),
            "Likely Benign": ("#90ee90", "#333"),
            "Benign": ("#4caf50", "#fff"),
        }
        bg, fg = colors.get(classification, ("#ffd700", "#333"))

        st.markdown(
            f'<div style="background:{bg};color:{fg};padding:20px;border-radius:10px;'
            f'text-align:center;font-size:24px;font-weight:bold;margin:10px 0">'
            f'{classification}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Confidence:** {confidence}")
        st.markdown(f"**Reasoning:** {reasoning}")

    # --- Warnings ---
    warnings = final_state.get("warnings", [])
    if warnings:
        with st.expander(f"\u26A0\uFE0F Warnings ({len(warnings)})", expanded=False):
            for w in warnings:
                st.markdown(f"- {w}")

    # --- Disclaimer ---
    st.divider()
    disclaimer = final_state.get("disclaimer", "")
    if disclaimer:
        st.markdown(
            f'<p style="color:gray;font-size:12px">{disclaimer}</p>',
            unsafe_allow_html=True,
        )

    # --- Debug ---
    with st.expander("\U0001f41b Debug: Full State JSON", expanded=False):
        debug = {}
        for k, v in final_state.items():
            try:
                json.dumps(v)
                debug[k] = v
            except (TypeError, ValueError):
                debug[k] = str(v)
        st.json(debug)
