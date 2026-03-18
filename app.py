"""Streamlit UI for ClinVar ACMG Variant Classifier — LangGraph multi-agent."""

import json
import threading

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from cache.cache_manager import CacheManager
from graph.graph import run_graph_stream
from graph.state import VariantState, make_initial_state

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_cache = CacheManager()


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


def _transcript_label(tx: dict) -> str:
    """Build display label for transcript dropdown."""
    nm = tx.get("nm_accession", "")
    enst = tx.get("enst_accession", "")

    # ID part: NM / ENST
    if nm and enst:
        id_part = f"{nm} / {enst}"
    else:
        id_part = nm or enst

    # Annotation tags
    tags = []
    if tx.get("is_mane_select"):
        tags.append("MANE Select")
    if tx.get("is_mane_plus_clinical"):
        tags.append("MANE+Clinical")
    if tx.get("is_most_reported_pathogenic"):
        tags.append("Most Reported")
    if tx.get("is_canonical") and not tx.get("is_mane_select"):
        tags.append("Canonical")

    # Consequence
    consequence = tx.get("consequence_display", "")
    pos_detail = tx.get("position_detail", "")

    parts = [id_part]
    if tags:
        parts.append(" \u00b7 ".join(tags))
    if consequence:
        parts.append(consequence)
    if pos_detail:
        parts.append(pos_detail)

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
# Input section
# ---------------------------------------------------------------------------

input_method = st.radio(
    "Input method", ["Gene + HGVS", "Genomic Coordinates"], horizontal=True
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
        genome_build_sel = st.selectbox("Genome build", genome_build_options, key="build_hgvs")
    raw_input_str = (variant_text or "").strip()
else:
    genome_build_sel = st.selectbox("Genome build", genome_build_options, key="build_coord")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        chrom_val = st.text_input("Chromosome", placeholder="17", key="coord_chr")
    with c2:
        pos_val = st.text_input("Position", placeholder="43057062", key="coord_pos")
    with c3:
        ref_val = st.text_input("Ref", placeholder="G", key="coord_ref")
    with c4:
        alt_val = st.text_input("Alt", placeholder="GG", key="coord_alt")
    if chrom_val and pos_val and ref_val and alt_val:
        c = chrom_val.strip().lstrip("chr")
        raw_input_str = f"chr{c}:{pos_val.strip()}:{ref_val.strip().upper()}:{alt_val.strip().upper()}"

genome_build = "GRCh37" if "37" in genome_build_sel else "GRCh38"

# ---------------------------------------------------------------------------
# Look Up Transcripts button
# ---------------------------------------------------------------------------

lookup_button = st.button(
    "Look Up Transcripts", disabled=not raw_input_str, type="secondary"
)

if lookup_button and raw_input_str:
    from agents.input_parser import input_parser_node

    init = make_initial_state(raw_input_str)
    init["genome_build"] = genome_build
    with st.spinner("Resolving transcripts via Ensembl VEP..."):
        parser_output = input_parser_node(init)

    st.session_state["parser_output"] = parser_output
    st.session_state["raw_input_str"] = raw_input_str
    st.session_state["genome_build"] = genome_build
    # Clear any previous classification
    if "final_state" in st.session_state:
        del st.session_state["final_state"]

# ---------------------------------------------------------------------------
# Show transcript dropdown if we have results
# ---------------------------------------------------------------------------

if "parser_output" in st.session_state:
    po = st.session_state["parser_output"]
    transcripts = po.get("all_transcripts") or []

    if po.get("input_parse_error") and not po.get("gene_symbol"):
        st.error(f"Parse error: {po['input_parse_error']}")
    elif transcripts:
        gene_sym = po.get("gene_symbol", "")
        full_name = po.get("gene_full_name", "")
        aliases = po.get("gene_aliases", [])
        sel_tx_id = po.get("selected_transcript", "")

        # Find why it was pre-selected
        top = transcripts[0]
        top_tags = []
        if top.get("is_mane_select"):
            top_tags.append("MANE Select")
        if top.get("is_most_reported_pathogenic"):
            top_tags.append("Most Reported Pathogenic")
        if top.get("is_canonical"):
            top_tags.append("Canonical")

        st.info(
            f"**Gene:** {full_name} ({gene_sym})  \n"
            f"**Aliases:** {', '.join(aliases) if aliases else 'N/A'}  \n"
            f"**Transcripts found:** {len(transcripts)}  \n"
            f"**Pre-selected:** {sel_tx_id}"
            + (f" \u2014 {', '.join(top_tags)}" if top_tags else "")
        )

        # Transcript dropdown
        options = [_transcript_label(tx) for tx in transcripts]
        selected_idx = st.selectbox(
            "Select transcript",
            range(len(options)),
            format_func=lambda i: options[i],
            key="tx_select",
        )
        if selected_idx is not None:
            st.session_state["chosen_transcript"] = transcripts[selected_idx]

        # Show variant annotation for selected transcript
        chosen = transcripts[selected_idx] if selected_idx is not None else top
        annot_cols = st.columns(4)
        with annot_cols[0]:
            pos_type = chosen.get("position_type", "unknown")
            icon = "\U0001f7e2" if pos_type == "exonic" else "\U0001f7e1" if pos_type == "intronic" else "\u26AA"
            st.metric("Position", f"{icon} {chosen.get('position_detail', 'N/A')}")
        with annot_cols[1]:
            st.metric("Consequence", chosen.get("consequence_display", "N/A"))
        with annot_cols[2]:
            st.metric("Impact", chosen.get("impact", "N/A"))
        with annot_cols[3]:
            aa = chosen.get("amino_acids", "")
            st.metric("Amino Acid Change", aa if aa else "N/A")
    else:
        st.warning("No transcripts found. Will proceed with raw input.")

# ---------------------------------------------------------------------------
# Classify Variant button
# ---------------------------------------------------------------------------

classify_button = st.button(
    "Classify Variant", type="primary", disabled=not raw_input_str
)

if classify_button and raw_input_str:
    initial_state = make_initial_state(raw_input_str)
    initial_state["genome_build"] = genome_build

    # If user looked up transcripts and chose one, inject into state
    if "chosen_transcript" in st.session_state and "parser_output" in st.session_state:
        ct = st.session_state["chosen_transcript"]
        po = st.session_state["parser_output"]

        # Use the HGVS-c from VEP for the chosen transcript
        hgvsc = ct.get("hgvsc", "")
        nm = ct.get("nm_accession", "")
        enst = ct.get("enst_accession", "")
        tx_id = nm or enst

        if hgvsc:
            initial_state["hgvs_on_transcript"] = hgvsc
        if tx_id:
            initial_state["selected_transcript"] = tx_id
        if po.get("gene_symbol"):
            initial_state["gene_symbol"] = po["gene_symbol"]
        if po.get("gene_aliases"):
            initial_state["gene_aliases"] = po["gene_aliases"]
        if po.get("gene_full_name"):
            initial_state["gene_full_name"] = po["gene_full_name"]
        if po.get("all_transcripts"):
            initial_state["all_transcripts"] = po["all_transcripts"]

    final_state = dict(initial_state)

    # Progress tracking
    with st.status("Running classification pipeline...", expanded=True) as status:
        try:
            for node_name, node_output in run_graph_stream(initial_state):
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

    st.session_state["final_state"] = final_state

# ---------------------------------------------------------------------------
# Show results if we have them
# ---------------------------------------------------------------------------

if "final_state" in st.session_state:
    final_state = st.session_state["final_state"]

    if not final_state.get("classification"):
        st.error("No classification produced.")
        st.stop()

    # --- Variant summary bar ---
    gene_sym = final_state.get("gene_symbol", "")
    hgvs_disp = final_state.get("hgvs_on_transcript", "")
    build_disp = final_state.get("genome_build", "")
    sel_tx = final_state.get("selected_transcript", "")
    summary_parts = [p for p in [gene_sym, hgvs_disp, build_disp, sel_tx] if p]

    # Add VEP annotation to summary if available
    all_tx = final_state.get("all_transcripts") or []
    for t in all_tx:
        tid = t.get("nm_accession") or t.get("enst_accession")
        if tid == sel_tx:
            if t.get("consequence_display"):
                summary_parts.append(t["consequence_display"])
            if t.get("position_detail"):
                summary_parts.append(t["position_detail"])
            break
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

            # Show VEP annotation alongside ClinVar
            if all_tx:
                for t in all_tx:
                    tid = t.get("nm_accession") or t.get("enst_accession")
                    if tid == sel_tx:
                        st.divider()
                        vc1, vc2, vc3, vc4 = st.columns(4)
                        with vc1:
                            st.markdown(f"**Position:** {t.get('position_detail', 'N/A')}")
                        with vc2:
                            st.markdown(f"**Consequence:** {t.get('consequence_display', 'N/A')}")
                        with vc3:
                            st.markdown(f"**Impact:** {t.get('impact', 'N/A')}")
                        with vc4:
                            aa = t.get("amino_acids", "")
                            st.markdown(f"**AA Change:** {aa if aa else 'N/A'}")
                        break

    # --- Section 2: gnomAD + In Silico Predictors ---
    gnomad = final_state.get("gnomad")
    if gnomad and isinstance(gnomad, dict):
        with st.expander("\U0001f4ca Population Frequencies & In Silico Predictors", expanded=True):
            af_data = gnomad.get("allele_frequency", {})
            predictors = gnomad.get("insilico_predictors", {})
            conservation = gnomad.get("conservation", {})
            acmg_crit = gnomad.get("acmg_criteria", {})

            # Frequency summary
            st.markdown("#### gnomAD Allele Frequencies")
            if af_data.get("variant_in_gnomad"):
                fc1, fc2, fc3, fc4 = st.columns(4)
                with fc1:
                    gaf = af_data.get("global_af")
                    st.metric("Global AF", f"{gaf:.6f}" if gaf else "N/A")
                with fc2:
                    maf = af_data.get("max_pop_af", 0)
                    mpn = af_data.get("max_pop_name", "")
                    st.metric("Max Pop AF", f"{maf:.6f}" if maf else "0", delta=mpn)
                with fc3:
                    st.metric("Homozygotes", af_data.get("hom", 0))
                with fc4:
                    rsid = gnomad.get("rsid", "")
                    st.metric("rsID", rsid if rsid else "N/A")

                # Population breakdown table
                pops = af_data.get("populations", {})
                if pops:
                    pop_rows = []
                    for pid, pdata in sorted(pops.items()):
                        if isinstance(pdata, dict):
                            pop_rows.append({
                                "Population": pdata.get("name", pid),
                                "AF": f"{pdata.get('af', 0):.6f}",
                                "AC": pdata.get("ac", 0),
                                "Hom": pdata.get("hom", 0),
                            })
                    if pop_rows:
                        st.dataframe(pop_rows, use_container_width=True, hide_index=True)
            else:
                st.info("Variant **not found** in gnomAD — absent from population controls (supports PM2)")

            # Frequency ACMG flags
            freq_cols = st.columns(3)
            for i, (code, label) in enumerate([("BA1", "AF>5%"), ("BS1", "AF>1%"), ("PM2", "Absent/rare")]):
                with freq_cols[i]:
                    met = acmg_crit.get(f"{code}_met", False)
                    icon = "\u2705" if met else "\u274C"
                    st.markdown(f"{icon} **{code}** ({label})")

            # In silico predictors
            if predictors:
                st.markdown("#### In Silico Predictors")
                pred_rows = []
                for name, val in predictors.items():
                    if isinstance(val, dict):
                        pred_rows.append({
                            "Predictor": name.upper(),
                            "Score": val.get("score", val.get("interpretation", "")),
                            "Prediction": val.get("pred", val.get("interpretation", "")),
                        })
                    else:
                        pred_rows.append({"Predictor": name.upper(), "Score": val, "Prediction": ""})
                if pred_rows:
                    st.dataframe(pred_rows, use_container_width=True, hide_index=True)

                consensus = gnomad.get("insilico_consensus", "")
                if consensus:
                    color = {"Damaging": "red", "Benign": "green"}.get(consensus, "orange")
                    st.markdown(f"**Consensus:** :{color}[{consensus}]")

                pp3_met = acmg_crit.get("PP3_met", False)
                bp4_met = acmg_crit.get("BP4_met", False)
                p1, p2 = st.columns(2)
                with p1:
                    st.markdown(f"{'\u2705' if pp3_met else '\u274C'} **PP3** (computational damaging)")
                with p2:
                    st.markdown(f"{'\u2705' if bp4_met else '\u274C'} **BP4** (computational benign)")

            # Conservation
            if conservation:
                st.markdown("#### Conservation")
                cons_cols = st.columns(len(conservation))
                for i, (name, val) in enumerate(conservation.items()):
                    with cons_cols[i]:
                        st.metric(name, f"{val:.3f}" if isinstance(val, float) else str(val))

    # --- Section 3: ACMG Criteria ---
    with st.expander("\U0001f3af ACMG Criteria", expanded=True):
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
                    st.markdown(f"{icon} **{c.get('code', '')}** ({c.get('strength', '')}) "
                                f"&mdash; {c.get('justification', '')}")
            if unmet:
                with st.expander(f"Criteria Not Met / Not Evaluable ({len(unmet)})"):
                    for c in unmet:
                        st.markdown(f"\u2B1C **{c.get('code', '')}** &mdash; {c.get('justification', '')}")

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
        with st.expander(f"\u26A0\uFE0F Warnings ({len(warnings)})"):
            for w in warnings:
                st.markdown(f"- {w}")

    # --- Disclaimer ---
    st.divider()
    disclaimer = final_state.get("disclaimer", "")
    if disclaimer:
        st.markdown(f'<p style="color:gray;font-size:12px">{disclaimer}</p>', unsafe_allow_html=True)

    # --- Debug ---
    with st.expander("\U0001f41b Debug: Full State JSON"):
        debug = {}
        for k, v in final_state.items():
            try:
                json.dumps(v)
                debug[k] = v
            except (TypeError, ValueError):
                debug[k] = str(v)
        st.json(debug)
