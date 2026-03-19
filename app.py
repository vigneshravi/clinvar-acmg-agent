"""Streamlit UI for ClinVar ACMG Variant Classifier — LangGraph multi-agent."""

import json
import math

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from cache.cache_manager import CacheManager
from graph.graph import run_graph_stream
from graph.state import VariantState, make_initial_state

_cache = CacheManager()

# ---------------------------------------------------------------------------
# Consistent CSS — no big/small font inconsistency
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Override st.metric to use consistent sizing */
    [data-testid="stMetricValue"] { font-size: 1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.85rem !important; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem !important; }
    /* Consistent markdown sizing */
    .stMarkdown p { font-size: 0.95rem; }
    .stMarkdown h4 { font-size: 1.1rem !important; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Link helpers
# ---------------------------------------------------------------------------

def _clinvar_link(variant_id: str) -> str:
    if not variant_id:
        return "N/A"
    return f"[{variant_id}](https://www.ncbi.nlm.nih.gov/clinvar/variation/{variant_id}/)"

def _gnomad_link(variant_id: str, dataset: str = "gnomad_r4") -> str:
    if not variant_id:
        return "N/A"
    return f"[{variant_id}](https://gnomad.broadinstitute.org/variant/{variant_id}?dataset={dataset})"

def _dbsnp_link(rsid: str) -> str:
    if not rsid:
        return "N/A"
    return f"[{rsid}](https://www.ncbi.nlm.nih.gov/snp/{rsid})"

def _gene_link(gene: str) -> str:
    if not gene:
        return "N/A"
    return f"[{gene}](https://www.ncbi.nlm.nih.gov/gene/?term={gene}%5BGENE%5D+AND+human%5BORGN%5D)"

def _get_star_count(review_status: str) -> int:
    if not review_status:
        return 0
    s = review_status.lower()
    if "practice guideline" in s: return 4
    if "expert panel" in s: return 3
    if "multiple submitters, no conflicts" in s: return 2
    if "single submitter" in s or "conflicting" in s: return 1
    return 0

def _transcript_label(tx: dict) -> str:
    nm = tx.get("nm_accession", "")
    enst = tx.get("enst_accession", "")
    id_part = f"{nm} / {enst}" if nm and enst else (nm or enst)
    tags = []
    if tx.get("is_mane_select"): tags.append("MANE Select")
    if tx.get("is_mane_plus_clinical"): tags.append("MANE+Clinical")
    if tx.get("is_most_reported_pathogenic"): tags.append("Most Reported")
    if tx.get("is_canonical") and not tx.get("is_mane_select"): tags.append("Canonical")
    consequence = tx.get("consequence_display", "")
    pos_detail = tx.get("position_detail", "")
    parts = [id_part]
    if tags: parts.append(" \u00b7 ".join(tags))
    if consequence: parts.append(consequence)
    if pos_detail: parts.append(pos_detail)
    return " | ".join(parts)

NODE_LABELS = {
    "input_parser": "Parse Input", "supervisor": "Route",
    "clinvar_agent": "ClinVar", "gnomad_agent": "gnomAD + In Silico",
    "pubmed_agent": "PubMed", "alphafold_agent": "AlphaFold",
    "tcga_agent": "TCGA", "acmg_classifier": "ACMG Classifier",
}

POP_ORDER = ["afr", "amr", "asj", "eas", "fin", "mid", "nfe", "sas", "remaining", "ami"]
POP_NAMES = {
    "afr": "African/African American", "amr": "Latino/Admixed American",
    "asj": "Ashkenazi Jewish", "eas": "East Asian", "fin": "Finnish",
    "mid": "Middle Eastern", "nfe": "Non-Finnish European", "sas": "South Asian",
    "remaining": "Remaining", "ami": "Amish",
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="ClinVar ACMG Variant Classifier", page_icon="\U0001f9ec", layout="wide")
st.title("\U0001f9ec ClinVar ACMG Variant Classifier")
st.markdown("*AI-assisted variant classification using ACMG/AMP 2015 guidelines. **Not for clinical use.***")
st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
input_method = st.radio("Input method", ["Gene + HGVS", "Genomic Coordinates"], horizontal=True)
genome_build_options = ["GRCh38 / hg38", "GRCh37 / hg19"]
raw_input_str = ""

if input_method == "Gene + HGVS":
    c_in, c_bld = st.columns([3, 1])
    with c_in:
        variant_text = st.text_input("Gene + HGVS notation", placeholder="e.g. BRCA1 c.5266dupC", key="hgvs_input")
    with c_bld:
        genome_build_sel = st.selectbox("Genome build", genome_build_options, key="build_hgvs")
    raw_input_str = (variant_text or "").strip()
else:
    genome_build_sel = st.selectbox("Genome build", genome_build_options, key="build_coord")
    c1, c2, c3, c4 = st.columns(4)
    with c1: chrom_val = st.text_input("Chromosome", placeholder="17", key="coord_chr")
    with c2: pos_val = st.text_input("Position", placeholder="43057062", key="coord_pos")
    with c3: ref_val = st.text_input("Ref", placeholder="T", key="coord_ref")
    with c4: alt_val = st.text_input("Alt", placeholder="TG", key="coord_alt")
    if chrom_val and pos_val and ref_val and alt_val:
        raw_input_str = f"chr{chrom_val.strip().lstrip('chr')}:{pos_val.strip()}:{ref_val.strip().upper()}:{alt_val.strip().upper()}"

genome_build = "GRCh37" if "37" in genome_build_sel else "GRCh38"

# ---------------------------------------------------------------------------
# Look Up Transcripts
# ---------------------------------------------------------------------------
lookup_btn = st.button("Look Up Transcripts", disabled=not raw_input_str, type="secondary")
if lookup_btn and raw_input_str:
    from agents.input_parser import input_parser_node
    init = make_initial_state(raw_input_str)
    init["genome_build"] = genome_build
    with st.spinner("Resolving transcripts via Ensembl VEP..."):
        parser_output = input_parser_node(init)
    st.session_state["parser_output"] = parser_output
    st.session_state["raw_input_str"] = raw_input_str
    st.session_state["genome_build"] = genome_build
    if "final_state" in st.session_state:
        del st.session_state["final_state"]

# ---------------------------------------------------------------------------
# Transcript results
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
        top = transcripts[0]
        top_tags = []
        if top.get("is_mane_select"): top_tags.append("MANE Select")
        if top.get("is_most_reported_pathogenic"): top_tags.append("Most Reported Pathogenic")
        if top.get("is_canonical"): top_tags.append("Canonical")

        st.info(
            f"**Gene:** {full_name} ({_gene_link(gene_sym)})  \n"
            f"**Aliases:** {', '.join(aliases) if aliases else 'N/A'}  \n"
            f"**Transcripts found:** {len(transcripts)}  \n"
            f"**Pre-selected:** {sel_tx_id}"
            + (f" \u2014 {', '.join(top_tags)}" if top_tags else "")
        )

        options = [_transcript_label(tx) for tx in transcripts]
        selected_idx = st.selectbox("Select transcript", range(len(options)), format_func=lambda i: options[i], key="tx_select")
        if selected_idx is not None:
            st.session_state["chosen_transcript"] = transcripts[selected_idx]

        chosen = transcripts[selected_idx] if selected_idx is not None else top
        po_chrom = po.get("chrom")
        po_pos = po.get("pos")
        po_ref = po.get("ref")
        po_alt = po.get("alt")
        strand_val = chosen.get("strand")
        strand_str = "+" if strand_val == 1 else "\u2212" if strand_val == -1 else ""

        # Unified annotation bar — consistent markdown, no st.metric
        if po_chrom and po_pos:
            st.markdown(
                f"**Chr:** chr{po_chrom} &nbsp;|&nbsp; "
                f"**Pos:** {po_pos:,} &nbsp;|&nbsp; "
                f"**Ref/Alt:** {po_ref} \u2192 {po_alt} &nbsp;|&nbsp; "
                f"**Strand:** {strand_str} &nbsp;|&nbsp; "
                f"**Build:** {genome_build} &nbsp;|&nbsp; "
                f"**{chosen.get('position_detail', '')}** &nbsp;|&nbsp; "
                f"**{chosen.get('consequence_display', '')}** &nbsp;|&nbsp; "
                f"**Impact:** {chosen.get('impact', '')} &nbsp;|&nbsp; "
                f"**AA:** {chosen.get('amino_acids', '') or 'N/A'}"
            )

            # PVS1 caveat warning for last/penultimate exon truncating variants
            from agents.acmg_classifier import _assess_pvs1_applicability
            pvs1_info = _assess_pvs1_applicability(chosen)
            if pvs1_info["is_null_variant"]:
                if pvs1_info.get("is_last_exon"):
                    st.warning(
                        f"\u26A0\uFE0F **PVS1 Caveat — Last exon variant** "
                        f"(exon {pvs1_info['exon_number']}/{pvs1_info['total_exons']}): "
                        f"Truncating variant may escape NMD. PVS1 downgraded to **Moderate**."
                    )
                elif pvs1_info.get("is_penultimate_exon"):
                    st.warning(
                        f"\u26A0\uFE0F **PVS1 Caveat — Penultimate exon** "
                        f"(exon {pvs1_info['exon_number']}/{pvs1_info['total_exons']}): "
                        f"If in last 50bp, may escape NMD. PVS1 downgraded to **Strong**."
                    )
                elif pvs1_info["total_exons"] == 1:
                    st.warning(
                        "\u26A0\uFE0F **PVS1 Caveat — Single-exon gene**: "
                        "NMD not applicable. PVS1 downgraded to **Moderate**."
                    )
    else:
        st.warning("No transcripts found. Will proceed with raw input.")

# ---------------------------------------------------------------------------
# Classify Variant
# ---------------------------------------------------------------------------
classify_btn = st.button("Classify Variant", type="primary", disabled=not raw_input_str)
if classify_btn and raw_input_str:
    initial_state = make_initial_state(raw_input_str)
    initial_state["genome_build"] = genome_build
    if "chosen_transcript" in st.session_state and "parser_output" in st.session_state:
        ct = st.session_state["chosen_transcript"]
        po = st.session_state["parser_output"]
        if ct.get("hgvsc"): initial_state["hgvs_on_transcript"] = ct["hgvsc"]
        tx_id = ct.get("nm_accession") or ct.get("enst_accession")
        if tx_id: initial_state["selected_transcript"] = tx_id
        for k in ["gene_symbol", "gene_aliases", "gene_full_name", "all_transcripts",
                   "chrom", "pos", "ref", "alt"]:
            if po.get(k): initial_state[k] = po[k]

    final_state = dict(initial_state)
    with st.status("Running classification pipeline...", expanded=True) as status:
        try:
            for node_name, node_output in run_graph_stream(initial_state):
                final_state.update(node_output)
                st.write(f"\u2705 **{NODE_LABELS.get(node_name, node_name)}** completed")
                if node_name == "supervisor" and final_state.get("input_parse_error"):
                    if not final_state.get("gene_symbol"):
                        status.update(label="Pipeline stopped", state="error")
                        st.error(f"Parse error: {final_state['input_parse_error']}")
                        st.stop()
            status.update(label="Pipeline complete!", state="complete")
        except Exception as e:
            status.update(label="Pipeline error", state="error")
            st.error(f"Error: {e}")
            st.stop()
    st.session_state["final_state"] = final_state

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
if "final_state" in st.session_state:
    fs = st.session_state["final_state"]
    if not fs.get("classification"):
        st.error("No classification produced.")
        st.stop()

    all_tx = fs.get("all_transcripts") or []
    sel_tx = fs.get("selected_transcript", "")
    clinvar = fs.get("clinvar") or {}
    gnomad = fs.get("gnomad") or {}
    af_data = gnomad.get("allele_frequency", {})
    predictors = gnomad.get("insilico_predictors", {})
    conservation = gnomad.get("conservation", {})
    acmg_crit = gnomad.get("acmg_criteria", {})

    # Summary bar
    parts = [fs.get("gene_symbol", ""), fs.get("hgvs_on_transcript", ""), fs.get("genome_build", "")]
    for t in all_tx:
        if (t.get("nm_accession") or t.get("enst_accession")) == sel_tx:
            if t.get("consequence_display"): parts.append(t["consequence_display"])
            if t.get("position_detail"): parts.append(t["position_detail"])
            break
    st.markdown(f"**{' | '.join(p for p in parts if p)}**")

    # --- ClinVar ---
    with st.expander("\U0001f4cb ClinVar Record", expanded=True):
        if not clinvar:
            st.warning(fs.get("clinvar_error", "No ClinVar data"))
        else:
            vid = clinvar.get("variant_id", "")
            rsid = clinvar.get("rsid", "")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"**Gene:** {_gene_link(clinvar.get('gene', ''))}")
                st.markdown(f"**HGVS:** {clinvar.get('hgvs', 'N/A')}")
                st.markdown(f"**Clinical Significance:** {clinvar.get('clinical_significance', 'N/A')}")
                st.markdown(f"**Condition:** {clinvar.get('condition', 'N/A')}")
            with c2:
                rs = clinvar.get("review_status", "N/A")
                stars = _get_star_count(rs)
                st.markdown(f"**Review Status:** {'\u2B50' * stars}{'\u2606' * (4 - stars)} ({rs})")
                st.markdown(f"**Submitters:** {clinvar.get('submitter_count', 0)}")
                st.markdown(f"**Last Evaluated:** {clinvar.get('last_evaluated', 'N/A')}")
                st.markdown(f"**ClinVar ID:** {_clinvar_link(vid)} &nbsp;|&nbsp; **rsID:** {_dbsnp_link(rsid)}")
            if clinvar.get("conflicting_interpretations"):
                st.warning("\u26A0\uFE0F Conflicting interpretations among submitters")

    # --- gnomAD + In Silico ---
    if gnomad:
        with st.expander("\U0001f4ca Population Frequencies & In Silico Predictors", expanded=True):
            gn_vid = gnomad.get("gnomad_variant_id", "")
            rsid_val = gnomad.get("rsid", "")
            coord_str = gnomad.get("coordinates", "")
            strand_info = ""
            for t in all_tx:
                if (t.get("nm_accession") or t.get("enst_accession")) == sel_tx:
                    s = t.get("strand")
                    strand_info = "+" if s == 1 else "\u2212" if s == -1 else ""
                    break

            link_parts = []
            if coord_str: link_parts.append(f"**Coords:** {coord_str}")
            if strand_info: link_parts.append(f"**Strand:** {strand_info}")
            gn_dataset = gnomad.get("dataset", "gnomad_r4")
            if gn_vid: link_parts.append(f"**gnomAD:** {_gnomad_link(gn_vid, gn_dataset)}")
            if rsid_val: link_parts.append(f"**dbSNP:** {_dbsnp_link(rsid_val)}")
            if link_parts:
                st.markdown(" &nbsp;|&nbsp; ".join(link_parts))

            st.markdown("#### gnomAD Allele Frequencies")

            # Three cohorts: overall, non-cancer, controls
            ov_c = gnomad.get("overall", {})
            nc_c = gnomad.get("non_cancer", {})
            ct_c = gnomad.get("controls", {})
            ov_avail = ov_c.get("available", False)
            nc_avail = nc_c.get("available", False)
            ct_avail = ct_c.get("available", False)

            if gnomad.get("variant_in_gnomad") or af_data.get("variant_in_gnomad"):
                # Helper to format AF
                def _af(val):
                    return f"{val:.6f}" if val is not None else "\u2014"

                def _ac_an(block, key):
                    """Format AC/AN for exome or genome."""
                    sub = (block or {}).get(key) or {}
                    if sub.get("an"):
                        return f"{sub.get('ac',0)}/{sub.get('an',0)}"
                    return "\u2014"

                # Summary line: all 3 cohorts
                parts = []
                if ov_avail:
                    parts.append(f"**Overall AF:** {_af(ov_c.get('global_af'))}")
                if nc_avail:
                    parts.append(f"**Non-cancer AF:** {_af(nc_c.get('global_af'))}")
                if ct_avail:
                    parts.append(f"**Controls AF:** {_af(ct_c.get('global_af'))}")
                st.markdown(" &nbsp;|&nbsp; ".join(parts))

                # Exome / Genome / Hom breakdown
                detail_rows = []
                for label, block in [("Overall", ov_c), ("Non-cancer", nc_c), ("Controls", ct_c)]:
                    if not block.get("available"):
                        continue
                    ds_label = block.get("dataset_label", label)
                    detail_rows.append({
                        "Cohort": ds_label,
                        "Global AF": _af(block.get("global_af")),
                        "Exome AC/AN": _ac_an(block, "exome"),
                        "Genome AC/AN": _ac_an(block, "genome"),
                        "Hom": block.get("hom", 0),
                    })
                if detail_rows:
                    st.dataframe(detail_rows, use_container_width=True, hide_index=True)

                # Population table: all 3 cohorts side by side
                # Use the first available cohort for population list
                ref_pops = (ov_c if ov_avail else nc_c if nc_avail else ct_c).get("populations", {})
                nc_pops = nc_c.get("populations", {})
                ct_pops = ct_c.get("populations", {})
                ov_pops = ov_c.get("populations", {})

                if ref_pops:
                    pop_rows = []
                    for pid in POP_ORDER:
                        # Find population data from any cohort
                        name = POP_NAMES.get(pid, pid)
                        ov_pd = ov_pops.get(pid, {}) if isinstance(ov_pops.get(pid), dict) else {}
                        nc_pd = nc_pops.get(pid, {}) if isinstance(nc_pops.get(pid), dict) else {}
                        ct_pd = ct_pops.get(pid, {}) if isinstance(ct_pops.get(pid), dict) else {}

                        if not (ov_pd or nc_pd or ct_pd):
                            continue

                        row = {"Population": name}
                        # Overall
                        if ov_avail:
                            row["AF"] = _af(ov_pd.get("af")) if ov_pd else "\u2014"
                            row["AC"] = ov_pd.get("ac", "\u2014") if ov_pd else "\u2014"
                            row["AN"] = ov_pd.get("an", "\u2014") if ov_pd else "\u2014"
                        # Non-cancer
                        if nc_avail:
                            row["NC AF"] = _af(nc_pd.get("af")) if nc_pd else "\u2014"
                            row["NC AC"] = nc_pd.get("ac", "\u2014") if nc_pd else "\u2014"
                        # Controls
                        if ct_avail:
                            row["Ctrl AF"] = _af(ct_pd.get("af")) if ct_pd else "\u2014"
                            row["Ctrl AC"] = ct_pd.get("ac", "\u2014") if ct_pd else "\u2014"

                        pop_rows.append(row)

                    if pop_rows:
                        st.dataframe(pop_rows, use_container_width=True, hide_index=True)
            else:
                st.info("Variant **not found** in gnomAD — absent from population controls (supports PM2)")

            # ACMG frequency flags — show which cohort and AF was used
            freq_cohort = acmg_crit.get("freq_cohort", "controls")
            freq_af = acmg_crit.get("freq_af_used")
            freq_af_str = f"{freq_af:.6f}" if freq_af is not None else "N/A"
            st.markdown(
                f"{'\u2705' if acmg_crit.get('BA1_met') else '\u274C'} **BA1** (global AF>5%) &nbsp;&nbsp; "
                f"{'\u2705' if acmg_crit.get('BS1_met') else '\u274C'} **BS1** (global AF>1%) &nbsp;&nbsp; "
                f"{'\u2705' if acmg_crit.get('PM2_met') else '\u274C'} **PM2** (global AF<0.01%) &nbsp;&nbsp; "
                f"*(evaluated on **{freq_cohort}** global AF = {freq_af_str})*"
            )

            # In silico predictors
            if predictors:
                st.markdown("#### In Silico Predictors")
                pred_rows = []
                for name, val in predictors.items():
                    if isinstance(val, dict):
                        pred_rows.append({
                            "Predictor": name.replace("_", " ").upper(),
                            "Score": val.get("score", val.get("interpretation", "")),
                            "Prediction": val.get("pred", val.get("interpretation", "")),
                        })
                    else:
                        pred_rows.append({"Predictor": name.upper(), "Score": val, "Prediction": ""})
                if pred_rows:
                    st.dataframe(pred_rows, use_container_width=True, hide_index=True)

                consensus = gnomad.get("insilico_consensus", "")
                color = {"Damaging": "red", "Benign": "green"}.get(consensus, "orange")
                st.markdown(
                    f"**Consensus:** :{color}[{consensus}] &nbsp;&nbsp; "
                    f"{'\u2705' if acmg_crit.get('PP3_met') else '\u274C'} **PP3** &nbsp;&nbsp; "
                    f"{'\u2705' if acmg_crit.get('BP4_met') else '\u274C'} **BP4**"
                )

            if conservation:
                st.markdown("#### Conservation")
                cons_parts = [f"**{k}:** {v:.3f}" if isinstance(v, float) else f"**{k}:** {v}" for k, v in conservation.items()]
                st.markdown(" &nbsp;|&nbsp; ".join(cons_parts))

    # --- Gene Constraint & Protein Domains ---
    if gnomad:
        constraint = gnomad.get("gene_constraint")
        uniprot_data = gnomad.get("uniprot", {})
        if constraint or uniprot_data.get("domains"):
            with st.expander("\U0001f9ec Gene Constraint & Protein Domains", expanded=True):
                # Constraint metrics
                if constraint:
                    uniprot_acc = uniprot_data.get("accession", "")
                    uniprot_link = f"[{uniprot_acc}](https://www.uniprot.org/uniprotkb/{uniprot_acc})" if uniprot_acc else ""
                    st.markdown(
                        f"**Missense Z:** {constraint.get('mis_z', 0):.2f} "
                        f"({constraint.get('missense_interpretation', '')}) &nbsp;|&nbsp; "
                        f"**o/e missense:** {constraint.get('oe_mis', 0):.3f} &nbsp;|&nbsp; "
                        f"**pLI:** {constraint.get('pli', 0):.4f} &nbsp;|&nbsp; "
                        f"**LOEUF:** {constraint.get('loeuf', 0):.3f} "
                        f"({constraint.get('lof_interpretation', '')}) &nbsp;|&nbsp; "
                        f"**UniProt:** {uniprot_link}"
                    )

                # Domains
                domains = uniprot_data.get("domains", [])
                if domains:
                    st.markdown(f"**Functional domains** (protein length: {uniprot_data.get('protein_length', '?')} aa):")
                    domain_rows = []
                    for d in domains:
                        is_hit = (uniprot_data.get("variant_in_domain") or {}).get("start") == d["start"]
                        domain_rows.append({
                            "Type": d["type"],
                            "Name": d["description"],
                            "Position": f"aa {d['start']}-{d['end']}",
                            "Variant": "\U0001f534 IN DOMAIN" if is_hit else "",
                        })
                    st.dataframe(domain_rows, use_container_width=True, hide_index=True)

                # PM1 / PM4 / BP3 flags
                pm1_met = acmg_crit.get("PM1_met", False)
                pm4_met = acmg_crit.get("PM4_met", False)
                bp3_met = acmg_crit.get("BP3_met", False)
                st.markdown(
                    f"{'\u2705' if pm1_met else '\u274C'} **PM1** (functional domain) "
                    f"{'(' + acmg_crit.get('PM1_strength', '') + ')' if pm1_met else ''} &nbsp;&nbsp; "
                    f"{'\u2705' if pm4_met else '\u274C'} **PM4** (protein length change) &nbsp;&nbsp; "
                    f"{'\u2705' if bp3_met else '\u274C'} **BP3** (in-frame in repeat)"
                )

    # --- Literature Evidence (LitVar) ---
    pubmed_data = fs.get("pubmed")
    if pubmed_data and isinstance(pubmed_data, dict) and pubmed_data.get("available"):
        with st.expander("\U0001f4da Literature Evidence (LitVar)", expanded=True):
            rsid_lit = pubmed_data.get("rsid", "")
            litvar_link = f"[LitVar](https://www.ncbi.nlm.nih.gov/research/litvar2/docsum?query={rsid_lit})" if rsid_lit else ""

            st.markdown(
                f"**Publications:** {pubmed_data.get('pmids_count', 0)} &nbsp;|&nbsp; "
                f"**Case Reports:** {pubmed_data.get('case_report_count', 0)} &nbsp;|&nbsp; "
                f"**Functional Studies:** {pubmed_data.get('functional_study_count', 0)} &nbsp;|&nbsp; "
                f"**Reviews:** {pubmed_data.get('review_count', 0)} &nbsp;|&nbsp; "
                f"**First Published:** {pubmed_data.get('first_published', 'N/A')} &nbsp;|&nbsp; "
                f"**Literature Significance:** {pubmed_data.get('clinical_significance', 'N/A')} &nbsp;|&nbsp; "
                f"{litvar_link}"
            )

            # Disease associations
            diseases = pubmed_data.get("diseases", [])
            if diseases:
                st.markdown("**Disease Associations:**")
                disease_rows = []
                for dname, dcount in diseases[:10]:
                    disease_rows.append({"Disease": dname, "Publications": dcount})
                st.dataframe(disease_rows, use_container_width=True, hide_index=True)

            # Recent publications
            pubs = pubmed_data.get("publications", [])
            if pubs:
                st.markdown(f"**Recent Publications** ({len(pubs)} of {pubmed_data.get('pmids_count', 0)}):")
                pub_rows = []
                for pub in pubs:
                    pmid = pub.get("pmid", "")
                    title = pub.get("title", "")[:100]
                    year = pub.get("year", "")
                    journal = pub.get("journal", "")
                    types = ", ".join(pub.get("pub_types", []))
                    pmid_link = f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                    pub_rows.append({
                        "PMID": pmid_link,
                        "Year": year,
                        "Title": title,
                        "Journal": journal,
                        "Type": types,
                    })
                st.dataframe(pub_rows, use_container_width=True, hide_index=True)

            # Related genes and chemicals
            related_genes = pubmed_data.get("related_genes", [])
            related_chems = pubmed_data.get("related_chemicals", [])
            if related_genes or related_chems:
                rel_parts = []
                if related_genes:
                    gene_strs = [f"{g['name']} ({g['count']})" for g in related_genes[:5]]
                    rel_parts.append(f"**Related Genes:** {', '.join(gene_strs)}")
                if related_chems:
                    chem_strs = [f"{c['name']} ({c['count']})" for c in related_chems[:5]]
                    rel_parts.append(f"**Related Chemicals:** {', '.join(chem_strs)}")
                st.markdown(" &nbsp;|&nbsp; ".join(rel_parts))

    # --- Case-Control Analysis ---
    if gnomad and af_data.get("variant_in_gnomad"):
        with st.expander("\U0001f9ea Case-Control Analysis", expanded=False):
            st.markdown("Compare your cohort's variant frequency against gnomAD populations.")

            # Ethnicity-specific toggle (above overall so auto-sum can work)
            use_eth = st.checkbox("Provide ethnicity-specific counts", key="use_eth_input")

            # Collect per-ethnicity data first (if enabled)
            eth_case_data: dict[str, dict[str, int]] = {}
            if use_eth:
                st.markdown("**Per-ancestry case counts** *(fill in ancestries you have data for)*")
                pops_available = af_data.get("populations", {})
                eth_cols = st.columns(3)
                col_idx = 0
                for pid in POP_ORDER:
                    if pid not in pops_available:
                        continue
                    pname = POP_NAMES.get(pid, pid)
                    with eth_cols[col_idx % 3]:
                        st.markdown(f"*{pname}*")
                        e1, e2 = st.columns(2)
                        with e1:
                            ec = st.number_input("Carriers", min_value=0, value=0, step=1, key=f"cc_{pid}_c")
                        with e2:
                            et = st.number_input("Total", min_value=0, value=0, step=1, key=f"cc_{pid}_t")
                        if et > 0:
                            eth_case_data[pid] = {"carriers": ec, "total": et}
                    col_idx += 1

            # Auto-sum from ethnicity rows if any are filled
            eth_sum_carriers = sum(d["carriers"] for d in eth_case_data.values())
            eth_sum_total = sum(d["total"] for d in eth_case_data.values())

            # Overall fields — auto-populated from ethnicity sum, but user can override
            st.markdown("**Overall cohort**" + (" *(auto-summed from ancestries, editable)*" if eth_case_data else ""))
            oc1, oc2 = st.columns(2)
            with oc1:
                default_carriers = eth_sum_carriers if eth_case_data else 1
                overall_carriers = st.number_input(
                    "Carriers", min_value=0, value=default_carriers, step=1, key="cc_overall_carriers"
                )
            with oc2:
                default_total = eth_sum_total if eth_case_data else 100
                overall_total = st.number_input(
                    "Total samples", min_value=1, value=max(default_total, 1), step=1, key="cc_overall_total"
                )

            # Build case_data dict
            case_data = {"overall": {"carriers": overall_carriers, "total": overall_total}}
            case_data.update(eth_case_data)

            # Count how many ethnicities have user data
            eth_with_data = [p for p in case_data if p != "overall"]
            n_eth_with_data = len(eth_with_data)

            run_cc = st.button("Run Case-Control Analysis", key="run_cc")

            if run_cc and overall_total > 0:
                from tools.case_control import run_case_control_analysis

                # Use controls cohort (already fetched by gnomad_agent)
                ctrl_block = gnomad.get("controls", {})
                if not ctrl_block.get("available"):
                    st.warning("Controls dataset not available — using overall gnomAD")
                    ctrl_block = gnomad.get("overall", {})

                # Build gnomad_data-like dict from controls block
                ctrl_exome = ctrl_block.get("exome") or {}
                ctrl_genome = ctrl_block.get("genome") or {}
                ds_data = {
                    "ac": ctrl_exome.get("ac", 0) + ctrl_genome.get("ac", 0),
                    "an": ctrl_exome.get("an", 0) + ctrl_genome.get("an", 0),
                    "populations": ctrl_block.get("populations", {}),
                    "dataset_label": ctrl_block.get("dataset_label", "Controls"),
                }

                ds_label = ctrl_block.get("dataset_label", "Controls")
                ctrl_ds = ctrl_block.get("dataset", gnomad.get("dataset", ""))
                gn_link = _gnomad_link(gnomad.get("gnomad_variant_id", ""), ctrl_ds)
                st.markdown(f"**Control cohort:** {ds_label} {gn_link}")

                analysis_case_data = {"overall": case_data["overall"]}
                analysis_case_data.update(eth_case_data)

                cc_result = run_case_control_analysis(analysis_case_data, ds_data)

                # Overall Fisher's
                ov = cc_result["overall_fishers"]
                or_str = f"{ov['odds_ratio']:.2f}" if ov.get("odds_ratio") and ov["odds_ratio"] != float("inf") else "Inf" if ov.get("odds_ratio") == float("inf") else "N/A"
                pv = ov.get("p_value")
                pv_str = f"{pv:.2e}" if pv and pv < 0.001 else f"{pv:.4f}" if pv else "N/A"
                sig_color = "red" if ov.get("significant") else "green"
                st.markdown(
                    f"**Overall Fisher's:** Case AF={ov['case_af']:.4f} vs Control AF={ov['control_af']:.6f} "
                    f"| OR={or_str} | :{sig_color}[p={pv_str}]"
                )

                # Per-ancestry table
                anc = cc_result.get("ancestry_fishers", [])
                if anc:
                    rows = []
                    plot_data = []
                    for ar in anc:
                        or_v = ar.get("odds_ratio")
                        pv = ar.get("p_value")
                        pop_id = ar.get("population_id", "")
                        has_user_data = pop_id in eth_case_data
                        source_tag = "" if has_user_data else " (using overall)"

                        rows.append({
                            "Population": ar.get("population", "") + source_tag,
                            "Case (carriers/total)": f"{ar['case_ac']}/{ar['case_an']//2}",
                            "Control (AC/AN)": f"{ar['control_ac']}/{ar['control_an']}",
                            "Case AF": f"{ar['case_af']:.4f}",
                            "Control AF": f"{ar['control_af']:.6f}",
                            "OR": f"{or_v:.2f}" if or_v and or_v != float("inf") else "Inf",
                            "p-value": f"{pv:.2e}" if pv and pv < 0.001 else f"{pv:.4f}" if pv else "N/A",
                            "Sig": "\u2705" if ar.get("significant") else "",
                        })

                        if has_user_data and or_v and or_v != float("inf") and or_v > 0:
                            plot_data.append({
                                "pop": ar.get("population", ""),
                                "or": or_v,
                                "ci_lo": ar.get("ci_lower", or_v * 0.5),
                                "ci_hi": ar.get("ci_upper", or_v * 2),
                                "sig": ar.get("significant", False),
                            })

                    st.dataframe(rows, use_container_width=True, hide_index=True)

                    # Forest plot — only ethnicities with user data
                    if plot_data:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        pops_sorted = sorted(plot_data, key=lambda x: x["or"])
                        fig.add_trace(go.Scatter(
                            x=[d["or"] for d in pops_sorted],
                            y=[d["pop"] for d in pops_sorted],
                            mode="markers",
                            marker=dict(size=10, color=["#ff4b4b" if d["sig"] else "#888" for d in pops_sorted]),
                            error_x=dict(
                                type="data", symmetric=False,
                                array=[d["ci_hi"] - d["or"] for d in pops_sorted],
                                arrayminus=[d["or"] - d["ci_lo"] for d in pops_sorted],
                            ),
                            hovertemplate="%{y}<br>OR=%{x:.2f}<extra></extra>",
                        ))
                        fig.add_vline(x=1, line_dash="dash", line_color="gray")
                        fig.update_layout(
                            title="Forest Plot — OR by Ancestry (user-specified, vs controls)",
                            xaxis_title="Odds Ratio (log scale)", xaxis_type="log",
                            height=max(300, len(pops_sorted) * 50 + 100),
                            showlegend=False, margin=dict(l=200), font=dict(size=13),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Weighted GLM
                if n_eth_with_data >= 2:
                    glm = cc_result.get("weighted_glm", {})
                    if glm.get("p_value") is not None:
                        glm_pv = glm["p_value"]
                        glm_color = "red" if glm_pv < 0.05 else "green"
                        st.markdown(f"**Weighted GLM:** {glm['interpretation']} :{glm_color}[p={glm_pv:.4f}]")
                    elif glm.get("interpretation"):
                        st.markdown(f"**Weighted GLM:** {glm['interpretation']}")
                elif use_eth:
                    st.info("Provide at least 2 ethnicities with data to enable weighted GLM." if n_eth_with_data < 2 else "")

    # --- ACMG Criteria ---
    with st.expander("\U0001f3af ACMG Criteria", expanded=True):
        criteria = fs.get("criteria_triggered") or []
        met = [c for c in criteria if c.get("met", True)]
        unmet = [c for c in criteria if not c.get("met", True)]
        if not met and not unmet:
            st.info("No ACMG criteria evaluated.")
        else:
            if met:
                st.markdown("**Criteria Met:**")
                for c in met:
                    d = c.get("direction", "")
                    icon = "\U0001f534" if d == "pathogenic" else "\U0001f7e2"
                    st.markdown(f"{icon} **{c.get('code','')}** ({c.get('strength','')}) \u2014 {c.get('justification','')}")
            if unmet:
                with st.expander(f"Criteria Not Met / Not Evaluable ({len(unmet)})"):
                    for c in unmet:
                        st.markdown(f"\u2B1C **{c.get('code','')}** \u2014 {c.get('justification','')}")

    # --- Classification ---
    with st.expander("\U0001f3c6 Final Classification", expanded=True):
        classification = fs.get("classification", "VUS")
        colors = {
            "Pathogenic": ("#ff4b4b", "#fff"), "Likely Pathogenic": ("#ff8c00", "#fff"),
            "VUS": ("#ffd700", "#333"), "Likely Benign": ("#90ee90", "#333"), "Benign": ("#4caf50", "#fff"),
        }
        bg, fg = colors.get(classification, ("#ffd700", "#333"))
        st.markdown(
            f'<div style="background:{bg};color:{fg};padding:16px;border-radius:8px;'
            f'text-align:center;font-size:1.3rem;font-weight:bold;margin:8px 0">'
            f'{classification}</div>', unsafe_allow_html=True,
        )
        st.markdown(f"**Confidence:** {fs.get('confidence','Low')}")
        st.markdown(f"**Reasoning:** {fs.get('reasoning','')}")

    # Warnings
    warnings = fs.get("warnings", [])
    if warnings:
        with st.expander(f"\u26A0\uFE0F Warnings ({len(warnings)})"):
            for w in warnings: st.markdown(f"- {w}")

    # Disclaimer
    st.divider()
    if fs.get("disclaimer"):
        st.markdown(f'<p style="color:gray;font-size:0.75rem">{fs["disclaimer"]}</p>', unsafe_allow_html=True)

    # Debug
    with st.expander("\U0001f41b Debug: Full State JSON"):
        debug = {}
        for k, v in fs.items():
            try: json.dumps(v); debug[k] = v
            except: debug[k] = str(v)
        st.json(debug)
