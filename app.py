"""PathoMAN 2.0 — AI-Powered ACMG Variant Classification (Streamlit UI)."""

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
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(page_title="PathoMAN 2.0", page_icon="\U0001f7e0", layout="wide")

# ---------------------------------------------------------------------------
# CSS + Logo
# ---------------------------------------------------------------------------
PACMAN_LOGO = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 52 48" width="44" height="40">'
    '<defs><linearGradient id="pg" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0%" stop-color="#FFB300"/><stop offset="100%" stop-color="#FF8F00"/>'
    '</linearGradient></defs>'
    # Pac-man body: circle with wedge mouth cut out (facing right)
    '<path d="M26 4 A20 20 0 1 1 26 44 A20 20 0 1 1 26 4 Z" fill="url(#pg)"/>'
    # Mouth wedge (white triangle cut-out opening to the right)
    '<path d="M26 24 L48 10 L48 38 Z" fill="white"/>'
    # Eye
    '<circle cx="30" cy="14" r="3" fill="#333"/>'
    '<circle cx="31" cy="13" r="1.2" fill="white"/>'
    # AI badge (purple pill)
    '<rect x="13" y="2" rx="5" ry="5" width="16" height="10" fill="#7C3AED" opacity="0.85"/>'
    '<text x="21" y="10" text-anchor="middle" fill="white" font-size="7" '
    'font-weight="bold" font-family="Arial,sans-serif">AI</text>'
    # DNA double-helix trail behind
    '<path d="M2 18 Q6 14 10 18 Q14 22 18 18" stroke="#FF8F00" stroke-width="1.5" fill="none" opacity="0.5"/>'
    '<path d="M2 30 Q6 34 10 30 Q14 26 18 30" stroke="#FF8F00" stroke-width="1.5" fill="none" opacity="0.5"/>'
    '<line x1="6" y1="18" x2="6" y2="30" stroke="#FF8F00" stroke-width="0.8" opacity="0.3"/>'
    '<line x1="10" y1="18" x2="10" y2="30" stroke="#FF8F00" stroke-width="0.8" opacity="0.3"/>'
    '<line x1="14" y1="18" x2="14" y2="30" stroke="#FF8F00" stroke-width="0.8" opacity="0.3"/>'
    '</svg>'
)

# Pac-man character SVG for the animation (smaller, just the face)
PACMAN_ANIM_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 28" width="24" height="24" style="vertical-align:middle;">'
    '<path d="M14 2 A12 12 0 1 1 14 26 A12 12 0 1 1 14 2 Z" fill="#FFB300"/>'
    '<path d="M14 14 L27 6 L27 22 Z" fill="white"/>'
    '<circle cx="16" cy="8" r="2" fill="#333"/>'
    '<circle cx="16.7" cy="7.3" r="0.8" fill="white"/>'
    '</svg>'
)

st.markdown(f"""
<style>
    .block-container {{ padding-top: 1rem !important; }}
    /* Hide default Streamlit header bar to prevent clipping */
    header[data-testid="stHeader"] {{ background: transparent; }}
    [data-testid="stMetricValue"] {{ font-size: 1rem !important; }}
    [data-testid="stMetricLabel"] {{ font-size: 0.85rem !important; }}
    .stMarkdown p {{ font-size: 0.95rem; }}
    .stMarkdown h4 {{ font-size: 1.1rem !important; margin-top: 1rem; }}
    /* ACMG criteria pills */
    .acmg-pill {{ display:inline-block; padding:4px 12px; border-radius:16px; margin:2px;
                 font-weight:600; font-size:0.8rem; cursor:default; position:relative; }}
    .acmg-pill.path-met {{ background:#ff4b4b; color:white; }}
    .acmg-pill.ben-met {{ background:#4caf50; color:white; }}
    .acmg-pill.not-met {{ background:#e8e8e8; color:#888; }}
    .acmg-pill[title]:hover::after {{
        content: attr(title); position:absolute; bottom:130%; left:50%;
        transform:translateX(-50%); background:#333; color:white;
        padding:6px 10px; border-radius:6px; font-size:0.7rem; font-weight:400;
        white-space:pre-wrap; max-width:300px; z-index:100; box-shadow:0 2px 8px rgba(0,0,0,0.2);
    }}
    /* Classification hero */
    .classification-hero {{ padding:16px 24px; border-radius:10px; text-align:center;
                           font-size:1.4rem; font-weight:700; margin:8px 0; }}
    .classification-sub {{ text-align:center; margin-top:4px; font-size:0.95rem; }}
    /* Header */
    .pathoman-header {{ display:flex; align-items:center; gap:14px; margin-bottom:8px; margin-top:0.5rem; }}
    .pathoman-header h1 {{ margin:0; font-size:1.8rem; line-height:1.2; }}
    .pathoman-header .subtitle {{ color:#666; font-size:0.9rem; margin:0; }}
    .pathoman-header svg {{ flex-shrink:0; }}
    /* Pac-man animation */
    .pacman-track {{ display:flex; align-items:center; gap:0; padding:12px 0; overflow-x:auto; }}
    .pacman-dot {{ width:12px; height:12px; border-radius:50%; margin:0 8px;
                  transition: all 0.3s ease; }}
    .pacman-dot.pending {{ background:#FFD54F; }}
    .pacman-dot.eaten {{ background:#e0e0e0; opacity:0.4; transform:scale(0.5); }}
    .pacman-dot.current {{ background:#FF8F00; animation: pulse 0.6s infinite alternate; }}
    .pacman-label {{ font-size:0.7rem; color:#666; text-align:center; min-width:60px; }}
    .pacman-step {{ display:flex; flex-direction:column; align-items:center; }}
    @keyframes pulse {{ from {{ transform:scale(1); }} to {{ transform:scale(1.3); }} }}
    .pacman-char {{ display:inline-block; margin:0 4px; vertical-align:middle; }}
</style>
<div class="pathoman-header">
    {PACMAN_LOGO}
    <div>
        <div style="font-size:1.8rem; font-weight:700; line-height:1.2; color:#1a1a1a;">PathoMAN 2.0</div>
        <div style="color:#666; font-size:0.9rem;">Pathogenicity of Mutation Analyzer &mdash; for Clinical Cancer Genomics &middot; <em>Not for clinical use</em></div>
    </div>
</div>
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

def _af(val):
    """Format allele frequency."""
    return f"{val:.6f}" if val is not None else "\u2014"

def _ac_an(block, key):
    """Format AC/AN for exome or genome."""
    sub = (block or {}).get(key) or {}
    if sub.get("an"):
        return f"{sub.get('ac',0)}/{sub.get('an',0)}"
    return "\u2014"

# Pipeline node labels + order for pac-man animation
NODE_LABELS = {
    "input_parser": "Parse", "supervisor": "Route",
    "clinvar_agent": "ClinVar", "gnomad_agent": "gnomAD",
    "pubmed_agent": "Literature", "alphafold_agent": "Protein",
    "tcga_agent": "Clinical",
    "pathway_agent": "Pathways",
    "acmg_classifier": "ACMG",
}

PIPELINE_STEPS = [
    "input_parser", "supervisor", "clinvar_agent", "gnomad_agent",
    "pubmed_agent", "alphafold_agent", "tcga_agent", "pathway_agent",
    "acmg_classifier",
]

POP_ORDER = ["afr", "amr", "asj", "eas", "fin", "mid", "nfe", "sas", "remaining", "ami"]
POP_NAMES = {
    "afr": "African/African American", "amr": "Latino/Admixed American",
    "asj": "Ashkenazi Jewish", "eas": "East Asian", "fin": "Finnish",
    "mid": "Middle Eastern", "nfe": "Non-Finnish European", "sas": "South Asian",
    "remaining": "Remaining", "ami": "Amish",
}


def _pacman_html(completed_nodes: list[str], current_node: str = "") -> str:
    """Generate pac-man pipeline animation HTML."""
    steps_html = []
    for step in PIPELINE_STEPS:
        label = NODE_LABELS.get(step, step)
        if step in completed_nodes:
            dot_cls = "eaten"
        elif step == current_node:
            dot_cls = "current"
        else:
            dot_cls = "pending"
        steps_html.append(
            f'<div class="pacman-step">'
            f'<div class="pacman-dot {dot_cls}"></div>'
            f'<div class="pacman-label">{label}</div>'
            f'</div>'
        )

    # Place pac-man character at the position of the current/last completed node
    pacman_pos = 0
    if current_node and current_node in PIPELINE_STEPS:
        pacman_pos = PIPELINE_STEPS.index(current_node)
    elif completed_nodes:
        last = completed_nodes[-1]
        if last in PIPELINE_STEPS:
            pacman_pos = PIPELINE_STEPS.index(last) + 1

    # Insert pac-man character (SVG face, not emoji)
    pacman_char = f'<div class="pacman-char">{PACMAN_ANIM_SVG}</div>'
    steps_html.insert(min(pacman_pos, len(steps_html)), pacman_char)

    return f'<div class="pacman-track">{"".join(steps_html)}</div>'


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
c_lookup, c_classify = st.columns(2)
with c_lookup:
    lookup_btn = st.button("Look Up Transcripts", disabled=not raw_input_str, type="secondary")
with c_classify:
    classify_btn = st.button("Classify Variant", type="primary", disabled=not raw_input_str)

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
# Classify Variant — with Pac-Man animation
# ---------------------------------------------------------------------------
if classify_btn and raw_input_str:
    initial_state = make_initial_state(raw_input_str)
    initial_state["genome_build"] = genome_build

    # If transcripts were not looked up yet, resolve them now automatically
    if "parser_output" not in st.session_state:
        from agents.input_parser import input_parser_node
        init_parse = make_initial_state(raw_input_str)
        init_parse["genome_build"] = genome_build
        with st.spinner("Resolving transcripts via Ensembl VEP..."):
            parser_output = input_parser_node(init_parse)
        st.session_state["parser_output"] = parser_output
        st.session_state["raw_input_str"] = raw_input_str
        st.session_state["genome_build"] = genome_build
        # Auto-select the top transcript (most reported / MANE Select)
        transcripts = parser_output.get("all_transcripts") or []
        if transcripts:
            st.session_state["chosen_transcript"] = transcripts[0]

    # Apply chosen transcript (or top/default) to initial state
    po = st.session_state.get("parser_output", {})
    ct = st.session_state.get("chosen_transcript")

    # If no transcript was explicitly chosen, use the first one from parser output
    if not ct:
        transcripts = po.get("all_transcripts") or []
        if transcripts:
            ct = transcripts[0]
            st.session_state["chosen_transcript"] = ct

    if ct and po:
        if ct.get("hgvsc"): initial_state["hgvs_on_transcript"] = ct["hgvsc"]
        tx_id = ct.get("nm_accession") or ct.get("enst_accession")
        if tx_id: initial_state["selected_transcript"] = tx_id
        for k in ["gene_symbol", "gene_aliases", "gene_full_name", "all_transcripts",
                   "chrom", "pos", "ref", "alt"]:
            if po.get(k): initial_state[k] = po[k]

    final_state = dict(initial_state)
    completed_nodes: list[str] = []
    anim_placeholder = st.empty()
    anim_placeholder.markdown(_pacman_html([], PIPELINE_STEPS[0]), unsafe_allow_html=True)

    try:
        for node_name, node_output in run_graph_stream(initial_state):
            final_state.update(node_output)
            completed_nodes.append(node_name)
            # Determine next node for animation
            idx = PIPELINE_STEPS.index(node_name) if node_name in PIPELINE_STEPS else -1
            next_node = PIPELINE_STEPS[idx + 1] if idx + 1 < len(PIPELINE_STEPS) else ""
            anim_placeholder.markdown(_pacman_html(completed_nodes, next_node), unsafe_allow_html=True)

            if node_name == "supervisor" and final_state.get("input_parse_error"):
                if not final_state.get("gene_symbol"):
                    anim_placeholder.empty()
                    st.error(f"Parse error: {final_state['input_parse_error']}")
                    st.stop()

        # Final: all done
        anim_placeholder.markdown(
            _pacman_html(completed_nodes, "") +
            '<div style="text-align:center;font-size:0.85rem;color:#4caf50;font-weight:600;">'
            'Classification complete!</div>',
            unsafe_allow_html=True,
        )
    except Exception as e:
        anim_placeholder.empty()
        st.error(f"Pipeline error: {e}")
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
    pubmed_data = fs.get("pubmed")

    # Summary bar
    summary_parts = [fs.get("gene_symbol", ""), fs.get("hgvs_on_transcript", ""), fs.get("genome_build", "")]
    for t in all_tx:
        if (t.get("nm_accession") or t.get("enst_accession")) == sel_tx:
            if t.get("consequence_display"): summary_parts.append(t["consequence_display"])
            if t.get("position_detail"): summary_parts.append(t["position_detail"])
            break
    st.markdown(f"**{' | '.join(p for p in summary_parts if p)}**")

    # ===================================================================
    # CLASSIFICATION HERO BANNER (first thing shown)
    # ===================================================================
    classification = fs.get("classification", "VUS")
    colors = {
        "Pathogenic": ("#ff4b4b", "#fff"), "Likely Pathogenic": ("#ff8c00", "#fff"),
        "VUS": ("#ffd700", "#333"), "Likely Benign": ("#90ee90", "#333"), "Benign": ("#4caf50", "#fff"),
    }
    bg, fg = colors.get(classification, ("#ffd700", "#333"))
    confidence = fs.get("confidence", "Low")
    reasoning = fs.get("reasoning", "")

    st.markdown(
        f'<div class="classification-hero" style="background:{bg};color:{fg};">'
        f'{classification}</div>'
        f'<div class="classification-sub">'
        f'<strong>Confidence:</strong> {confidence} &mdash; {reasoning[:200]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ===================================================================
    # ACMG CRITERIA PILLS
    # ===================================================================
    criteria = fs.get("criteria_triggered") or []
    if criteria:
        path_criteria = [c for c in criteria if c.get("direction") == "pathogenic"]
        ben_criteria = [c for c in criteria if c.get("direction") == "benign"]
        other_criteria = [c for c in criteria if c.get("direction") not in ("pathogenic", "benign")]

        def _pill(c):
            met = c.get("met", False)
            direction = c.get("direction", "")
            code = c.get("code", "")
            strength = c.get("strength", "")
            justification = c.get("justification", "").replace('"', '&quot;')
            if met and direction == "pathogenic":
                cls = "path-met"
            elif met and direction == "benign":
                cls = "ben-met"
            else:
                cls = "not-met"
            title = f"{strength}: {justification}" if justification else strength
            return f'<span class="acmg-pill {cls}" title="{title}">{code}</span>'

        pills_html = ""
        # Pathogenic row
        path_pills = "".join(_pill(c) for c in path_criteria)
        if path_pills:
            pills_html += f'<div style="margin:4px 0">{path_pills}</div>'
        # Benign row
        ben_pills = "".join(_pill(c) for c in ben_criteria)
        if ben_pills:
            pills_html += f'<div style="margin:4px 0">{ben_pills}</div>'
        # Other
        other_pills = "".join(_pill(c) for c in other_criteria)
        if other_pills:
            pills_html += f'<div style="margin:4px 0">{other_pills}</div>'

        st.markdown(pills_html, unsafe_allow_html=True)

    st.divider()

    # ===================================================================
    # TABBED EVIDENCE LAYOUT
    # ===================================================================
    tab_freq, tab_clinvar, tab_lit, tab_struct, tab_pubdata, tab_pathways, tab_cc = st.tabs(
        ["Frequencies", "ClinVar", "Literature", "Structure", "Public Datasets", "Pathways", "Case-Control"]
    )

    # ---- TAB: Frequencies ----
    with tab_freq:
        if gnomad:
            col_gn, col_pred = st.columns([3, 2])

            with col_gn:
                gn_vid = gnomad.get("gnomad_variant_id", "")
                rsid_val = gnomad.get("rsid", "")
                coord_str = gnomad.get("coordinates", "")
                gn_dataset = gnomad.get("dataset", "gnomad_r4")

                link_parts = []
                if coord_str: link_parts.append(f"**Coords:** {coord_str}")
                if gn_vid: link_parts.append(f"**gnomAD:** {_gnomad_link(gn_vid, gn_dataset)}")
                if rsid_val: link_parts.append(f"**dbSNP:** {_dbsnp_link(rsid_val)}")
                if link_parts:
                    st.markdown(" &nbsp;|&nbsp; ".join(link_parts))

                st.markdown("#### gnomAD Cohorts")

                # Use ordered cohorts list
                cohorts = gnomad.get("cohorts", [])
                if cohorts:
                    cohort_rows = []
                    for coh in cohorts:
                        row = {
                            "Cohort": coh.get("label", coh.get("key", "")),
                            "Status": "Found" if coh.get("available") else "Not Available",
                            "Global AF": _af(coh.get("global_af")) if coh.get("available") else "\u2014",
                            "Exome AC/AN": _ac_an(coh, "exome") if coh.get("available") else "\u2014",
                            "Genome AC/AN": _ac_an(coh, "genome") if coh.get("available") else "\u2014",
                            "Hom": coh.get("hom", 0) if coh.get("available") else "\u2014",
                        }
                        cohort_rows.append(row)
                    st.dataframe(cohort_rows, use_container_width=True, hide_index=True)
                elif gnomad.get("variant_in_gnomad"):
                    # Fallback to legacy 3-cohort display
                    detail_rows = []
                    for key, label in [("overall", "Overall"), ("non_cancer", "Non-cancer"), ("controls", "Controls")]:
                        block = gnomad.get(key, {})
                        if block.get("available"):
                            detail_rows.append({
                                "Cohort": block.get("label", label),
                                "Global AF": _af(block.get("global_af")),
                                "Exome AC/AN": _ac_an(block, "exome"),
                                "Genome AC/AN": _ac_an(block, "genome"),
                                "Hom": block.get("hom", 0),
                            })
                    if detail_rows:
                        st.dataframe(detail_rows, use_container_width=True, hide_index=True)
                else:
                    st.info("Variant **not found** in gnomAD (supports PM2)")

                # ACMG frequency flags with cohort label
                freq_cohort_label = gnomad.get("freq_cohort_label", acmg_crit.get("freq_cohort", "controls"))
                freq_af = acmg_crit.get("freq_af_used")
                freq_af_str = f"{freq_af:.6f}" if freq_af is not None else "N/A"
                st.markdown(
                    f"{'\u2705' if acmg_crit.get('BA1_met') else '\u274C'} **BA1** (>5%) &nbsp;&nbsp; "
                    f"{'\u2705' if acmg_crit.get('BS1_met') else '\u274C'} **BS1** (>1%) &nbsp;&nbsp; "
                    f"{'\u2705' if acmg_crit.get('PM2_met') else '\u274C'} **PM2** (<0.01%) &nbsp;&nbsp; "
                    f"*(Using **{freq_cohort_label}** as reference &middot; AF = {freq_af_str})*"
                )

                # Population table (use first available cohort with populations)
                ref_cohort = None
                for coh in cohorts:
                    if coh.get("available") and coh.get("populations"):
                        ref_cohort = coh
                        break
                if ref_cohort:
                    pop_rows = []
                    for pid in POP_ORDER:
                        pd = ref_cohort["populations"].get(pid, {})
                        if isinstance(pd, dict) and pd:
                            pop_rows.append({
                                "Population": POP_NAMES.get(pid, pid),
                                "AF": _af(pd.get("af")),
                                "AC": pd.get("ac", "\u2014"),
                                "AN": pd.get("an", "\u2014"),
                            })
                    if pop_rows:
                        with st.expander(f"Population breakdown ({ref_cohort.get('label', '')})"):
                            st.dataframe(pop_rows, use_container_width=True, hide_index=True)

            with col_pred:
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

                # Gene constraint + domains
                constraint = gnomad.get("gene_constraint")
                uniprot_data = gnomad.get("uniprot", {})
                if constraint:
                    st.markdown("#### Gene Constraint")
                    uniprot_acc = uniprot_data.get("accession", "")
                    uniprot_link = f"[{uniprot_acc}](https://www.uniprot.org/uniprotkb/{uniprot_acc})" if uniprot_acc else ""
                    st.markdown(
                        f"**mis_z:** {constraint.get('mis_z', 0):.2f} "
                        f"({constraint.get('missense_interpretation', '')}) &nbsp;|&nbsp; "
                        f"**pLI:** {constraint.get('pli', 0):.4f} &nbsp;|&nbsp; "
                        f"**LOEUF:** {constraint.get('loeuf', 0):.3f} "
                        f"({constraint.get('lof_interpretation', '')})"
                    )
                    if uniprot_link:
                        st.markdown(f"**UniProt:** {uniprot_link}")

                domains = uniprot_data.get("domains", [])
                if domains:
                    st.markdown(f"#### Domains ({uniprot_data.get('protein_length', '?')} aa)")
                    domain_rows = []
                    for d in domains:
                        is_hit = (uniprot_data.get("variant_in_domain") or {}).get("start") == d["start"]
                        domain_rows.append({
                            "Name": d["description"],
                            "Position": f"{d['start']}-{d['end']}",
                            "Hit": "\U0001f534" if is_hit else "",
                        })
                    st.dataframe(domain_rows, use_container_width=True, hide_index=True)

                # PM1 / PM4 / BP3
                pm1_met = acmg_crit.get("PM1_met", False)
                pm4_met = acmg_crit.get("PM4_met", False)
                bp3_met = acmg_crit.get("BP3_met", False)
                if pm1_met or pm4_met or bp3_met:
                    st.markdown(
                        f"{'\u2705' if pm1_met else '\u274C'} **PM1** "
                        f"{'\u2705' if pm4_met else '\u274C'} **PM4** "
                        f"{'\u2705' if bp3_met else '\u274C'} **BP3**"
                    )
        else:
            st.info("No gnomAD data available.")

    # ---- TAB: ClinVar ----
    with tab_clinvar:
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

    # ---- TAB: Literature ----
    with tab_lit:
        if pubmed_data and isinstance(pubmed_data, dict) and pubmed_data.get("available"):
            col_dis, col_pubs = st.columns(2)
            with col_dis:
                rsid_lit = pubmed_data.get("rsid", "")
                litvar_link = f"[LitVar](https://www.ncbi.nlm.nih.gov/research/litvar2/docsum?query={rsid_lit})" if rsid_lit else ""
                st.markdown(
                    f"**Publications:** {pubmed_data.get('pmids_count', 0)} &nbsp;|&nbsp; "
                    f"**Case Reports:** {pubmed_data.get('case_report_count', 0)} &nbsp;|&nbsp; "
                    f"**Functional Studies:** {pubmed_data.get('functional_study_count', 0)} &nbsp;|&nbsp; "
                    f"{litvar_link}"
                )

                diseases = pubmed_data.get("diseases", [])
                if diseases:
                    st.markdown("**Disease Associations:**")
                    disease_rows = [{"Disease": dname, "Pubs": dcount} for dname, dcount in diseases[:10]]
                    st.dataframe(disease_rows, use_container_width=True, hide_index=True)

                related_genes = pubmed_data.get("related_genes", [])
                related_chems = pubmed_data.get("related_chemicals", [])
                if related_genes:
                    gene_strs = [f"{g['name']} ({g['count']})" for g in related_genes[:5]]
                    st.markdown(f"**Related Genes:** {', '.join(gene_strs)}")
                if related_chems:
                    chem_strs = [f"{c['name']} ({c['count']})" for c in related_chems[:5]]
                    st.markdown(f"**Related Chemicals:** {', '.join(chem_strs)}")

            with col_pubs:
                pubs = pubmed_data.get("publications", [])
                if pubs:
                    st.markdown(f"**Recent Publications** ({len(pubs)} of {pubmed_data.get('pmids_count', 0)}):")
                    pub_rows = []
                    for pub in pubs:
                        pmid = pub.get("pmid", "")
                        pmid_link = f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)"
                        pub_rows.append({
                            "PMID": pmid_link, "Year": pub.get("year", ""),
                            "Title": pub.get("title", "")[:100],
                            "Journal": pub.get("journal", ""),
                            "Type": ", ".join(pub.get("pub_types", [])),
                        })
                    st.dataframe(pub_rows, use_container_width=True, hide_index=True)

                # BioMCP articles
                biomcp_arts = pubmed_data.get("biomcp_articles", [])
                if biomcp_arts:
                    st.markdown(f"**Europe PMC** ({len(biomcp_arts)} articles)")
                    art_rows = []
                    for art in biomcp_arts:
                        pmid = art.get("pmid", "")
                        pmid_link = f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else ""
                        art_rows.append({
                            "PMID": pmid_link, "Date": art.get("date", ""),
                            "Title": art.get("title", "")[:120],
                            "Citations": art.get("citation_count", 0),
                        })
                    st.dataframe(art_rows, use_container_width=True, hide_index=True)

            # PubTator3
            if pubmed_data.get("pubtator3_articles"):
                with st.expander(f"PubTator3 NLP-Annotated ({len(pubmed_data['pubtator3_articles'])} articles)"):
                    pt3_arts = pubmed_data["pubtator3_articles"]
                    pt3_rows = []
                    for art in pt3_arts:
                        pmid = art.get("pmid", "")
                        pmid_link = f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else ""
                        pt3_rows.append({
                            "PMID": pmid_link,
                            "Score": f"{art.get('score', 0):.0f}",
                            "Title": art.get("title", "")[:120],
                        })
                    st.dataframe(pt3_rows, use_container_width=True, hide_index=True)

                    pt3_annots = pubmed_data.get("pubtator3_annotations", [])
                    if pt3_annots:
                        all_variants = set()
                        all_diseases = set()
                        all_genes = set()
                        for ann_article in pt3_annots:
                            for ann in ann_article.get("annotations", []):
                                atype = ann.get("type", "")
                                atext = ann.get("text", "")
                                if atype == "Variant": all_variants.add(atext)
                                elif atype == "Disease": all_diseases.add(atext)
                                elif atype == "Gene": all_genes.add(atext)
                        if all_variants:
                            st.markdown(f"**Variants:** {', '.join(sorted(all_variants)[:15])}")
                        if all_diseases:
                            st.markdown(f"**Diseases:** {', '.join(sorted(all_diseases)[:15])}")
        else:
            st.info("No literature data available.")

    # ---- TAB: Structure ----
    with tab_struct:
        protein_info = fs.get("protein_info")
        if protein_info:
            col_func, col_pdb = st.columns(2)
            with col_func:
                accession = protein_info.get("accession", "")
                uniprot_link = f"[{accession}](https://www.uniprot.org/uniprot/{accession})" if accession else ""
                af_link = f"[AlphaFold](https://alphafold.ebi.ac.uk/entry/{accession})" if accession else ""
                st.markdown(
                    f"**{protein_info.get('name', 'N/A')}** &nbsp;|&nbsp; "
                    f"**Length:** {protein_info.get('length', 0)} aa &nbsp;|&nbsp; "
                    f"{uniprot_link} &nbsp;|&nbsp; {af_link}"
                )

                func_text = protein_info.get("function", "")
                if func_text:
                    st.markdown(f"**Function:** {func_text[:600]}{'...' if len(func_text) > 600 else ''}")

                interpro = protein_info.get("interpro_domains", [])
                if interpro:
                    st.markdown(f"**InterPro Domains** ({len(interpro)})")
                    domain_rows = [{"Accession": d.get("accession", ""), "Name": d.get("name", ""), "Type": d.get("domain_type", "")} for d in interpro]
                    st.dataframe(domain_rows, use_container_width=True, hide_index=True)

            with col_pdb:
                pdb_data = fs.get("pdb") or {}
                pdb_structs = protein_info.get("pdb_structures", [])
                if pdb_structs:
                    st.markdown(f"**PDB Structures** ({pdb_data.get('count', len(pdb_structs))} total)")
                    pdb_rows = []
                    for s in pdb_structs[:10]:
                        pdb_id = str(s).split(" ")[0] if isinstance(s, str) else str(s)
                        pdb_link = f"[{pdb_id}](https://www.rcsb.org/structure/{pdb_id})"
                        pdb_rows.append({"Structure": pdb_link, "Details": str(s)})
                    st.dataframe(pdb_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No protein structure data available.")

    # ---- TAB: Public Datasets ----
    with tab_pubdata:
        gwas_data = fs.get("gwas")
        clingen_data = fs.get("clingen")
        civic_data = fs.get("civic")

        # --- GWAS Catalog ---
        st.markdown("#### GWAS Catalog")
        if gwas_data:
            st.markdown(f"**{len(gwas_data)} associations** from NHGRI-EBI GWAS Catalog")
            gwas_rows = []
            for a in gwas_data[:20]:
                pv = a.get("p_value")
                pv_str = f"{pv:.2e}" if pv is not None and pv < 0.001 else str(pv) if pv is not None else ""
                pmid = a.get("pmid", "")
                pmid_link = f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid}/)" if pmid else ""
                gwas_rows.append({
                    "Trait": a.get("trait_name", ""), "p-value": pv_str,
                    "Effect": f"{a.get('effect_size', '')} ({a.get('effect_type', '')})" if a.get("effect_size") else "",
                    "Risk Allele": a.get("risk_allele", ""),
                    "RAF": a.get("risk_allele_frequency", ""),
                    "PMID": pmid_link,
                })
            st.dataframe(gwas_rows, use_container_width=True, hide_index=True)
        else:
            st.info("No GWAS associations found for this variant.")

        st.divider()

        # --- ClinGen ---
        col_clingen, col_civic = st.columns(2)
        with col_clingen:
            st.markdown("#### ClinGen")
            if clingen_data:
                validity = clingen_data.get("validity", [])
                if validity:
                    st.markdown("**Gene-Disease Validity**")
                    val_rows = [{"Disease": v.get("disease", ""), "Classification": v.get("classification", ""),
                                 "MOI": v.get("moi", ""), "Date": v.get("review_date", "")} for v in validity]
                    st.dataframe(val_rows, use_container_width=True, hide_index=True)

                haplo = clingen_data.get("haploinsufficiency", "")
                triplo = clingen_data.get("triplosensitivity", "")
                if haplo or triplo:
                    st.markdown("**Dosage Sensitivity**")
                    st.markdown(f"**Haploinsufficiency:** {haplo or 'N/A'} &nbsp;|&nbsp; **Triplosensitivity:** {triplo or 'N/A'}")
            else:
                st.info("No ClinGen data available.")

        with col_civic:
            st.markdown("#### CIViC")
            if civic_data:
                cached = civic_data.get("cached_evidence", [])
                if cached:
                    st.markdown(f"**{len(cached)} evidence items**")
                    by_type: dict[str, list] = {}
                    for ev in cached:
                        by_type.setdefault(ev.get("evidence_type", "OTHER"), []).append(ev)
                    for etype, items in sorted(by_type.items()):
                        st.markdown(f"*{etype}* ({len(items)})")
                        ev_rows = [{
                            "ID": ev.get("name", ""),
                            "Level": ev.get("evidence_level", ""),
                            "Significance": ev.get("significance", ""),
                            "Disease": ev.get("disease", ""),
                            "Therapies": ", ".join(ev.get("therapies", [])) or "\u2014",
                        } for ev in items]
                        st.dataframe(ev_rows, use_container_width=True, hide_index=True)

                assertions = civic_data.get("graphql_assertions", [])
                if assertions:
                    st.markdown("**CIViC Assertions**")
                    assert_rows = [{
                        "ID": a.get("name", ""),
                        "AMP Level": a.get("amp_level", ""),
                        "Disease": a.get("disease", ""),
                        "Therapies": ", ".join(a.get("therapies", [])) or "\u2014",
                    } for a in assertions]
                    st.dataframe(assert_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No CIViC data available.")

        st.divider()

        # --- TCGA ---
        st.markdown("#### TCGA")
        st.markdown(
            '<div style="background:#f0f2f6; border-radius:8px; padding:16px; text-align:center; '
            'color:#555; font-size:0.95rem; border:1px dashed #ccc;">'
            '<strong>TCGA germline and somatic prevalence coming soon</strong><br>'
            '<span style="font-size:0.8rem; color:#888;">'
            'Planned: germline carrier frequency across 33 cancer types, '
            'somatic mutation hotspots, and co-occurrence analysis from TCGA PanCanAtlas'
            '</span></div>',
            unsafe_allow_html=True,
        )

    # ---- TAB: Pathways ----
    with tab_pathways:
        pathway_data = fs.get("pathways")
        drug_data = fs.get("druggability")

        col_pw, col_drug = st.columns(2)

        with col_pw:
            if pathway_data:
                st.markdown(f"**Reactome Pathways** ({len(pathway_data)})")
                pw_rows = []
                for p in pathway_data[:20]:
                    pid = p.get("id", "")
                    reactome_link = f"[{pid}](https://reactome.org/content/detail/{pid})" if pid.startswith("R-") else pid
                    pw_rows.append({"ID": reactome_link, "Name": p.get("name", "")})
                st.dataframe(pw_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No pathway data available.")

        with col_drug:
            if drug_data and (drug_data.get("categories") or drug_data.get("interactions")):
                st.markdown("**Druggability (DGIdb)**")
                cats = drug_data.get("categories", [])
                if cats:
                    st.markdown(f"**Categories:** {', '.join(cats)}")

                interactions = drug_data.get("interactions", [])
                if interactions:
                    approved = [i for i in interactions if i.get("approved")]
                    st.markdown(f"**Interactions:** {len(interactions)} total, **{len(approved)} approved**")
                    if approved:
                        drug_rows = [{
                            "Drug": i.get("drug", ""),
                            "Types": ", ".join(i.get("interaction_types", [])) if i.get("interaction_types") else "\u2014",
                            "Score": i.get("score", ""),
                        } for i in approved[:10]]
                        st.dataframe(drug_rows, use_container_width=True, hide_index=True)
            else:
                st.info("No druggability data available.")

    # ---- TAB: Case-Control ----
    with tab_cc:
        if gnomad and af_data.get("variant_in_gnomad"):
            st.markdown("Compare your cohort's variant frequency against gnomAD populations.")

            # Fisher's control selection — explicit label
            # Determine which controls cohort will be used
            ctrl_block = gnomad.get("controls", {})
            ctrl_label = ctrl_block.get("label", "Controls")
            if not ctrl_block.get("available"):
                ctrl_block = gnomad.get("overall", {})
                ctrl_label = ctrl_block.get("label", "Overall")

            st.info(f"Using **{ctrl_label}** as reference population for Fisher's test")

            use_eth = st.checkbox("Provide ethnicity-specific counts", key="use_eth_input")

            eth_case_data: dict[str, dict[str, int]] = {}
            if use_eth:
                st.markdown("**Per-ancestry case counts**")
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

            eth_sum_carriers = sum(d["carriers"] for d in eth_case_data.values())
            eth_sum_total = sum(d["total"] for d in eth_case_data.values())

            st.markdown("**Overall cohort**" + (" *(auto-summed)*" if eth_case_data else ""))
            oc1, oc2 = st.columns(2)
            with oc1:
                default_carriers = eth_sum_carriers if eth_case_data else 1
                overall_carriers = st.number_input("Carriers", min_value=0, value=default_carriers, step=1, key="cc_overall_carriers")
            with oc2:
                default_total = eth_sum_total if eth_case_data else 100
                overall_total = st.number_input("Total samples", min_value=1, value=max(default_total, 1), step=1, key="cc_overall_total")

            case_data = {"overall": {"carriers": overall_carriers, "total": overall_total}}
            case_data.update(eth_case_data)
            eth_with_data = [p for p in case_data if p != "overall"]
            n_eth_with_data = len(eth_with_data)

            run_cc = st.button("Run Case-Control Analysis", key="run_cc")

            if run_cc and overall_total > 0:
                from tools.case_control import run_case_control_analysis

                ctrl_exome = ctrl_block.get("exome") or {}
                ctrl_genome = ctrl_block.get("genome") or {}
                ds_data = {
                    "ac": ctrl_exome.get("ac", 0) + ctrl_genome.get("ac", 0),
                    "an": ctrl_exome.get("an", 0) + ctrl_genome.get("an", 0),
                    "populations": ctrl_block.get("populations", {}),
                    "dataset_label": ctrl_label,
                }

                ctrl_ds = ctrl_block.get("dataset", gnomad.get("dataset", ""))
                gn_link = _gnomad_link(gnomad.get("gnomad_variant_id", ""), ctrl_ds)
                st.markdown(f"**Control cohort:** {ctrl_label} {gn_link}")

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
                            "Case": f"{ar['case_ac']}/{ar['case_an']//2}",
                            "Control": f"{ar['control_ac']}/{ar['control_an']}",
                            "Case AF": f"{ar['case_af']:.4f}",
                            "Ctrl AF": f"{ar['control_af']:.6f}",
                            "OR": f"{or_v:.2f}" if or_v and or_v != float("inf") else "Inf",
                            "p": f"{pv:.2e}" if pv and pv < 0.001 else f"{pv:.4f}" if pv else "N/A",
                            "Sig": "\u2705" if ar.get("significant") else "",
                        })
                        if has_user_data and or_v and or_v != float("inf") and or_v > 0:
                            plot_data.append({
                                "pop": ar.get("population", ""), "or": or_v,
                                "ci_lo": ar.get("ci_lower", or_v * 0.5),
                                "ci_hi": ar.get("ci_upper", or_v * 2),
                                "sig": ar.get("significant", False),
                            })

                    st.dataframe(rows, use_container_width=True, hide_index=True)

                    if plot_data:
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        pops_sorted = sorted(plot_data, key=lambda x: x["or"])
                        fig.add_trace(go.Scatter(
                            x=[d["or"] for d in pops_sorted], y=[d["pop"] for d in pops_sorted],
                            mode="markers",
                            marker=dict(size=10, color=["#ff4b4b" if d["sig"] else "#888" for d in pops_sorted]),
                            error_x=dict(type="data", symmetric=False,
                                array=[d["ci_hi"] - d["or"] for d in pops_sorted],
                                arrayminus=[d["or"] - d["ci_lo"] for d in pops_sorted]),
                            hovertemplate="%{y}<br>OR=%{x:.2f}<extra></extra>",
                        ))
                        fig.add_vline(x=1, line_dash="dash", line_color="gray")
                        fig.update_layout(
                            title="Forest Plot — OR by Ancestry (vs controls)",
                            xaxis_title="Odds Ratio (log scale)", xaxis_type="log",
                            height=max(300, len(pops_sorted) * 50 + 100),
                            showlegend=False, margin=dict(l=200), font=dict(size=13),
                        )
                        st.plotly_chart(fig, use_container_width=True)

                if n_eth_with_data >= 2:
                    glm = cc_result.get("weighted_glm", {})
                    if glm.get("p_value") is not None:
                        glm_pv = glm["p_value"]
                        glm_color = "red" if glm_pv < 0.05 else "green"
                        st.markdown(f"**Weighted GLM:** {glm['interpretation']} :{glm_color}[p={glm_pv:.4f}]")
                    elif glm.get("interpretation"):
                        st.markdown(f"**Weighted GLM:** {glm['interpretation']}")
                elif use_eth:
                    st.info("Provide at least 2 ethnicities for weighted GLM.")
        else:
            st.info("Variant must be present in gnomAD for case-control analysis.")

    # ===================================================================
    # Warnings + Disclaimer + Debug (compact footer)
    # ===================================================================
    warnings = fs.get("warnings", [])
    if warnings:
        with st.expander(f"\u26A0\uFE0F Warnings ({len(warnings)})"):
            for w in warnings: st.markdown(f"- {w}")

    st.divider()
    if fs.get("disclaimer"):
        st.markdown(f'<p style="color:gray;font-size:0.75rem">{fs["disclaimer"]}</p>', unsafe_allow_html=True)

    with st.expander("\U0001f41b Debug: Full State JSON"):
        debug = {}
        for k, v in fs.items():
            try: json.dumps(v); debug[k] = v
            except: debug[k] = str(v)
        st.json(debug)
