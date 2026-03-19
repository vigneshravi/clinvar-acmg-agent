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


def _get_selected_transcript_info(state: VariantState) -> dict[str, Any]:
    """Extract VEP annotation for the selected transcript."""
    sel_tx = state.get("selected_transcript", "")
    for tx in state.get("all_transcripts") or []:
        tid = tx.get("nm_accession") or tx.get("enst_accession")
        if tid == sel_tx:
            return tx
    return {}


def _assess_pvs1_applicability(tx_info: dict[str, Any]) -> dict[str, Any]:
    """Assess PVS1 applicability based on exon position and consequence.

    ClinGen PVS1 decision tree caveats:
    - Last exon: truncating variants may escape NMD → downgrade to PVS1_Moderate
    - Last 50bp of penultimate exon: may also escape NMD
    - Missense/in-frame: PVS1 does not apply (not a null variant)
    - Single-exon gene: NMD does not apply → downgrade
    """
    result = {
        "is_null_variant": False,
        "pvs1_applicable": False,
        "pvs1_strength": "Very Strong",  # default if applicable
        "pvs1_caveat": None,
        "exon_number": None,
        "total_exons": None,
        "is_last_exon": False,
        "is_penultimate_exon": False,
        "consequence": "",
    }

    consequences = tx_info.get("consequence_terms", [])
    exon_str = tx_info.get("exon", "")  # e.g. "19/23"
    result["consequence"] = ", ".join(consequences)

    # Check if this is a null/truncating variant
    null_consequences = {
        "frameshift_variant", "stop_gained", "splice_donor_variant",
        "splice_acceptor_variant", "start_lost", "transcript_ablation",
    }
    is_null = bool(null_consequences & set(consequences))
    result["is_null_variant"] = is_null

    if not is_null:
        result["pvs1_caveat"] = "Not a null/truncating variant — PVS1 does not apply"
        return result

    # Parse exon position
    if exon_str and "/" in exon_str:
        try:
            parts = exon_str.split("/")
            exon_num = int(parts[0])
            total_exons = int(parts[1])
            result["exon_number"] = exon_num
            result["total_exons"] = total_exons

            # Single-exon gene
            if total_exons == 1:
                result["pvs1_applicable"] = True
                result["pvs1_strength"] = "Moderate"
                result["pvs1_caveat"] = (
                    "Single-exon gene — NMD not applicable. "
                    "PVS1 downgraded to Moderate per ClinGen guidelines."
                )
                return result

            # Last exon
            if exon_num == total_exons:
                result["is_last_exon"] = True
                result["pvs1_applicable"] = True
                result["pvs1_strength"] = "Moderate"
                result["pvs1_caveat"] = (
                    f"Variant in last exon ({exon_str}) — truncated protein "
                    f"may escape NMD and produce a stable but altered protein. "
                    f"PVS1 downgraded to Moderate per ClinGen PVS1 decision tree."
                )
                return result

            # Penultimate exon (last 50bp rule handled by noting it)
            if exon_num == total_exons - 1:
                result["is_penultimate_exon"] = True
                result["pvs1_applicable"] = True
                result["pvs1_strength"] = "Strong"
                result["pvs1_caveat"] = (
                    f"Variant in penultimate exon ({exon_str}) — if within "
                    f"last 50bp, truncated protein may escape NMD. "
                    f"PVS1 downgraded to Strong per ClinGen guidelines."
                )
                return result

        except (ValueError, IndexError):
            pass

    # Standard PVS1 — null variant not in last/penultimate exon
    result["pvs1_applicable"] = True
    result["pvs1_strength"] = "Very Strong"
    return result


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

    # Transcript annotation (VEP)
    tx_info = _get_selected_transcript_info(state)
    pvs1 = _assess_pvs1_applicability(tx_info)

    if tx_info:
        vep_lines = [
            "## Variant Annotation (VEP)",
            f"- Consequence: {tx_info.get('consequence_display', 'N/A')}",
            f"- Impact: {tx_info.get('impact', 'N/A')}",
            f"- Exon: {tx_info.get('exon', 'N/A')}",
            f"- Amino acid change: {tx_info.get('amino_acids', 'N/A')}",
            f"- Position type: {tx_info.get('position_type', 'N/A')} ({tx_info.get('position_detail', '')})",
        ]
        if tx_info.get("intron"):
            vep_lines.append(f"- Intron: {tx_info.get('intron', '')}")
        parts.append("\n".join(vep_lines))

    # PVS1 assessment
    pvs1_lines = ["## PVS1 Assessment (Null Variant)"]
    pvs1_lines.append(f"- Is null/truncating variant: {pvs1['is_null_variant']}")
    if pvs1["is_null_variant"]:
        pvs1_lines.append(f"- PVS1 applicable: {pvs1['pvs1_applicable']}")
        pvs1_lines.append(f"- PVS1 strength: {pvs1['pvs1_strength']}")
        if pvs1["exon_number"]:
            pvs1_lines.append(f"- Exon position: {pvs1['exon_number']}/{pvs1['total_exons']}")
            pvs1_lines.append(f"- Last exon: {pvs1['is_last_exon']}")
            pvs1_lines.append(f"- Penultimate exon: {pvs1['is_penultimate_exon']}")
        if pvs1["pvs1_caveat"]:
            pvs1_lines.append(f"- CAVEAT: {pvs1['pvs1_caveat']}")
    else:
        pvs1_lines.append("- PVS1 does not apply (not a null/truncating variant)")
    parts.append("\n".join(pvs1_lines))

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

    # gnomAD + In Silico Predictors
    gnomad = state.get("gnomad")
    if gnomad and isinstance(gnomad, dict):
        af_data = gnomad.get("allele_frequency", {})
        predictors = gnomad.get("insilico_predictors", {})
        conservation = gnomad.get("conservation", {})
        acmg_freq = gnomad.get("acmg_criteria", {})

        gn_lines = ["## gnomAD Population Frequencies"]
        gn_lines.append(f"- Global AF: {af_data.get('global_af', 'Not found')}")
        gn_lines.append(f"- Variant in gnomAD: {af_data.get('variant_in_gnomad', False)}")
        gn_lines.append(f"- Homozygote count: {af_data.get('hom', 0)}")
        gn_lines.append(f"- Max population AF: {af_data.get('max_pop_af', 0)} ({af_data.get('max_pop_name', '')})")
        pops = af_data.get("populations", {})
        if pops:
            gn_lines.append("- Population breakdown:")
            for pop_id, pop_data in sorted(pops.items()):
                if isinstance(pop_data, dict):
                    gn_lines.append(f"  - {pop_data.get('name', pop_id)}: AF={pop_data.get('af', 0)}, AC={pop_data.get('ac', 0)}")
                else:
                    gn_lines.append(f"  - {pop_id}: AF={pop_data}")
        gn_lines.append(f"- Pre-computed: BA1={'MET' if acmg_freq.get('BA1_met') else 'not met'}, "
                        f"BS1={'MET' if acmg_freq.get('BS1_met') else 'not met'}, "
                        f"PM2={'MET' if acmg_freq.get('PM2_met') else 'not met'}")
        parts.append("\n".join(gn_lines))

        # In silico predictors
        if predictors:
            is_lines = ["## In Silico Predictors (dbNSFP + CADD)"]
            for name, val in predictors.items():
                if isinstance(val, dict):
                    score = val.get("score", val.get("interpretation", ""))
                    pred = val.get("pred", val.get("interpretation", ""))
                    is_lines.append(f"- {name}: score={score}, prediction={pred}")
                else:
                    is_lines.append(f"- {name}: {val}")
            is_lines.append(f"- Consensus: {gnomad.get('insilico_consensus', 'N/A')}")
            is_lines.append(f"- PP3={'MET' if acmg_freq.get('PP3_met') else 'not met'}, "
                            f"BP4={'MET' if acmg_freq.get('BP4_met') else 'not met'}")
            parts.append("\n".join(is_lines))
        else:
            parts.append("## In Silico Predictors\nNo predictor scores available for this variant type.")

        # Conservation
        if conservation:
            cv_lines = ["## Conservation Scores"]
            for name, val in conservation.items():
                cv_lines.append(f"- {name}: {val}")
            parts.append("\n".join(cv_lines))
    else:
        parts.append(
            "## gnomAD + In Silico Evidence\n"
            "Not available — BA1, BS1, PM2, PP3, BP4 cannot be evaluated."
        )

    # Gene constraint + domain annotations
    if gnomad and isinstance(gnomad, dict):
        constraint = gnomad.get("gene_constraint")
        uniprot_data = gnomad.get("uniprot", {})
        acmg_crit = gnomad.get("acmg_criteria", {})

        gc_lines = ["## Gene Constraint & Protein Domains"]
        if constraint:
            gc_lines.append(f"- Missense Z-score: {constraint.get('mis_z', 'N/A')}")
            gc_lines.append(f"- Missense o/e: {constraint.get('oe_mis', 'N/A')}")
            gc_lines.append(f"- Missense interpretation: {constraint.get('missense_interpretation', 'N/A')}")
            gc_lines.append(f"- pLI: {constraint.get('pli', 'N/A')}")
            gc_lines.append(f"- LOEUF: {constraint.get('loeuf', 'N/A')}")
            gc_lines.append(f"- LOF interpretation: {constraint.get('lof_interpretation', 'N/A')}")
        else:
            gc_lines.append("- Gene constraint data not available")

        if uniprot_data.get("domains"):
            gc_lines.append(f"- UniProt ({uniprot_data.get('accession', '')}): "
                            f"{len(uniprot_data['domains'])} functional domains, "
                            f"protein length {uniprot_data.get('protein_length', '?')} aa")
            for d in uniprot_data["domains"]:
                gc_lines.append(f"  - {d['type']}: {d['description']} (aa {d['start']}-{d['end']})")
            if uniprot_data.get("variant_in_domain"):
                d = uniprot_data["variant_in_domain"]
                gc_lines.append(f"- **Variant in domain:** {d['type']} '{d['description']}' (aa {d['start']}-{d['end']})")
            else:
                gc_lines.append("- Variant NOT in a known functional domain")
        if uniprot_data.get("variant_in_repeat"):
            gc_lines.append("- Variant in repeat region")

        gc_lines.append(f"- PM1 (functional domain): {'MET' if acmg_crit.get('PM1_met') else 'not met'} "
                        f"({acmg_crit.get('PM1_strength', '')}) — {acmg_crit.get('PM1_detail', '')}")
        gc_lines.append(f"- PM4 (protein length change): {'MET' if acmg_crit.get('PM4_met') else 'not met'} "
                        f"— {acmg_crit.get('PM4_BP3_detail', '')}")
        gc_lines.append(f"- BP3 (in-frame in repeat): {'MET' if acmg_crit.get('BP3_met') else 'not met'}")

        parts.append("\n".join(gc_lines))

    # Literature evidence (LitVar)
    pubmed = state.get("pubmed")
    if pubmed and isinstance(pubmed, dict) and pubmed.get("available"):
        lit_lines = [
            "## Literature Evidence (LitVar)",
            f"- Total publications: {pubmed.get('pmids_count', 0)}",
            f"- Case reports: {pubmed.get('case_report_count', 0)}",
            f"- Functional studies: {pubmed.get('functional_study_count', 0)}",
            f"- Reviews: {pubmed.get('review_count', 0)}",
            f"- Clinical significance (literature): {pubmed.get('clinical_significance', 'N/A')}",
            f"- First published: {pubmed.get('first_published', 'N/A')}",
        ]
        diseases = pubmed.get("diseases", [])
        if diseases:
            lit_lines.append("- Disease associations:")
            for dname, dcount in diseases[:5]:
                lit_lines.append(f"  - {dname}: {dcount} publications")

        pubs = pubmed.get("publications", [])
        if pubs:
            lit_lines.append(f"- Recent publications ({len(pubs)} shown):")
            for pub in pubs[:10]:
                title = pub.get("title", "")[:80]
                year = pub.get("year", "")
                types = ", ".join(pub.get("pub_types", []))
                lit_lines.append(f"  - [{year}] {title}... ({types})")

        related_genes = pubmed.get("related_genes", [])
        if related_genes:
            lit_lines.append("- Related genes in literature: " +
                             ", ".join(g["name"] for g in related_genes[:5]))

        # PubTator3 articles (direct API — NLP annotated)
        pubtator_articles = pubmed.get("pubtator3_articles", [])
        if pubtator_articles:
            lit_lines.append(f"\n### PubTator3 Articles (NLP-annotated) — {len(pubtator_articles)} results:")
            for art in pubtator_articles[:10]:
                lit_lines.append(
                    f"  - [{art.get('date', '')[:10]}] PMID:{art.get('pmid', '')} "
                    f"(score={art.get('score', 0):.0f}) "
                    f"— {art.get('title', '')} ({art.get('journal', '')})"
                )

        # PubTator3 NLP annotations (variants, diseases, genes found in articles)
        pubtator_annots = pubmed.get("pubtator3_annotations", [])
        if pubtator_annots:
            all_variants = set()
            all_diseases = set()
            for ann_article in pubtator_annots:
                for ann in ann_article.get("annotations", []):
                    if ann.get("type") == "Variant":
                        all_variants.add(ann.get("text", ""))
                    elif ann.get("type") == "Disease":
                        all_diseases.add(ann.get("text", ""))
            if all_variants:
                lit_lines.append(f"- PubTator3 mined variants: {', '.join(sorted(all_variants)[:10])}")
            if all_diseases:
                lit_lines.append(f"- PubTator3 mined diseases: {', '.join(sorted(all_diseases)[:10])}")

        # BioMCP Europe PMC articles
        biomcp_articles = pubmed.get("biomcp_articles", [])
        if biomcp_articles:
            lit_lines.append(f"\n### Europe PMC Articles (BioMCP) — {len(biomcp_articles)} results:")
            for art in biomcp_articles[:10]:
                lit_lines.append(
                    f"  - [{art.get('date', '')}] PMID:{art.get('pmid', '')} "
                    f"— {art.get('title', '')} ({art.get('journal', '')})"
                )

        parts.append("\n".join(lit_lines))
    else:
        parts.append(
            "## Literature Evidence\n"
            "Not available — PS3/BS3 (functional studies), PS4 (case reports) "
            "cannot be fully evaluated from literature."
        )

    # Structural Evidence (BioMCP: UniProt + InterPro + PDB)
    protein_info = state.get("protein_info")
    if protein_info:
        se_lines = [
            "## Structural Evidence (BioMCP: UniProt + InterPro + PDB)",
            f"- Protein: {protein_info.get('name', 'N/A')} ({protein_info.get('accession', '')})",
            f"- Length: {protein_info.get('length', 0)} aa",
        ]
        func = protein_info.get("function", "")
        if func:
            se_lines.append(f"- Function: {func[:500]}")
        domains = protein_info.get("interpro_domains", [])
        if domains:
            se_lines.append(f"- InterPro domains ({len(domains)}):")
            for d in domains[:10]:
                se_lines.append(f"  - {d.get('name', '')} ({d.get('accession', '')}) [{d.get('domain_type', '')}]")
        pdb_count = protein_info.get("pdb_structure_count", 0)
        pdb_structs = protein_info.get("pdb_structures", [])
        if pdb_structs:
            se_lines.append(f"- PDB structures: {pdb_count} total, top: {', '.join(str(s) for s in pdb_structs[:5])}")
        alphafold_url = protein_info.get("alphafold_url", "")
        if alphafold_url:
            se_lines.append(f"- AlphaFold: {alphafold_url}")
        parts.append("\n".join(se_lines))
    else:
        parts.append(
            "## Structural Evidence\n"
            "Not available — protein structural data could not be retrieved."
        )

    # CIViC Clinical Evidence
    civic = state.get("civic")
    if civic:
        cv_lines = ["## CIViC Clinical Evidence"]
        cached = civic.get("cached_evidence", [])
        if cached:
            cv_lines.append(f"- Evidence items: {len(cached)}")
            for ev in cached[:15]:
                therapies = ", ".join(ev.get("therapies", []))
                therapy_str = f" | Therapies: {therapies}" if therapies else ""
                cv_lines.append(
                    f"  - {ev.get('name', '')}: {ev.get('evidence_type', '')} "
                    f"({ev.get('evidence_level', '')}) — {ev.get('disease', '')}"
                    f"{therapy_str}"
                )
        assertions = civic.get("graphql_assertions", [])
        if assertions:
            cv_lines.append(f"- Assertions: {len(assertions)}")
            for a in assertions:
                cv_lines.append(
                    f"  - {a.get('name', '')}: AMP {a.get('amp_level', '')} — "
                    f"{a.get('disease', '')}"
                )
        parts.append("\n".join(cv_lines))
    else:
        parts.append("## CIViC Clinical Evidence\nNot available for this variant.")

    # ClinGen Gene-Disease Validity
    clingen = state.get("clingen")
    if clingen:
        cg_lines = ["## ClinGen Gene-Disease Validity"]
        validity = clingen.get("validity", [])
        for v in validity:
            cg_lines.append(
                f"- {v.get('disease', '')}: {v.get('classification', '')} "
                f"(MOI: {v.get('moi', 'N/A')}, reviewed {v.get('review_date', '')})"
            )
        haplo = clingen.get("haploinsufficiency", "")
        triplo = clingen.get("triplosensitivity", "")
        if haplo:
            cg_lines.append(f"- Haploinsufficiency: {haplo}")
        if triplo:
            cg_lines.append(f"- Triplosensitivity: {triplo}")
        parts.append("\n".join(cg_lines))
    else:
        parts.append("## ClinGen\nNot available for this gene.")

    # GWAS Catalog
    gwas = state.get("gwas")
    if gwas:
        gw_lines = [f"## GWAS Catalog ({len(gwas)} associations)"]
        for a in gwas[:10]:
            gw_lines.append(
                f"- {a.get('trait_name', '')}: p={a.get('p_value', 'N/A')}, "
                f"effect={a.get('effect_size', 'N/A')} ({a.get('effect_type', '')}), "
                f"risk allele: {a.get('risk_allele', 'N/A')}"
            )
        parts.append("\n".join(gw_lines))
    else:
        parts.append("## GWAS Catalog\nNo associations found.")

    # Pathways (Reactome via BioMCP)
    pathways = state.get("pathways")
    if pathways:
        pw_lines = [f"## Reactome Pathways ({len(pathways)})"]
        for p in pathways[:10]:
            pw_lines.append(f"- {p.get('name', '')} ({p.get('id', '')})")
        parts.append("\n".join(pw_lines))

    # Druggability (DGIdb via BioMCP)
    druggability = state.get("druggability")
    if druggability:
        dg_lines = ["## Druggability (DGIdb)"]
        cats = druggability.get("categories", [])
        if cats:
            dg_lines.append(f"- Categories: {', '.join(cats)}")
        interactions = druggability.get("interactions", [])
        if interactions:
            approved = [i for i in interactions if i.get("approved")]
            dg_lines.append(f"- Drug interactions: {len(interactions)} total, {len(approved)} approved")
        parts.append("\n".join(dg_lines))

    # Instructions
    parts.append(
        "## Instructions\n"
        "Evaluate ALL of the following ACMG criteria based on the evidence above:\n\n"
        "### PVS1 — Null variant:\n"
        "- PVS1: Null variant (frameshift, nonsense, splice site, initiation codon) "
        "in a gene where loss of function is a known mechanism of disease.\n"
        "- **CRITICAL CAVEATS (ClinGen PVS1 decision tree):**\n"
        "  - LAST EXON: Truncating variants in the last exon may escape NMD "
        "and produce a stable truncated protein → downgrade PVS1 to **Moderate**\n"
        "  - PENULTIMATE EXON (last 50bp): May also escape NMD → downgrade to **Strong**\n"
        "  - SINGLE-EXON GENE: NMD does not apply → downgrade to **Moderate**\n"
        "  - NOT a null variant (missense, in-frame, synonymous): PVS1 does NOT apply\n"
        "- Use the PVS1 Assessment section above which pre-computes these caveats. "
        "If it says PVS1_strength=Moderate or Strong, use that strength, NOT Very Strong.\n\n"
        "### From Literature (LitVar):\n"
        "- PS3: Well-established functional studies showing damaging effect (Strong Pathogenic)\n"
        "  Look for research support / comparative / functional study publication types.\n"
        "  High publication count + disease associations supports PS3.\n"
        "- PS4: Variant prevalence significantly increased in affected vs controls (Strong Pathogenic)\n"
        "  Look for case reports, epidemiological studies.\n"
        "  Multiple case reports across different populations supports PS4.\n"
        "- If literature data unavailable, set PS3/PS4 as not evaluable.\n\n"
        "### From ClinVar:\n"
        "- PS1: Same amino acid change as established pathogenic variant\n"
        "- PP5: Reputable source reports variant as pathogenic "
        "(upgrade to Strong for expert panel 3+ stars, Moderate for 2 stars)\n"
        "- BP6: Reputable source reports variant as benign "
        "(upgrade to Strong for expert panel 3+ stars, Moderate for 2 stars)\n"
        "- PM5: Novel missense at same position as known pathogenic missense\n\n"
        "### From gene constraint + protein domains:\n"
        "- PM1: Missense variant in a functional domain (UniProt) in a missense-constrained gene\n"
        "  Domain + constraint Z>2.0 = Moderate; domain only = Supporting\n"
        "- PM4: In-frame deletion/insertion in non-repeat region causing protein length change (Moderate)\n"
        "- BP3: In-frame deletion/insertion in a repetitive region (Supporting Benign)\n"
        "  Use pre-computed PM1_met, PM4_met, BP3_met flags.\n\n"
        "### From gnomAD allele frequency (using controls cohort):\n"
        "- BA1: Allele frequency >5% in any gnomAD population (Stand-alone Benign)\n"
        "- BS1: Allele frequency >1%, greater than expected for rare disease (Strong Benign)\n"
        "- PM2: Absent from gnomAD or extremely rare AF<0.01% (Supporting Pathogenic)\n\n"
        "### From in silico predictors:\n"
        "- PP3: Multiple computational tools predict damaging effect (Supporting Pathogenic)\n"
        "  Use pre-computed consensus. REVEL>0.75, CADD>25, AlphaMissense>0.564 = damaging\n"
        "- BP4: Multiple computational tools predict no impact (Supporting Benign)\n"
        "  REVEL<0.15, CADD<15 = benign\n\n"
        "### From CIViC clinical evidence:\n"
        "- CIViC evidence items with evidence levels A/B support clinical significance.\n"
        "- Predictive evidence items indicate therapy response/resistance.\n"
        "- Diagnostic/prognostic evidence strengthens disease association.\n"
        "- CIViC assertions with AMP TIER_I indicate strong clinical evidence.\n\n"
        "### From ClinGen gene-disease validity:\n"
        "- Definitive gene-disease validity supports PVS1 applicability.\n"
        "- Haploinsufficiency evidence supports LOF mechanism of disease.\n"
        "- ClinGen classifications strengthen overall confidence.\n\n"
        "### From GWAS Catalog:\n"
        "- Significant associations (p < 5e-8) support PS4 when applicable.\n\n"
        "### Rules:\n"
        "- Use the pre-computed ACMG criteria flags (BA1_met, PM2_met, PP3_met, PVS1 strength) "
        "from the evidence sections — do not override them unless you have a specific reason.\n"
        "- For any criterion that cannot be evaluated, set met=false and explain why.\n"
        "- Apply ACMG combining rules to determine final classification.\n"
        "- BA1 alone is sufficient for Benign (Stand-alone).\n"
        "- Consider CIViC, ClinGen, and structural data as supporting context for your reasoning.\n"
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
