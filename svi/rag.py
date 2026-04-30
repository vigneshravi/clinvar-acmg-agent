"""RAG layer — FAISS index over ACMG/SVI/VCEP guideline corpus.

Paths adapted from FinalTermProject_28Apr2026 (2026-04-30 integration):
    KB_DIR    -> PathoMAN2.0/svi/knowledge_base/
    INDEX_DIR -> PathoMAN2.0/cache/svi_faiss_index/

What goes into the index (per the Ravichandran 2026 systematic review S4.3):
- Richards et al. 2015 ACMG/AMP standards (full text)
- ClinGen SVI working group amendments (PVS1 decision tree, PM2 downgrade,
  Pejaver PP3 calibration)
- ENIGMA VCEP gene-specific specifications for BRCA1/BRCA2
- Riggs 2020 ClinGen Dosage / HI Score documentation
- Tavtigian 2018 Bayesian framework

What is NOT indexed (corpus circularity / wrong-tool):
- ClinVar P/LP labels (deprecated PP5/BP6 by ClinGen SVI 2018)
- gnomAD frequencies (deterministic API call, not RAG-shaped)
- REVEL / CADD scores (numeric features, not language)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# --- Path constants (adapted for PathoMAN2.0 layout) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # PathoMAN2.0/
KB_DIR = PROJECT_ROOT / "svi" / "knowledge_base"
INDEX_DIR = PROJECT_ROOT / "cache" / "svi_faiss_index"

# Per Week 12 lecture (Carlo Unda 2026)
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 80


def load_embeddings():
    """Load the sentence-transformers embedding model.

    Lazy-imported so this module can be imported even when langchain_huggingface
    is not installed (UI degrades gracefully and shows a guardrails error
    rather than crashing the whole Streamlit app).
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents() -> List[Any]:
    """Load all knowledge-base documents with source metadata."""
    from langchain_core.documents import Document  # type: ignore
    docs: List[Any] = []
    for fp in sorted(KB_DIR.glob("*.md")) + sorted(KB_DIR.glob("*.txt")):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            continue

        source_type = "guideline"
        nm = fp.name.lower()
        if "richards" in nm:
            source_type = "Richards2015"
        elif "abou_tayoun" in nm or "pvs1" in nm:
            source_type = "AbouTayoun2018_PVS1"
        elif "pejaver" in nm or "pp3" in nm:
            source_type = "Pejaver2022_PP3calibration"
        elif "enigma" in nm or "vcep" in nm:
            source_type = "ENIGMA_VCEP_BRCA12"
        elif "riggs" in nm or "clingen_dosage" in nm:
            source_type = "Riggs2020_ClinGenDosage"
        elif "tavtigian" in nm or "bayesian" in nm:
            source_type = "Tavtigian2018_Bayesian"

        docs.append(Document(
            page_content=text,
            metadata={"source": source_type, "filename": fp.name, "path": str(fp)},
        ))
    logger.info("Loaded %d source documents from %s", len(docs), KB_DIR)
    return docs


def chunk_documents(docs: List[Any]) -> List[Any]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_id"] = f"{c.metadata.get('source', 'doc')}-{i:04d}"
    logger.info("Split %d docs into %d chunks", len(docs), len(chunks))
    return chunks


def build_index(force_rebuild: bool = False):
    """Build (or load cached) FAISS index from svi/knowledge_base/."""
    from langchain_community.vectorstores import FAISS  # type: ignore
    embeddings = load_embeddings()

    if INDEX_DIR.exists() and not force_rebuild:
        try:
            logger.info("Loading cached FAISS index from %s", INDEX_DIR)
            return FAISS.load_local(
                str(INDEX_DIR),
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception as e:
            logger.warning("Cache load failed (%s); rebuilding", e)

    docs = load_documents()
    if not docs:
        raise RuntimeError(
            f"No documents found in {KB_DIR}. Add .md or .txt files first."
        )
    chunks = chunk_documents(docs)
    logger.info("Embedding %d chunks (this may take ~30s on first run)", len(chunks))
    vs = FAISS.from_documents(chunks, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(INDEX_DIR))
    logger.info("Saved FAISS index to %s", INDEX_DIR)
    return vs


def retrieve(vs, query: str, k: int = 6, score_floor: float = 0.0) -> List[Dict[str, Any]]:
    """Retrieve top-k chunks for a query. Returns list of dicts with citations."""
    results = vs.similarity_search_with_score(query, k=k)
    out = []
    for doc, score in results:
        # FAISS returns L2 distance; convert to similarity (lower distance = more similar)
        similarity = 1.0 / (1.0 + float(score))
        if similarity < score_floor:
            continue
        out.append({
            "chunk_id": doc.metadata.get("chunk_id"),
            "source": doc.metadata.get("source"),
            "filename": doc.metadata.get("filename"),
            "score": similarity,
            "text": doc.page_content,
        })
    return out


def build_query(state: Dict[str, Any]) -> str:
    """Construct a retrieval query from the current pipeline state."""
    parts = ["ACMG variant classification"]

    # Pull consequence_terms from selected_transcript or vep
    cs: List[str] = []
    sel_tx = state.get("selected_transcript")
    if state.get("all_transcripts"):
        for tx in state["all_transcripts"]:
            tid = tx.get("nm_accession") or tx.get("enst_accession")
            if tid == sel_tx:
                cs = tx.get("consequence_terms", []) or []
                break
    if not cs and state.get("consequence_terms"):
        cs = state.get("consequence_terms") or []

    if cs:
        parts.append(" ".join(cs))
    if "frameshift_variant" in cs or "stop_gained" in cs or \
       "splice_donor_variant" in cs or "splice_acceptor_variant" in cs:
        parts.append("PVS1 null variant loss of function decision tree NMD")

    gnomad = state.get("gnomad") or {}
    af_data = gnomad.get("allele_frequency", {}) if isinstance(gnomad, dict) else {}
    if af_data.get("global_af") is not None or state.get("global_af") is not None:
        parts.append("PM2 BA1 BS1 allele frequency threshold")

    if (gnomad and gnomad.get("insilico_predictors")) or state.get("revel") is not None:
        parts.append("PP3 BP4 REVEL Pejaver computational predictor calibration")

    gene = state.get("gene_symbol")
    if gene in {"BRCA1", "BRCA2"}:
        parts.append(f"ENIGMA VCEP {gene} specification")

    return " | ".join(parts)
