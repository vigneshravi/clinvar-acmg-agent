"""Node 5: Structural Agent — protein structure & domain evidence via BioMCP.

Queries BioMCP for:
- Protein info (function, length, domains from InterPro, PDB structures)
- AlphaFold predicted structures (via UniProt accession)

Supports ACMG criteria:
- PM1: Variant in established functional domain (complementary to UniProt data)
"""

import logging
from typing import Any

from graph.state import VariantState
from tools.biomcp import get_protein_info

logger = logging.getLogger(__name__)


def alphafold_agent_node(state: VariantState) -> dict[str, Any]:
    """Query BioMCP for protein structural evidence.

    Fetches protein function, InterPro domains, PDB structures via BioMCP's
    unified protein endpoint (UniProt + InterPro + PDB + AlphaFold).
    """
    logger.info("alphafold_agent_node: starting structural lookup via BioMCP")

    updates: dict[str, Any] = {"current_node": "alphafold_agent"}
    warnings: list[str] = list(state.get("warnings", []))

    gene_symbol = state.get("gene_symbol")
    if not gene_symbol:
        updates["structural_error"] = "No gene symbol available for protein lookup"
        warnings.append("Structural data not available — no gene symbol")
        updates["warnings"] = warnings
        return updates

    try:
        protein_data = get_protein_info(gene_symbol)

        if not protein_data:
            updates["protein_info"] = None
            updates["structural_error"] = (
                f"BioMCP returned no protein data for {gene_symbol}"
            )
            warnings.append(
                f"BioMCP protein lookup returned no data for {gene_symbol}"
            )
            updates["warnings"] = warnings
            return updates

        # Extract structured fields
        structures = protein_data.get("structures", [])
        domains = protein_data.get("domains", [])
        function_desc = protein_data.get("function", "")
        accession = protein_data.get("accession", "")
        protein_length = protein_data.get("length", 0)
        structure_count = protein_data.get("structure_count", len(structures))

        # Build AlphaFold entry URL if we have a UniProt accession
        alphafold_url = ""
        if accession:
            alphafold_url = f"https://alphafold.ebi.ac.uk/entry/{accession}"

        updates["protein_info"] = {
            "source": "BioMCP (UniProt + InterPro + PDB)",
            "accession": accession,
            "name": protein_data.get("name", ""),
            "gene_symbol": protein_data.get("gene_symbol", gene_symbol),
            "length": protein_length,
            "function": function_desc,
            "interpro_domains": domains,
            "pdb_structures": structures[:10],
            "pdb_structure_count": structure_count,
            "alphafold_url": alphafold_url,
            "evidence_urls": protein_data.get("_meta", {}).get("evidence_urls", []),
        }

        # PDB structures
        if structures:
            updates["pdb"] = {
                "available": True,
                "count": structure_count,
                "top_structures": structures[:10],
            }
        else:
            updates["pdb"] = {"available": False, "count": 0}

        # AlphaFold info
        if accession:
            updates["alphafold"] = {
                "available": True,
                "accession": accession,
                "url": alphafold_url,
            }
        else:
            updates["alphafold"] = {"available": False}

        logger.info(
            "alphafold_agent: %s — %d InterPro domains, %d PDB structures, "
            "protein length %d aa",
            gene_symbol, len(domains), structure_count, protein_length,
        )

    except Exception as e:
        error_msg = f"BioMCP protein query failed: {str(e)}"
        updates["protein_info"] = None
        updates["structural_error"] = error_msg
        updates["errors"] = state.get("errors", []) + [error_msg]
        warnings.append(error_msg)
        logger.exception("alphafold_agent: %s", error_msg)

    updates["warnings"] = warnings
    return updates
