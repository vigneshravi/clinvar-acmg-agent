"""SQLite cache layer for transcript and variant lookups."""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "transcript_cache.db")

HEREDITARY_CANCER_GENES = [
    "BRCA1", "BRCA2", "TP53", "PALB2", "CHEK2", "ATM",
    "MLH1", "MSH2", "MSH6", "PMS2", "EPCAM",
    "APC", "MUTYH", "PTEN", "STK11", "CDH1",
    "RAD51C", "RAD51D", "BARD1", "BRIP1",
    "NBN", "NF1", "RB1", "RET", "VHL",
    "MEN1", "SDHB", "SDHC", "SDHD", "TMEM127",
]


class CacheManager:
    """SQLite-backed cache for transcript and variant data."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create cache tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transcripts (
                    gene_symbol TEXT NOT NULL,
                    genome_build TEXT NOT NULL,
                    data JSON NOT NULL,
                    cached_at TIMESTAMP NOT NULL,
                    ttl_days INTEGER DEFAULT 30,
                    PRIMARY KEY (gene_symbol, genome_build)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS variant_lookups (
                    hgvs TEXT NOT NULL,
                    gene_symbol TEXT,
                    genome_build TEXT NOT NULL,
                    data JSON NOT NULL,
                    cached_at TIMESTAMP NOT NULL,
                    ttl_days INTEGER DEFAULT 7,
                    PRIMARY KEY (hgvs, genome_build)
                )
            """)
            conn.commit()

    def is_stale(self, cached_at: str, ttl_days: int) -> bool:
        """Check if a cached entry has exceeded its TTL."""
        try:
            cached_time = datetime.fromisoformat(cached_at)
            return datetime.now() - cached_time > timedelta(days=ttl_days)
        except (ValueError, TypeError):
            return True

    # --- Transcript cache ---

    def get_transcripts(
        self, gene: str, genome_build: str = "GRCh38"
    ) -> Optional[list[dict[str, Any]]]:
        """Retrieve cached transcripts for a gene. Returns None if stale/missing."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data, cached_at, ttl_days FROM transcripts "
                "WHERE gene_symbol = ? AND genome_build = ?",
                (gene.upper(), genome_build),
            ).fetchone()

        if row is None:
            return None

        data_json, cached_at, ttl_days = row
        if self.is_stale(cached_at, ttl_days):
            logger.info("Cache stale for transcripts: %s", gene)
            return None

        return json.loads(data_json)

    def set_transcripts(
        self,
        gene: str,
        genome_build: str,
        data: list[dict[str, Any]],
        ttl_days: int = 30,
    ) -> None:
        """Cache transcript data for a gene."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO transcripts "
                "(gene_symbol, genome_build, data, cached_at, ttl_days) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    gene.upper(),
                    genome_build,
                    json.dumps(data),
                    datetime.now().isoformat(),
                    ttl_days,
                ),
            )
            conn.commit()

    # --- Variant cache ---

    def get_variant(
        self, hgvs: str, genome_build: str = "GRCh38"
    ) -> Optional[dict[str, Any]]:
        """Retrieve a cached variant lookup. Returns None if stale/missing."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT data, cached_at, ttl_days FROM variant_lookups "
                "WHERE hgvs = ? AND genome_build = ?",
                (hgvs, genome_build),
            ).fetchone()

        if row is None:
            return None

        data_json, cached_at, ttl_days = row
        if self.is_stale(cached_at, ttl_days):
            return None

        return json.loads(data_json)

    def set_variant(
        self,
        hgvs: str,
        genome_build: str,
        data: dict[str, Any],
        gene_symbol: Optional[str] = None,
        ttl_days: int = 7,
    ) -> None:
        """Cache a variant lookup result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO variant_lookups "
                "(hgvs, gene_symbol, genome_build, data, cached_at, ttl_days) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (hgvs, gene_symbol, genome_build, json.dumps(data),
                 datetime.now().isoformat(), ttl_days),
            )
            conn.commit()

    # --- Prewarm ---

    def needs_prewarm(
        self,
        gene_list: list[str],
        genome_build: str = "GRCh38",
    ) -> bool:
        """Check if any genes in the list are missing from cache."""
        with sqlite3.connect(self.db_path) as conn:
            for gene in gene_list[:5]:  # spot-check first 5
                row = conn.execute(
                    "SELECT cached_at, ttl_days FROM transcripts "
                    "WHERE gene_symbol = ? AND genome_build = ?",
                    (gene.upper(), genome_build),
                ).fetchone()
                if row is None:
                    return True
                if self.is_stale(row[0], row[1]):
                    return True
        return False

    def prewarm(
        self,
        gene_list: Optional[list[str]] = None,
        genome_build: str = "GRCh38",
    ) -> None:
        """Pre-populate the cache with transcript data for common genes."""
        from tools.ensembl import get_transcripts_for_gene

        if gene_list is None:
            gene_list = HEREDITARY_CANCER_GENES

        logger.info("CacheManager: prewarming %d genes", len(gene_list))
        for gene in gene_list:
            cached = self.get_transcripts(gene, genome_build)
            if cached is not None:
                continue
            try:
                transcripts = get_transcripts_for_gene(gene, genome_build)
                if transcripts:
                    self.set_transcripts(gene, genome_build, transcripts)
                    logger.info("Prewarmed: %s (%d transcripts)", gene, len(transcripts))
                time.sleep(0.5)  # respect rate limits
            except Exception as e:
                logger.warning("Prewarm failed for %s: %s", gene, e)
