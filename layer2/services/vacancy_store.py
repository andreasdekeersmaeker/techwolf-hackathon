"""Vacancy store: loads vacancies, builds / loads FAISS index, performs retrieval."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config import (
    EMBEDDINGS_PATH,
    EXCLUSION_TITLE_KEYWORDS,
    FAISS_INDEX_PATH,
    JOBBERT_BATCH_SIZE,
    RETRIEVAL_TOP_K,
    VACANCIES_PATH,
    VACANCY_META_PATH,
    UNIQUE_TITLES_PATH,
    DATA_DIR,
)
from models.schemas import VacancyRecord

log = logging.getLogger(__name__)


class VacancyStore:
    """Manages the vacancy database and FAISS index for retrieval."""

    def __init__(self) -> None:
        self._records: list[dict[str, Any]] = []
        self._index: faiss.IndexFlatIP | None = None
        self._embeddings: np.ndarray | None = None
        self._title_to_indices: dict[str, list[int]] = {}
        self._loaded = False

    @property
    def is_indexed(self) -> bool:
        return FAISS_INDEX_PATH.exists() and VACANCY_META_PATH.exists()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_index(self) -> None:
        """Load pre-built FAISS index and vacancy metadata."""
        if self._loaded:
            return

        log.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
        self._index = faiss.read_index(str(FAISS_INDEX_PATH))

        log.info("Loading vacancy metadata from %s", VACANCY_META_PATH)
        self._records = []
        with open(VACANCY_META_PATH, "r", encoding="utf-8") as f:
            for line in f:
                self._records.append(json.loads(line))

        log.info("Loading embeddings from %s", EMBEDDINGS_PATH)
        self._embeddings = np.load(str(EMBEDDINGS_PATH))

        # Build title lookup
        for i, rec in enumerate(self._records):
            title = rec.get("enriched_job_title", "").lower()
            self._title_to_indices.setdefault(title, []).append(i)

        self._loaded = True
        log.info(
            "Vacancy store loaded: %d records, index dim=%d",
            len(self._records),
            self._index.d,
        )

    def get_record(self, idx: int) -> dict[str, Any]:
        return self._records[idx]

    def get_record_by_id(self, identifier: str) -> dict[str, Any] | None:
        for rec in self._records:
            if rec.get("identifier") == identifier:
                return rec
        return None

    def total_records(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self, query_embeddings: np.ndarray, top_k: int = RETRIEVAL_TOP_K
    ) -> list[list[tuple[int, float]]]:
        """Search the FAISS index.

        Args:
            query_embeddings: (Q, D) float32 array, L2-normalized
            top_k: number of nearest neighbors per query

        Returns:
            List of Q lists, each containing (record_index, cosine_score) tuples.
        """
        if self._index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        scores, indices = self._index.search(query_embeddings, top_k)
        results = []
        for q in range(len(query_embeddings)):
            hits = []
            for k in range(top_k):
                idx = int(indices[q, k])
                if idx == -1:
                    continue
                hits.append((idx, float(scores[q, k])))
            results.append(hits)
        return results

    def get_embedding(self, idx: int) -> np.ndarray:
        """Get the pre-computed embedding for a vacancy by index."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded.")
        return self._embeddings[idx]

    def get_embeddings(self, indices: list[int]) -> np.ndarray:
        """Get pre-computed embeddings for multiple vacancies."""
        if self._embeddings is None:
            raise RuntimeError("Embeddings not loaded.")
        return self._embeddings[indices]

    # ------------------------------------------------------------------
    # Preprocessing (one-time)
    # ------------------------------------------------------------------

    def preprocess(self, max_records: int | None = None) -> None:
        """One-time: load raw vacancies, embed titles, build FAISS index."""
        from services.jobbert_service import embed_titles

        DATA_DIR.mkdir(parents=True, exist_ok=True)

        # 1. Load raw vacancies
        log.info("Loading raw vacancies from %s", VACANCIES_PATH)
        raw_records: list[dict[str, Any]] = []
        with gzip.open(str(VACANCIES_PATH), "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_records and i >= max_records:
                    break
                rec = json.loads(line)
                raw_records.append(rec)
                if (i + 1) % 200000 == 0:
                    log.info("  loaded %d records...", i + 1)

        log.info("Total raw records loaded: %d", len(raw_records))

        # 2. Extract unique enriched job titles and map to records
        title_to_record_indices: dict[str, list[int]] = {}
        for i, rec in enumerate(raw_records):
            title = rec.get("enriched_job_title", "").strip()
            if title:
                title_to_record_indices.setdefault(title, []).append(i)

        unique_titles = list(title_to_record_indices.keys())
        log.info("Unique enriched job titles: %d", len(unique_titles))

        # 3. Embed unique titles with JobBERT
        log.info("Embedding %d unique titles with JobBERT...", len(unique_titles))
        title_embeddings = embed_titles(unique_titles, batch_size=JOBBERT_BATCH_SIZE)
        log.info("Embedding complete. Shape: %s", title_embeddings.shape)

        # 4. Expand: create per-record embeddings and metadata
        log.info("Building per-record metadata and embedding matrix...")
        all_embeddings = []
        metadata_records = []
        for title_idx, title in enumerate(unique_titles):
            emb = title_embeddings[title_idx]
            for rec_idx in title_to_record_indices[title]:
                rec = raw_records[rec_idx]
                metadata_records.append({
                    "identifier": rec.get("identifier", ""),
                    "title": rec.get("title", ""),
                    "enriched_job_title": rec.get("enriched_job_title", ""),
                    "description": rec.get("description", "")[:500],
                    "enriched_skills": rec.get("enriched_skills", ""),
                    "enriched_tasks": rec.get("enriched_tasks", ""),
                    "enriched_industry": rec.get("enriched_industry", ""),
                    "enriched_contract_type": rec.get("enriched_contract_type", ""),
                    "country": rec.get("country", ""),
                    "locality": rec.get("address_addresslocality", ""),
                })
                all_embeddings.append(emb)

        embeddings_matrix = np.array(all_embeddings, dtype=np.float32)
        log.info("Embeddings matrix shape: %s", embeddings_matrix.shape)

        # 5. Build FAISS index (Inner Product = cosine for normalized vectors)
        log.info("Building FAISS index...")
        dim = embeddings_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings_matrix)
        log.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

        # 6. Save everything
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        log.info("Saved FAISS index to %s", FAISS_INDEX_PATH)

        np.save(str(EMBEDDINGS_PATH), embeddings_matrix)
        log.info("Saved embeddings to %s", EMBEDDINGS_PATH)

        with open(VACANCY_META_PATH, "w", encoding="utf-8") as f:
            for rec in metadata_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("Saved vacancy metadata to %s", VACANCY_META_PATH)

        # Save unique titles for reference
        with open(UNIQUE_TITLES_PATH, "w", encoding="utf-8") as f:
            json.dump(unique_titles, f, ensure_ascii=False)
        log.info("Saved unique titles to %s", UNIQUE_TITLES_PATH)

        log.info("Preprocessing complete.")


def is_excluded_role(title: str, description: str = "") -> bool:
    """Check if a vacancy matches the exclusion criteria."""
    title_lower = title.lower()
    for kw in EXCLUSION_TITLE_KEYWORDS:
        if kw in title_lower:
            return True
    return False
