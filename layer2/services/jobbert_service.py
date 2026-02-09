"""JobBERT embedding service using sentence-transformers.

TechWolf/JobBERT-v2 specifics:
- Max sequence length: 64 tokens
- Output dimension: 1024 (after asymmetric Dense projection)
- Anchor encoder: for job titles
- Positive encoder: for comma-separated skill lists
- Similarity metric: cosine
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from config import JOBBERT_MODEL, JOBBERT_BATCH_SIZE

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        log.info("Loading JobBERT model: %s", JOBBERT_MODEL)
        _model = SentenceTransformer(JOBBERT_MODEL)
        log.info(
            "JobBERT loaded. Max seq length: %s, embedding dim: %s",
            _model.max_seq_length,
            _model.get_sentence_embedding_dimension(),
        )
    return _model


def embed_titles(titles: list[str], batch_size: int = JOBBERT_BATCH_SIZE) -> np.ndarray:
    """Embed job titles using the anchor encoder.

    Returns shape (N, 1024) float32 array, L2-normalized.
    """
    model = get_model()
    embeddings = model.encode(
        titles,
        batch_size=batch_size,
        show_progress_bar=len(titles) > 1000,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def embed_skill_lists(skill_lists: list[str], batch_size: int = JOBBERT_BATCH_SIZE) -> np.ndarray:
    """Embed comma-separated skill strings using the model.

    JobBERT v2 was trained with skill lists as the 'positive' side.
    The sentence-transformers encode() applies the appropriate projection
    based on the model architecture. For symmetric usage (which is what
    encode() does), both go through the same path after mean pooling.

    Each skill_list string should be ≤64 tokens (roughly ≤25 comma-separated terms).
    Returns shape (N, 1024) float32 array, L2-normalized.
    """
    model = get_model()
    embeddings = model.encode(
        skill_lists,
        batch_size=batch_size,
        show_progress_bar=len(skill_lists) > 1000,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between two sets of L2-normalized vectors.

    a: (M, D), b: (N, D) -> returns (M, N) similarity matrix.
    Since vectors are already normalized, this is just a dot product.
    """
    return a @ b.T
