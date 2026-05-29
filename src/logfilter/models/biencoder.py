"""
SecureBERT2.0-BiEncoder wrapper.

Two roles:
  1. Deduplication — embed each log, compare to a sliding 5-minute FAISS
     index of recent embeddings; if cosine similarity > threshold, mark as
     near-duplicate and skip expensive Tier-3 models.

  2. ATT&CK candidate retrieval — embed MITRE ATT&CK technique descriptions
     once at startup, then for each log retrieve the top-k most similar
     techniques (by cosine similarity) as candidates for the cross-encoder.

Model: cisco-ai/SecureBERT2.0-biencoder
Type:  BiEncoder / Sentence Transformer (768-dim output)
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

# numpy and faiss are heavy optional deps — imported lazily inside methods
# so the module can be imported in test environments without them installed.
try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class DedupResult:
    is_duplicate: bool
    similarity: float  # cosine sim to nearest recent event (0–1)


@dataclass
class ATTACKCandidate:
    technique_id: str
    name: str
    description: str
    similarity: float


class BiEncoderModel:
    """
    Lazy-loading wrapper around SecureBERT2.0-biencoder.

    Maintains:
      - A rolling FAISS index of recent embeddings for deduplication.
      - A static FAISS index of MITRE ATT&CK technique embeddings.
    """

    MODEL_ID = "cisco-ai/SecureBERT2.0-biencoder"

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str = "cpu",
        batch_size: int = 64,
        dedup_threshold: float = 0.95,
        dedup_window_minutes: float = 5.0,
        faiss_top_k: int = 3,
        mitre_techniques_path: str | Path = "config/mitre_techniques.json",
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.dedup_threshold = dedup_threshold
        self.dedup_window_seconds = dedup_window_minutes * 60
        self.faiss_top_k = faiss_top_k
        self.mitre_techniques_path = Path(mitre_techniques_path)

        self._model: Any | None = None
        self._faiss_dedup: Any | None = None  # rolling dedup index
        self._faiss_attack: Any | None = None  # static ATT&CK index
        self._attack_techniques: list[dict[str, str]] = []

        # Rolling dedup window: deque of (timestamp, embedding)
        self._dedup_window: deque[tuple[float, np.ndarray]] = deque()

    # ── initialisation ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        logger.info("Loading BiEncoder model", model_id=self.model_id)
        self._model = SentenceTransformer(self.model_id, device=self.device)
        self._dim = self._model.get_sentence_embedding_dimension()

        # FAISS index for deduplication (flat L2 — will normalise → cosine)
        self._faiss_dedup = faiss.IndexFlatIP(self._dim)

        # Load and index MITRE ATT&CK techniques
        self._load_attack_techniques()
        logger.info("BiEncoder model loaded", embedding_dim=self._dim)

    def _load_attack_techniques(self) -> None:
        import faiss

        if not self.mitre_techniques_path.exists():
            logger.warning(
                "MITRE techniques file not found",
                path=str(self.mitre_techniques_path),
            )
            self._faiss_attack = faiss.IndexFlatIP(self._dim)
            return

        with open(self.mitre_techniques_path) as f:
            self._attack_techniques = json.load(f)

        descriptions = [t["description"] for t in self._attack_techniques]
        logger.info("Indexing ATT&CK techniques", count=len(descriptions))
        embeddings = self._model.encode(  # type: ignore[union-attr]
            descriptions,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._faiss_attack = faiss.IndexFlatIP(self._dim)
        self._faiss_attack.add(embeddings.astype(np.float32))
        logger.info("ATT&CK index ready")

    # ── public API ─────────────────────────────────────────────────────────────
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts → normalised embedding matrix (N × dim)."""
        if self._model is None:
            self._load()
        embeddings = self._model.encode(  # type: ignore[union-attr]
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.astype(np.float32)

    def check_dedup(self, embedding: np.ndarray) -> DedupResult:
        """
        Check if this embedding is a near-duplicate of a recent event.
        Prunes stale entries from the dedup window before checking.
        """
        if self._model is None:
            self._load()

        self._prune_dedup_window()

        if self._faiss_dedup.ntotal == 0:  # type: ignore[union-attr]
            self._add_to_dedup(embedding)
            return DedupResult(is_duplicate=False, similarity=0.0)

        query = embedding.reshape(1, -1)
        distances, _ = self._faiss_dedup.search(query, k=1)  # type: ignore[union-attr]
        sim = float(distances[0][0])

        self._add_to_dedup(embedding)
        return DedupResult(is_duplicate=sim >= self.dedup_threshold, similarity=sim)

    def retrieve_attack_candidates(self, embedding: np.ndarray) -> list[ATTACKCandidate]:
        """Return top-k MITRE ATT&CK techniques most similar to this embedding."""
        if self._model is None:
            self._load()

        if not self._attack_techniques or self._faiss_attack.ntotal == 0:  # type: ignore[union-attr]
            return []

        k = min(self.faiss_top_k, len(self._attack_techniques))
        query = embedding.reshape(1, -1)
        distances, indices = self._faiss_attack.search(query, k=k)  # type: ignore[union-attr]

        candidates = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self._attack_techniques):
                continue
            t = self._attack_techniques[idx]
            candidates.append(
                ATTACKCandidate(
                    technique_id=t["id"],
                    name=t["name"],
                    description=t["description"],
                    similarity=float(dist),
                )
            )
        return candidates

    def check_dedup_and_retrieve_batch(
        self, texts: list[str]
    ) -> list[tuple[DedupResult, list[ATTACKCandidate]]]:
        """
        Batch version: encode all texts, check dedup, retrieve ATT&CK candidates.
        Returns list of (DedupResult, [ATTACKCandidate, ...]).
        """
        embeddings = self.encode(texts)
        results = []
        for emb in embeddings:
            dedup = self.check_dedup(emb)
            if not dedup.is_duplicate:
                candidates = self.retrieve_attack_candidates(emb)
            else:
                candidates = []
            results.append((dedup, candidates))
        return results

    # ── internal ───────────────────────────────────────────────────────────────
    def _add_to_dedup(self, embedding: np.ndarray) -> None:
        now = time.monotonic()
        self._dedup_window.append((now, embedding.copy()))
        # Rebuild FAISS index from window (window typically small)
        self._rebuild_dedup_faiss()

    def _prune_dedup_window(self) -> None:
        cutoff = time.monotonic() - self.dedup_window_seconds
        while self._dedup_window and self._dedup_window[0][0] < cutoff:
            self._dedup_window.popleft()

    def _rebuild_dedup_faiss(self) -> None:
        import faiss

        self._faiss_dedup = faiss.IndexFlatIP(self._dim)
        if self._dedup_window:
            matrix = np.vstack([e for _, e in self._dedup_window]).astype(np.float32)
            self._faiss_dedup.add(matrix)
