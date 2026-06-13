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

from logfilter.monitoring.novelty_detector import NoveltyDetector, NoveltyResult

# numpy and faiss are heavy optional deps — imported lazily inside methods
# so the module can be imported in test environments without them installed.
np: Any
try:
    import numpy as _np

    np = _np

    _NUMPY_AVAILABLE = True
except ImportError:
    np = None
    _NUMPY_AVAILABLE = False

logger = structlog.get_logger(__name__)


class _NumpyIndexFlatIP:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def add(self, vectors: np.ndarray) -> None:
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != self.dim:
            raise ValueError(f"expected matrix with shape (n, {self.dim}), got {matrix.shape}")
        self._vectors = np.vstack([self._vectors, matrix])

    def search(self, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 2 or q.shape[1] != self.dim:
            raise ValueError(f"expected query with shape (n, {self.dim}), got {q.shape}")
        if self.ntotal == 0:
            distances = np.zeros((q.shape[0], k), dtype=np.float32)
            indices = np.full((q.shape[0], k), -1, dtype=np.int64)
            return distances, indices

        scores = q @ self._vectors.T
        top_k = min(k, self.ntotal)
        partition = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
        partition_scores = np.take_along_axis(scores, partition, axis=1)
        order = np.argsort(-partition_scores, axis=1)
        indices = np.take_along_axis(partition, order, axis=1).astype(np.int64)
        distances = np.take_along_axis(scores, indices, axis=1).astype(np.float32)

        if top_k < k:
            pad = k - top_k
            indices = np.hstack([indices, np.full((q.shape[0], pad), -1, dtype=np.int64)])
            distances = np.hstack([distances, np.zeros((q.shape[0], pad), dtype=np.float32)])
        return distances, indices


def _cpu_supports_avx2() -> bool:
    try:
        return "avx2" in Path("/proc/cpuinfo").read_text(errors="ignore").lower()
    except OSError:
        return True


def _new_inner_product_index(dim: int) -> Any:
    if _cpu_supports_avx2():
        try:
            import faiss

            return faiss.IndexFlatIP(dim)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FAISS unavailable; using NumPy index fallback", error=str(exc))
    else:
        logger.warning("CPU lacks AVX2; using NumPy index fallback instead of FAISS")
    return _NumpyIndexFlatIP(dim)


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
        cache_dir: str | Path | None = None,
        revision: str | None = None,
        novelty_detector: NoveltyDetector | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.dedup_threshold = dedup_threshold
        self.dedup_window_seconds = dedup_window_minutes * 60
        self.faiss_top_k = faiss_top_k
        self.mitre_techniques_path = Path(mitre_techniques_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.revision = revision

        self._model: Any | None = None
        self._faiss_dedup: Any | None = None  # rolling dedup index
        self._faiss_attack: Any | None = None  # static ATT&CK index
        self._attack_techniques: list[dict[str, str]] = []
        self._dim: int | None = None

        # Rolling dedup window: deque of (timestamp, embedding)
        self._dedup_window: deque[tuple[float, np.ndarray]] = deque()

        self.novelty_detector = novelty_detector

    # ── initialisation ─────────────────────────────────────────────────────────
    def _load(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading BiEncoder model", model_id=self.model_id)
        st_kwargs: dict[str, Any] = {"device": self.device}
        if self.cache_dir is not None:
            st_kwargs["cache_folder"] = str(self.cache_dir)
        if self.revision is not None:
            st_kwargs["revision"] = self.revision
        self._model = SentenceTransformer(self.model_id, **st_kwargs)
        dim = self._model.get_embedding_dimension()
        if dim is None:
            raise RuntimeError("BiEncoder model did not report an embedding dimension")
        self._dim = int(dim)

        # FAISS index for deduplication (flat L2 — will normalise → cosine)
        self._faiss_dedup = _new_inner_product_index(self._dim)

        # Load and index MITRE ATT&CK techniques
        self._load_attack_techniques()
        logger.info("BiEncoder model loaded", embedding_dim=self._dim)

    def _load_attack_techniques(self) -> None:
        dim = self._dim
        if dim is None:
            raise RuntimeError("BiEncoder embedding dimension is not initialized")

        if not self.mitre_techniques_path.exists():
            logger.warning(
                "MITRE techniques file not found",
                path=str(self.mitre_techniques_path),
            )
            self._faiss_attack = _new_inner_product_index(dim)
            return

        with open(self.mitre_techniques_path) as f:
            self._attack_techniques = json.load(f)

        model = self._model
        if model is None:
            raise RuntimeError("BiEncoder model is not loaded")

        descriptions = [t["description"] for t in self._attack_techniques]
        logger.info("Indexing ATT&CK techniques", count=len(descriptions))
        embeddings = model.encode(
            descriptions,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._faiss_attack = _new_inner_product_index(dim)
        self._faiss_attack.add(embeddings.astype(np.float32))
        logger.info("ATT&CK index ready")

    # ── public API ─────────────────────────────────────────────────────────────
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode a list of texts → normalised embedding matrix (N × dim)."""
        if self._model is None:
            self._load()
        model = self._model
        if model is None:
            raise RuntimeError("BiEncoder model is not loaded")
        embeddings = model.encode(
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

        dedup_index = self._faiss_dedup
        if dedup_index is None:
            raise RuntimeError("BiEncoder dedup index is not initialized")

        if dedup_index.ntotal == 0:
            self._add_to_dedup(embedding)
            return DedupResult(is_duplicate=False, similarity=0.0)

        query = embedding.reshape(1, -1)
        distances, _ = dedup_index.search(query, k=1)
        sim = float(distances[0][0])

        self._add_to_dedup(embedding)
        return DedupResult(is_duplicate=sim >= self.dedup_threshold, similarity=sim)

    def retrieve_attack_candidates(self, embedding: np.ndarray) -> list[ATTACKCandidate]:
        """Return top-k MITRE ATT&CK techniques most similar to this embedding."""
        if self._model is None:
            self._load()

        attack_index = self._faiss_attack
        if attack_index is None:
            raise RuntimeError("BiEncoder ATT&CK index is not initialized")

        if not self._attack_techniques or attack_index.ntotal == 0:
            return []

        k = min(self.faiss_top_k, len(self._attack_techniques))
        query = embedding.reshape(1, -1)
        distances, indices = attack_index.search(query, k=k)

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
        return self._check_dedup_and_retrieve_embeddings(embeddings)

    def check_dedup_retrieve_and_score_novelty_batch(
        self, texts: list[str]
    ) -> list[tuple[DedupResult, list[ATTACKCandidate], NoveltyResult]]:
        embeddings = self.encode(texts)
        bi_results = self._check_dedup_and_retrieve_embeddings(embeddings)
        novelty_results = self._score_novelty_embeddings(embeddings)
        return [
            (dedup, candidates, novelty)
            for (dedup, candidates), novelty in zip(bi_results, novelty_results)
        ]

    def _check_dedup_and_retrieve_embeddings(
        self, embeddings: np.ndarray
    ) -> list[tuple[DedupResult, list[ATTACKCandidate]]]:
        results = []
        for emb in embeddings:
            dedup = self.check_dedup(emb)
            if not dedup.is_duplicate:
                candidates = self.retrieve_attack_candidates(emb)
            else:
                candidates = []
            results.append((dedup, candidates))
        return results

    def score_novelty_batch(
        self,
        texts: list[str],
        embeddings: np.ndarray | None = None,
    ) -> list[NoveltyResult]:
        """
        Compute novelty scores for a batch of texts using existing BiEncoder embeddings.
        
        This method reuses embeddings already computed during dedup/ATT&CK retrieval
        to avoid redundant computation. Call after check_dedup_and_retrieve_batch().
        """
        if embeddings is None:
            embeddings = self.encode(texts)
        return self._score_novelty_embeddings(embeddings)

    def _score_novelty_embeddings(self, embeddings: np.ndarray) -> list[NoveltyResult]:
        if self.novelty_detector is None:
            return [
                NoveltyResult(score=0.0, distance=0.0, baseline_size=0)
                for _ in range(len(embeddings))
            ]
        results = []
        for emb in embeddings:
            novelty = self.novelty_detector.compute_novelty(emb)
            self.novelty_detector.record_embedding(emb)
            results.append(novelty)
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
        if self._dim is None:
            raise RuntimeError("BiEncoder embedding dimension is not initialized")
        self._faiss_dedup = _new_inner_product_index(self._dim)
        if self._dedup_window:
            matrix = np.vstack([e for _, e in self._dedup_window]).astype(np.float32)
            self._faiss_dedup.add(matrix)
