"""Rolling centroid novelty detector for embedding-based anomaly detection.

Maintains a sliding window of "normal" embeddings and computes cosine distance
from the centroid. High distance indicates novel/rare events that may represent
zero-day attacks, emerging threats, or unusual behavior patterns.

The detector reuses the existing BiEncoder embeddings (768-dim) and does NOT
require training a separate model. This makes it lightweight and easy to
enable/disable in production.

Architecture:
    BiEncoder embedding → rolling baseline window → centroid computation
    → cosine distance → normalized novelty score (0.0-1.0)

Scoring:
    - 0.0 = normal (close to centroid of known events)
    - 1.0 = highly novel (far from any known pattern)

Thread-safe — all public methods acquire an internal lock.
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class NoveltyResult:
    """Result of novelty detection for a single event."""

    score: float  # 0.0 = normal, 1.0 = highly novel
    distance: float  # Raw cosine distance from centroid
    baseline_size: int  # Number of events in baseline

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "distance": round(self.distance, 4),
            "baseline_size": self.baseline_size,
        }


class NoveltyDetector:
    """Rolling centroid novelty detector for embedding-based anomaly detection.

    Maintains a sliding window of embeddings and computes cosine distance from
    the centroid. Events far from the centroid are considered novel.

    Parameters
    ----------
    window_size : int
        Maximum embeddings to retain in the rolling baseline.
    min_baseline : int
        Minimum events in baseline before novelty scoring activates.
        Prevents false positives during cold start.
    warmup_events : int
        Total events to process before enabling detection.
        Higher values improve baseline quality.
    distance_scale : float
        Scaling factor for distance-to-score conversion.
        Score = min(1.0, distance * distance_scale).
        Higher values make the detector more sensitive.
    """

    def __init__(
        self,
        window_size: int = 10000,
        min_baseline: int = 100,
        warmup_events: int = 500,
        distance_scale: float = 2.0,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if min_baseline < 1:
            raise ValueError("min_baseline must be >= 1")
        if warmup_events < 0:
            raise ValueError("warmup_events must be >= 0")
        if distance_scale <= 0:
            raise ValueError("distance_scale must be > 0")

        self.window_size = window_size
        self.min_baseline = min_baseline
        self.warmup_events = warmup_events
        self.distance_scale = distance_scale

        # Lock guards all mutable state
        self._lock = threading.Lock()
        self._baseline: deque[np.ndarray] = deque(maxlen=window_size)
        self._centroid: np.ndarray | None = None
        self._centroid_dirty = True
        self._event_count = 0
        self._total_novelty_score = 0.0
        self._score_count = 0

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_embedding(self, embedding: np.ndarray) -> None:
        """Record an embedding into the baseline window.

        Call this AFTER compute_novelty() to avoid including the current
        event in its own baseline.
        """
        with self._lock:
            self._baseline.append(embedding.copy())
            self._centroid_dirty = True
            self._event_count += 1

    def compute_novelty(self, embedding: np.ndarray) -> NoveltyResult:
        """Compute novelty score for an embedding against the baseline.

        Parameters
        ----------
        embedding : np.ndarray
            Normalized embedding vector (should be L2-normalized).

        Returns
        -------
        NoveltyResult
            Novelty score (0.0-1.0), raw distance, and baseline size.
        """
        with self._lock:
            # Warmup: no novelty scoring until enough events processed
            if self._event_count < self.warmup_events:
                return NoveltyResult(score=0.0, distance=0.0, baseline_size=0)

            # Not enough baseline data for meaningful centroid
            if len(self._baseline) < self.min_baseline:
                return NoveltyResult(
                    score=0.0,
                    distance=0.0,
                    baseline_size=len(self._baseline),
                )

            # Lazy centroid recomputation
            if self._centroid_dirty:
                self._recompute_centroid_locked()

            # Compute cosine distance (1 - cosine_similarity)
            if self._centroid is None:
                return NoveltyResult(score=0.0, distance=0.0, baseline_size=0)

            cos_sim = float(np.dot(embedding, self._centroid))
            cos_sim = max(-1.0, min(1.0, cos_sim))  # Clamp for numerical safety
            distance = 1.0 - cos_sim

            # Convert distance to 0-1 score using empirical scaling
            score = min(1.0, distance * self.distance_scale)

            # Track running average for monitoring
            self._total_novelty_score += score
            self._score_count += 1

        return NoveltyResult(
            score=score,
            distance=distance,
            baseline_size=len(self._baseline),
        )

    def get_stats(self) -> dict[str, Any]:
        """Return detector statistics for monitoring."""
        with self._lock:
            avg_score = (
                self._total_novelty_score / self._score_count
                if self._score_count > 0
                else 0.0
            )
            return {
                "window_size": self.window_size,
                "baseline_size": len(self._baseline),
                "event_count": self._event_count,
                "score_count": self._score_count,
                "avg_novelty_score": round(avg_score, 4),
                "centroid_dirty": self._centroid_dirty,
            }

    def reset(self) -> None:
        """Reset the detector state (clear baseline)."""
        with self._lock:
            self._baseline.clear()
            self._centroid = None
            self._centroid_dirty = True
            self._event_count = 0
            self._total_novelty_score = 0.0
            self._score_count = 0
            logger.info("Novelty detector reset")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _recompute_centroid(self) -> None:
        """Recompute the centroid from the current baseline window.

        Must be called while ``self._lock`` is held.
        """
        if not self._baseline:
            self._centroid = None
            return

        # Stack embeddings and compute mean
        embeddings = np.stack(list(self._baseline))
        centroid = np.mean(embeddings, axis=0)

        # L2-normalize the centroid
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        self._centroid = centroid
        self._centroid_dirty = False

    def _recompute_centroid_locked(self) -> None:
        """Recompute centroid. Alias for _recompute_centroid for clarity."""
        self._recompute_centroid()
