"""Lightweight PSI-based model drift detector for production ML systems.

Tracks score distributions over sliding windows and computes the Population
Stability Index (PSI) between a reference baseline and the current window.
When drift exceeds a threshold the detector can signal auto-fallback to a
simpler model tier, preventing degraded predictions from poisoning downstream
routing decisions.

PSI formula::

    PSI = Σ (Actual% - Expected%) * ln(Actual% / Expected%)

Typical thresholds:
    PSI < 0.1   — no significant drift
    PSI 0.1–0.25 — moderate drift (monitor closely)
    PSI > 0.25  — significant drift (alert / fallback)

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
class DriftStatus:
    """Immutable snapshot of the detector's current state."""

    drift_detected: bool = False
    psi: float = 0.0
    reference_count: int = 0
    current_count: int = 0
    last_check_scores: int = 0
    fallback_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_detected": self.drift_detected,
            "psi": round(self.psi, 4),
            "reference_count": self.reference_count,
            "current_count": self.current_count,
            "last_check_scores": self.last_check_scores,
            "fallback_active": self.fallback_active,
        }


class DriftDetector:
    """Sliding-window PSI drift detector for model score distributions.

    Parameters
    ----------
    window_size : int
        Maximum number of scores to retain in the *current* sliding window.
    psi_threshold : float
        PSI value above which drift is declared significant.
    check_interval : int
        Number of new scores required before a fresh PSI check is performed.
    auto_fallback : bool
        If ``True``, ``fallback_active`` becomes ``True`` while drift is
        detected.  Callers (e.g. ``LogScorer``) can use this to skip Tier-2.
    num_bins : int
        Number of equal-width histogram bins spanning ``[0.0, 1.0]``.
    epsilon : float
        Minimum probability mass per bin to avoid division-by-zero in ``ln``.
    """

    def __init__(
        self,
        window_size: int = 1000,
        psi_threshold: float = 0.25,
        check_interval: int = 100,
        auto_fallback: bool = True,
        num_bins: int = 10,
        epsilon: float = 1e-10,
        on_drift_detected: Any | None = None,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if psi_threshold < 0.0:
            raise ValueError("psi_threshold must be >= 0.0")
        if check_interval < 1:
            raise ValueError("check_interval must be >= 1")
        if num_bins < 2:
            raise ValueError("num_bins must be >= 2")

        self.window_size = window_size
        self.psi_threshold = psi_threshold
        self.check_interval = check_interval
        self.auto_fallback = auto_fallback
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.on_drift_detected = on_drift_detected

        # Lock guards all mutable state
        self._lock = threading.Lock()
        self._reference: deque[float] = deque(maxlen=window_size)
        self._current: deque[float] = deque(maxlen=window_size)
        self._scores_since_check = 0
        self._drift_detected = False
        self._last_psi = 0.0
        self._fallback_active = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def record_score(self, score: float) -> None:
        """Record a single model score (0.0–1.0) into the current window."""
        clamped = float(np.clip(score, 0.0, 1.0))
        newly_drifted: DriftStatus | None = None
        with self._lock:
            self._current.append(clamped)
            self._scores_since_check += 1

            # Prime the reference window with the first batch of scores so we
            # have a baseline even before the window fills completely.
            if len(self._reference) < self.window_size:
                self._reference.append(clamped)
                return

            if self._scores_since_check >= self.check_interval:
                newly_drifted = self._check_drift_locked()

        # Fire the user callback *outside* the lock: invoking it while holding
        # self._lock would deadlock (check_drift re-acquires it) and would run
        # arbitrary user code under the lock.
        if newly_drifted is not None and self.on_drift_detected is not None:
            try:
                self.on_drift_detected(newly_drifted)
            except Exception as exc:
                logger.error("Drift callback failed", error=str(exc))

    def check_drift(self) -> DriftStatus:
        """Return the current drift status (thread-safe snapshot)."""
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> DriftStatus:
        """Build a status snapshot. Caller must hold ``self._lock``."""
        return DriftStatus(
            drift_detected=self._drift_detected,
            psi=self._last_psi,
            reference_count=len(self._reference),
            current_count=len(self._current),
            last_check_scores=self._scores_since_check,
            fallback_active=self._fallback_active,
        )

    def reset_reference(self) -> None:
        """Promote the current window to the new reference baseline."""
        with self._lock:
            self._reference = deque(self._current, maxlen=self.window_size)
            self._drift_detected = False
            self._last_psi = 0.0
            if self.auto_fallback:
                self._fallback_active = False
            logger.info("Drift reference baseline reset", reference_size=len(self._reference))

    def is_fallback_active(self) -> bool:
        """True when auto-fallback is enabled and drift is currently detected."""
        with self._lock:
            return self.auto_fallback and self._drift_detected

    # ── Internal ─────────────────────────────────────────────────────────────────

    def _check_drift_locked(self) -> DriftStatus | None:
        """Compute PSI between reference and current windows.

        Must be called while ``self._lock`` is held. Returns a status snapshot
        when this check is the transition into a drifting state (so the caller
        can fire ``on_drift_detected`` after releasing the lock), else ``None``.
        """
        self._scores_since_check = 0

        if len(self._reference) < self.num_bins or len(self._current) < self.num_bins:
            # Not enough data for meaningful histograms yet.
            return None

        expected = self._hist(self._reference)
        actual = self._hist(self._current)

        psi = 0.0
        for e, a in zip(expected, actual):
            e_adj = max(e, self.epsilon)
            a_adj = max(a, self.epsilon)
            psi += (a_adj - e_adj) * np.log(a_adj / e_adj)

        self._last_psi = float(psi)
        was_drifting = self._drift_detected
        self._drift_detected = bool(psi > self.psi_threshold)

        if self.auto_fallback:
            self._fallback_active = bool(self._drift_detected)

        if self._drift_detected and not was_drifting:
            logger.warning(
                "Model drift detected",
                psi=round(psi, 4),
                threshold=self.psi_threshold,
                reference_size=len(self._reference),
                current_size=len(self._current),
            )
            return self._status_locked()
        elif was_drifting and not self._drift_detected:
            logger.info(
                "Model drift recovered",
                psi=round(psi, 4),
                threshold=self.psi_threshold,
            )
        return None

    def _hist(self, scores: deque[float]) -> np.ndarray:
        """Return a normalised histogram for *scores* using equal-width bins."""
        arr = np.array(list(scores), dtype=np.float64)
        counts, _ = np.histogram(arr, bins=self.num_bins, range=(0.0, 1.0))
        total = counts.sum()
        if total == 0:
            return np.full(self.num_bins, self.epsilon)
        return counts / total
