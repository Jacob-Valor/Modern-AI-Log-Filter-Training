"""Tests for the PSI-based drift detector."""

from __future__ import annotations

import random
import threading
from typing import Any

import pytest

from logfilter.monitoring.drift_detector import DriftDetector, DriftStatus


class TestDriftDetectorInit:
    def test_defaults(self) -> None:
        d = DriftDetector()
        assert d.window_size == 1000
        assert d.psi_threshold == 0.25
        assert d.check_interval == 100
        assert d.auto_fallback is True
        assert d.num_bins == 10

    def test_custom_params(self) -> None:
        d = DriftDetector(
            window_size=500, psi_threshold=0.1, check_interval=50, auto_fallback=False
        )
        assert d.window_size == 500
        assert d.psi_threshold == 0.1
        assert d.check_interval == 50
        assert d.auto_fallback is False

    @pytest.mark.parametrize(
        "kwargs,msg",
        [
            ({"window_size": 0}, "window_size"),
            ({"psi_threshold": -0.1}, "psi_threshold"),
            ({"check_interval": 0}, "check_interval"),
            ({"num_bins": 1}, "num_bins"),
        ],
    )
    def test_invalid_params_raise(self, kwargs: dict, msg: str) -> None:
        with pytest.raises(ValueError, match=msg):
            DriftDetector(**kwargs)


class TestRecordAndCheck:
    def test_record_score_clamps(self) -> None:
        d = DriftDetector(window_size=10)
        d.record_score(-0.5)
        d.record_score(1.5)
        status = d.check_drift()
        assert status.current_count == 2

    def test_reference_primes_from_first_scores(self) -> None:
        d = DriftDetector(window_size=5)
        for _ in range(5):
            d.record_score(0.5)
        status = d.check_drift()
        assert status.reference_count == 5
        assert status.current_count == 5

    def test_no_drift_for_identical_distribution(self) -> None:
        d = DriftDetector(window_size=100, check_interval=50)
        rng = random.Random(42)
        for _ in range(200):
            d.record_score(rng.random())
        status = d.check_drift()
        assert status.drift_detected is False
        assert status.psi < 0.1

    def test_drift_detected_for_shifted_distribution(self) -> None:
        d = DriftDetector(window_size=200, psi_threshold=0.25, check_interval=100)
        # Reference: uniform [0.0, 0.5]
        rng = random.Random(42)
        for _ in range(300):
            d.record_score(rng.random() * 0.5)
        # Current: uniform [0.5, 1.0] — strong shift
        for _ in range(300):
            d.record_score(0.5 + rng.random() * 0.5)
        status = d.check_drift()
        assert status.drift_detected is True
        assert status.psi > 0.25
        assert status.fallback_active is True

    def test_fallback_inactive_when_auto_fallback_disabled(self) -> None:
        d = DriftDetector(
            window_size=200, psi_threshold=0.25, check_interval=100, auto_fallback=False
        )
        rng = random.Random(42)
        for _ in range(300):
            d.record_score(rng.random() * 0.5)
        for _ in range(300):
            d.record_score(0.5 + rng.random() * 0.5)
        status = d.check_drift()
        assert status.drift_detected is True
        assert status.fallback_active is False

    def test_recovery_after_drift(self) -> None:
        d = DriftDetector(window_size=200, psi_threshold=0.25, check_interval=100)
        rng = random.Random(42)
        # Build reference
        for _ in range(300):
            d.record_score(rng.random() * 0.5)
        # Induce drift
        for _ in range(300):
            d.record_score(0.5 + rng.random() * 0.5)
        assert d.check_drift().drift_detected is True
        # Return to reference-like distribution
        for _ in range(400):
            d.record_score(rng.random() * 0.5)
        assert d.check_drift().drift_detected is False
        assert d.check_drift().fallback_active is False


class TestResetReference:
    def test_reset_clears_drift(self) -> None:
        d = DriftDetector(window_size=200, psi_threshold=0.25, check_interval=100)
        rng = random.Random(42)
        for _ in range(300):
            d.record_score(rng.random() * 0.5)
        for _ in range(300):
            d.record_score(0.5 + rng.random() * 0.5)
        assert d.check_drift().drift_detected is True
        d.reset_reference()
        status = d.check_drift()
        assert status.drift_detected is False
        assert status.psi == 0.0
        assert status.fallback_active is False


class TestThreadSafety:
    def test_concurrent_record(self) -> None:
        d = DriftDetector(window_size=1000, check_interval=50)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                rng = random.Random()
                for _ in range(100):
                    d.record_score(rng.random())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        status = d.check_drift()
        assert status.current_count == 1000  # window_size cap


class TestDriftStatus:
    def test_to_dict(self) -> None:
        s = DriftStatus(drift_detected=True, psi=0.35, reference_count=500, current_count=500)
        d = s.to_dict()
        assert d["drift_detected"] is True
        assert d["psi"] == 0.35
        assert d["reference_count"] == 500

    def test_defaults(self) -> None:
        s = DriftStatus()
        assert s.drift_detected is False
        assert s.psi == 0.0
        assert s.fallback_active is False


class TestEdgeCases:
    def test_empty_window_no_crash(self) -> None:
        d = DriftDetector()
        status = d.check_drift()
        assert status.psi == 0.0
        assert status.drift_detected is False

    def test_single_score_no_drift(self) -> None:
        d = DriftDetector(window_size=10, check_interval=5)
        d.record_score(0.5)
        status = d.check_drift()
        assert status.drift_detected is False

    def test_all_scores_same_bin(self) -> None:
        d = DriftDetector(window_size=20, check_interval=10)
        for _ in range(30):
            d.record_score(0.05)
        status = d.check_drift()
        # Identical distributions → PSI should be ~0
        assert status.psi < 0.01

    def test_is_fallback_active_reflects_state(self) -> None:
        d = DriftDetector(window_size=200, psi_threshold=0.25, check_interval=100)
        assert d.is_fallback_active() is False
        rng = random.Random(42)
        for _ in range(300):
            d.record_score(rng.random() * 0.5)
        for _ in range(300):
            d.record_score(0.5 + rng.random() * 0.5)
        assert d.is_fallback_active() is True


class TestDriftCallback:
    def test_callback_invoked_on_drift(self) -> None:
        calls: list[Any] = []

        def callback(status: Any) -> None:
            calls.append(status)

        d = DriftDetector(
            window_size=200, psi_threshold=0.25, check_interval=100, on_drift_detected=callback
        )
        rng = random.Random(42)
        for _ in range(300):
            d.record_score(rng.random() * 0.5)
        for _ in range(300):
            d.record_score(0.5 + rng.random() * 0.5)
        assert len(calls) == 1
        assert calls[0].drift_detected is True

    def test_callback_exception_logged_not_raised(self) -> None:
        def callback(_status: Any) -> None:
            raise RuntimeError("boom")

        d = DriftDetector(
            window_size=200, psi_threshold=0.25, check_interval=100, on_drift_detected=callback
        )
        rng = random.Random(42)
        for _ in range(300):
            d.record_score(rng.random() * 0.5)
        for _ in range(300):
            d.record_score(0.5 + rng.random() * 0.5)
        assert d.check_drift().drift_detected is True
