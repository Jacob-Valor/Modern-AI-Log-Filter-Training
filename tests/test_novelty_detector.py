"""Tests for the novelty detector module."""

from __future__ import annotations

import numpy as np
import pytest

from logfilter.monitoring.novelty_detector import NoveltyDetector, NoveltyResult


class TestNoveltyResult:
    def test_to_dict(self) -> None:
        result = NoveltyResult(score=0.75, distance=0.375, baseline_size=500)
        payload = result.to_dict()
        assert payload["score"] == 0.75
        assert payload["distance"] == 0.375
        assert payload["baseline_size"] == 500

    def test_to_dict_rounding(self) -> None:
        result = NoveltyResult(score=0.123456, distance=0.987654, baseline_size=100)
        payload = result.to_dict()
        assert payload["score"] == 0.1235
        assert payload["distance"] == 0.9877


class TestNoveltyDetector:
    def test_init_defaults(self) -> None:
        detector = NoveltyDetector()
        assert detector.window_size == 10000
        assert detector.min_baseline == 100
        assert detector.warmup_events == 500
        assert detector.distance_scale == 2.0

    def test_init_custom(self) -> None:
        detector = NoveltyDetector(
            window_size=5000,
            min_baseline=50,
            warmup_events=100,
            distance_scale=3.0,
        )
        assert detector.window_size == 5000
        assert detector.min_baseline == 50
        assert detector.warmup_events == 100
        assert detector.distance_scale == 3.0

    def test_init_invalid_window_size(self) -> None:
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            NoveltyDetector(window_size=0)

    def test_init_invalid_min_baseline(self) -> None:
        with pytest.raises(ValueError, match="min_baseline must be >= 1"):
            NoveltyDetector(min_baseline=0)

    def test_init_invalid_warmup_events(self) -> None:
        with pytest.raises(ValueError, match="warmup_events must be >= 0"):
            NoveltyDetector(warmup_events=-1)

    def test_init_invalid_distance_scale(self) -> None:
        with pytest.raises(ValueError, match="distance_scale must be > 0"):
            NoveltyDetector(distance_scale=0)

    def test_warmup_returns_zero_score(self) -> None:
        detector = NoveltyDetector(warmup_events=10)
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        for _ in range(5):
            result = detector.compute_novelty(embedding)
            detector.record_embedding(embedding)
            assert result.score == 0.0
            assert result.distance == 0.0
            assert result.baseline_size == 0

    def test_insufficient_baseline_returns_zero_score(self) -> None:
        detector = NoveltyDetector(warmup_events=5, min_baseline=10)
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        for _ in range(8):
            detector.record_embedding(embedding)

        result = detector.compute_novelty(embedding)
        assert result.score == 0.0
        assert result.baseline_size == 8

    def test_normal_event_low_novelty(self) -> None:
        detector = NoveltyDetector(warmup_events=5, min_baseline=5, distance_scale=2.0)

        base_embeddings = []
        for _ in range(10):
            emb = np.random.randn(768).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            base_embeddings.append(emb)
            detector.record_embedding(emb)

        centroid = np.mean(base_embeddings, axis=0)
        centroid = centroid / np.linalg.norm(centroid)

        result = detector.compute_novelty(centroid)
        assert result.score < 0.1
        assert result.distance < 0.05

    def test_novel_event_high_novelty(self) -> None:
        detector = NoveltyDetector(warmup_events=5, min_baseline=5, distance_scale=2.0)

        for _ in range(10):
            emb = np.random.randn(768).astype(np.float32)
            emb[0] = 1.0
            emb[1:] = 0.0
            detector.record_embedding(emb)

        novel_emb = np.zeros(768, dtype=np.float32)
        novel_emb[767] = 1.0

        result = detector.compute_novelty(novel_emb)
        assert result.score > 0.5

    def test_record_embedding_updates_baseline(self) -> None:
        detector = NoveltyDetector(warmup_events=5, min_baseline=5)
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        detector.record_embedding(embedding)
        detector.record_embedding(embedding)

        stats = detector.get_stats()
        assert stats["baseline_size"] == 2
        assert stats["event_count"] == 2

    def test_window_size_limit(self) -> None:
        detector = NoveltyDetector(window_size=5, warmup_events=1, min_baseline=1)

        for i in range(10):
            emb = np.random.randn(768).astype(np.float32)
            emb[0] = float(i)
            emb = emb / np.linalg.norm(emb)
            detector.record_embedding(emb)

        stats = detector.get_stats()
        assert stats["baseline_size"] == 5

    def test_get_stats(self) -> None:
        detector = NoveltyDetector(warmup_events=5, min_baseline=5)
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        detector.record_embedding(embedding)

        stats = detector.get_stats()
        assert "window_size" in stats
        assert "baseline_size" in stats
        assert "event_count" in stats
        assert "score_count" in stats
        assert "avg_novelty_score" in stats
        assert "centroid_dirty" in stats

    def test_reset(self) -> None:
        detector = NoveltyDetector(warmup_events=5, min_baseline=5)
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        detector.record_embedding(embedding)
        detector.reset()

        stats = detector.get_stats()
        assert stats["baseline_size"] == 0
        assert stats["event_count"] == 0

    def test_to_dict_method(self) -> None:
        detector = NoveltyDetector(warmup_events=5, min_baseline=5)
        embedding = np.random.randn(768).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        for _ in range(10):
            detector.record_embedding(embedding)

        result = detector.compute_novelty(embedding)
        assert isinstance(result, NoveltyResult)
        payload = result.to_dict()
        assert isinstance(payload, dict)

    def test_steady_stream_never_restacks_the_window(self, monkeypatch) -> None:
        """A steady stream must not re-stack the 10k x 768 window per event.

        The old implementation np.stack'd the whole window on every event
        (~7.7M float64 ops and ~30 MB per event, under the lock). The centroid
        is now derived from a running sum in O(dim).
        """
        detector = NoveltyDetector(window_size=10_000, min_baseline=1, warmup_events=0)
        stack_calls = {"count": 0}
        real_stack = np.stack

        def counting_stack(*args, **kwargs):
            stack_calls["count"] += 1
            return real_stack(*args, **kwargs)

        monkeypatch.setattr(np, "stack", counting_stack)

        rng = np.random.default_rng(7)
        for _ in range(300):
            vector = rng.random(768)
            vector = (vector / np.linalg.norm(vector)).astype(np.float32)
            detector.compute_novelty(vector)
            detector.record_embedding(vector)

        assert stack_calls["count"] == 0
        # Only the first event short-circuits on an empty baseline.
        assert detector.get_stats()["centroid_recompute_count"] == 299

    def test_running_sum_centroid_matches_full_recompute(self) -> None:
        """The incrementally maintained centroid equals a fresh full recompute."""
        detector = NoveltyDetector(window_size=64, min_baseline=1, warmup_events=0)
        rng = np.random.default_rng(11)
        vectors = []
        for _ in range(200):
            vector = rng.random(768)
            vector = vector / np.linalg.norm(vector)
            detector.compute_novelty(vector.astype(np.float32))
            detector.record_embedding(vector.astype(np.float32))
            vectors.append(vector)

        detector._recompute_centroid()
        incremental = detector._centroid.copy()

        # Independent full recompute from the retained window only.
        retained = np.stack(list(detector._baseline))
        expected = retained.mean(axis=0)
        expected = expected / np.linalg.norm(expected)

        assert np.allclose(incremental, expected, atol=1e-9)
        assert len(detector._baseline) == 64

    def test_centroid_matches_full_recompute_when_window_wraps(self) -> None:
        """Eviction from the rolling window keeps the running sum exact."""
        detector = NoveltyDetector(window_size=8, min_baseline=1, warmup_events=0)
        rng = np.random.default_rng(3)
        for _ in range(50):
            vector = rng.random(16)
            vector = vector / np.linalg.norm(vector)
            detector.compute_novelty(vector)
            detector.record_embedding(vector)

        detector._recompute_centroid()
        incremental = detector._centroid.copy()
        retained = np.stack(list(detector._baseline))
        expected = retained.mean(axis=0)
        expected = expected / np.linalg.norm(expected)

        assert np.allclose(incremental, expected, atol=1e-12)

    def test_reset_restores_running_sum(self) -> None:
        detector = NoveltyDetector(window_size=8, min_baseline=1, warmup_events=0)
        vector = np.ones(4, dtype=np.float32) / 2.0
        for _ in range(5):
            detector.record_embedding(vector)
        detector.compute_novelty(vector)
        detector.reset()

        assert detector._centroid_sum is None
        assert detector._centroid is None
        assert detector.get_stats()["centroid_recompute_count"] == 0

        detector.record_embedding(vector)
        detector.compute_novelty(vector)
        assert np.allclose(detector._centroid_sum, vector)
