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
