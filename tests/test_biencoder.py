"""Tests for BiEncoder logic using fake FAISS and sentence-transformer modules."""

from __future__ import annotations

import json
import sys
import types

import numpy as np

from logfilter.models.biencoder import BiEncoderModel


class FakeIndexFlatIP:
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, values) -> None:
        self.vectors = np.vstack([self.vectors, values.astype(np.float32)])

    def search(self, query, k: int):
        if self.ntotal == 0:
            return np.zeros((1, 0), dtype=np.float32), np.zeros((1, 0), dtype=np.int64)
        scores = query @ self.vectors.T
        order = np.argsort(scores[0])[::-1][:k]
        return scores[:, order], order.reshape(1, -1)


class FakeSentenceTransformer:
    def __init__(self, model_id: str, device: str = "cpu") -> None:
        self.model_id = model_id
        self.device = device

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts, **kwargs):
        mapping = {
            "technique one": [1.0, 0.0],
            "technique two": [0.0, 1.0],
            "log-one": [1.0, 0.0],
            "log-two": [0.0, 1.0],
        }
        return np.array([mapping.get(text, [0.5, 0.5]) for text in texts], dtype=np.float32)


def _install_fake_modules(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "faiss", types.SimpleNamespace(IndexFlatIP=FakeIndexFlatIP))
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )


def test_biencoder_loads_and_retrieves_attack_candidates(tmp_path, monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    path = tmp_path / "techniques.json"
    path.write_text(
        json.dumps(
            [
                {"id": "T1", "name": "One", "description": "technique one"},
                {"id": "T2", "name": "Two", "description": "technique two"},
            ]
        )
    )
    model = BiEncoderModel(mitre_techniques_path=path, faiss_top_k=2)

    embedding = model.encode(["log-one"])[0]
    candidates = model.retrieve_attack_candidates(embedding)

    assert [candidate.technique_id for candidate in candidates] == ["T1", "T2"]


def test_biencoder_missing_techniques_returns_empty_candidates(tmp_path, monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    model = BiEncoderModel(mitre_techniques_path=tmp_path / "missing.json")

    model.encode(["log-one"])

    assert model.retrieve_attack_candidates(np.array([1.0, 0.0], dtype=np.float32)) == []


def test_biencoder_dedup_and_batch_paths(tmp_path, monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    path = tmp_path / "techniques.json"
    path.write_text(json.dumps([{"id": "T1", "name": "One", "description": "technique one"}]))
    model = BiEncoderModel(
        mitre_techniques_path=path,
        dedup_threshold=0.9,
        dedup_window_minutes=0.001,
    )

    first = model.check_dedup(np.array([1.0, 0.0], dtype=np.float32))
    second = model.check_dedup(np.array([1.0, 0.0], dtype=np.float32))
    batch = model.check_dedup_and_retrieve_batch(["log-two"])

    assert not first.is_duplicate
    assert second.is_duplicate
    assert batch[0][0].is_duplicate is False
    assert batch[0][1][0].technique_id == "T1"
