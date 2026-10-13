"""Tests for BiEncoder logic using fake FAISS and sentence-transformer modules."""

from __future__ import annotations

import json
import sys
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from logfilter.models.biencoder import (
    BiEncoderModel,
    _cpu_supports_avx2,
    _new_inner_product_index,
    _NumpyIndexFlatIP,
)
from logfilter.models.ner import ExtractedEntities, NERModel
from logfilter.monitoring.novelty_detector import NoveltyDetector


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

    def get_embedding_dimension(self) -> int:
        return 2

    def encode(self, texts, **kwargs):
        del kwargs
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


def test_biencoder_numpy_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("logfilter.models.biencoder.np", None)
    monkeypatch.setattr("logfilter.models.biencoder._NUMPY_AVAILABLE", False)

    model = BiEncoderModel()
    assert model._dedup_window.maxlen is None


def test_biencoder_load_with_cache_dir_and_revision(monkeypatch) -> None:
    _install_fake_modules(monkeypatch)

    calls = []

    class FakeSTWithArgs(FakeSentenceTransformer):
        def __init__(self, model_id: str, **kwargs):
            calls.append((model_id, kwargs))
            super().__init__(model_id, kwargs.get("device", "cpu"))

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSTWithArgs),
    )

    model = BiEncoderModel(
        cache_dir="/tmp/cache",
        revision="v1",
    )
    model._load()

    assert calls[0][1]["cache_folder"] == "/tmp/cache"
    assert calls[0][1]["revision"] == "v1"


def test_biencoder_dedup_first_event_no_index(monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    model = BiEncoderModel(dedup_window_minutes=0.001)

    emb = np.array([1.0, 0.0], dtype=np.float32)
    result = model.check_dedup(emb)

    assert result.is_duplicate is False
    assert result.similarity == 0.0


def test_biencoder_retrieve_empty_when_no_techniques(monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    model = BiEncoderModel(mitre_techniques_path="missing.json")

    emb = np.array([1.0, 0.0], dtype=np.float32)
    candidates = model.retrieve_attack_candidates(emb)

    assert candidates == []


def test_biencoder_retrieve_skips_invalid_indices(tmp_path, monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    path = tmp_path / "techniques.json"
    path.write_text(json.dumps([{"id": "T1", "name": "One", "description": "technique one"}]))

    model = BiEncoderModel(mitre_techniques_path=path, faiss_top_k=5)
    emb = np.array([1.0, 0.0], dtype=np.float32)
    candidates = model.retrieve_attack_candidates(emb)

    assert len(candidates) == 1
    assert candidates[0].technique_id == "T1"


def test_biencoder_duplicate_branch_in_batch(tmp_path, monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    path = tmp_path / "techniques.json"
    path.write_text(json.dumps([{"id": "T1", "name": "One", "description": "technique one"}]))

    model = BiEncoderModel(
        mitre_techniques_path=path,
        dedup_threshold=0.9,
        dedup_window_minutes=0.001,
    )

    batch = model.check_dedup_and_retrieve_batch(["log-one", "log-one"])

    assert batch[0][0].is_duplicate is False
    assert batch[1][0].is_duplicate is True
    assert batch[1][1] == []


def test_biencoder_rebuild_dedup_faiss_empty_window(monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    model = BiEncoderModel()
    model._dim = 2

    model._prune_dedup_window()
    model._rebuild_dedup_faiss()

    assert model._faiss_dedup is not None
    assert model._faiss_dedup.ntotal == 0


def test_numpy_index_add_and_search_top_k() -> None:
    index = _NumpyIndexFlatIP(dim=2)
    assert index.ntotal == 0

    index.add(np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32))
    assert index.ntotal == 3

    distances, indices = index.search(np.array([[1.0, 0.0]], dtype=np.float32), k=2)
    assert indices[0][0] == 0
    assert distances[0][0] == pytest.approx(1.0)
    assert indices[0][1] == 2


def test_numpy_index_search_empty_returns_sentinel() -> None:
    index = _NumpyIndexFlatIP(dim=3)
    distances, indices = index.search(np.zeros((1, 3), dtype=np.float32), k=2)
    assert distances.shape == (1, 2)
    assert indices.tolist() == [[-1, -1]]


def test_numpy_index_search_pads_when_k_exceeds_ntotal() -> None:
    index = _NumpyIndexFlatIP(dim=2)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))

    distances, indices = index.search(np.array([[1.0, 0.0]], dtype=np.float32), k=3)
    assert indices[0][0] == 0
    assert indices[0][1] == -1
    assert indices[0][2] == -1
    assert distances[0][1] == 0.0


def test_numpy_index_add_rejects_bad_shape() -> None:
    index = _NumpyIndexFlatIP(dim=2)
    with pytest.raises(ValueError, match="expected matrix with shape"):
        index.add(np.zeros((2, 3), dtype=np.float32))


def test_numpy_index_search_rejects_bad_shape() -> None:
    index = _NumpyIndexFlatIP(dim=2)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="expected query with shape"):
        index.search(np.zeros((1, 5), dtype=np.float32), k=1)


def test_new_inner_product_index_uses_faiss_when_available(monkeypatch) -> None:
    monkeypatch.setattr("logfilter.models.biencoder._cpu_supports_avx2", lambda: True)

    class FaissIndex:
        def __init__(self, dim: int) -> None:
            self.dim = dim

    monkeypatch.setitem(sys.modules, "faiss", types.SimpleNamespace(IndexFlatIP=FaissIndex))

    index = _new_inner_product_index(4)
    assert isinstance(index, FaissIndex)
    assert index.dim == 4


def test_new_inner_product_index_falls_back_when_faiss_missing(monkeypatch) -> None:
    monkeypatch.setattr("logfilter.models.biencoder._cpu_supports_avx2", lambda: True)
    # Setting the module entry to None forces `import faiss` to raise ImportError.
    monkeypatch.setitem(sys.modules, "faiss", None)

    index = _new_inner_product_index(4)
    assert isinstance(index, _NumpyIndexFlatIP)


def test_new_inner_product_index_falls_back_without_avx2(monkeypatch) -> None:
    monkeypatch.setattr("logfilter.models.biencoder._cpu_supports_avx2", lambda: False)
    index = _new_inner_product_index(8)
    assert isinstance(index, _NumpyIndexFlatIP)
    assert index.dim == 8


def test_cpu_supports_avx2_falls_back_to_numpy_when_cpuinfo_unreadable(monkeypatch) -> None:
    """A platform without /proc/cpuinfo (macOS/Windows) must take the NumPy path.

    Assuming AVX2 here would silently force the portable fallback on Linux
    hosts that simply cannot read the file, so the OSError branch returns False.
    """

    class BoomPath:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def read_text(self, **_kwargs):
            raise OSError("no /proc/cpuinfo")

    monkeypatch.setattr("logfilter.models.biencoder.Path", BoomPath)
    assert _cpu_supports_avx2() is False


def test_score_novelty_batch_without_detector_returns_zeros(monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    model = BiEncoderModel()

    results = model.score_novelty_batch(["log-one", "log-two"])

    assert len(results) == 2
    assert all(r.score == 0.0 and r.baseline_size == 0 for r in results)


def test_score_novelty_batch_with_detector_records_and_scores(monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    detector = NoveltyDetector(window_size=100, min_baseline=1, warmup_events=0)
    model = BiEncoderModel(novelty_detector=detector)

    results = model.score_novelty_batch(["log-one", "log-two"])

    assert len(results) == 2
    assert detector.get_stats()["baseline_size"] == 2


def _attack_file(tmp_path) -> Path:
    path = tmp_path / "techniques.json"
    path.write_text(
        json.dumps(
            [
                {"id": "T1", "name": "One", "description": "technique one"},
                {"id": "T2", "name": "Two", "description": "technique two"},
            ]
        )
    )
    return path


def test_missing_mitre_file_is_reported_as_not_ready(tmp_path, monkeypatch) -> None:
    """A missing techniques file must be observable, not silently empty."""
    _install_fake_modules(monkeypatch)
    model = BiEncoderModel(mitre_techniques_path=tmp_path / "absent.json")

    # Force the lazy load path.
    model.encode(["log-one"])

    state = model.is_ready()
    assert state["attack_index_ready"] is False
    assert "absent.json" in state["attack_index_error"]
    assert model.retrieve_attack_candidates(np.array([1.0, 0.0], dtype=np.float32)) == []


def test_loaded_mitre_index_reports_ready(tmp_path, monkeypatch) -> None:
    _install_fake_modules(monkeypatch)
    model = BiEncoderModel(mitre_techniques_path=_attack_file(tmp_path))

    model.encode(["log-one"])

    state = model.is_ready()
    assert state["attack_index_ready"] is True
    assert state["attack_technique_count"] == 2
    assert state["attack_index_error"] is None


class YieldingIndex:
    """Index wrapper whose add/search yield the GIL, widening the race window.

    ``check_dedup`` searches the shared index and then appends + rebuilds it.
    Yielding inside the index makes the read-then-append window wide enough
    that an unguarded implementation reliably interleaves, so the test is a
    real regression guard rather than a probabilistic one.
    """

    def __init__(self, inner) -> None:
        self.inner = inner

    @property
    def ntotal(self) -> int:
        return self.inner.ntotal

    def add(self, values) -> None:
        time.sleep(0.0005)
        self.inner.add(values)

    def search(self, query, k: int):
        time.sleep(0.0005)
        return self.inner.search(query, k)


def _concurrent_dedup_verdicts(model: BiEncoderModel, vector, threads: int = 4) -> list[bool]:
    """Run *threads* workers through check_dedup; return the verdicts in order."""
    verdicts: list[bool] = []
    verdicts_lock = threading.Lock()

    def worker() -> None:
        for _ in range(4):
            result = model.check_dedup(vector)
            with verdicts_lock:
                verdicts.append(result.is_duplicate)

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join(timeout=30.0)
        assert not worker_thread.is_alive()
    return verdicts


def test_check_dedup_is_serialised_across_threads(tmp_path, monkeypatch) -> None:
    """A repeated embedding must read as a duplicate on every thread but one.

    The scorer is a process singleton shared by FastAPI handlers and the Kafka
    consumer thread. If check_dedup's read-then-append is unguarded, callers
    read a stale index and report an already-seen event as novel, which
    re-enables the Tier-3 work dedup exists to suppress.
    """
    _install_fake_modules(monkeypatch)
    path = tmp_path / "techniques.json"
    path.write_text(json.dumps([{"id": "T1", "name": "One", "description": "technique one"}]))
    model = BiEncoderModel(mitre_techniques_path=path, dedup_threshold=0.95)
    model.encode(["log-one"])
    model._faiss_dedup = YieldingIndex(model._faiss_dedup)

    verdicts = _concurrent_dedup_verdicts(model, np.array([1.0, 0.0], dtype=np.float32))

    # Exactly one caller seeds the empty index; every other caller must see it.
    assert len(verdicts) == 16
    assert verdicts.count(False) == 1, verdicts
    assert verdicts.count(True) == 15, verdicts


def test_check_dedup_window_matches_index_after_concurrency(tmp_path, monkeypatch) -> None:
    """Every appended embedding must survive into both window and index."""
    _install_fake_modules(monkeypatch)
    model = BiEncoderModel(mitre_techniques_path=_attack_file(tmp_path))
    model.encode(["log-one"])

    vectors = [np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0], dtype=np.float32)]
    threads = [threading.Thread(target=model.check_dedup, args=(v,)) for v in vectors * 8]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
        assert not thread.is_alive()

    assert len(model._dedup_window) == len(vectors) * 8
    assert model._faiss_dedup.ntotal == len(vectors) * 8



# ── NER wrapper ───────────────────────────────────────────────────────────────


class FakeNERTokenizer:
    """Tokenizer double that enforces a real token ceiling like the HF one."""

    def __init__(self, model_max_length: int) -> None:
        self.model_max_length = model_max_length


def _install_fake_ner(monkeypatch, model_max_length: int = 8192) -> list[list[str]]:
    """Install fake transformers modules; return the captured pipeline inputs."""
    seen: list[list[str]] = []

    def fake_pipeline(_task: str, **kwargs):
        def run(batch):
            seen.append(list(batch))
            return [[] for _ in batch]

        return run

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            pipeline=fake_pipeline,
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda model_id, **_kw: FakeNERTokenizer(model_max_length)
            ),
        ),
    )
    return seen


def test_ner_extract_batch_truncates_over_long_text(monkeypatch) -> None:
    """A line past the token ceiling must not raise; it is truncated instead.

    The pipeline in transformers v5 has no `truncation` kwarg, so the guard is
    applied to the raw text before the pipeline ever sees it. Previously an
    over-long line raised and took down the whole batch, and passing
    `truncation=True` to the pipeline raised for *every* input.
    """
    seen = _install_fake_ner(monkeypatch)
    model = NERModel()
    long_line = "A" * (model._max_input_chars * 3)

    results = model.extract_batch([long_line, "short line"])

    assert len(results) == 2
    assert all(isinstance(r, ExtractedEntities) for r in results)
    assert len(seen[0]) == 2
    assert len(seen[0][0]) <= model._max_input_chars
    assert seen[0][1] == "short line"


def test_ner_truncation_never_exceeds_the_token_ceiling(monkeypatch) -> None:
    """The char budget must be small enough to fit the model's token limit."""
    seen = _install_fake_ner(monkeypatch, model_max_length=512)
    model = NERModel()
    model.extract_batch(["B" * 100_000])

    truncated = seen[0][0]
    assert len(truncated) <= 512 * NERModel.CONSERVATIVE_CHARS_PER_TOKEN
    assert truncated.endswith(NERModel.TRUNCATION_MARKER)


def test_ner_unbounded_tokenizer_uses_default_budget(monkeypatch) -> None:
    """A tokenizer declaring no finite limit must not disable truncation."""
    seen = _install_fake_ner(monkeypatch, model_max_length=int(1e30))
    model = NERModel()
    model.extract_batch(["C" * 10_000_000])

    assert len(seen[0][0]) <= NERModel.DEFAULT_MAX_TOKENS * NERModel.CONSERVATIVE_CHARS_PER_TOKEN


def test_ner_short_texts_are_untouched(monkeypatch) -> None:
    seen = _install_fake_ner(monkeypatch)
    model = NERModel()
    texts = ["first", "second", "third"]

    results = model.extract_batch(texts)

    assert seen[0] == texts
    assert len(results) == 3


def test_ner_strips_all_leading_hashes_from_subword_tokens() -> None:
    """All leading word-piece markers are removed, whatever the run length.

    ``lstrip("#")`` removes every leading '#', which is what a subword token needs:
    "##evil" and "#root" denote the same entities as "evil" and "root", and a bare
    "###" separator must collapse to nothing rather than survive as a stray "##"
    fragment in the indicator list.
    """
    model = NERModel()

    def indicator(word: str) -> list[str]:
        return model._parse_entities(
            [{"entity_group": "B-Indicator", "score": 0.99, "word": word}]
        ).indicators

    assert indicator("##evil") == ["evil"]
    assert indicator("###evil") == ["evil"]
    assert indicator("#root") == ["root"]
    assert indicator("###") == []  # separator only, not an entity
    assert indicator("evil.exe") == ["evil.exe"]



def test_ner_budget_follows_model_position_embeddings(monkeypatch) -> None:
    """A tokenizer claiming 1e30 must not disable truncation.

    transformers reports model_max_length = 1e30 whenever no finite limit was
    declared, and the pipeline then self-truncates against that "limit" — so the
    text reaches the model's position-embedding table and raises. The budget
    must come from the model config, not the tokenizer declaration.
    """
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            pipeline=lambda *_a, **_k: types.SimpleNamespace(model=None),
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda model_id, **_kw: FakeNERTokenizer(int(1e30))
            ),
        ),
    )
    model = NERModel()

    # Without a model to inspect, fall back to the class default — never to 1e30.
    assert model._max_input_chars == NERModel.DEFAULT_MAX_TOKENS * (
        NERModel.CONSERVATIVE_CHARS_PER_TOKEN
    )


def test_ner_budget_uses_model_config_when_tokenizer_is_unbounded(monkeypatch) -> None:
    class FakeModel:
        class config:
            max_position_embeddings = 512

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(
            pipeline=lambda *_a, **_k: types.SimpleNamespace(model=FakeModel()),
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda model_id, **_kw: FakeNERTokenizer(int(1e30))
            ),
        ),
    )
    model = NERModel()
    model._load()

    # 512 positions, not the 8192 class default and not the 1e30 sentinel.
    assert model._max_input_chars == 512 * NERModel.CONSERVATIVE_CHARS_PER_TOKEN
