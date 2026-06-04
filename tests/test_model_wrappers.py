"""Tests for lightweight model-wrapper logic with heavy models mocked out."""

from __future__ import annotations

import sys
import types

import numpy as np

from logfilter.models.cross_encoder import CrossEncoderModel
from logfilter.models.ner import ExtractedEntities, NERModel


def test_extracted_entities_flat_string_deduplicates_in_order() -> None:
    entities = ExtractedEntities(
        indicators=["10.0.0.1", "10.0.0.1"],
        malware=["loader"],
        vulnerabilities=["CVE-2026-0001"],
        organizations=["ExampleCo"],
        systems=["Linux"],
    )

    assert entities.flat_entity_string() == "10.0.0.1,loader,CVE-2026-0001,ExampleCo,Linux"


def test_ner_parse_entities_filters_confidence_and_groups() -> None:
    model = NERModel(min_confidence=0.8)

    parsed = model._parse_entities(
        [
            {"score": 0.95, "entity_group": "Indicator", "word": "10.0.0.5"},
            {"score": 0.90, "entity_group": "Malware", "word": "##loader"},
            {"score": 0.99, "entity_group": "Vulnerability", "word": "CVE-2026-0001"},
            {"score": 0.85, "entity_group": "Organization", "word": "ExampleCo"},
            {"score": 0.82, "entity_group": "System", "word": "Linux"},
            {"score": 0.10, "entity_group": "Indicator", "word": "filtered"},
            {"score": 0.90, "entity_group": "O", "word": "ignored"},
            {"score": 0.90, "entity_group": "Indicator", "word": "###"},
        ]
    )

    assert parsed.indicators == ["10.0.0.5"]
    assert parsed.malware == ["loader"]
    assert parsed.vulnerabilities == ["CVE-2026-0001"]
    assert parsed.organizations == ["ExampleCo"]
    assert parsed.systems == ["Linux"]
    assert parsed.confidence == 0.99
    assert parsed.has_high_value_entities


def test_ner_extract_batch_handles_single_text_pipeline_shape() -> None:
    model = NERModel(batch_size=2, min_confidence=0.1)
    model._pipeline = lambda batch: [{"score": 0.9, "entity_group": "Indicator", "word": batch[0]}]

    results = model.extract_batch(["10.0.0.5"])

    assert results[0].indicators == ["10.0.0.5"]


def test_ner_load_uses_cuda_device(monkeypatch) -> None:
    calls = []

    def fake_pipeline(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    module = types.SimpleNamespace(pipeline=fake_pipeline)
    monkeypatch.setitem(sys.modules, "transformers", module)

    model = NERModel(model_id="ner", device="cuda")
    model._load()

    assert calls[0][1]["device"] == 0


def test_cross_encoder_score_empty_candidates_skips_model_load() -> None:
    model = CrossEncoderModel()

    assert model.score("text", []) == []


def test_cross_encoder_score_batch_sorts_and_sigmoids() -> None:
    class FakeModel:
        def predict(self, pairs, **kwargs):
            assert pairs == [
                ("log-a", "desc-low"),
                ("log-a", "desc-high"),
                ("log-b", "desc-mid"),
            ]
            return np.array([-2.0, 2.0, 0.0], dtype=np.float32)

    model = CrossEncoderModel(batch_size=4)
    model._model = FakeModel()

    results = model.score_batch(
        ["log-a", "log-b", "log-c"],
        [
            [
                {"id": "T1", "name": "Low", "description": "desc-low"},
                {"id": "T2", "name": "High", "description": "desc-high"},
            ],
            [{"id": "T3", "name": "Mid", "description": "desc-mid"}],
            [],
        ],
    )

    assert [score.technique_id for score in results[0]] == ["T2", "T1"]
    assert round(results[1][0].score, 3) == 0.5
    assert results[2] == []


def test_cross_encoder_load_uses_sentence_transformer(monkeypatch) -> None:
    calls = []

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    module = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    model = CrossEncoderModel(model_id="cross", device="cpu")
    model._load()

    assert calls[0][0] == ("cross",)
    assert calls[0][1]["max_length"] == 1024


def test_cross_encoder_load_with_cache_dir_and_revision(monkeypatch) -> None:
    calls = []

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

    module = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    model = CrossEncoderModel(
        model_id="cross", device="cpu", cache_dir="/tmp/cache", revision="v1"
    )
    model._load()

    assert calls[0][1]["cache_folder"] == "/tmp/cache"
    assert calls[0][1]["revision"] == "v1"


def test_cross_encoder_score_batch_all_pairs_empty() -> None:
    model = CrossEncoderModel()
    model._model = None

    results = model.score_batch(["log-a", "log-b"], [[], []])
    assert results == [[], []]


def test_cross_encoder_score_batch_with_model_load(monkeypatch) -> None:
    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs, **kwargs):
            return np.array([1.0], dtype=np.float32)

    module = types.SimpleNamespace(CrossEncoder=FakeModel)
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    model = CrossEncoderModel()
    results = model.score_batch(
        ["log-a"],
        [[{"id": "T1", "name": "Test", "description": "desc"}]],
    )
    assert len(results) == 1
    assert results[0][0].technique_id == "T1"


def test_ner_load_with_cache_dir_and_revision(monkeypatch) -> None:
    calls = []

    def fake_pipeline(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    module = types.SimpleNamespace(pipeline=fake_pipeline)
    monkeypatch.setitem(sys.modules, "transformers", module)

    model = NERModel(model_id="ner", device="cpu", cache_dir="/tmp/cache", revision="v1")
    model._load()

    assert calls[0][1]["cache_dir"] == "/tmp/cache"
    assert calls[0][1]["revision"] == "v1"


def test_ner_extract_batch_single_text_fallback() -> None:
    model = NERModel(batch_size=2)
    model._pipeline = lambda batch: [
        {"score": 0.9, "entity_group": "Indicator", "word": "10.0.0.5"}
    ]

    results = model.extract_batch(["single text"])
    assert results[0].indicators == ["10.0.0.5"]


def test_ner_extract_batch_multi_text() -> None:
    model = NERModel(batch_size=2)
    model._pipeline = lambda batch: [
        [{"score": 0.9, "entity_group": "Indicator", "word": "10.0.0.5"}],
        [{"score": 0.8, "entity_group": "Malware", "word": "trojan"}],
    ]

    results = model.extract_batch(["text1", "text2"])
    assert results[0].indicators == ["10.0.0.5"]
    assert results[1].malware == ["trojan"]


def test_ner_parse_entities_filters_low_confidence() -> None:
    model = NERModel(min_confidence=0.8)

    parsed = model._parse_entities(
        [
            {"score": 0.95, "entity_group": "Indicator", "word": "10.0.0.5"},
            {"score": 0.10, "entity_group": "Indicator", "word": "filtered"},
        ]
    )

    assert parsed.indicators == ["10.0.0.5"]


def test_ner_parse_entities_skips_empty_word_and_o_label() -> None:
    model = NERModel(min_confidence=0.1)

    parsed = model._parse_entities(
        [
            {"score": 0.9, "entity_group": "O", "word": "ignored"},
            {"score": 0.9, "entity_group": "Indicator", "word": ""},
            {"score": 0.9, "entity_group": "Indicator", "word": "ok"},
        ]
    )

    assert parsed.indicators == ["ok"]


def test_ner_parse_entities_strips_hashes() -> None:
    model = NERModel(min_confidence=0.1)

    parsed = model._parse_entities(
        [
            {"score": 0.9, "entity_group": "Indicator", "word": "##10.0.0.5"},
            {"score": 0.9, "entity_group": "Indicator", "word": "###"},
        ]
    )

    assert parsed.indicators == ["10.0.0.5"]


def test_ner_parse_entities_all_entity_types() -> None:
    model = NERModel(min_confidence=0.1)

    parsed = model._parse_entities(
        [
            {"score": 0.9, "entity_group": "Indicator", "word": "10.0.0.5"},
            {"score": 0.9, "entity_group": "Malware", "word": "trojan"},
            {"score": 0.9, "entity_group": "Vulnerability", "word": "CVE-2026-0001"},
            {"score": 0.9, "entity_group": "Organization", "word": "ExampleCo"},
            {"score": 0.9, "entity_group": "System", "word": "Linux"},
        ]
    )

    assert parsed.indicators == ["10.0.0.5"]
    assert parsed.malware == ["trojan"]
    assert parsed.vulnerabilities == ["CVE-2026-0001"]
    assert parsed.organizations == ["ExampleCo"]
    assert parsed.systems == ["Linux"]
    assert parsed.has_high_value_entities is True


def test_ner_extract_calls_load_lazily(monkeypatch) -> None:
    calls = []

    class FakePipeline:
        def __init__(self, *args, **kwargs):
            calls.append((args, kwargs))

        def __call__(self, batch):
            return [{"score": 0.9, "entity_group": "Indicator", "word": "10.0.0.5"}]

    module = types.SimpleNamespace(pipeline=FakePipeline)
    monkeypatch.setitem(sys.modules, "transformers", module)

    model = NERModel()
    assert model._pipeline is None
    result = model.extract("test")
    assert model._pipeline is not None
    assert result.indicators == ["10.0.0.5"]
    assert len(calls) == 1
