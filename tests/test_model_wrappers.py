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
