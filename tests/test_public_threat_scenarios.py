"""Tests for label-independent public threat scenario adapters."""

from __future__ import annotations

import importlib
import json
import sys

from logfilter.pipeline.normalizer import LogNormalizer

public_eval = importlib.import_module("scripts.evaluate_public_threat_scenarios")


def test_csic_post_is_adapted_to_structured_waf_with_redaction() -> None:
    request = (
        "POST /login HTTP/1.1\r\n"
        "Host: example.test\r\n"
        "Content-Type: application/x-www-form-urlencoded\r\n"
        "\r\n"
        "password=do-not-expose&query=%2527%2520OR%25201%253D1--\r\n"
    )
    model_input = public_eval.csic_to_waf(request)
    assert model_input is not None
    envelope = json.loads(model_input)
    assert envelope["request"]["body"].startswith("password=do-not-expose")

    event = LogNormalizer().normalize(model_input)

    assert event.fields["http_lexical_score"] >= 0.7
    assert "do-not-expose" not in event.text
    assert "do-not-expose" not in event.fields["request_body_redacted"]
    assert "REDACTED" in event.fields["request_body_redacted"]


def test_csic_get_adapter_is_deterministic_and_has_empty_body() -> None:
    request = "GET /products?page=2 HTTP/1.1\nHost: example.test\nUser-Agent: test\n"
    first = public_eval.csic_to_waf(request)
    second = public_eval.csic_to_waf(request)
    assert first == second
    assert first is not None
    assert json.loads(first)["request"]["body"] == ""


def test_malformed_csic_request_is_rejected() -> None:
    assert public_eval.csic_to_waf("not-http") is None


def test_cic_evaluator_defaults_to_immutable_held_out_cutoffs(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["evaluate_public_threat_scenarios.py"])
    args = public_eval.parse_args()
    assert args.ddos_min_row == 175002
    assert args.bot_min_row == 152828
    assert args.web_min_row == 70002


def test_deduplication_representative_does_not_depend_on_ground_truth() -> None:
    common = {
        "scenario": "test",
        "source_file": "source",
        "row_number": 1,
        "exact_label": "label",
        "family": "family",
        "model_input": "same-model-input",
        "adapter": "adapter",
    }
    attack = public_eval.Sample(sample_id="b", is_attack=True, **common)
    benign = public_eval.Sample(sample_id="a", is_attack=False, **common)

    selected, removed, conflicts = public_eval.deduplicate_samples([attack, benign], seed=7)

    assert [sample.sample_id for sample in selected] == ["a"]
    assert removed == 1
    assert conflicts == 1
