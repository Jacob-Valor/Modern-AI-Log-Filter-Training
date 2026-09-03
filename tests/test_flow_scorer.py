"""Integration coverage for flow-aware tier-1 scoring."""

# pyright: reportMissingImports=false
from __future__ import annotations

from pathlib import Path

import pytest

from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer

pytestmark = pytest.mark.skipif(
    not Path("models/flow/flow_classifier.onnx").exists(),
    reason="flow model artifacts are provisioned with 'make train-flow'",
)


def _config() -> dict:
    return {
        "models": {
            "flow": {
                "enabled": True,
                "path": "models/flow/flow_classifier.onnx",
                "scaler_path": "models/flow/scaler.json",
                "feature_names_path": "models/flow/feature_names.json",
            },
            "ner": {"enabled": False},
            "biencoder": {"enabled": False},
            "cross_encoder": {"enabled": False},
        },
        "scoring": {
            "weights": {
                "classifier": 1.0,
                "entity_boost": 0.0,
                "cross_encoder": 0.0,
                "novelty": 0.0,
            },
            "routing": {"high": 0.85, "medium": 0.5, "low": 0.2},
            "flow_window": {"enabled": False},
        },
    }


def test_flow_model_scores_label_blind_ddos_observation():
    raw = (
        "CEF:0|UNB|CICFlowMeter|1.0|FLOW|Network flow observation|1|"
        "dpt=80\tduration=894580\tfwdPackets=3\tbwdPackets=5\t"
        "fwdBytes=26\tbwdBytes=11607\tflowBytesPerSec=13003.86774\t"
        "flowPacketsPerSec=8.942744081\tpacketLengthMean=1292.555556\t"
        "packetLengthStd=2650.585648\tsynCount=0\tackCount=0\trstCount=0\t"
        "activeMean=0\tidleMean=0\tfinCount=0\tpshCount=1\turgCount=0\t"
        "flowIatMean=127797.1429\tflowIatStd=337732.7702\tflowIatMax=893703\t"
        "flowIatMin=4\tfwdIatMean=316\tfwdIatStd=284.256926\t"
        "bwdIatMean=223630.75\tbwdIatStd=446714.9202\tpacketLengthMin=0\t"
        "packetLengthMax=7215\tfwdPacketLengthMean=8.666666667\t"
        "bwdPacketLengthMean=2321.4\tinitWinFwd=8192\tinitWinBwd=229\t"
        "downUpRatio=1"
    )
    result = LogScorer(_config()).score(LogNormalizer().normalize(raw))
    assert result.flow_score > 0.5
    assert result.classifier_score == pytest.approx(result.flow_score)
    assert result.ai_priority == "MEDIUM"
    assert "DDoS" not in result.normalized_text


def test_non_flow_firewall_event_keeps_text_classifier_path():
    raw = "CEF:0|Vendor|Firewall|1|ALLOW|Connection allowed|1|src=192.0.2.1 dpt=443"
    result = LogScorer(_config()).score(LogNormalizer().normalize(raw))
    assert result.flow_score == 0.0
