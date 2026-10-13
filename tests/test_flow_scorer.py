"""Flow-aware tier-1 scoring.

These tests build a small synthetic flow model instead of depending on
``make train-flow`` output, so the flow routing/fusion logic executes on a clean
checkout and in CI rather than being skipped forever.
"""

# pyright: reportMissingImports=false
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.linear_model import LogisticRegression

from logfilter.models.classifier import SafeMaxAbsScaler
from logfilter.pipeline.flow_features import FLOW_FEATURE_NAMES, extract_flow_vector
from logfilter.pipeline.normalizer import LogNormalizer
from logfilter.pipeline.scorer import LogScorer

_DDOS_CEF = (
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
_FIREWALL_CEF = "CEF:0|Vendor|Firewall|1|ALLOW|Connection allowed|1|src=192.0.2.1 dpt=443"


def _write_synthetic_flow_artifacts(target: Path) -> Path:
    """Write a tiny but loadable flow model bundle and return its directory.

    The model is a logistic regression fitted on the *actual* extracted vector
    of ``_DDOS_CEF`` (positive) versus a quiet flow (negative), exported through
    the same skl2onnx path ``training/train_flow_classifier.py`` uses. It is
    deliberately weak — the tests assert *routing* (which classifier scored the
    event), not detection accuracy.
    """
    width = len(FLOW_FEATURE_NAMES)
    target.mkdir(parents=True, exist_ok=True)

    attack = extract_flow_vector(LogNormalizer().normalize(_DDOS_CEF))
    assert attack is not None, "the DDoS sample must yield a flow vector"
    quiet = attack.copy()
    quiet[0] = 443.0
    quiet[2:6] = 1.0  # tiny packet/byte counts
    quiet[10:12] = 0.0  # no burstiness

    features = np.vstack([attack, quiet, attack, quiet, attack, quiet])
    labels = np.array([1, 0, 1, 0, 1, 0])

    model = LogisticRegression(C=1.0).fit(features, labels)
    onnx_model = to_onnx(
        model,
        initial_types=[("float_input", FloatTensorType([None, width]))],
        options={id(model): {"zipmap": False}},
        target_opset=15,
    )
    (target / "flow_classifier.onnx").write_bytes(onnx_model.SerializeToString())

    # SafeMaxAbsScaler rejects a zero scale; use the per-feature observed max with
    # a floor of 1.0 for features that are constant across these samples.
    scale = np.maximum(np.abs(features).max(axis=0), 1.0)
    SafeMaxAbsScaler(scale).to_json(target / "scaler.json")
    (target / "feature_names.json").write_text(json.dumps(list(FLOW_FEATURE_NAMES)))
    return target


def _config(model_dir: Path) -> dict:
    return {
        "models": {
            "flow": {
                "enabled": True,
                "path": str(model_dir / "flow_classifier.onnx"),
                "scaler_path": str(model_dir / "scaler.json"),
                "feature_names_path": str(model_dir / "feature_names.json"),
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


def _score(config: dict, raw: str):
    return LogScorer(config).score(LogNormalizer().normalize(raw))


@pytest.fixture(scope="module")
def flow_model_dir(tmp_path_factory) -> Path:
    return _write_synthetic_flow_artifacts(tmp_path_factory.mktemp("flow-model"))


def test_flow_model_scores_label_blind_ddos_observation(flow_model_dir: Path) -> None:
    """A CEF flow observation is scored by the flow classifier, not the text model."""
    result = _score(_config(flow_model_dir), _DDOS_CEF)

    assert result.flow_score > 0.5
    assert result.classifier_score == pytest.approx(result.flow_score)
    assert result.ai_priority == "MEDIUM"
    assert "DDoS" not in result.normalized_text


def test_non_flow_firewall_event_keeps_text_classifier_path(flow_model_dir: Path) -> None:
    """A firewall allow event has no flow vector, so flow_score stays 0.0."""
    result = _score(_config(flow_model_dir), _FIREWALL_CEF)

    assert result.flow_score == 0.0
    assert result.score_degraded is False


def test_flow_scoring_failure_degrades_and_falls_back(flow_model_dir: Path) -> None:
    """An artifact whose feature order diverges from the runtime schema degrades.

    FlowClassifier refuses a bundle whose feature_names.json does not match
    FLOW_FEATURE_NAMES; the scorer must catch that and keep the event rather
    than letting the ValueError escape score().
    """
    reordered = list(FLOW_FEATURE_NAMES)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    (flow_model_dir / "feature_names_reordered.json").write_text(json.dumps(reordered))

    config = _config(flow_model_dir)
    config["models"]["flow"]["feature_names_path"] = str(
        flow_model_dir / "feature_names_reordered.json"
    )

    result = _score(config, _DDOS_CEF)

    assert result.flow_score == 0.0
    assert result.score_degraded is True
    assert 0.0 <= result.classifier_score <= 1.0