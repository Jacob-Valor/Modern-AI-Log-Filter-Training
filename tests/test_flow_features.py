"""Tests for leakage-safe flow features and bounded DDoS windows."""

# pyright: reportMissingImports=false
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from logfilter.pipeline.flow_features import (
    FLOW_FEATURE_NAMES,
    FlowWindowAggregator,
    extract_flow_vector,
)
from logfilter.pipeline.normalizer import LogNormalizer


def _flow(*, source: str = "192.0.2.1", destination: str = "198.51.100.10"):
    line = (
        "CEF:0|UNB|CICFlowMeter|1.0|FLOW|Network flow observation|1|"
        f"src={source}\tdst={destination}\tdpt=443\tproto=TCP\tduration=1000\t"
        "fwdPackets=100\tbwdPackets=20\tfwdBytes=50000\tbwdBytes=10000\t"
        "flowBytesPerSec=60000\tflowPacketsPerSec=120\tpacketLengthMean=500\t"
        "packetLengthStd=50\tsynCount=10\tackCount=8\trstCount=0\t"
        "activeMean=500\tidleMean=100"
    )
    return LogNormalizer().normalize(line)


def test_extract_flow_vector_uses_canonical_order():
    vector = extract_flow_vector(_flow())
    assert vector is not None
    assert vector.dtype == np.float32
    assert vector.shape == (len(FLOW_FEATURE_NAMES),)
    assert vector[0] == 443
    assert vector[2] == 100
    assert vector[10] == 10


def test_extract_flow_vector_rejects_incomplete_or_non_flow_input():
    event = LogNormalizer().normalize("Jan 01 00:00:00 host sshd: accepted login")
    assert extract_flow_vector(event) is None


def test_extract_flow_vector_replaces_invalid_values_without_label_features():
    event = _flow()
    event.fields["duration"] = "NaN"
    event.fields["synCount"] = -1
    event.fields["attackCategory"] = "DDoS"
    vector = extract_flow_vector(event)
    assert vector is not None
    assert vector[1] == 0
    assert vector[10] == 0
    assert len(vector) == len(FLOW_FEATURE_NAMES)


def test_window_expires_old_entries_and_is_bounded():
    aggregator = FlowWindowAggregator(window_seconds=10, max_keys=2, max_events_per_key=2)
    first = aggregator.update(_flow(source="192.0.2.1"), now=0)
    assert first.flow_count == 1
    aggregator.update(_flow(source="192.0.2.2"), now=1)
    capped = aggregator.update(_flow(source="192.0.2.3"), now=2)
    assert capped.flow_count == 2
    expired = aggregator.update(_flow(source="192.0.2.4"), now=20)
    assert expired.flow_count == 1


def test_window_requires_two_corroborating_volume_signals():
    aggregator = FlowWindowAggregator(
        window_seconds=10,
        flow_threshold=2,
        packet_threshold=200,
        byte_threshold=100_000,
        source_threshold=2,
    )
    aggregator.update(_flow(source="192.0.2.1"), now=0)
    snapshot = aggregator.update(_flow(source="192.0.2.2"), now=1)
    assert snapshot.flow_count == 2
    assert snapshot.unique_sources == 2
    assert snapshot.risk_score == pytest.approx(1.0)


def test_window_expires_out_of_order_entries_using_latest_watermark():
    aggregator = FlowWindowAggregator(window_seconds=10)
    aggregator.update(_flow(source="192.0.2.1"), now=20)
    aggregator.update(_flow(source="192.0.2.2"), now=0)
    snapshot = aggregator.update(_flow(source="192.0.2.3"), now=25)
    assert snapshot.flow_count == 2


def test_window_does_not_merge_flows_with_missing_destination_identity():
    aggregator = FlowWindowAggregator()
    event = _flow()
    event.fields.pop("dst")
    snapshot = aggregator.update(event, now=1)
    assert snapshot.flow_count == 0
    assert aggregator.key_count == 0


def test_window_is_thread_safe():
    aggregator = FlowWindowAggregator(max_events_per_key=1000)
    event = _flow()
    with ThreadPoolExecutor(max_workers=8) as pool:
        snapshots = list(pool.map(lambda _: aggregator.update(event, now=5), range(100)))
    assert max(snapshot.flow_count for snapshot in snapshots) == 100
