"""Leakage-safe numeric flow features and bounded DDoS time-window aggregation."""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from logfilter.pipeline.normalizer import NormalizedEvent

FLOW_FEATURE_NAMES: tuple[str, ...] = (
    "destination_port",
    "flow_duration_us",
    "forward_packets",
    "backward_packets",
    "forward_bytes",
    "backward_bytes",
    "flow_bytes_per_second",
    "flow_packets_per_second",
    "packet_length_mean",
    "packet_length_std",
    "syn_count",
    "ack_count",
    "rst_count",
    "active_mean_us",
    "idle_mean_us",
    "fin_count",
    "psh_count",
    "urg_count",
    "flow_iat_mean_us",
    "flow_iat_std_us",
    "flow_iat_max_us",
    "flow_iat_min_us",
    "forward_iat_mean_us",
    "forward_iat_std_us",
    "backward_iat_mean_us",
    "backward_iat_std_us",
    "packet_length_min",
    "packet_length_max",
    "forward_packet_length_mean",
    "backward_packet_length_mean",
    "initial_window_forward_bytes",
    "initial_window_backward_bytes",
    "down_up_ratio",
)

_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "destination_port": ("dpt", "destination_port", "Destination Port"),
    "flow_duration_us": ("duration", "flow_duration", "Flow Duration"),
    "forward_packets": ("fwdPackets", "forward_packets", "Total Fwd Packets"),
    "backward_packets": ("bwdPackets", "backward_packets", "Total Backward Packets"),
    "forward_bytes": ("fwdBytes", "forward_bytes", "Total Length of Fwd Packets"),
    "backward_bytes": ("bwdBytes", "backward_bytes", "Total Length of Bwd Packets"),
    "flow_bytes_per_second": ("flowBytesPerSec", "flow_bytes_per_second", "Flow Bytes/s"),
    "flow_packets_per_second": (
        "flowPacketsPerSec",
        "flow_packets_per_second",
        "Flow Packets/s",
    ),
    "packet_length_mean": ("packetLengthMean", "packet_length_mean", "Packet Length Mean"),
    "packet_length_std": ("packetLengthStd", "packet_length_std", "Packet Length Std"),
    "syn_count": ("synCount", "syn_count", "SYN Flag Count"),
    "ack_count": ("ackCount", "ack_count", "ACK Flag Count"),
    "rst_count": ("rstCount", "rst_count", "RST Flag Count"),
    "active_mean_us": ("activeMean", "active_mean", "Active Mean"),
    "idle_mean_us": ("idleMean", "idle_mean", "Idle Mean"),
    "fin_count": ("finCount", "fin_count", "FIN Flag Count"),
    "psh_count": ("pshCount", "psh_count", "PSH Flag Count"),
    "urg_count": ("urgCount", "urg_count", "URG Flag Count"),
    "flow_iat_mean_us": ("flowIatMean", "flow_iat_mean", "Flow IAT Mean"),
    "flow_iat_std_us": ("flowIatStd", "flow_iat_std", "Flow IAT Std"),
    "flow_iat_max_us": ("flowIatMax", "flow_iat_max", "Flow IAT Max"),
    "flow_iat_min_us": ("flowIatMin", "flow_iat_min", "Flow IAT Min"),
    "forward_iat_mean_us": ("fwdIatMean", "forward_iat_mean", "Fwd IAT Mean"),
    "forward_iat_std_us": ("fwdIatStd", "forward_iat_std", "Fwd IAT Std"),
    "backward_iat_mean_us": ("bwdIatMean", "backward_iat_mean", "Bwd IAT Mean"),
    "backward_iat_std_us": ("bwdIatStd", "backward_iat_std", "Bwd IAT Std"),
    "packet_length_min": ("packetLengthMin", "packet_length_min", "Min Packet Length"),
    "packet_length_max": ("packetLengthMax", "packet_length_max", "Max Packet Length"),
    "forward_packet_length_mean": (
        "fwdPacketLengthMean",
        "forward_packet_length_mean",
        "Fwd Packet Length Mean",
    ),
    "backward_packet_length_mean": (
        "bwdPacketLengthMean",
        "backward_packet_length_mean",
        "Bwd Packet Length Mean",
    ),
    "initial_window_forward_bytes": (
        "initWinFwd",
        "initial_window_forward_bytes",
        "Init_Win_bytes_forward",
    ),
    "initial_window_backward_bytes": (
        "initWinBwd",
        "initial_window_backward_bytes",
        "Init_Win_bytes_backward",
    ),
    "down_up_ratio": ("downUpRatio", "down_up_ratio", "Down/Up Ratio"),
}


def _finite_nonnegative(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed < 0:
        return None
    return parsed if parsed <= 3.4028235e38 else 3.4028235e38


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


def extract_flow_vector(
    event: NormalizedEvent,
    *,
    minimum_present: int = 8,
) -> np.ndarray | None:
    """Return the canonical flow vector or ``None`` for non-flow/incomplete events."""
    values: list[float] = []
    present = 0
    for feature_name in FLOW_FEATURE_NAMES:
        parsed = None
        for alias in _FIELD_ALIASES[feature_name]:
            if alias in event.fields:
                parsed = _finite_nonnegative(event.fields[alias])
                if parsed is not None:
                    present += 1
                break
        values.append(parsed if parsed is not None else 0.0)
    if present < minimum_present:
        return None
    return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True)
class FlowWindowSnapshot:
    """Bounded aggregate for one destination/protocol key."""

    flow_count: int
    packet_count: float
    byte_count: float
    unique_sources: int
    syn_ack_ratio: float
    window_seconds: float
    risk_score: float


@dataclass(frozen=True)
class _WindowEntry:
    observed_at: float
    packets: float
    byte_count: float
    syn_count: float
    ack_count: float
    source: str


class FlowWindowAggregator:
    """Thread-safe, bounded ingest-time flow aggregation for DDoS risk signals.

    State is intentionally local to one process. Distributed deployments must use
    a shared stream/state store before enabling enforcement from this signal.
    """

    def __init__(
        self,
        *,
        window_seconds: float = 10.0,
        max_keys: int = 10_000,
        max_events_per_key: int = 5_000,
        flow_threshold: float = 200.0,
        packet_threshold: float = 20_000.0,
        byte_threshold: float = 10_000_000.0,
        source_threshold: float = 50.0,
    ) -> None:
        self.window_seconds = _positive_float(window_seconds, "window_seconds")
        self.max_keys = _positive_int(max_keys, "max_keys")
        self.max_events_per_key = _positive_int(max_events_per_key, "max_events_per_key")
        self.flow_threshold = _positive_float(flow_threshold, "flow_threshold")
        self.packet_threshold = _positive_float(packet_threshold, "packet_threshold")
        self.byte_threshold = _positive_float(byte_threshold, "byte_threshold")
        self.source_threshold = _positive_float(source_threshold, "source_threshold")
        self._windows: OrderedDict[str, deque[_WindowEntry]] = OrderedDict()
        self._lock = threading.RLock()

    @staticmethod
    def _field(event: NormalizedEvent, *names: str, default: str = "unknown") -> str:
        for name in names:
            value = event.fields.get(name)
            if value not in (None, ""):
                return str(value)
        return default

    def _key(self, event: NormalizedEvent) -> str | None:
        destination = self._field(event, "dst", "destinationAddress", "dst_ip")
        port = self._field(event, "dpt", "destination_port")
        protocol = self._field(event, "proto", "protocol")
        if "unknown" in (destination, port, protocol):
            return None
        return f"{destination}|{port}|{protocol}"[:512]

    @staticmethod
    def _numeric(event: NormalizedEvent, *names: str) -> float:
        for name in names:
            if name in event.fields:
                parsed = _finite_nonnegative(event.fields[name])
                return parsed if parsed is not None else 0.0
        return 0.0

    def update(self, event: NormalizedEvent, *, now: float | None = None) -> FlowWindowSnapshot:
        try:
            observed_at = time.monotonic() if now is None else float(now)
        except (TypeError, ValueError) as exc:
            raise ValueError("Flow observation time must be numeric") from exc
        if not math.isfinite(observed_at):
            raise ValueError("Flow observation time must be finite")
        entry = _WindowEntry(
            observed_at=observed_at,
            packets=self._numeric(event, "fwdPackets", "forward_packets")
            + self._numeric(event, "bwdPackets", "backward_packets"),
            byte_count=self._numeric(event, "fwdBytes", "forward_bytes")
            + self._numeric(event, "bwdBytes", "backward_bytes"),
            syn_count=self._numeric(event, "synCount", "syn_count"),
            ack_count=self._numeric(event, "ackCount", "ack_count"),
            source=self._field(event, "src", "sourceAddress", "src_ip"),
        )
        key = self._key(event)
        if key is None:
            return FlowWindowSnapshot(
                flow_count=0,
                packet_count=0.0,
                byte_count=0.0,
                unique_sources=0,
                syn_ack_ratio=0.0,
                window_seconds=self.window_seconds,
                risk_score=0.0,
            )
        with self._lock:
            window = self._windows.pop(key, deque())
            window.append(entry)
            # Updates can arrive out of order across request threads. Sort before
            # expiry and use the latest observation as the event-time watermark.
            window = deque(sorted(window, key=lambda item: item.observed_at))
            cutoff = window[-1].observed_at - self.window_seconds
            while window and window[0].observed_at < cutoff:
                window.popleft()
            while len(window) > self.max_events_per_key:
                window.popleft()
            self._windows[key] = window
            while len(self._windows) > self.max_keys:
                self._windows.popitem(last=False)

            packets = sum(item.packets for item in window)
            byte_count = sum(item.byte_count for item in window)
            sources = {item.source for item in window if item.source != "unknown"}
            syn_count = sum(item.syn_count for item in window)
            ack_count = sum(item.ack_count for item in window)
            ratios = sorted(
                (
                    min(len(window) / self.flow_threshold, 1.0),
                    min(packets / self.packet_threshold, 1.0),
                    min(byte_count / self.byte_threshold, 1.0),
                    min(len(sources) / self.source_threshold, 1.0),
                ),
                reverse=True,
            )
            # Require corroboration from two independent volume/cardinality signals.
            risk_score = (ratios[0] + ratios[1]) / 2.0
            return FlowWindowSnapshot(
                flow_count=len(window),
                packet_count=packets,
                byte_count=byte_count,
                unique_sources=len(sources),
                syn_ack_ratio=syn_count / max(ack_count, 1.0),
                window_seconds=self.window_seconds,
                risk_score=risk_score,
            )

    @property
    def key_count(self) -> int:
        with self._lock:
            return len(self._windows)

    def clear(self) -> None:
        with self._lock:
            self._windows.clear()
