"""Deterministic, label-free endpoint telemetry feature extraction."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np

from logfilter.pipeline.normalizer import NormalizedEvent

_HASH_BINS = 128
_NUMERIC_FIELDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("process_id", ("process_id", "processId", "ProcessId")),
    ("thread_id", ("thread_id", "threadId", "ThreadId")),
    ("parent_process_id", ("parent_process_id", "parentProcessId", "ParentProcessId")),
    ("user_id", ("user_id", "userId", "UserId")),
    ("mount_namespace", ("mount_namespace", "mountNamespace")),
    ("event_id", ("event_id", "eventId", "EventID")),
    ("args_count", ("args_count", "argsNum")),
    ("return_value", ("return_value", "returnValue")),
)
ENDPOINT_FEATURE_NAMES: tuple[str, ...] = tuple(name for name, _ in _NUMERIC_FIELDS) + tuple(
    f"token_hash_{index:03d}" for index in range(_HASH_BINS)
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./:-]{2,128}")


def _number(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(parsed):
        return 0.0
    return max(-3.4028235e38, min(parsed, 3.4028235e38))


def _first(mapping: Mapping[str, Any], aliases: tuple[str, ...]) -> Any:
    for alias in aliases:
        if alias in mapping:
            return mapping[alias]
    return 0


def _token_bucket(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8", errors="replace"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % _HASH_BINS


def endpoint_vector_from_mapping(mapping: Mapping[str, Any]) -> np.ndarray:
    """Build a stable endpoint vector without host, timestamp, or label fields."""
    vector = np.zeros(len(ENDPOINT_FEATURE_NAMES), dtype=np.float32)
    for index, (_, aliases) in enumerate(_NUMERIC_FIELDS):
        vector[index] = _number(_first(mapping, aliases))

    text_values = (
        mapping.get("process") or mapping.get("processName") or mapping.get("ProcessName") or "",
        mapping.get("parent_process") or mapping.get("ParentProcessName") or "",
        mapping.get("event_type") or mapping.get("eventName") or mapping.get("EventType") or "",
        mapping.get("cmdline") or mapping.get("CommandLine") or mapping.get("args") or "",
    )
    offset = len(_NUMERIC_FIELDS)
    for token in _TOKEN_RE.findall(" ".join(str(value) for value in text_values).lower()):
        vector[offset + _token_bucket(token)] += 1.0
    return vector


def extract_endpoint_vector(event: NormalizedEvent) -> np.ndarray | None:
    if event.source_type.value != "endpoint":
        return None
    return endpoint_vector_from_mapping(event.fields)
