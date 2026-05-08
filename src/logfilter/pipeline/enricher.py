"""
LEEF 2.0 enricher.

Takes a ScoredEvent and produces a LEEF 2.0 formatted string with AI-generated
custom properties embedded.  The enriched event preserves the original log via
a forensic reference field pointing to the Elasticsearch document ID.

LEEF 2.0 format:
  LEEF:2.0|Vendor|Product|Version|EventID|delimiter|key=value\tkey=value...

QRadar-compatible custom properties added:
  ai_threat_score      — 0.0–1.0 composite threat score
  ai_priority          — HIGH / MEDIUM / LOW / INFO
  ai_mitre_technique   — Top ATT&CK technique ID (e.g. T1021.002)
  ai_entities          — Comma-separated IOCs/malware/CVEs extracted by NER
  ai_ner_confidence    — NER model confidence
  ai_dedup_flag        — true/false (was this a near-duplicate event?)
  ai_sigma_match       — true/false (matched a Sigma detection rule?)
  ai_sigma_rules       — comma-separated matched rule IDs
  raw_log_ref          — Elasticsearch document ID for forensic chain-of-custody
"""

from __future__ import annotations

import base64
import re
from datetime import datetime, timezone

import structlog

from logfilter.pipeline.scorer import ScoredEvent

logger = structlog.get_logger(__name__)

# LEEF 2.0 allows a custom delimiter character declared in the header.
# We use ^ (caret) as delimiter inside the attribute section.
_LEEF_DELIMITER = "^"
_LEEF_ATTR_SEP = "\t"  # tab separates key=value pairs

# Fields that may contain the ^ delimiter — sanitise before embedding
_SANITISE_PATTERN = re.compile(r"[|\^\t\r\n]")


def _sanitise(value: str) -> str:
    """Remove LEEF-unsafe characters from field values."""
    return _SANITISE_PATTERN.sub(" ", str(value))


class LEEFEnricher:
    """
    Produces LEEF 2.0 strings from ScoredEvent objects.

    Parameters
    ----------
    vendor : str
        Your organisation name (e.g. "YourCo")
    product : str
        Product name for the AI layer (e.g. "AIPreprocessor")
    version : str
        Product version string
    """

    def __init__(
        self,
        vendor: str = "YourCo",
        product: str = "AIPreprocessor",
        version: str = "1.0",
    ) -> None:
        self.vendor = vendor
        self.product = product
        self.version = version

    def enrich(self, scored: ScoredEvent, es_doc_id: str = "") -> str:
        """
        Build a LEEF 2.0 string from a scored event.

        Parameters
        ----------
        scored : ScoredEvent
            Output of LogScorer.score() or score_batch().
        es_doc_id : str
            Elasticsearch document ID of the archived raw log. If empty,
            the raw log is base64-encoded inline as raw_log_b64.

        Returns
        -------
        str
            Ready-to-send LEEF 2.0 formatted string (single line, tab-delimited attrs).
        """
        # ── Determine source event ID ─────────────────────────────────────────
        # Use the original EventID if present in parsed fields, else generate one
        event_id = str(
            scored.fields.get("event_id", "")
            or scored.fields.get("EventID", "")
            or scored.fields.get("signature_id", "")
            or "LOG_EVENT"
        )

        # ── LEEF header ───────────────────────────────────────────────────────
        header = (
            f"LEEF:2.0|{_sanitise(self.vendor)}|{_sanitise(self.product)}"
            f"|{_sanitise(self.version)}|{_sanitise(event_id)}|{_LEEF_DELIMITER}|"
        )

        # ── Standard LEEF attributes ──────────────────────────────────────────
        attrs: dict[str, str] = {}

        # Map known fields to standard LEEF keys
        if "src_ip" in scored.fields:
            attrs["src"] = _sanitise(scored.fields["src_ip"])
        if scored.host and scored.host != "unknown":
            attrs["devTime"] = _sanitise(scored.timestamp or _utcnow())
            attrs["devTimeFormat"] = "MMM dd yyyy HH:mm:ss"
        if "dst" in scored.fields or "dst_ip" in scored.fields:
            attrs["dst"] = _sanitise(scored.fields.get("dst", scored.fields.get("dst_ip", "")))
        if "user" in scored.fields:
            attrs["usrName"] = _sanitise(scored.fields["user"])
        if "src_port" in scored.fields or "spt" in scored.fields:
            attrs["srcPort"] = _sanitise(
                scored.fields.get("spt", scored.fields.get("src_port", ""))
            )
        if "dst_port" in scored.fields or "dpt" in scored.fields:
            attrs["dstPort"] = _sanitise(
                scored.fields.get("dpt", scored.fields.get("dst_port", ""))
            )
        if "proto" in scored.fields or "protocol" in scored.fields:
            attrs["proto"] = _sanitise(
                scored.fields.get("proto", scored.fields.get("protocol", ""))
            )

        # ── AI custom properties ──────────────────────────────────────────────
        attrs["ai_threat_score"] = f"{scored.ai_threat_score:.4f}"
        attrs["ai_priority"] = scored.ai_priority
        attrs["ai_mitre_technique"] = _sanitise(scored.ai_mitre_technique)
        attrs["ai_entities"] = _sanitise(scored.ai_entities)
        attrs["ai_ner_confidence"] = (
            f"{scored.entities.get('confidence', 0.0):.4f}" if scored.entities else "0.0000"
        )
        attrs["ai_dedup_flag"] = "true" if scored.is_duplicate else "false"
        attrs["ai_sigma_match"] = "true" if scored.sigma_matched else "false"
        attrs["ai_sigma_rules"] = _sanitise(",".join(scored.sigma_rule_ids))
        attrs["ai_source_type"] = _sanitise(scored.source_type)
        attrs["ai_scoring_latency_ms"] = f"{scored.scoring_latency_ms:.1f}"

        # ── Forensic chain-of-custody ──────────────────────────────────────────
        if es_doc_id:
            attrs["raw_log_ref"] = _sanitise(es_doc_id)
        else:
            # Embed raw log as base64 when no ES reference is available
            raw_b64 = base64.b64encode(scored.raw.encode("utf-8", errors="replace")).decode()
            attrs["raw_log_b64"] = raw_b64[:4096]  # cap at 4KB

        # ── Assemble final LEEF string ─────────────────────────────────────────
        attr_string = _LEEF_ATTR_SEP.join(f"{k}={v}" for k, v in attrs.items() if v)
        return header + attr_string

    def enrich_batch(
        self, scored_events: list[ScoredEvent], es_doc_ids: list[str] | None = None
    ) -> list[str]:
        """Enrich a batch of events. es_doc_ids must match scored_events length."""
        if es_doc_ids is None:
            es_doc_ids = [""] * len(scored_events)
        return [self.enrich(ev, doc_id) for ev, doc_id in zip(scored_events, es_doc_ids)]


def _utcnow() -> str:
    return datetime.now(tz=timezone.utc).isoformat()
