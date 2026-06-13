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

import re
from datetime import datetime, timezone

import structlog

from logfilter.pipeline.events import ScoredEvent

logger = structlog.get_logger(__name__)

# LEEF 2.0 allows a custom delimiter character declared in the header.
# B19: tab is the QRadar-default attribute delimiter; declare the same char in
# the header (after the EventID) so consumers know the split. We previously
# declared "^" as header delimiter while using "\t" as attr separator — that
# ambiguity broke parsers. Now BOTH use "\t".
_LEEF_DELIMITER = "\t"  # declared in the LEEF header
_LEEF_ATTR_SEP = "\t"  # tab separates key=value pairs

# Sanitise LEEF-unsafe characters from values: | ends the header, tab is the
# delimiter/separator (B19), and \r\n would break single-line framing.
_SANITISE_PATTERN = re.compile(r"[|\t\r\n]")


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

    def enrich(self, scored: ScoredEvent, es_doc_id: str) -> str:
        """
        Build a LEEF 2.0 string from a scored event.

        Parameters
        ----------
        scored : ScoredEvent
            Output of LogScorer.score() or score_batch().
        es_doc_id : str
            Elasticsearch document ID of the archived raw log. **Required**:
            the chain-of-custody reference. Callers must archive the raw event
            FIRST (via ``LogArchive.write_with_id``) and pass the same ID.

        Returns
        -------
        str
            Ready-to-send LEEF 2.0 formatted string (single line, tab-delimited attrs).
        """
        if not es_doc_id:
            raise ValueError(
                "es_doc_id is required for chain-of-custody. Archive the raw event "
                "first via LogArchive.write_with_id() and pass the returned ref."
            )

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
            # B15: emit epoch milliseconds (13-digit) which QRadar accepts as
            # an unambiguous timestamp; we no longer declare a devTimeFormat
            # because the value is already a numeric epoch.
            attrs["devTime"] = _epoch_millis(scored.timestamp)
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
        attrs["ai_novelty_score"] = f"{scored.novelty_score:.4f}"
        attrs["degraded"] = "1" if scored.score_degraded else "0"

        attrs["raw_log_ref"] = _sanitise(es_doc_id)

        # ── Assemble final LEEF string ─────────────────────────────────────────
        attr_string = _LEEF_ATTR_SEP.join(f"{k}={v}" for k, v in attrs.items() if v)
        return header + attr_string

    def enrich_batch(
        self, scored_events: list[ScoredEvent], es_doc_ids: list[str]
    ) -> list[str]:
        """Enrich a batch of events. es_doc_ids must match scored_events length (B8)."""
        if len(es_doc_ids) != len(scored_events):
            raise ValueError(
                f"enrich_batch: es_doc_ids length {len(es_doc_ids)} != "
                f"scored_events length {len(scored_events)}"
            )
        return [self.enrich(ev, doc_id) for ev, doc_id in zip(scored_events, es_doc_ids)]


def _epoch_millis(timestamp: str | None) -> str:
    """Convert an ISO8601 timestamp to epoch milliseconds (13-digit string)."""
    if not timestamp:
        return str(int(datetime.now(tz=timezone.utc).timestamp() * 1000))
    try:
        # ``fromisoformat`` (3.10) handles "+00:00" but not trailing "Z".
        normalised = timestamp.replace("Z", "+00:00")
        dt = datetime.fromisoformat(normalised)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return str(int(dt.timestamp() * 1000))
    except (ValueError, TypeError):
        return str(int(datetime.now(tz=timezone.utc).timestamp() * 1000))
