"""
Log normalizer — converts heterogeneous log formats into a canonical JSON event dict
and a human-readable text representation suitable for the SecureBERT2.0 models.

Supported input formats:
  - syslog     : RFC 3164 / RFC 5424 text
  - winevent   : Windows Event Log JSON (from NXLog, WEF, or similar)
  - firewall   : CEF or vendor syslog (Cisco ASA, Palo Alto, Suricata)
  - endpoint   : CrowdStrike / Carbon Black / Sysmon JSON
  - cloudtrail : AWS CloudTrail JSON
  - web        : Apache Combined Log Format / Nginx
  - generic    : fallback — use raw message as text
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ── RFC 3164 syslog regex ─────────────────────────────────────────────────────
_SYSLOG_3164 = re.compile(
    r"^(?:<(?P<priority>\d{1,3})>)?"
    r"(?P<month>[A-Za-z]{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+"
    r"(?:(?P<process>\S+?)(?:\[(?P<pid>\d+)\])?:\s*(?P<message>.+)"  # with process:message
    r"|(?P<message_only>.+))$",  # or bare message
    re.DOTALL,
)

# ── RFC 5424 syslog regex ─────────────────────────────────────────────────────
_SYSLOG_5424 = re.compile(
    r"^<(?P<priority>\d{1,3})>(?P<version>\d)\s+"
    r"(?P<timestamp>\S+)\s+(?P<host>\S+)\s+(?P<app>\S+)\s+"
    r"(?P<procid>\S+)\s+(?P<msgid>\S+)\s+"
    r"(?P<structured_data>\[.*?\]|-)\s*(?P<message>.+)?$",
    re.DOTALL,
)

# ── CEF header regex ──────────────────────────────────────────────────────────
_CEF_HEADER = re.compile(
    r"^(?:<\d+>)?(?:\S+ \d+ [\d:]+ \S+ )?"
    r"CEF:(?P<cef_version>\d+)\|"
    r"(?P<device_vendor>[^|]*)\|(?P<device_product>[^|]*)\|"
    r"(?P<device_version>[^|]*)\|(?P<signature_id>[^|]*)\|"
    r"(?P<name>[^|]*)\|(?P<severity>[^|]*)\|"
    r"(?P<extensions>.*)$",
    re.DOTALL,
)

# ── Apache/Nginx combined log ─────────────────────────────────────────────────
_APACHE_COMBINED = re.compile(
    r"^(?P<src_ip>\S+)\s+-\s+(?P<user>\S+)\s+"
    r"\[(?P<timestamp>[^\]]+)\]\s+"
    r'"(?P<method>\S+)\s+(?P<uri>\S+)\s+(?P<protocol>[^"]+)"\s+'
    r"(?P<status>\d{3})\s+(?P<bytes>\S+)"
    r'(?:\s+"(?P<referrer>[^"]*)"\s+"(?P<ua>[^"]*)")?'
)


class LogSourceType(str, Enum):
    SYSLOG = "syslog"
    WINEVENT = "winevent"
    FIREWALL = "firewall"
    ENDPOINT = "endpoint"
    CLOUDTRAIL = "cloudtrail"
    WEB = "web"
    GENERIC = "generic"


@dataclass
class NormalizedEvent:
    """
    Canonical event representation produced by the normalizer.

    Attributes
    ----------
    source_type : LogSourceType
    timestamp   : ISO-8601 UTC timestamp (or empty string if unparseable)
    host        : Source hostname / IP
    text        : Human-readable text fed to SecureBERT2.0 models
    raw         : Original raw log string (preserved for forensic replay)
    fields      : Parsed key-value pairs (for LEEF enrichment and ES archiving)
    """

    source_type: LogSourceType
    timestamp: str
    host: str
    text: str
    raw: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "timestamp": self.timestamp,
            "host": self.host,
            "text": self.text,
            "raw": self.raw,
            "fields": self.fields,
        }


# ── CEF extension parser ──────────────────────────────────────────────────────
def _parse_cef_extensions(ext_str: str) -> dict[str, str]:
    """Parse CEF key=value extensions, handling escaped spaces in values."""
    result: dict[str, str] = {}
    parts = re.split(r"(?<!\\)\s+(?=\w+=)", ext_str.strip())
    for part in parts:
        if "=" in part:
            k, _, v = part.partition("=")
            result[k.strip()] = v.replace(r"\=", "=").replace(r"\ ", " ").strip()
    return result


# ── Normalizer class ──────────────────────────────────────────────────────────
class LogNormalizer:
    """
    Normalize a raw log string into a NormalizedEvent.

    Parameters
    ----------
    source_type_hint : LogSourceType | None
        If the ingestion pipeline already knows the source type (e.g., from the
        Kafka topic or log source configuration), pass it here to skip detection.
    """

    def __init__(self, source_type_hint: LogSourceType | None = None) -> None:
        self._hint = source_type_hint

    # ── public API ─────────────────────────────────────────────────────────────
    def normalize(self, raw: str, source_type_hint: LogSourceType | None = None) -> NormalizedEvent:
        """Parse *raw* log string and return a NormalizedEvent."""
        raw = raw.strip()
        hint = source_type_hint or self._hint

        # Try each parser in order of specificity
        parsers = [
            (LogSourceType.WINEVENT, self._try_winevent),
            (LogSourceType.CLOUDTRAIL, self._try_cloudtrail),
            (LogSourceType.ENDPOINT, self._try_endpoint),
            (LogSourceType.FIREWALL, self._try_cef),
            (LogSourceType.WEB, self._try_apache),
            (LogSourceType.SYSLOG, self._try_syslog_5424),
            (LogSourceType.SYSLOG, self._try_syslog_3164),
        ]

        if hint:
            # Honour hint first, then fall through to general detection
            hint_parsers = [(t, p) for t, p in parsers if t == hint]
            remaining = [(t, p) for t, p in parsers if t != hint]
            ordered = hint_parsers + remaining
        else:
            ordered = parsers

        for src_type, parser in ordered:
            result = parser(raw)
            if result is not None:
                result.source_type = src_type
                return result

        return self._generic(raw)

    # ── syslog ─────────────────────────────────────────────────────────────────
    def _try_syslog_5424(self, raw: str) -> NormalizedEvent | None:
        m = _SYSLOG_5424.match(raw)
        if not m:
            return None
        d = m.groupdict()
        host = d.get("host") or "unknown"
        app = d.get("app") or ""
        msg = (d.get("message") or "").strip()
        text = f"Host {host} App {app}: {msg}"
        return NormalizedEvent(
            source_type=LogSourceType.SYSLOG,
            timestamp=d.get("timestamp", ""),
            host=host,
            text=text,
            raw=raw,
            fields={k: v for k, v in d.items() if v and v != "-"},
        )

    def _try_syslog_3164(self, raw: str) -> NormalizedEvent | None:
        m = _SYSLOG_3164.match(raw)
        if not m:
            return None
        d = m.groupdict()
        host = d.get("host") or "unknown"
        process = d.get("process") or ""
        pid = d.get("pid") or ""
        # Support bare message (no process:pid: prefix) via message_only group
        msg = (d.get("message") or d.get("message_only") or "").strip()
        ts = f"{d.get('month', '')} {d.get('day', '')} {d.get('time', '')}"
        if process:
            text = f"Host {host} Process {process}[{pid}]: {msg}"
        else:
            text = f"Host {host}: {msg}"
        return NormalizedEvent(
            source_type=LogSourceType.SYSLOG,
            timestamp=ts,
            host=host,
            text=text,
            raw=raw,
            fields={k: v for k, v in d.items() if v and k != "message_only"},
        )

    # ── Windows Event ──────────────────────────────────────────────────────────
    def _try_winevent(self, raw: str) -> NormalizedEvent | None:
        """Expect JSON with EventID or System.EventID key."""
        if not raw.startswith("{"):
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        # Support NXLog schema and direct WEF schema
        event_id = (
            data.get("EventID") or data.get("System", {}).get("EventID") or data.get("event_id", "")
        )
        if not event_id:
            return None

        host = (
            data.get("Computer")
            or data.get("System", {}).get("Computer")
            or data.get("hostname", "unknown")
        )
        message = data.get("Message") or data.get("message") or json.dumps(data)
        user = data.get("SubjectUserName") or data.get("TargetUserName") or data.get("user", "")
        timestamp = (
            data.get("TimeCreated")
            or data.get("System", {}).get("TimeCreated", {}).get("@SystemTime", "")
            or data.get("@timestamp", "")
        )

        text = f"Windows EventID {event_id}: {message} on host {host} by user {user}"
        return NormalizedEvent(
            source_type=LogSourceType.WINEVENT,
            timestamp=str(timestamp),
            host=str(host),
            text=text.strip(),
            raw=raw,
            fields={"event_id": str(event_id), "host": str(host), "user": str(user)},
        )

    # ── CEF (firewall / IDS) ───────────────────────────────────────────────────
    def _try_cef(self, raw: str) -> NormalizedEvent | None:
        m = _CEF_HEADER.match(raw)
        if not m:
            return None
        d = m.groupdict()
        ext = _parse_cef_extensions(d.get("extensions", ""))
        src = ext.get("src", ext.get("sourceAddress", "unknown"))
        dst = ext.get("dst", ext.get("destinationAddress", "unknown"))
        src_port = ext.get("spt", ext.get("sourcePort", ""))
        dst_port = ext.get("dpt", ext.get("destinationPort", ""))
        act = ext.get("act", ext.get("deviceAction", ""))
        proto = ext.get("proto", ext.get("protocol", ""))
        host = ext.get("dvchost", ext.get("dvc", "unknown"))
        ts = ext.get("rt", ext.get("end", ""))

        text = (
            f"Firewall {d['device_product']} Action {act} "
            f"src {src}:{src_port} dst {dst}:{dst_port} "
            f"proto {proto} sig {d['signature_id']} name {d['name']} "
            f"sev {d['severity']}"
        )
        fields = {**d, **ext}
        return NormalizedEvent(
            source_type=LogSourceType.FIREWALL,
            timestamp=ts,
            host=host,
            text=text,
            raw=raw,
            fields=fields,
        )

    # ── Endpoint (CrowdStrike / CB / Sysmon JSON) ──────────────────────────────
    def _try_endpoint(self, raw: str) -> NormalizedEvent | None:
        if not raw.startswith("{"):
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        # Require at least one endpoint-specific key
        endpoint_keys = {
            "ProcessName",
            "process_name",
            "ParentProcessName",
            "ImageFileName",
            "CommandLine",
            "command_line",
            "event_type",
            "Sysmon",
        }
        if not endpoint_keys.intersection(data.keys()):
            return None

        host = data.get("ComputerName") or data.get("hostname") or data.get("host", "unknown")
        process = (
            data.get("ProcessName") or data.get("process_name") or data.get("ImageFileName", "")
        )
        parent = data.get("ParentProcessName") or data.get("parent_process_name", "")
        cmdline = data.get("CommandLine") or data.get("command_line", "")
        user = data.get("UserName") or data.get("user", "")
        event_type = data.get("event_type") or data.get("EventType", "")
        ts = data.get("@timestamp") or data.get("timestamp", "")

        text = (
            f"Endpoint {event_type}: Process {process} spawned by {parent} "
            f"executed '{cmdline}' as user {user} on {host}"
        )
        return NormalizedEvent(
            source_type=LogSourceType.ENDPOINT,
            timestamp=str(ts),
            host=str(host),
            text=text.strip(),
            raw=raw,
            fields={
                "process": process,
                "parent_process": parent,
                "cmdline": cmdline,
                "user": user,
                "event_type": event_type,
            },
        )

    # ── AWS CloudTrail ─────────────────────────────────────────────────────────
    def _try_cloudtrail(self, raw: str) -> NormalizedEvent | None:
        if not raw.startswith("{"):
            return None
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None

        # CloudTrail events have eventSource + eventName
        if "eventSource" not in data or "eventName" not in data:
            return None

        user_identity = data.get("userIdentity", {})
        user = user_identity.get("userName") or user_identity.get("arn", "unknown")
        event_name = data.get("eventName", "")
        event_source = data.get("eventSource", "")
        resource_type = ""
        resources = data.get("resources", [])
        if resources and isinstance(resources, list):
            resource_type = resources[0].get("type", "")
        src_ip = data.get("sourceIPAddress", "unknown")
        ts = data.get("eventTime", "")
        region = data.get("awsRegion", "")
        error = data.get("errorCode", "")

        text = (
            f"CloudTrail: User {user} performed {event_name} on {event_source} "
            f"resource {resource_type} from {src_ip} region {region}"
        )
        if error:
            text += f" error {error}"

        return NormalizedEvent(
            source_type=LogSourceType.CLOUDTRAIL,
            timestamp=ts,
            host=src_ip,
            text=text,
            raw=raw,
            fields={
                "event_name": event_name,
                "event_source": event_source,
                "user": user,
                "src_ip": src_ip,
                "region": region,
                "error_code": error,
            },
        )

    # ── Apache / Nginx web logs ────────────────────────────────────────────────
    def _try_apache(self, raw: str) -> NormalizedEvent | None:
        m = _APACHE_COMBINED.match(raw)
        if not m:
            return None
        d = m.groupdict()
        src_ip = d.get("src_ip", "unknown")
        method = d.get("method", "")
        uri = d.get("uri", "")
        status = d.get("status", "")
        ua = d.get("ua", "")
        ts = d.get("timestamp", "")

        text = f"Web request {method} {uri} from {src_ip} status {status} user-agent '{ua}'"
        return NormalizedEvent(
            source_type=LogSourceType.WEB,
            timestamp=ts,
            host=src_ip,
            text=text,
            raw=raw,
            fields=d,
        )

    # ── Generic fallback ───────────────────────────────────────────────────────
    def _generic(self, raw: str) -> NormalizedEvent:
        return NormalizedEvent(
            source_type=LogSourceType.GENERIC,
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            host="unknown",
            text=raw,
            raw=raw,
            fields={},
        )


# ── Module-level singleton ────────────────────────────────────────────────────
_default_normalizer = LogNormalizer()


def normalize(raw: str, source_type_hint: LogSourceType | None = None) -> NormalizedEvent:
    """Convenience wrapper around the default LogNormalizer."""
    return _default_normalizer.normalize(raw, source_type_hint)
