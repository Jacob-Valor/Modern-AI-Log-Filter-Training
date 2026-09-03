"""
CIC-IDS2017/2018 CSV → syslog/CEF converter.

Reads the labelled network-flow CSV from the Canadian Institute for Cybersecurity
datasets and converts each attack row into a CEF syslog message that the
LogNormalizer's ``_try_cef`` parser can handle.

Attack types covered
--------------------
- DDoS (LOIC / HOIC / HTTP flood)
- DoS (Hulk, GoldenEye, Slowloris, SlowHTTPTest)
- Botnet (Zeus, Ares)
- Brute-force (SSH, FTP, web)
- Heartbleed
- Web attacks (SQLi, XSS, brute-force)
- Infiltration
- Port scan

Usage
-----
    from logfilter.demo.cic_converter import CicCsvToSyslog

    converter = CicCsvToSyslog()
    syslog_lines = converter.csv_to_syslog("path/to/cic_csv.csv", max_rows=500)
"""

from __future__ import annotations

import csv
import logging
import random
import socket
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

logger = logging.getLogger(__name__)

# ── Attack → CEF mapping ───────────────────────────────────────────────────────

# Severity mapping (CEF severity 0-10)
_ATTACK_SEVERITY: dict[str, int] = {
    "DDOS": 10,
    "DOS": 8,
    "BOTNET": 9,
    "HEARTBLEED": 10,
    "WEB ATTACK": 8,
    "BRUTE FORCE": 7,
    "INFILTRATION": 9,
    "PORT SCAN": 5,
    "BENIGN": 1,
}

# MITRE ATT&CK technique mapping for each attack type
_ATTACK_MITRE: dict[str, str] = {
    "DDOS": "T1498",
    "DOS": "T1499",
    "BOTNET": "T1583.001",
    "HEARTBLEED": "T1190",
    "WEB ATTACK": "T1190",
    "BRUTE FORCE": "T1110",
    "INFILTRATION": "T1078",
    "PORT SCAN": "T1046",
    "BENIGN": "NONE",
}

# Attack families in the CIC-IDS2017 dataset (Label column values)
_ATTACK_FAMILIES = {
    "ddos": "DDOS",
    "dos": "DOS",
    "hulk": "DOS",
    "goldeneye": "DOS",
    "slowloris": "DOS",
    "slowhttptest": "DOS",
    "heartbleed": "HEARTBLEED",
    "webattack": "WEB ATTACK",
    "web attack": "WEB ATTACK",
    "brute force": "BRUTE FORCE",
    "ssh-patator": "BRUTE FORCE",
    "ftp-patator": "BRUTE FORCE",
    "botnet": "BOTNET",
    "infiltration": "INFILTRATION",
    "portscan": "PORT SCAN",
    "port_scan": "PORT SCAN",
    "benign": "BENIGN",
}


def classify_attack(label: str) -> tuple[str, int, str]:
    """Normalize a CIC label into (attack_category, severity, mitre_technique)."""
    cleaned = label.strip().lower().replace(" ", "").replace("-", "").replace("_", "")
    for key, category in _ATTACK_FAMILIES.items():
        k = key.lower().replace(" ", "").replace("-", "").replace("_", "")
        if k in cleaned or cleaned in k:
            sev = _ATTACK_SEVERITY.get(category, 5)
            mitre = _ATTACK_MITRE.get(category, "T1068")
            return category, sev, mitre
    return "BENIGN", 1, "NONE"


# ── CIC CSV column reader ──────────────────────────────────────────────────────

# Columns we care about from the CIC-2017 CSV
_REQUIRED_COLS = {
    " Destination Port",
    " Protocol",
    " Label",
    " Flow Duration",
    " Total Fwd Packets",
    " Total Backward Packets",
    "Fwd Packet Length Mean",
    " Bwd Packet Length Mean",
    " Flow Bytes/s",
    " Flow Packets/s",
}

_IP_COLS = {
    " Source IP": "1.1.1.1",
    " Destination IP": "2.2.2.2",
}

# Alternate column names (CIC-2018 format without leading spaces)
_REQUIRED_COLS_ALT = {c.strip() for c in _REQUIRED_COLS}
_IP_COLS_ALT = {c.strip(): v for c, v in _IP_COLS.items()}


@dataclass
class CicFlowRecord:
    """A single parsed flow from the CIC CSV."""

    timestamp: str = ""
    src_ip: str = ""
    dst_ip: str = ""
    src_port: int = 0
    dst_port: int = 0
    protocol: str = ""
    label: str = ""
    flow_duration: float = 0.0
    fwd_packets: int = 0
    bwd_packets: int = 0
    fwd_bytes_mean: float = 0.0
    bwd_bytes_mean: float = 0.0
    flow_bytes_sec: float = 0.0
    flow_pkts_sec: float = 0.0
    attack_category: str = "BENIGN"
    severity: int = 1
    mitre_technique: str = "NONE"
    extra: dict[str, Any] = field(default_factory=dict)


def parse_cic_row(row: dict[str, str]) -> CicFlowRecord | None:
    """Parse a single CIC CSV row dict into a CicFlowRecord."""
    # Detect column format (CIC-2017 has leading spaces, CIC-2018 doesn't)
    cols = set(row.keys())

    if " Label" in cols or " Label" in str(list(cols)[:1]):
        # CIC-2017 format (spaces in column names)
        src_key, dst_key = " Source IP", " Destination IP"
        label_key = " Label"
    else:
        # CIC-2018 format or stripped
        src_key, dst_key = "Source", "Destination"
        label_key = "Label"
        # Fallback to simpler detection
        for k in cols:
            kl = k.lower()
            if "label" in kl and "label" not in label_key.lower():
                label_key = k
            if "src" in kl or "source" in kl:
                if "ip" in kl:
                    src_key = k
            if "dst" in kl or "dest" in kl or "destination" in kl:
                if "ip" in kl:
                    dst_key = k

    label = row.get(label_key, "BENIGN").strip()
    if not label:
        return None

    cat, sev, mitre = classify_attack(label)

    try:
        sport = int(row.get(" Source Port", row.get("SourcePort", row.get("src_port", 0))))
    except (ValueError, TypeError):
        sport = 0

    try:
        dport = int(
            row.get(" Destination Port", row.get("DestinationPort", row.get("dst_port", 0)))
        )
    except (ValueError, TypeError):
        dport = 0

    def _float(v: str) -> float:
        try:
            return float(v) if v.strip() else 0.0
        except (ValueError, TypeError):
            return 0.0

    def _int(v: str) -> int:
        try:
            return int(float(v)) if v.strip() else 0
        except (ValueError, TypeError):
            return 0

    ts = row.get(" Timestamp", row.get("Timestamp", row.get("timestamp", "")))
    if ts:
        try:
            # CIC timestamp format: "7/7/2017 8:00" or ISO
            dt = datetime.fromisoformat(ts)
            ts = dt.isoformat()
        except (ValueError, TypeError):
            ts = ts.strip()

    return CicFlowRecord(
        timestamp=ts,
        src_ip=row.get(src_key, row.get("src_ip", "0.0.0.0")).strip(),
        dst_ip=row.get(dst_key, row.get("dst_ip", "0.0.0.0")).strip(),
        src_port=sport,
        dst_port=dport,
        protocol=str(_int(row.get(" Protocol", row.get("Protocol", "0")))),
        label=label,
        flow_duration=_float(row.get(" Flow Duration", row.get("FlowDuration", "0"))),
        fwd_packets=_int(row.get(" Total Fwd Packets", row.get("TotalFwdPackets", "0"))),
        bwd_packets=_int(row.get(" Total Backward Packets", row.get("TotalBackwardPackets", "0"))),
        fwd_bytes_mean=_float(
            row.get("Fwd Packet Length Mean", row.get("FwdPacketLengthMean", "0"))
        ),
        bwd_bytes_mean=_float(
            row.get(" Bwd Packet Length Mean", row.get("BwdPacketLengthMean", "0"))
        ),
        flow_bytes_sec=_float(row.get(" Flow Bytes/s", row.get("FlowBytes/s", "0"))),
        flow_pkts_sec=_float(row.get(" Flow Packets/s", row.get("FlowPackets/s", "0"))),
        attack_category=cat,
        severity=sev,
        mitre_technique=mitre,
        extra={k: v for k, v in row.items() if k not in (label_key, src_key, dst_key)},
    )


# ── CEF syslog string builder ──────────────────────────────────────────────────

_PROTO_MAP = {"0": "ip", "1": "icmp", "6": "tcp", "17": "udp", "47": "gre"}


def to_cef_syslog(record: CicFlowRecord, hostname: str = "cic-collector") -> str:
    """
    Convert a CicFlowRecord to a CEF syslog line parsable by LogNormalizer.

    Output format (RFC 3164 + CEF):
        <priority>MMM DD HH:MM:SS hostname CEF:0|CIC|IDS2017|1.0|{attack_id}|{name}|{sev}|...
    """
    now = datetime.now(tz=timezone.utc)
    timestamp = now.strftime("%b %d %H:%M:%S")

    attack_id = (
        record.mitre_technique
        if record.mitre_technique != "NONE"
        else f"ATK-{record.attack_category[:4]}"
    )
    attack_name = record.label.replace(" ", "_") or record.attack_category

    proto_num = record.protocol or "0"
    proto = _PROTO_MAP.get(proto_num, f"proto_{proto_num}")

    # Build CEF extensions
    ext_parts = [
        f"src={record.src_ip}",
        f"dst={record.dst_ip}",
        f"spt={record.src_port}",
        f"dpt={record.dst_port}",
        f"proto={proto}",
        f"act={record.attack_category.lower()}",
        f"cat={record.label[:60]}" if record.label else "",
        f"cn1={record.fwd_packets}",
        "cn1Label=FwdPkts",
        f"cn2={record.bwd_packets}",
        "cn2Label=BwdPkts",
        f"cf1={record.flow_bytes_sec:.1f}",
        "cf1Label=FlowBytesPerSec",
        f"cf2={record.flow_pkts_sec:.1f}",
        "cf2Label=FlowPktsPerSec",
        f"cs1={record.mitre_technique}",
        "cs1Label=MitreTechnique",
        f"cs2={record.attack_category}",
        "cs2Label=AttackCategory",
        f"flexNumber1={record.severity}",
        "flexNumber1Label=Severity",
        f"deviceExternalId={uuid.uuid4().hex[:8]}",
    ]

    ext_str = "\t".join(ext_parts)

    # RFC 3164 syslog header + CEF
    priority = 134  # facility=16 (local0), severity=6 (informational)
    cef = (
        f"<{priority}>{timestamp} {hostname} "
        f"CEF:0|CIC|IDS2017|1.0|{attack_id}|{attack_name}|{record.severity}|"
        f"{ext_str}"
    )
    return cef


# ── Public API ─────────────────────────────────────────────────────────────────


class CicCsvToSyslog:
    """
    Convert CIC-IDS2017/2018 CSV files into CEF syslog lines.

    Parameters
    ----------
    hostname : str
        Syslog hostname used in generated messages.
    shuffle : bool
        Whether to randomise row order (mixes attack types together).
    """

    def __init__(self, hostname: str = "cic-collector", shuffle: bool = True) -> None:
        self.hostname = hostname
        self.shuffle = shuffle

    def csv_lines(
        self,
        source: str | Path | TextIO,
        max_rows: int | None = None,
        *,
        attack_only: bool = True,
        attack_types: list[str] | None = None,
    ) -> list[str]:
        """
        Read a CIC CSV and return CEF syslog lines.

        Parameters
        ----------
        source : str | Path | TextIO
            CSV file path or open file handle.
        max_rows : int | None
            Maximum number of rows to convert (None = all).
        attack_only : bool
            If True, skip BENIGN rows.
        attack_types : list[str] | None
            If set, only include these attack categories (e.g. ``["DDOS", "BOTNET"]``).

        Returns
        -------
        list[str]
            CEF syslog lines ready to send to the collector.
        """
        if isinstance(source, (str, Path)):
            with open(source, encoding="utf-8", errors="replace") as f:
                return self._process(f, max_rows, attack_only, attack_types)
        return self._process(source, max_rows, attack_only, attack_types)

    def _process(
        self,
        f: TextIO,
        max_rows: int | None,
        attack_only: bool,
        attack_types: list[str] | None,
    ) -> list[str]:
        reader = csv.DictReader(f)
        records: list[CicFlowRecord] = []

        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            # Skip empty rows
            if not any(v.strip() for v in row.values()):
                continue
            try:
                rec = parse_cic_row(row)
            except Exception:
                logger.warning("Skipping unparseable row %d", i)
                continue
            if rec is None:
                continue
            if attack_only and rec.attack_category == "BENIGN":
                continue
            if attack_types and rec.attack_category not in attack_types:
                continue
            records.append(rec)

        if self.shuffle:
            random.shuffle(records)

        lines = [to_cef_syslog(r, self.hostname) for r in records]
        return lines

    @staticmethod
    def write_lines(
        lines: list[str],
        output: str | Path | None = None,
    ) -> str | None:
        """
        Write syslog lines to a file or return as a single string.

        If *output* is None, returns the concatenated string.
        """
        text = "\n".join(lines)
        if output:
            Path(output).write_text(text, encoding="utf-8")
            return None
        return text

    @staticmethod
    def feed_udp(
        lines: list[str],
        host: str = "127.0.0.1",
        port: int = 5140,
        delay: float = 0.0,
    ) -> int:
        """
        Send syslog lines via UDP to a collector.

        Parameters
        ----------
        lines : list[str]
            CEF syslog lines to send.
        host : str
            Collector host.
        port : int
            Collector UDP port (default 5140 — matches project default).
        delay : float
            Seconds to wait between each line (0 = firehose).

        Returns
        -------
        int
            Number of lines sent.
        """
        sent = 0
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            for line in lines:
                data = (line + "\n").encode("utf-8", errors="replace")
                sock.sendto(data, (host, port))
                sent += 1
                if delay:
                    time.sleep(delay)
        finally:
            sock.close()
        return sent

    @staticmethod
    def list_csv_files(dataset_dir: str | Path) -> list[Path]:
        """List CSV files in a CIC dataset directory."""
        p = Path(dataset_dir)
        return sorted(p.glob("*.csv")) if p.is_dir() else []

    @staticmethod
    def summary(lines: list[str]) -> dict[str, int]:
        """Count attack categories in generated syslog lines."""
        counts: dict[str, int] = {}
        for line in lines:
            if "cs2Label=AttackCategory" not in line:
                continue
            # Extract attack category from CEF cs2 field
            for part in line.split("\t"):
                if part.startswith("cs2=") and not part.startswith("cs2Label="):
                    cat = part.split("=", 1)[1].strip()
                    counts[cat] = counts.get(cat, 0) + 1
        return counts
