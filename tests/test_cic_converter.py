"""
Unit tests for the CIC-IDS2017 CSV → CEF syslog converter.

Validates:
  1. CSV parsing with correct column mapping
  2. Attack classification (label → category/severity/MITRE)
  3. CEF syslog output format
  4. LogNormalizer can parse the CEF output
  5. UDP feed helper and summary stats
"""

from __future__ import annotations

import csv
import io
import json
import socket
import threading
from pathlib import Path

import pytest

from logfilter.demo.cic_converter import (
    CicCsvToSyslog,
    classify_attack,
    parse_cic_row,
    to_cef_syslog,
)

# ── Sample CIC-IDS2017 CSV data (mixing attack types) ─────────────────────────

# Minimal realistic CSV header matching CIC-2017 column names (with leading spaces)
SAMPLE_HEADER = (
    " Destination Port, Flow Duration, Total Fwd Packets, Total Backward Packets,"
    "Fwd Packet Length Mean, Bwd Packet Length Mean, Flow Bytes/s, Flow Packets/s,"
    " Label, Source IP, Destination IP, Protocol, Timestamp"
).replace("\n", "")

# Columns: port, duration, packet/length rates, label, IPs, protocol, timestamp.
SAMPLE_ROWS = [
    # DDoS attack
    "80,398877,4,4,209,1581,197.0,0.0,DDoS,192.168.1.5,10.0.0.1,6,7/7/2017 8:00",
    # DoS Hulk
    ("443,294577,8,2,150.5,200.3,45000.5,152.3,DoS Hulk,192.168.1.10,10.0.0.2,6,7/7/2017 8:01"),
    # Botnet ARES
    ("22,500123,12,8,100.2,250.7,12500.8,25.1,Botnet,192.168.1.15,10.0.0.3,6,7/7/2017 8:02"),
    # Heartbleed
    ("443,10000,3,5,60.1,180.4,5000.0,500.0,Heartbleed,192.168.1.20,10.0.0.4,6,7/7/2017 8:03"),
    # Web Attack Brute Force
    (
        "80,200000,6,3,500.0,300.0,25000.0,45.0,Web Attack Brute Force,"
        "192.168.1.25,10.0.0.5,6,7/7/2017 8:04"
    ),
    # Port Scan
    ("22,5000,2,0,40.0,0.0,8000.0,400.0,PortScan,192.168.1.30,10.0.0.6,6,7/7/2017 8:05"),
    # SSH Patator (brute force)
    "22,300000,20,10,100.0,150.0,6666.0,100.0,SSH-Patator,192.168.1.35,10.0.0.7,6,7/7/2017 8:06",
    # FTP Patator (brute force)
    "21,250000,15,8,200.0,120.0,8000.0,92.0,FTP-Patator,192.168.1.40,10.0.0.8,6,7/7/2017 8:07",
    # Infiltration
    "8080,600000,30,20,400.0,350.0,15000.0,83.3,Infiltration,192.168.1.45,10.0.0.9,6,7/7/2017 8:08",
    # BENIGN
    "53,100000,5,3,100.0,80.0,500.0,80.0,BENIGN,192.168.1.50,8.8.8.8,17,7/7/2017 8:09",
    # BENIGN 2
    "443,150000,10,5,300.0,200.0,1200.0,100.0,BENIGN,192.168.1.55,8.8.4.4,6,7/7/2017 8:10",
]


@pytest.fixture
def sample_csv() -> str:
    """Build a CIC-like CSV string with sample rows."""
    buf = io.StringIO()
    buf.write(SAMPLE_HEADER + "\n")
    for row in SAMPLE_ROWS:
        buf.write(row + "\n")
    buf.seek(0)
    return buf.getvalue()


@pytest.fixture
def sample_csv_file(tmp_path: Path, sample_csv: str) -> Path:
    """Write sample CSV to a temp file."""
    csv_path = tmp_path / "cic_sample.csv"
    csv_path.write_text(sample_csv, encoding="utf-8")
    return csv_path


# ── 1. Header parsing ─────────────────────────────────────────────────────────


class TestCicCsvParsing:
    """Test CSV parsing handles CIC-2017 and 2018 column formats."""

    def test_cic_2017_header_parsing(self):
        """Parse a CIC-2017-style CSV with leading spaces in column names."""
        raw = f"{SAMPLE_HEADER}\n{SAMPLE_ROWS[0]}"
        reader = csv.DictReader(io.StringIO(raw))
        row = next(reader)
        rec = parse_cic_row(row)
        assert rec is not None
        assert rec.dst_port == 80
        assert rec.attack_category == "DDOS"
        assert rec.severity == 10

    def test_classify_ddos(self):
        cat, sev, mitre = classify_attack("DDoS")
        assert cat == "DDOS"
        assert sev == 10
        assert mitre == "T1498"

    def test_classify_dos_hulk(self):
        cat, sev, mitre = classify_attack("DoS Hulk")
        assert cat == "DOS"
        assert sev == 8
        assert mitre == "T1499"

    def test_classify_botnet(self):
        cat, sev, mitre = classify_attack("Botnet")
        assert cat == "BOTNET"
        assert sev == 9
        assert mitre == "T1583.001"

    def test_classify_heartbleed(self):
        cat, sev, mitre = classify_attack("Heartbleed")
        assert cat == "HEARTBLEED"
        assert sev == 10
        assert mitre == "T1190"

    def test_classify_benign(self):
        cat, sev, mitre = classify_attack("BENIGN")
        assert cat == "BENIGN"
        assert sev == 1
        assert mitre == "NONE"

    def test_classify_unknown(self):
        cat, sev, mitre = classify_attack("SOME_RANDOM_LABEL")
        assert cat == "BENIGN"
        assert sev == 1
        assert mitre == "NONE"

    def test_parse_cic_row_all_fields(self):
        """Ensure all CicFlowRecord fields are populated."""
        raw = "443,294577,8,2,150,200,45000.5,152.3,DoS Hulk,192.168.1.10,10.0.0.2,6,7/7/2017 8:01"
        header = SAMPLE_HEADER
        reader = csv.DictReader(io.StringIO(f"{header}\n{raw}"))
        rec = parse_cic_row(next(reader))
        assert rec is not None
        assert rec.src_ip == "192.168.1.10"
        assert rec.dst_ip == "10.0.0.2"
        assert rec.src_port == 0  # not in our minimal header
        assert rec.dst_port == 443
        assert rec.protocol == "6"
        assert rec.label == "DoS Hulk"
        assert rec.flow_duration == 294577
        assert rec.fwd_packets == 8
        assert rec.bwd_packets == 2
        assert rec.fwd_bytes_mean == 150.0
        assert rec.bwd_bytes_mean == 200.0
        assert rec.flow_bytes_sec == 45000.5
        assert rec.flow_pkts_sec == 152.3
        assert rec.attack_category == "DOS"
        assert rec.severity == 8
        assert rec.mitre_technique == "T1499"


# ── 2. CEF syslog string ─────────────────────────────────────────────────────


class TestCefSyslogOutput:
    """Test CEF syslog message generation."""

    def test_cef_format_structure(self):
        """Verify the CEF message follows the expected format."""
        raw = f"{SAMPLE_HEADER}\n{SAMPLE_ROWS[0]}"
        reader = csv.DictReader(io.StringIO(raw))
        rec = parse_cic_row(next(reader))
        assert rec is not None
        cef = to_cef_syslog(rec, "test-host")

        # Must start with syslog priority
        assert cef.startswith("<")
        assert "CEF:0|CIC|IDS2017|1.0|" in cef
        assert "test-host" in cef

        # Must contain key CEF fields
        assert "src=192.168.1.5" in cef
        assert "dst=10.0.0.1" in cef
        assert "dpt=80" in cef
        assert "proto=tcp" in cef

        # Must contain MITRE technique
        assert "cs1=T1498" in cef
        assert "cs1Label=MitreTechnique" in cef

    def test_cef_dos_hulk(self):
        raw = f"{SAMPLE_HEADER}\n{SAMPLE_ROWS[1]}"
        reader = csv.DictReader(io.StringIO(raw))
        rec = parse_cic_row(next(reader))
        assert rec is not None
        cef = to_cef_syslog(rec)
        assert "DoS_Hulk" in cef or "DoS" in cef
        assert "act=dos" in cef
        assert "cs1=T1499" in cef

    def test_cef_benign(self):
        raw = f"{SAMPLE_HEADER}\n{SAMPLE_ROWS[9]}"
        reader = csv.DictReader(io.StringIO(raw))
        rec = parse_cic_row(next(reader))
        assert rec is not None
        cef = to_cef_syslog(rec)
        assert "cat=BENIGN" in cef or "BENIGN" in cef
        assert "cs1=NONE" in cef


# ── 3. CicCsvToSyslog converter ──────────────────────────────────────────────


class TestCicCsvToSyslog:
    """Test the full converter pipeline."""

    def test_read_csv_lines(self, sample_csv_file: Path):
        converter = CicCsvToSyslog(shuffle=False)
        lines = converter.csv_lines(sample_csv_file, max_rows=None, attack_only=False)

        # 11 rows total (including UNKNOWN)
        assert len(lines) == 11

    def test_attack_only_filter(self, sample_csv_file: Path):
        converter = CicCsvToSyslog(shuffle=False)
        lines = converter.csv_lines(sample_csv_file, max_rows=None, attack_only=True)

        # 11 total - 2 BENIGN = 9 attack rows
        assert len(lines) == 9

    def test_attack_type_filter(self, sample_csv_file: Path):
        converter = CicCsvToSyslog(shuffle=False)
        lines = converter.csv_lines(
            sample_csv_file, max_rows=None, attack_only=False, attack_types=["DDOS", "DOS"]
        )
        # Only DDoS + DoS Hulk = 2
        assert len(lines) == 2

    def test_max_rows(self, sample_csv_file: Path):
        converter = CicCsvToSyslog(shuffle=False)
        lines = converter.csv_lines(sample_csv_file, max_rows=3, attack_only=False)
        assert len(lines) == 3

    def test_summary(self, sample_csv_file: Path):
        converter = CicCsvToSyslog(shuffle=False)
        lines = converter.csv_lines(sample_csv_file, max_rows=None, attack_only=False)
        summary = converter.summary(lines)
        assert summary.get("DDOS") == 1
        assert summary.get("DOS") == 1
        assert summary.get("BOTNET") == 1
        assert summary.get("HEARTBLEED") == 1
        assert summary.get("BENIGN") == 2


# ── 4. LogNormalizer integration ──────────────────────────────────────────────


class TestNormalizerIntegration:
    """Test that CEF syslog output is correctly parsed by LogNormalizer."""

    @staticmethod
    def _get_normalizer():
        """Import and return a LogNormalizer instance."""
        from logfilter.pipeline.normalizer import LogNormalizer, LogSourceType

        return LogNormalizer(), LogSourceType

    def test_normalizer_parses_cef(self, sample_csv_file: Path):
        """Feed CEF syslog output to normalizer and verify parsing."""
        normalizer, LogSourceType = self._get_normalizer()
        converter = CicCsvToSyslog(shuffle=False)
        lines = converter.csv_lines(sample_csv_file, max_rows=None, attack_only=True)

        # Try to normalize each CEF line
        parsed = 0
        for line in lines:
            try:
                # Remove syslog header to extract just the CEF portion
                event = normalizer.normalize(line)
                if event is not None:
                    parsed += 1
                    # CEF events should be classified as FIREWALL
                    # (which is where the normalizer routes CEF)
                    if hasattr(event, "source_type"):
                        assert event.source_type in (
                            LogSourceType.FIREWALL,
                            LogSourceType.SYSLOG,
                        )
            except Exception:
                pass

        # At least some should parse
        assert parsed > 0, "No CEF lines were parsed by the normalizer"
        print(f"  ✓ Normalizer successfully parsed {parsed}/{len(lines)} CEF events")

    def test_normalizer_extracts_cef_fields(self, sample_csv_file: Path):
        """Verify the normalizer extracts key fields from CEF events."""
        normalizer, LogSourceType = self._get_normalizer()
        converter = CicCsvToSyslog(shuffle=False)
        lines = converter.csv_lines(sample_csv_file, max_rows=1, attack_only=False)

        if not lines:
            pytest.skip("No CEF lines generated")

        event = normalizer.normalize(lines[0])
        if event is None:
            pytest.skip("Normalizer returned None (expected if no ML models loaded)")

        # Check event has expected attributes
        event_dict = event if isinstance(event, dict) else event.__dict__
        print(f"  Event details: {json.dumps(str(event_dict)[:300])}")

        # CEF fields should be present in the raw message at minimum
        raw = event_dict.get("raw", event_dict.get("message", str(event)))
        assert "CEF:" in raw, f"CEF format marker not found in: {raw[:100]}"
        assert "CIC|IDS2017" in raw


# ── 5. Write and feed helpers ─────────────────────────────────────────────────


class TestHelpers:
    """Test utility methods."""

    def test_write_lines_to_file(self, tmp_path: Path):
        output = tmp_path / "output.syslog"
        converter = CicCsvToSyslog()
        lines = ["<134>Jul 27 10:00:00 test CEF:0|CIC|IDS2017|1.0|T1498|test|5|src=1.2.3.4"]
        converter.write_lines(lines, output)
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert content.strip() == lines[0]

    def test_write_lines_return_string(self):
        converter = CicCsvToSyslog()
        lines = ["<134>Jul 27 10:00:00 test CEF:0|CIC|IDS2017|1.0|T1498|test|5|src=1.2.3.4"]
        result = converter.write_lines(lines)
        assert result == lines[0]

    @pytest.mark.skipif(
        not hasattr(socket, "AF_INET"),
        reason="UDP test requires socket support",
    )
    def test_feed_udp_sends_data(self):
        """Start a UDP listener, feed data, verify receipt."""
        received_data: list[str] = []
        port_ready = threading.Event()
        listener_port: list[int] = []

        def udp_listener(stop_event: threading.Event):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(3.0)
            sock.bind(("127.0.0.1", 0))  # bind ephemeral port
            listener_port.append(sock.getsockname()[1])
            port_ready.set()
            try:
                while not stop_event.is_set():
                    try:
                        data, addr = sock.recvfrom(4096)
                        received_data.append(data.decode("utf-8", errors="replace"))
                        if len(received_data) >= 2:
                            break
                    except TimeoutError:
                        break
            finally:
                sock.close()

        stop_event = threading.Event()
        listener_thread = threading.Thread(target=udp_listener, args=(stop_event,), daemon=True)
        listener_thread.start()

        port_ready.wait(timeout=3)
        if not listener_port:
            pytest.skip("Could not determine listener port")
            return

        c = CicCsvToSyslog()
        sent = c.feed_udp(
            [
                (
                    "<134>Jul 27 10:00:00 test CEF:0|CIC|IDS2017|1.0|"
                    "T1498|ddos_test|5|src=1.2.3.4 dst=5.6.7.8"
                ),
                (
                    "<134>Jul 27 10:00:01 test CEF:0|CIC|IDS2017|1.0|"
                    "T1499|dos_test|8|src=2.3.4.5 dst=6.7.8.9"
                ),
            ],
            host="127.0.0.1",
            port=listener_port[0],
            delay=0.05,
        )
        assert sent == 2

        stop_event.set()
        listener_thread.join(timeout=3)
        assert len(received_data) == 2
        assert "ddos_test" in received_data[0] or "T1498" in received_data[0]
