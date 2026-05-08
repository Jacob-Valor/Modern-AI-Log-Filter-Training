"""Unit tests for the log normalizer."""

from __future__ import annotations

import json

import pytest

from logfilter.pipeline.normalizer import LogNormalizer, LogSourceType, normalize


@pytest.fixture
def normalizer():
    return LogNormalizer()


# ── Syslog RFC 3164 ────────────────────────────────────────────────────────────


class TestSyslog3164:
    def test_basic_syslog(self, normalizer):
        raw = (
            "Jan 15 11:07:53 prod-server01 sshd[22345]: Failed password "
            "for root from 10.0.0.5 port 44382 ssh2"
        )
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.SYSLOG
        assert ev.host == "prod-server01"
        assert "Failed password" in ev.text
        assert "10.0.0.5" in ev.text
        assert ev.raw == raw

    def test_syslog_with_priority(self, normalizer):
        raw = "<13>Jan 18 11:07:53 192.168.1.1 kernel: Out of memory: Kill process"
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.SYSLOG
        assert "192.168.1.1" == ev.host

    def test_syslog_no_process(self, normalizer):
        raw = "Mar  5 09:14:22 myhost Some message without process"
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.SYSLOG
        assert ev.host == "myhost"


# ── Syslog RFC 5424 ────────────────────────────────────────────────────────────


class TestSyslog5424:
    def test_rfc5424(self, normalizer):
        raw = "<34>1 2026-01-15T11:07:53.520Z prod-server01 sshd 22345 - - Failed password"
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.SYSLOG
        assert ev.host == "prod-server01"
        assert "2026-01-15" in ev.timestamp


# ── Windows Event ──────────────────────────────────────────────────────────────


class TestWindowsEvent:
    def test_winevent_json(self, normalizer):
        payload = {
            "EventID": "4625",
            "Computer": "WINSERVER01",
            "Message": "An account failed to log on.",
            "SubjectUserName": "SYSTEM",
            "TimeCreated": "2026-01-15T11:07:53",
        }
        raw = json.dumps(payload)
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.WINEVENT
        assert ev.host == "WINSERVER01"
        assert "4625" in ev.text
        assert "failed to log on" in ev.text.lower()

    def test_winevent_hint(self, normalizer):
        payload = {"EventID": "4688", "Computer": "WS01", "Message": "Process created"}
        raw = json.dumps(payload)
        ev = normalizer.normalize(raw, source_type_hint=LogSourceType.WINEVENT)
        assert ev.source_type == LogSourceType.WINEVENT

    def test_non_event_json_not_matched(self, normalizer):
        # JSON without EventID should not be parsed as winevent
        raw = json.dumps({"foo": "bar", "baz": 123})
        ev = normalizer.normalize(raw)
        assert ev.source_type != LogSourceType.WINEVENT


# ── CEF (Firewall) ─────────────────────────────────────────────────────────────


class TestCEF:
    def test_cef_basic(self, normalizer):
        raw = (
            "CEF:0|Cisco|ASA|9.14|106023|Deny tcp src outside:10.0.0.5/44382 "
            "dst inside:172.16.1.10/22 by access-group outside_access_in|7|"
            "src=10.0.0.5 dst=172.16.1.10 spt=44382 dpt=22 proto=tcp act=Deny"
        )
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.FIREWALL
        assert "Deny" in ev.text
        assert "10.0.0.5" in ev.text
        assert ev.fields.get("src") == "10.0.0.5" or "src" in ev.fields

    def test_cef_with_syslog_prefix(self, normalizer):
        raw = (
            "<134>Jan 15 11:07:53 firewall1 "
            "CEF:0|PaloAlto|NGFW|10.0|threat|SSH brute force|8|"
            "src=10.0.0.5 dst=192.168.1.1 spt=55000 dpt=22"
        )
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.FIREWALL


# ── Endpoint ───────────────────────────────────────────────────────────────────


class TestEndpoint:
    def test_crowdstrike_style(self, normalizer):
        payload = {
            "event_type": "ProcessCreate",
            "ComputerName": "ENDPOINT01",
            "ProcessName": "cmd.exe",
            "ParentProcessName": "powershell.exe",
            "CommandLine": "cmd.exe /c net user hacker P@ss /add",
            "UserName": "DOMAIN\\admin",
            "@timestamp": "2026-01-15T11:07:53Z",
        }
        raw = json.dumps(payload)
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.ENDPOINT
        assert ev.host == "ENDPOINT01"
        assert "cmd.exe" in ev.text
        assert "powershell" in ev.text

    def test_non_endpoint_json(self, normalizer):
        raw = json.dumps({"eventSource": "s3.amazonaws.com", "eventName": "GetObject"})
        ev = normalizer.normalize(raw)
        # Should be recognised as CloudTrail, not endpoint
        assert ev.source_type == LogSourceType.CLOUDTRAIL


# ── CloudTrail ─────────────────────────────────────────────────────────────────


class TestCloudTrail:
    def test_cloudtrail(self, normalizer):
        payload = {
            "eventSource": "iam.amazonaws.com",
            "eventName": "CreateUser",
            "userIdentity": {"userName": "attacker"},
            "sourceIPAddress": "1.2.3.4",
            "awsRegion": "us-east-1",
            "eventTime": "2026-01-15T11:07:53Z",
        }
        raw = json.dumps(payload)
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.CLOUDTRAIL
        assert "CreateUser" in ev.text
        assert "attacker" in ev.text
        assert "1.2.3.4" in ev.text


# ── Web / Apache ───────────────────────────────────────────────────────────────


class TestWeb:
    def test_apache_combined(self, normalizer):
        raw = (
            "10.0.0.1 - - [15/Jan/2026:11:07:53 +0000] "
            '"GET /wp-admin/admin.php HTTP/1.1" 200 1234 '
            '"-" "Mozilla/5.0 (Hydra)"'
        )
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.WEB
        assert ev.host == "10.0.0.1"
        assert "GET" in ev.text
        assert "/wp-admin" in ev.text

    def test_apache_no_ua(self, normalizer):
        raw = '192.168.1.5 - frank [15/Jan/2026:11:07:53 +0000] "POST /login HTTP/1.0" 403 512'
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.WEB
        assert "POST" in ev.text
        assert "403" in ev.text


# ── Generic fallback ───────────────────────────────────────────────────────────


class TestGeneric:
    def test_unknown_format_is_generic(self, normalizer):
        raw = "This is some completely unrecognised log format"
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.GENERIC
        assert ev.text == raw

    def test_module_level_normalize(self):
        raw = "Jan 15 11:07:53 host1 cron: (root) CMD (/usr/lib/apt/apt.systemd.daily)"
        ev = normalize(raw)
        assert ev.source_type == LogSourceType.SYSLOG


# ── Edge cases ─────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_string_is_generic(self, normalizer):
        ev = normalizer.normalize("   ")
        assert ev.source_type == LogSourceType.GENERIC

    def test_raw_preserved(self, normalizer):
        raw = "Jan 15 11:07:53 host1 test: hello world"
        ev = normalizer.normalize(raw)
        assert ev.raw == raw

    def test_invalid_json_fallback(self, normalizer):
        raw = '{"broken": json'
        ev = normalizer.normalize(raw)
        # Should not raise, should fall through to generic
        assert ev.raw == raw
