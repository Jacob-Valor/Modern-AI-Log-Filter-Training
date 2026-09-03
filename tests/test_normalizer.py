"""Unit tests for the log normalizer."""

from __future__ import annotations

import json

import pytest  # type: ignore[import-not-found]

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

    def test_apache_decodes_percent_encoded_security_payload(self, normalizer):
        raw = (
            "192.0.2.1 - - [01/Jan/2010:00:00:00 +0000] "
            '"GET /search?q=%3Cscript%3Ealert%281%29%3C%2Fscript%3E HTTP/1.1" '
            '200 0 "-" "Mozilla/5.0"'
        )
        ev = normalizer.normalize(raw)
        assert ev.source_type == LogSourceType.WEB
        assert "<script>alert(1)</script>" in ev.text
        assert ev.fields["decoded_uri"] == "/search?q=<script>alert(1)</script>"

    def test_apache_query_string_credentials_are_redacted(self, normalizer):
        raw = (
            "192.0.2.1 - - [01/Jan/2010:00:00:00 +0000] "
            '"GET /login?token=VERYSECRET123&next=%2Fadmin HTTP/1.1" '
            '200 0 "-" "Mozilla/5.0"'
        )
        event = normalizer.normalize(raw)
        assert "VERYSECRET123" not in event.text
        assert "VERYSECRET123" not in event.fields["uri"]
        assert "token=<REDACTED>" in event.fields["uri"]
        assert "next=/admin" in event.fields["uri"]


# ── Structured WAF JSON ────────────────────────────────────────────────────────


class TestWafJson:
    def test_modsecurity_post_body_is_redacted_and_model_visible(self, normalizer):
        raw = json.dumps(
            {
                "transaction": {
                    "time_stamp": "2026-07-30T00:00:00Z",
                    "client_ip": "192.0.2.50",
                    "request": {
                        "method": "POST",
                        "uri": "/login%3Fnext%3D%2Fadmin",
                        "headers": {"Host": "app.example", "Content-Type": "application/json"},
                        "body": {
                            "username": "admin",
                            "password": "do-not-expose",
                            "query": "' OR '1'='1",
                        },
                    },
                },
                "messages": [{"details": {"ruleId": "942100"}}],
            }
        )
        event = normalizer.normalize(raw)
        assert event.source_type == LogSourceType.FIREWALL
        assert event.fields["method"] == "POST"
        assert event.fields["rule_id"] == "942100"
        assert "OR" in event.text
        assert "do-not-expose" not in event.text
        assert "<REDACTED>" in event.fields["request_body_redacted"]

    @pytest.mark.parametrize(
        ("content_type", "body"),
        [
            ("application/json", '{"password":"secret123","query":"x"}'),
            ("text/plain", "password secret123 query x"),
            ("text/plain", "Authorization: Basic dXNlcjpwYXNzd29yZA=="),
            ("text/plain", "Authorization: Bearer eyJhbGciOiJub25lIn0.payload"),
            ("application/x-www-form-urlencoded", "username=admin&password=secret123"),
        ],
    )
    def test_string_waf_body_redacts_credentials(self, normalizer, content_type, body):
        raw = json.dumps(
            {
                "event_type": "waf",
                "request": {
                    "method": "POST",
                    "uri": "/login",
                    "headers": {"Content-Type": content_type},
                    "body": body,
                },
            }
        )
        event = normalizer.normalize(raw)
        assert "secret123" not in event.text
        assert "secret123" not in event.fields["request_body_redacted"]
        assert "dXNlcjpwYXNzd29yZA==" not in event.text
        assert "eyJhbGciOiJub25lIn0.payload" not in event.text
        assert "REDACTED" in event.fields["request_body_redacted"]

    def test_waf_query_string_credentials_are_redacted(self, normalizer):
        raw = json.dumps(
            {
                "event_type": "waf",
                "request": {
                    "method": "GET",
                    "uri": "/login?token=VERYSECRET123&next=%2Fadmin",
                    "headers": {"Host": "app.example"},
                },
            }
        )
        event = normalizer.normalize(raw)
        assert "VERYSECRET123" not in event.text
        assert "VERYSECRET123" not in event.fields["uri"]
        assert "token=<REDACTED>" in event.fields["uri"]
        assert "next=/admin" in event.fields["uri"]

    def test_binary_waf_body_is_not_model_visible(self, normalizer):
        raw = json.dumps(
            {
                "event_type": "waf",
                "action": "BLOCK",
                "request": {
                    "method": "POST",
                    "uri": "/upload",
                    "headers": {"Content-Type": "application/octet-stream"},
                    "body": "binary-secret",
                },
            }
        )
        event = normalizer.normalize(raw)
        assert event.fields["request_body_redacted"] == ""
        assert "binary-secret" not in event.text


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
