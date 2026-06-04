"""
Log router — forwards enriched LEEF events to IBM QRadar via syslog.

Routing modes (set in config.yaml qradar.mode):
  enrich_only  — forward ALL events to QRadar, regardless of ai_priority.
                 QRadar custom rules use ai_threat_score to filter/prioritise.
                 Zero false-negative risk. Recommended for compliance environments.
  suppress_low — suppress events with ai_priority=INFO from QRadar forwarding.
                 Reduces EPS license cost but risks false negatives.

In both modes, ALL raw logs have already been archived to Elasticsearch
by the archive consumer BEFORE scoring (archive-first pattern).
"""

from __future__ import annotations

import socket
import ssl
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from logfilter.pipeline.events import ScoredEvent

logger = structlog.get_logger(__name__)


class RoutingMode(str, Enum):
    ENRICH_ONLY = "enrich_only"
    SUPPRESS_LOW = "suppress_low"


@dataclass
class RoutingDecision:
    forward_to_qradar: bool
    reason: str
    priority: str


class SyslogSender:
    """
    RFC 5424 / TCP syslog sender with optional TLS.

    Wraps the underlying socket with retry logic (tenacity) and
    automatic reconnect on broken connections.
    """

    FACILITY = 1  # user-level messages
    SEVERITY_MAP = {"HIGH": 2, "MEDIUM": 4, "LOW": 5, "INFO": 6}

    def __init__(
        self,
        host: str,
        port: int = 514,
        protocol: str = "tcp",  # tcp | udp | tls
        timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.protocol = protocol
        self.timeout = timeout
        self._sock: socket.socket | None = None
        self._tls_context: ssl.SSLContext | None = None

        if protocol == "tls":
            self._tls_context = ssl.create_default_context()

    def _connect(self) -> None:
        self._disconnect()
        if self.protocol == "udp":
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            raw_sock.settimeout(self.timeout)
            raw_sock.connect((self.host, self.port))
            if self._tls_context:
                self._sock = self._tls_context.wrap_socket(raw_sock, server_hostname=self.host)
            else:
                self._sock = raw_sock
        logger.info("Syslog connected", host=self.host, port=self.port, protocol=self.protocol)

    def _disconnect(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.5, max=4),
        reraise=True,
    )
    def send(self, message: str, priority: str = "INFO") -> None:
        """Send a single syslog message."""
        sev = self.SEVERITY_MAP.get(priority, 6)
        pri = self.FACILITY * 8 + sev
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        syslog_frame = f"<{pri}>1 {ts} logfilter-ai - - - {message}\n"
        payload = syslog_frame.encode("utf-8", errors="replace")

        try:
            if self.protocol == "udp":
                if self._sock is None:
                    self._connect()
                self._sock.sendto(payload, (self.host, self.port))  # type: ignore[union-attr]
            else:
                if self._sock is None:
                    self._connect()
                # TCP syslog framing: octet-count prefix (RFC 6587)
                frame = f"{len(payload)} ".encode() + payload
                self._sock.sendall(frame)  # type: ignore[union-attr]
        except (OSError, BrokenPipeError):
            self._disconnect()
            raise  # tenacity will retry

    def send_batch(self, messages: list[tuple[str, str]]) -> int:
        """
        Send a batch of (message, priority) tuples.

        Returns the number of successfully sent messages.
        """
        sent = 0
        for msg, prio in messages:
            try:
                self.send(msg, prio)
                sent += 1
            except (OSError, BrokenPipeError) as exc:
                logger.error("Failed to send syslog message", error=str(exc))
        return sent

    def close(self) -> None:
        self._disconnect()


class LogRouter:
    """
    Routes enriched LEEF events to QRadar and handles mode-based suppression.

    Parameters
    ----------
    config : dict
        The 'qradar' section of config.yaml.
    sender : SyslogSender | None
        Pre-instantiated syslog sender (optional; created from config if None).
    """

    def __init__(
        self,
        config: dict[str, Any],
        sender: SyslogSender | None = None,
    ) -> None:
        qradar_cfg = config.get("qradar", config)  # tolerate full or section-only config
        self.mode = RoutingMode(qradar_cfg.get("mode", "enrich_only"))
        self.sender = sender or SyslogSender(
            host=qradar_cfg.get("syslog_host", "localhost"),
            port=int(qradar_cfg.get("syslog_port", 514)),
            protocol=qradar_cfg.get("syslog_protocol", "tcp"),
        )

    def decide(self, scored: ScoredEvent) -> RoutingDecision:
        """Determine whether this event should be forwarded to QRadar."""
        if self.mode == RoutingMode.ENRICH_ONLY:
            return RoutingDecision(
                forward_to_qradar=True,
                reason="enrich_only mode — all events forwarded",
                priority=scored.ai_priority,
            )

        # suppress_low mode
        if scored.ai_priority == "INFO":
            return RoutingDecision(
                forward_to_qradar=False,
                reason=f"score {scored.ai_threat_score:.3f} below low threshold",
                priority=scored.ai_priority,
            )

        return RoutingDecision(
            forward_to_qradar=True,
            reason=f"priority={scored.ai_priority}",
            priority=scored.ai_priority,
        )

    def route(self, leef_message: str, scored: ScoredEvent) -> RoutingDecision:
        """
        Apply routing decision and send to QRadar if appropriate.

        Parameters
        ----------
        leef_message : str
            Enriched LEEF string from LEEFEnricher.
        scored : ScoredEvent
            The corresponding scored event (for routing decision).

        Returns
        -------
        RoutingDecision
        """
        decision = self.decide(scored)

        if decision.forward_to_qradar:
            try:
                self.sender.send(leef_message, decision.priority)
                logger.debug(
                    "Event routed to QRadar",
                    priority=decision.priority,
                    score=round(scored.ai_threat_score, 3),
                    mitre=scored.ai_mitre_technique,
                )
            except (OSError, BrokenPipeError) as exc:
                logger.error("Failed to route event to QRadar", error=str(exc))
        else:
            logger.debug(
                "Event suppressed (not forwarded to QRadar)",
                reason=decision.reason,
                score=round(scored.ai_threat_score, 3),
            )

        return decision

    def route_batch(
        self,
        leef_messages: list[str],
        scored_events: list[ScoredEvent],
    ) -> list[RoutingDecision]:
        """Route a batch of enriched events."""
        decisions = []
        forward_pairs: list[tuple[str, str]] = []
        forward_indices: list[int] = []

        for i, (msg, scored) in enumerate(zip(leef_messages, scored_events)):
            decision = self.decide(scored)
            decisions.append(decision)
            if decision.forward_to_qradar:
                forward_pairs.append((msg, decision.priority))
                forward_indices.append(i)

        if forward_pairs:
            sent = self.sender.send_batch(forward_pairs)
            logger.info(
                "Batch routed",
                total=len(scored_events),
                forwarded=len(forward_pairs),
                sent=sent,
                suppressed=len(scored_events) - len(forward_pairs),
            )

        return decisions

    def close(self) -> None:
        self.sender.close()
