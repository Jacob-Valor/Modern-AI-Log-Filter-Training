"""Shared fixtures for integration tests backed by Docker Compose infrastructure."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

# ── Docker Compose lifecycle ────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPOSE_FILE = str(REPO_ROOT / "docker-compose.test.yml")


def _docker_available() -> tuple[bool, str]:
    """Return (usable, reason). ``docker`` must exist *and* the daemon must answer."""
    if shutil.which("docker") is None:
        return False, "docker CLI not found on PATH"
    probe = subprocess.run(
        ["docker", "info", "--format", "{{.ServerVersion}}"],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if probe.returncode != 0:
        return False, f"docker daemon unreachable: {probe.stderr.decode(errors='ignore').strip()}"
    return True, ""


def _compose_up() -> None:
    subprocess.run(
        ["docker", "compose", "-f", _COMPOSE_FILE, "up", "-d", "--wait"],
        check=True,
        capture_output=True,
    )


def _compose_down() -> None:
    subprocess.run(
        ["docker", "compose", "-f", _COMPOSE_FILE, "down", "-v"],
        check=False,
        capture_output=True,
    )


def _wait_for_kafka(bootstrap: str, timeout: float = 60.0) -> None:
    """Poll kafka-python until connection succeeds."""
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            from kafka import KafkaAdminClient

            client = KafkaAdminClient(bootstrap_servers=bootstrap)
            client.close()
            return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0)
    raise RuntimeError(f"Kafka not ready after {timeout}s: {last_err}")


def _wait_for_es(host: str, timeout: float = 60.0) -> None:
    """Poll Elasticsearch until cluster health is green/yellow."""
    import urllib.request

    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            req = urllib.request.Request(f"{host}/_cluster/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
                if body["status"] in ("green", "yellow"):
                    return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.0)
    raise RuntimeError(f"Elasticsearch not ready after {timeout}s: {last_err}")


@pytest.fixture(scope="session")
def compose_stack():
    """Start the Docker Compose test stack once per session and tear it down after.

    Deliberately NOT autouse: tests that never touch Kafka or Elasticsearch
    (e.g. the circuit-breaker resilience suite) must not require Docker. Every
    service endpoint fixture below depends on this one, so declaring
    ``kafka_bootstrap``/``es_host`` is how a test opts into the stack.

    Skips rather than errors when Docker is unavailable, so a Docker-less
    machine gets a clean skip instead of ``FileNotFoundError: 'docker'``.
    """
    available, reason = _docker_available()
    if not available:
        pytest.skip(f"Docker unavailable ({reason}); integration stack not started")
    _compose_up()
    try:
        yield
    finally:
        _compose_down()


# ── Service clients ─────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def kafka_bootstrap(compose_stack) -> str:
    """Opt in to the compose stack: any use of this fixture starts it."""
    return "localhost:19092"


@pytest.fixture(scope="session")
def es_host(compose_stack) -> str:
    """Opt in to the compose stack: any use of this fixture starts it."""
    return "http://localhost:19200"


@pytest.fixture(scope="session")
def es_client(es_host: str):
    """Yield a connected Elasticsearch client."""
    _wait_for_es(es_host)
    from elasticsearch import Elasticsearch

    client = Elasticsearch([es_host])
    yield client
    client.close()


@pytest.fixture
def kafka_producer(kafka_bootstrap: str):
    """Yield a fresh KafkaProducer per test (auto-closed)."""
    from kafka import KafkaProducer

    producer = KafkaProducer(
        bootstrap_servers=kafka_bootstrap,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    yield producer
    producer.close()


@pytest.fixture
def kafka_consumer(kafka_bootstrap: str):
    """Yield a fresh KafkaConsumer per test (auto-closed)."""
    from kafka import KafkaConsumer

    consumer = KafkaConsumer(
        bootstrap_servers=kafka_bootstrap,
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    )
    yield consumer
    consumer.close()


@pytest.fixture(scope="session")
def kafka_admin(kafka_bootstrap: str):
    """Yield a connected KafkaAdminClient."""
    _wait_for_kafka(kafka_bootstrap)
    from kafka import KafkaAdminClient

    client = KafkaAdminClient(bootstrap_servers=kafka_bootstrap)
    yield client
    client.close()
