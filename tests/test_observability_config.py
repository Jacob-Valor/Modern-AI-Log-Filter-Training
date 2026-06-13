from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def _load_yaml(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text())


def test_alertmanager_webhook_has_no_localhost_fallback() -> None:
    config_text = (ROOT / "config" / "alertmanager.yml").read_text()
    config = yaml.safe_load(config_text)

    assert "localhost:9095" not in config_text
    for receiver in config["receivers"]:
        for webhook in receiver.get("webhook_configs", []):
            assert webhook["url"] == "${ALERT_WEBHOOK_URL}"


def test_prometheus_has_blackbox_deadman_scrape_job() -> None:
    config = _load_yaml("config/prometheus.yml")
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}

    assert "logfilter-blackbox" in jobs
    blackbox = jobs["logfilter-blackbox"]
    assert blackbox["metrics_path"] == "/probe"
    assert blackbox["params"] == {"module": ["http_2xx"]}
    assert blackbox["static_configs"][0]["targets"] == ["http://logfilter-api:8080/health"]


def test_compose_defines_blackbox_exporter_service() -> None:
    compose = _load_yaml("docker-compose.yml")
    blackbox = compose["services"]["blackbox-exporter"]

    assert blackbox["image"].startswith("prom/blackbox-exporter:")
    assert "blackbox-exporter" in compose["services"]["prometheus"]["depends_on"]


def test_prometheus_exporter_scrapes_have_matching_compose_services() -> None:
    compose = _load_yaml("docker-compose.yml")
    prometheus = _load_yaml("config/prometheus.yml")
    services = compose["services"]
    jobs = {job["job_name"]: job for job in prometheus["scrape_configs"]}

    assert "kafka-exporter" in services
    assert services["kafka-exporter"]["image"].startswith("danielqsj/kafka-exporter:")
    assert jobs["kafka"]["static_configs"][0]["targets"] == ["kafka-exporter:9308"]
    assert "kafka-exporter" in services["prometheus"]["depends_on"]

    assert "elasticsearch-exporter" in services
    assert services["elasticsearch-exporter"]["image"].startswith(
        "quay.io/prometheuscommunity/elasticsearch-exporter:"
    )
    assert jobs["elasticsearch"]["static_configs"][0]["targets"] == [
        "elasticsearch-exporter:9114"
    ]
    assert "elasticsearch-exporter" in services["prometheus"]["depends_on"]


def test_otel_collector_exports_traces_to_jaeger() -> None:
    compose = _load_yaml("docker-compose.yml")
    otel = _load_yaml("config/otel-collector.yml")

    assert "jaeger" in compose["services"]
    assert compose["services"]["jaeger"]["image"].startswith("jaegertracing/all-in-one:")
    assert "otlp/jaeger" in otel["exporters"]
    assert otel["exporters"]["otlp/jaeger"]["endpoint"] == "jaeger:4317"
    assert "otlp/jaeger" in otel["service"]["pipelines"]["traces"]["exporters"]
    assert "debug" in otel["service"]["pipelines"]["traces"]["exporters"]
