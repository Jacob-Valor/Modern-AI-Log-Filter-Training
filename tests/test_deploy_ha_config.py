from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def _prod_compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.prod.yml").read_text())


def test_prod_compose_defines_three_kafka_brokers_with_rf_three() -> None:
    compose = _prod_compose()
    services = compose["services"]
    broker_names = {"kafka", "kafka-2", "kafka-3"}

    assert broker_names.issubset(services)
    for broker in broker_names:
        env = services[broker]["environment"]
        assert int(env["KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR"]) >= 3
        assert int(env["KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR"]) >= 3
        assert int(env["KAFKA_TRANSACTION_STATE_LOG_MIN_ISR"]) >= 2


def test_prod_compose_creates_topics_with_replication_factor_three() -> None:
    command = "\n".join(_prod_compose()["services"]["kafka-init"]["command"])

    assert "--replication-factor 3" in command
    assert "--bootstrap-server kafka:29092,kafka-2:29092,kafka-3:29092" in command


def test_prod_compose_defines_three_elasticsearch_nodes_without_single_node() -> None:
    compose = _prod_compose()
    services = compose["services"]
    node_names = {"elasticsearch", "elasticsearch-2", "elasticsearch-3"}

    assert node_names.issubset(services)
    for node in node_names:
        env_values = services[node]["environment"]
        rendered = "\n".join(env_values if isinstance(env_values, list) else env_values.values())
        assert "discovery.type=single-node" not in rendered
        expected_master_nodes = (
            "cluster.initial_master_nodes=elasticsearch,elasticsearch-2,elasticsearch-3"
        )
        assert expected_master_nodes in rendered
