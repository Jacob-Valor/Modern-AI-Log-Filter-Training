from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent


def _prod_compose() -> dict:
    """Parse the HA overlay, rejecting duplicate keys.

    ``yaml.safe_load`` silently keeps the LAST of a duplicated mapping key, which
    hid a real defect: ``logfilter-api`` was defined twice and the whole first
    block (3-broker bootstrap, LOGFILTER_MODELS_STRICT, flow/novelty enablement)
    was discarded before Compose ever merged the overlay. This loader refuses to
    parse a file with duplicate keys so that class of bug fails the test instead
    of passing it.
    """

    class _NoDuplicateLoader(yaml.SafeLoader):
        pass

    def _no_duplicates(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
        # Iterate the raw (key, value) node pairs rather than construct_mapping:
        # construct_mapping builds a dict first, so a duplicated key is already
        # collapsed by the time it is returned and cannot be detected.
        mapping: dict = {}
        seen: set = set()
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in seen:
                raise AssertionError(f"duplicate key in docker-compose.prod.yml: {key!r}")
            seen.add(key)
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _NoDuplicateLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _no_duplicates
    )
    return yaml.load((ROOT / "docker-compose.prod.yml").read_text(), Loader=_NoDuplicateLoader)


def test_prod_compose_api_targets_all_three_kafka_brokers() -> None:
    """The API must bootstrap against every broker the overlay defines.

    This assertion is the one the duplicated ``logfilter-api`` key silently broke:
    the surviving block had no KAFKA_BOOTSTRAP_SERVERS at all.
    """
    env = _prod_compose()["services"]["logfilter-api"]["environment"]
    assert env["KAFKA_BOOTSTRAP_SERVERS"] == "kafka:29092,kafka-2:29092,kafka-3:29092"


def test_prod_compose_api_enables_strict_models_and_detectors() -> None:
    """Production must fail closed on model load and run both extra detectors."""
    env = _prod_compose()["services"]["logfilter-api"]["environment"]
    assert str(env["LOGFILTER_MODELS_STRICT"]) == "1"
    assert str(env["LOGFILTER_FLOW_ENABLED"]).lower() == "true"
    assert str(env["LOGFILTER_ENDPOINT_NOVELTY_ENABLED"]).lower() == "true"


def test_prod_compose_api_archive_replicas_match_archive_service() -> None:
    """The API provisions the archive index, so both must agree on replicas."""
    services = _prod_compose()["services"]
    assert (
        services["logfilter-api"]["environment"]["LOGFILTER_ES_INDEX_REPLICAS"]
        == services["archive"]["environment"]["LOGFILTER_ES_INDEX_REPLICAS"]
    )


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
    """Topic creation must be pinned to 3 replicas across all three brokers.

    Asserted structurally against the command *list*: joining it into one string
    first would let a substring match survive a reordered or partially-removed
    command.
    """
    command = _prod_compose()["services"]["kafka-init"]["command"]
    assert isinstance(command, list) and command, "kafka-init must define a command list"

    joined = "\n".join(str(part) for part in command)
    assert "--replication-factor 3" in joined
    assert "--bootstrap-server kafka:29092,kafka-2:29092,kafka-3:29092" in joined


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
