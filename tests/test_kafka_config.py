from __future__ import annotations

import pytest

from logfilter.kafka.config import kafka_security_kwargs


def test_kafka_security_kwargs_defaults_to_plaintext() -> None:
    assert kafka_security_kwargs({}) == {"security_protocol": "PLAINTEXT"}


def test_kafka_security_kwargs_omits_empty_ssl_values() -> None:
    kwargs = kafka_security_kwargs(
        {
            "security": {
                "protocol": "SSL",
                "ssl": {
                    "cafile": "/etc/kafka/ca.pem",
                    "certfile": "",
                    "keyfile": None,
                    "check_hostname": "false",
                },
            }
        }
    )

    assert kwargs == {
        "security_protocol": "SSL",
        "ssl_check_hostname": False,
        "ssl_cafile": "/etc/kafka/ca.pem",
    }


def test_kafka_security_kwargs_maps_sasl_ssl_username_password() -> None:
    kwargs = kafka_security_kwargs(
        {
            "security": {
                "protocol": "SASL_SSL",
                "sasl": {
                    "mechanism": "SCRAM-SHA-512",
                    "username": "logfilter",
                    "password": "secret",
                },
                "ssl": {
                    "cafile": "/etc/kafka/ca.pem",
                    "certfile": "/etc/kafka/client.pem",
                    "keyfile": "/etc/kafka/client.key",
                    "password": "key-secret",
                },
            }
        }
    )

    assert kwargs == {
        "security_protocol": "SASL_SSL",
        "sasl_mechanism": "SCRAM-SHA-512",
        "sasl_plain_username": "logfilter",
        "sasl_plain_password": "secret",
        "ssl_check_hostname": True,
        "ssl_cafile": "/etc/kafka/ca.pem",
        "ssl_certfile": "/etc/kafka/client.pem",
        "ssl_keyfile": "/etc/kafka/client.key",
        "ssl_password": "key-secret",
    }


def test_kafka_security_kwargs_rejects_unknown_protocol() -> None:
    with pytest.raises(ValueError, match="Unsupported Kafka security protocol"):
        kafka_security_kwargs({"security": {"protocol": "NOPE"}})


def test_kafka_security_kwargs_rejects_invalid_check_hostname() -> None:
    with pytest.raises(ValueError, match="Invalid boolean config value"):
        kafka_security_kwargs(
            {
                "security": {
                    "protocol": "SSL",
                    "ssl": {"check_hostname": "maybe"},
                }
            }
        )


def test_kafka_security_kwargs_requires_sasl_credentials() -> None:
    with pytest.raises(ValueError, match="username/password"):
        kafka_security_kwargs(
            {
                "security": {
                    "protocol": "SASL_SSL",
                    "sasl": {"mechanism": "PLAIN", "username": "logfilter"},
                }
            }
        )
