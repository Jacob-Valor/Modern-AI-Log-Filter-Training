from __future__ import annotations

import pytest

from logfilter.kafka.config import _bool_config, _clean, kafka_security_kwargs


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


def test_clean_handles_none() -> None:
    assert _clean(None) == ""


def test_clean_strips_whitespace() -> None:
    assert _clean("  value  ") == "value"


def test_bool_config_returns_default_for_none() -> None:
    assert _bool_config(None, default=True) is True
    assert _bool_config(None, default=False) is False


def test_bool_config_returns_bool_directly() -> None:
    assert _bool_config(True, default=False) is True
    assert _bool_config(False, default=True) is False


def test_bool_config_parses_true_values() -> None:
    for value in ["1", "true", "True", "TRUE", "yes", "YES", "on", "ON"]:
        assert _bool_config(value) is True


def test_bool_config_parses_false_values() -> None:
    for value in ["0", "false", "False", "FALSE", "no", "NO", "off", "OFF"]:
        assert _bool_config(value, default=True) is False


def test_bool_config_raises_on_invalid() -> None:
    with pytest.raises(ValueError, match="Invalid boolean"):
        _bool_config("maybe")


def test_kafka_security_kwargs_security_not_mapping() -> None:
    with pytest.raises(ValueError, match="must be a mapping"):
        kafka_security_kwargs({"security": "bad"})


def test_kafka_security_kwargs_sasl_not_mapping() -> None:
    with pytest.raises(ValueError, match="sasl must be a mapping"):
        kafka_security_kwargs(
            {"security": {"protocol": "SASL_PLAINTEXT", "sasl": "bad"}}
        )


def test_kafka_security_kwargs_ssl_not_mapping() -> None:
    with pytest.raises(ValueError, match="ssl must be a mapping"):
        kafka_security_kwargs({"security": {"protocol": "SSL", "ssl": "bad"}})


def test_kafka_security_kwargs_sasl_missing_mechanism() -> None:
    with pytest.raises(ValueError, match="mechanism"):
        kafka_security_kwargs(
            {"security": {"protocol": "SASL_PLAINTEXT", "sasl": {}}}
        )


def test_kafka_security_kwargs_sasl_plain_missing_password() -> None:
    with pytest.raises(ValueError, match="username/password"):
        kafka_security_kwargs(
            {
                "security": {
                    "protocol": "SASL_PLAINTEXT",
                    "sasl": {"mechanism": "PLAIN", "username": "user"},
                }
            }
        )


def test_kafka_security_kwargs_sasl_plain_missing_username() -> None:
    with pytest.raises(ValueError, match="username/password"):
        kafka_security_kwargs(
            {
                "security": {
                    "protocol": "SASL_PLAINTEXT",
                    "sasl": {"mechanism": "PLAIN", "password": "secret"},
                }
            }
        )


def test_kafka_security_kwargs_ssl_full_config() -> None:
    kwargs = kafka_security_kwargs(
        {
            "security": {
                "protocol": "SSL",
                "ssl": {
                    "cafile": "/etc/ca.pem",
                    "certfile": "/etc/cert.pem",
                    "keyfile": "/etc/key.pem",
                    "password": "secret",
                    "check_hostname": "false",
                },
            }
        }
    )

    assert kwargs["security_protocol"] == "SSL"
    assert kwargs["ssl_cafile"] == "/etc/ca.pem"
    assert kwargs["ssl_certfile"] == "/etc/cert.pem"
    assert kwargs["ssl_keyfile"] == "/etc/key.pem"
    assert kwargs["ssl_password"] == "secret"
    assert kwargs["ssl_check_hostname"] is False


def test_kafka_security_kwargs_ssl_skips_empty_values() -> None:
    kwargs = kafka_security_kwargs(
        {
            "security": {
                "protocol": "SSL",
                "ssl": {"cafile": "", "certfile": "/etc/cert.pem"},
            }
        }
    )

    assert "ssl_cafile" not in kwargs
    assert kwargs["ssl_certfile"] == "/etc/cert.pem"


def test_kafka_security_kwargs_sasl_ssl_full() -> None:
    kwargs = kafka_security_kwargs(
        {
            "security": {
                "protocol": "SASL_SSL",
                "sasl": {
                    "mechanism": "SCRAM-SHA-256",
                    "username": "user",
                    "password": "secret",
                },
                "ssl": {"cafile": "/etc/ca.pem"},
            }
        }
    )

    assert kwargs["security_protocol"] == "SASL_SSL"
    assert kwargs["sasl_mechanism"] == "SCRAM-SHA-256"
    assert kwargs["sasl_plain_username"] == "user"
    assert kwargs["sasl_plain_password"] == "secret"
    assert kwargs["ssl_cafile"] == "/etc/ca.pem"
