from __future__ import annotations

import httpx
import pytest

from logfilter.kafka.config import (
    _bool_config,
    _clean,
    is_transient_failure,
    kafka_consumer_health_kwargs,
    kafka_security_kwargs,
)
from logfilter.utils.circuit_breaker import CircuitBreakerOpen


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "http://api")
    return httpx.HTTPStatusError(
        f"status {status}", request=request, response=httpx.Response(status, request=request)
    )


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
        kafka_security_kwargs({"security": {"protocol": "SASL_PLAINTEXT", "sasl": "bad"}})


def test_kafka_security_kwargs_ssl_not_mapping() -> None:
    with pytest.raises(ValueError, match="ssl must be a mapping"):
        kafka_security_kwargs({"security": {"protocol": "SSL", "ssl": "bad"}})


def test_kafka_security_kwargs_sasl_missing_mechanism() -> None:
    with pytest.raises(ValueError, match="mechanism"):
        kafka_security_kwargs({"security": {"protocol": "SASL_PLAINTEXT", "sasl": {}}})


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


def test_kafka_consumer_health_kwargs_defaults_are_well_above_the_library_default() -> None:
    """kafka-python's 300s default poll interval is too short for a 500-record batch."""
    kwargs = kafka_consumer_health_kwargs({})

    assert kwargs["max_poll_interval_ms"] == 900_000
    assert kwargs["max_poll_interval_ms"] > 300_000
    assert kwargs["session_timeout_ms"] == 45_000
    assert kwargs["max_poll_interval_ms"] > kwargs["session_timeout_ms"]


def test_kafka_consumer_health_kwargs_reads_string_values_from_env_resolution() -> None:
    kwargs = kafka_consumer_health_kwargs(
        {"max_poll_interval_ms": "1200000", "session_timeout_ms": "60000"}
    )

    assert kwargs == {"max_poll_interval_ms": 1200000, "session_timeout_ms": 60000}


def test_kafka_consumer_health_kwargs_treats_empty_values_as_unset() -> None:
    kwargs = kafka_consumer_health_kwargs({"max_poll_interval_ms": "", "session_timeout_ms": None})

    assert kwargs["max_poll_interval_ms"] == 900_000
    assert kwargs["session_timeout_ms"] == 45_000


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_is_transient_failure_flags_retryable_statuses(status) -> None:
    assert is_transient_failure(_status_error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_is_transient_failure_rejects_permanent_statuses(status) -> None:
    assert is_transient_failure(_status_error(status)) is False


def test_is_transient_failure_flags_transport_and_breaker_failures() -> None:
    from kafka.errors import KafkaError

    assert is_transient_failure(CircuitBreakerOpen("api")) is True
    assert is_transient_failure(httpx.ConnectError("down")) is True
    assert is_transient_failure(httpx.ReadTimeout("slow")) is True
    assert is_transient_failure(ConnectionResetError("reset")) is True
    assert is_transient_failure(KafkaError("broker gone")) is True


def test_is_transient_failure_treats_record_errors_as_permanent() -> None:
    from elastic_transport import ApiResponseMeta
    from elasticsearch import exceptions as es_exceptions

    meta = ApiResponseMeta(
        status=400, http_version="1.1", headers={}, duration=0.0, node=None
    )
    bad_request = es_exceptions.BadRequestError("bad", meta, {})

    assert is_transient_failure(ValueError("bad record")) is False
    assert is_transient_failure(bad_request) is False


def test_is_transient_failure_follows_the_exception_cause_chain() -> None:
    """RouteBatchError wraps the real cause; that is what classifies the failure."""
    cause = httpx.ConnectError("api down")
    wrapped = RuntimeError("routing failed")
    wrapped.__cause__ = cause

    assert is_transient_failure(wrapped) is True


def test_is_transient_failure_survives_a_cyclic_cause_chain() -> None:
    first = RuntimeError("a")
    second = RuntimeError("b")
    first.__cause__ = second
    second.__cause__ = first

    assert is_transient_failure(first) is False


