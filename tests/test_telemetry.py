"""Tests for OpenTelemetry soft-fallback helpers."""

from __future__ import annotations

from logfilter import telemetry


def test_telemetry_helpers_are_safe_without_optional_dependencies(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    assert telemetry.setup_tracing() is False
    assert telemetry.inject_kafka_headers() == []
    assert telemetry.extract_kafka_context([]) is None
    assert telemetry.inject_http_headers({"X-API-Token": "token"}) == {"X-API-Token": "token"}
    assert telemetry.extract_http_context({}) is None

    with telemetry.start_as_current_span("test.noop") as span:
        span.set_attribute("key", "value")
        telemetry.record_exception(span, RuntimeError("boom"))


def test_kafka_header_injection_preserves_existing_headers(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    headers = telemetry.inject_kafka_headers([("existing", b"value")])

    assert headers == [("existing", b"value")]


def test_http_header_injection_preserves_auth_header(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    headers = telemetry.inject_http_headers({"X-API-Token": "secret"})

    assert headers["X-API-Token"] == "secret"


def test_traced_decorator_sync_no_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    @telemetry.traced(name="test.sync")
    def sample_func():
        return 42

    assert sample_func() == 42


def test_traced_decorator_async_no_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    @telemetry.traced(name="test.async")
    async def sample_async():
        return 42

    import asyncio

    assert asyncio.run(sample_async()) == 42


def test_get_tracer_returns_none_without_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    assert telemetry.get_tracer() is None


def test_setup_tracing_idempotent(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    assert telemetry.setup_tracing() is False
    assert telemetry.setup_tracing() is False


def test_instrument_kafka_clients_returns_false_without_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    assert telemetry.instrument_kafka_clients() is False


def test_attach_context_returns_none_without_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    assert telemetry.attach_context(None) is None


def test_attach_span_context_returns_none_without_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    assert telemetry.attach_span_context(None, None) is None


def test_detach_context_noop_without_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    telemetry.detach_context(None)


def test_set_span_attributes_with_none_span() -> None:
    telemetry.set_span_attributes(None, {"key": "value"})


def test_set_span_attributes_with_none_attributes() -> None:
    telemetry.set_span_attributes(telemetry._NoopSpan(), None)


def test_record_exception_with_none_span() -> None:
    telemetry.record_exception(None, RuntimeError("boom"))


def test_instrument_fastapi_app_returns_false_without_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    assert telemetry.instrument_fastapi_app(None) is False


def test_start_as_current_span_with_exception_no_telemetry(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    with telemetry.start_as_current_span("test"):
        pass


def test_traced_decorator_uses_module_name(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    def sample():
        return 1

    sample.__qualname__ = "SampleClass.method"
    decorated = telemetry.traced()(sample)
    assert decorated() == 1


def test_traced_decorator_uses_fallback_name(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)

    class Callable:
        def __call__(self):
            return 1

    obj = Callable()
    decorated = telemetry.traced()(obj)
    assert decorated() == 1


def test_noop_span_set_attribute() -> None:
    span = telemetry._NoopSpan()
    span.set_attribute("key", "value")


def test_noop_span_set_attributes() -> None:
    span = telemetry._NoopSpan()
    span.set_attributes({"key": "value"})


def test_noop_span_record_exception() -> None:
    span = telemetry._NoopSpan()
    span.record_exception(RuntimeError("test"))


def test_noop_span_set_status() -> None:
    span = telemetry._NoopSpan()
    span.set_status("ok")


def test_noop_span_context_manager() -> None:
    span = telemetry._NoopSpan()
    with span:
        pass


def test_set_span_attributes_skips_none_values(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    span = telemetry._NoopSpan()
    telemetry.set_span_attributes(span, {"key": None, "other": "value"})


def test_inject_http_headers_returns_empty_without_input(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    result = telemetry.inject_http_headers()
    assert result == {}


def test_inject_kafka_headers_returns_empty_without_input(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    result = telemetry.inject_kafka_headers()
    assert result == []


def test_extract_kafka_context_returns_none_with_none(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    result = telemetry.extract_kafka_context(None)
    assert result is None


def test_extract_http_context_returns_none_with_none(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", False)
    result = telemetry.extract_http_context(None)
    assert result is None
