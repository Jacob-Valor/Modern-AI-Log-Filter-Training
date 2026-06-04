"""Tests for OpenTelemetry soft-fallback helpers."""

from __future__ import annotations

import pytest

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


def test_setup_tracing_requires_explicit_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert telemetry.setup_tracing() is False
    assert telemetry.get_tracer() is None

    with telemetry.start_as_current_span("test.no_endpoint") as span:
        assert isinstance(span, telemetry._NoopSpan)


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


def test_setup_tracing_with_otel_available(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.setattr(telemetry, "_KAFKA_INSTRUMENTED", False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    fake_resource = type("Resource", (), {"create": staticmethod(lambda d: d)})()
    fake_provider = type("TracerProvider", (), {
        "add_span_processor": lambda self, p: None,
        "resource": None,
    })()
    fake_exporter = type("OTLPSpanExporter", (), {})()
    fake_processor = type("BatchSpanProcessor", (), {})()
    fake_trace = type("trace", (), {
        "set_tracer_provider": lambda p: None,
        "set_span_in_context": lambda span, context=None: None,
    })()
    fake_propagate = type(
        "propagate", (), {"inject": lambda *a, **k: None, "extract": lambda *a, **k: None}
    )()
    fake_context = type("context", (), {"attach": lambda c: "token", "detach": lambda t: None})()
    fake_instrumentor = type("KafkaInstrumentor", (), {"instrument": lambda self: None})()
    fake_fastapi_instrumentor = type(
        "FastAPIInstrumentor", (), {"instrument_app": lambda self, app: None}
    )()

    monkeypatch.setattr(telemetry, "Resource", fake_resource)
    monkeypatch.setattr(telemetry, "TracerProvider", lambda resource: fake_provider)
    monkeypatch.setattr(telemetry, "OTLPSpanExporter", lambda endpoint: fake_exporter)
    monkeypatch.setattr(telemetry, "BatchSpanProcessor", lambda exporter: fake_processor)
    monkeypatch.setattr(telemetry, "trace", fake_trace)
    monkeypatch.setattr(telemetry, "propagate", fake_propagate)
    monkeypatch.setattr(telemetry, "context", fake_context)
    monkeypatch.setattr(telemetry, "KafkaInstrumentor", lambda: fake_instrumentor)
    monkeypatch.setattr(telemetry, "FastAPIInstrumentor", lambda: fake_fastapi_instrumentor)

    assert telemetry.setup_tracing() is True
    assert telemetry.setup_tracing() is True  # idempotent


def test_setup_tracing_with_service_name_and_endpoint(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.setattr(telemetry, "_KAFKA_INSTRUMENTED", False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    fake_resource = type("Resource", (), {"create": staticmethod(lambda d: d)})()
    fake_provider = type("TracerProvider", (), {"add_span_processor": lambda self, p: None})()
    fake_trace = type("trace", (), {"set_tracer_provider": lambda p: None})()
    fake_propagate = type("propagate", (), {"inject": lambda *a, **k: None})()
    fake_context = type("context", (), {"attach": lambda c: "token", "detach": lambda t: None})()
    fake_instrumentor = type("KafkaInstrumentor", (), {"instrument": lambda self: None})()

    monkeypatch.setattr(telemetry, "Resource", fake_resource)
    monkeypatch.setattr(telemetry, "TracerProvider", lambda resource: fake_provider)
    monkeypatch.setattr(telemetry, "OTLPSpanExporter", lambda endpoint: object())
    monkeypatch.setattr(telemetry, "BatchSpanProcessor", lambda exporter: object())
    monkeypatch.setattr(telemetry, "trace", fake_trace)
    monkeypatch.setattr(telemetry, "propagate", fake_propagate)
    monkeypatch.setattr(telemetry, "context", fake_context)
    monkeypatch.setattr(telemetry, "KafkaInstrumentor", lambda: fake_instrumentor)

    assert telemetry.setup_tracing(service_name="test", endpoint="http://otel:4317") is True


def test_setup_tracing_trace_provider_already_configured(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.setattr(telemetry, "_KAFKA_INSTRUMENTED", False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    fake_resource = type("Resource", (), {"create": staticmethod(lambda d: d)})()
    fake_provider = type("TracerProvider", (), {"add_span_processor": lambda self, p: None})()

    def raise_exc(p):
        raise RuntimeError("already set")

    fake_trace = type("trace", (), {"set_tracer_provider": raise_exc})()
    fake_propagate = type("propagate", (), {"inject": lambda *a, **k: None})()
    fake_context = type("context", (), {"attach": lambda c: "token", "detach": lambda t: None})()
    fake_instrumentor = type("KafkaInstrumentor", (), {"instrument": lambda self: None})()

    monkeypatch.setattr(telemetry, "Resource", fake_resource)
    monkeypatch.setattr(telemetry, "TracerProvider", lambda resource: fake_provider)
    monkeypatch.setattr(telemetry, "OTLPSpanExporter", lambda endpoint: object())
    monkeypatch.setattr(telemetry, "BatchSpanProcessor", lambda exporter: object())
    monkeypatch.setattr(telemetry, "trace", fake_trace)
    monkeypatch.setattr(telemetry, "propagate", fake_propagate)
    monkeypatch.setattr(telemetry, "context", fake_context)
    monkeypatch.setattr(telemetry, "KafkaInstrumentor", lambda: fake_instrumentor)

    assert telemetry.setup_tracing() is True


def test_instrument_kafka_clients_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_KAFKA_INSTRUMENTED", False)
    fake_instrumentor = type("KafkaInstrumentor", (), {"instrument": lambda self: None})()
    monkeypatch.setattr(telemetry, "KafkaInstrumentor", lambda: fake_instrumentor)

    assert telemetry.instrument_kafka_clients() is True
    assert telemetry.instrument_kafka_clients() is True  # idempotent


def test_instrument_kafka_clients_exception(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_KAFKA_INSTRUMENTED", False)

    class FailingInstrumentor:
        def instrument(self) -> None:
            raise RuntimeError("fail")

    monkeypatch.setattr(telemetry, "KafkaInstrumentor", FailingInstrumentor)

    assert telemetry.instrument_kafka_clients() is False


def test_get_tracer_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    fake_tracer = object()
    fake_trace = type(
        "trace", (), {"get_tracer": lambda name, instrumenting_module_version=None: fake_tracer}
    )()
    monkeypatch.setattr(telemetry, "trace", fake_trace)

    assert telemetry.get_tracer() is fake_tracer


def test_get_tracer_returns_none_when_setup_fails(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert telemetry.get_tracer() is None


def test_start_as_current_span_with_otel_tracer_none(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    with telemetry.start_as_current_span("test") as span:
        assert isinstance(span, telemetry._NoopSpan)


def test_start_as_current_span_with_otel_and_exception(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    fake_span = type("Span", (), {
        "set_attribute": lambda self, k, v: None,
        "set_attributes": lambda self, d: None,
        "record_exception": lambda self, e: None,
        "set_status": lambda self, s: None,
        "__enter__": lambda self: self,
        "__exit__": lambda self, *a: None,
    })()

    class FakeTracer:
        def start_as_current_span(self, name, context=None):
            return type("CtxMgr", (), {
                "__enter__": lambda self: fake_span,
                "__exit__": lambda self, *a: None,
            })()

    fake_trace = type(
        "trace", (), {"get_tracer": lambda name, instrumenting_module_version=None: FakeTracer()}
    )()
    monkeypatch.setattr(telemetry, "trace", fake_trace)

    with pytest.raises(RuntimeError):
        with telemetry.start_as_current_span("test"):
            raise RuntimeError("boom")


def test_inject_kafka_headers_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)

    class FakePropagate:
        @staticmethod
        def inject(carrier, context=None, setter=None):
            carrier.append(("trace", b"id"))

    monkeypatch.setattr(telemetry, "propagate", FakePropagate())
    monkeypatch.setattr(
        telemetry, "trace", type("trace", (), {"set_span_in_context": lambda span: None})()
    )

    headers = telemetry.inject_kafka_headers([("existing", b"value")], span=None)
    assert ("existing", b"value") in headers


def test_extract_kafka_context_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)

    fake_context = object()

    class FakePropagateExtract:
        @staticmethod
        def extract(carrier, getter=None):
            return fake_context

    monkeypatch.setattr(telemetry, "propagate", FakePropagateExtract())

    result = telemetry.extract_kafka_context([("trace", b"id")])
    assert result is fake_context


def test_inject_http_headers_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)

    class FakePropagateInject:
        @staticmethod
        def inject(carrier, context=None):
            carrier.update({"trace": "id"})

    monkeypatch.setattr(telemetry, "propagate", FakePropagateInject())

    headers = telemetry.inject_http_headers({"existing": "value"})
    assert headers["existing"] == "value"
    assert headers["trace"] == "id"


def test_extract_http_context_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)

    fake_context = object()

    class FakePropagateExtract:
        @staticmethod
        def extract(carrier, context=None):
            return fake_context

    monkeypatch.setattr(telemetry, "propagate", FakePropagateExtract())

    result = telemetry.extract_http_context({"trace": "id"})
    assert result is fake_context


def test_attach_context_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)

    class FakeContext:
        @staticmethod
        def attach(c):
            return "token"

    monkeypatch.setattr(telemetry, "context", FakeContext())

    assert telemetry.attach_context({"trace": "id"}) == "token"


def test_attach_span_context_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)

    class FakeTrace:
        @staticmethod
        def set_span_in_context(span, extracted):
            return {"ctx": "data"}

    class FakeContext:
        @staticmethod
        def attach(c):
            return "token"

    monkeypatch.setattr(telemetry, "trace", FakeTrace())
    monkeypatch.setattr(telemetry, "context", FakeContext())

    assert telemetry.attach_span_context(None, {"trace": "id"}) == "token"


def test_detach_context_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)

    class FakeContext:
        @staticmethod
        def detach(t):
            pass

    monkeypatch.setattr(telemetry, "context", FakeContext())

    telemetry.detach_context("token")


def test_record_exception_with_otel_span(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)

    class FakeStatus:
        def __init__(self, code, message):
            self.code = code
            self.message = message

    class FakeStatusCode:
        ERROR = "error"

    monkeypatch.setattr(telemetry, "Status", FakeStatus)
    monkeypatch.setattr(telemetry, "StatusCode", FakeStatusCode)

    class FakeSpan:
        def record_exception(self, exc):
            pass

        def set_status(self, status):
            pass

    telemetry.record_exception(FakeSpan(), RuntimeError("boom"))


def test_instrument_fastapi_app_with_otel(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    fake_trace = type("trace", (), {"set_tracer_provider": lambda p: None})()
    fake_propagate = type("propagate", (), {"inject": lambda *a, **k: None})()
    fake_context = type("context", (), {"attach": lambda c: "token", "detach": lambda t: None})()
    fake_instrumentor = type("KafkaInstrumentor", (), {"instrument": lambda self: None})()

    monkeypatch.setattr(telemetry, "trace", fake_trace)
    monkeypatch.setattr(telemetry, "propagate", fake_propagate)
    monkeypatch.setattr(telemetry, "context", fake_context)
    monkeypatch.setattr(telemetry, "KafkaInstrumentor", lambda: fake_instrumentor)

    class FakeApp:
        class state:
            pass

    fake_fastapi = type("FastAPIInstrumentor", (), {"instrument_app": lambda self, app: None})()
    monkeypatch.setattr(telemetry, "FastAPIInstrumentor", lambda: fake_fastapi)

    assert telemetry.instrument_fastapi_app(FakeApp()) is True
    assert telemetry.instrument_fastapi_app(FakeApp()) is True  # idempotent


def test_instrument_fastapi_app_returns_false_without_tracing(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)
    monkeypatch.setattr(telemetry, "_TRACING_INITIALIZED", False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)

    assert telemetry.instrument_fastapi_app(None) is False


def test_kafka_header_setter_and_getter(monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "OTEL_AVAILABLE", True)

    carrier = [("existing", b"value")]
    telemetry._kafka_setter.set(carrier, "trace", "id")
    assert ("trace", b"id") in carrier

    result = telemetry._kafka_getter.get(carrier, "trace")
    assert result == ["id"]

    result_none = telemetry._kafka_getter.get(carrier, "missing")
    assert result_none is None

    keys = telemetry._kafka_getter.keys(carrier)
    assert "existing" in keys
    assert "trace" in keys
