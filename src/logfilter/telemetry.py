"""OpenTelemetry helpers for logfilter runtime tracing.

The module is intentionally safe to import without OpenTelemetry installed. In
that case all helpers degrade to no-ops while preserving the same call shape.
"""

from __future__ import annotations

import asyncio
import functools
import importlib
import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

context: Any
propagate: Any
trace: Any
OTLPSpanExporter: Any
FastAPIInstrumentor: Any
KafkaInstrumentor: Any
Resource: Any
TracerProvider: Any
BatchSpanProcessor: Any
Status: Any
StatusCode: Any
Context = Any
Span = Any

try:  # pragma: no cover - exercised in environments with optional deps installed
    _otel = importlib.import_module("opentelemetry")
    context = importlib.import_module("opentelemetry.context")
    propagate = importlib.import_module("opentelemetry.propagate")
    trace = importlib.import_module("opentelemetry.trace")
    OTLPSpanExporter = importlib.import_module(
        "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    ).OTLPSpanExporter
    FastAPIInstrumentor = importlib.import_module(
        "opentelemetry.instrumentation.fastapi"
    ).FastAPIInstrumentor
    KafkaInstrumentor = importlib.import_module(
        "opentelemetry.instrumentation.kafka"
    ).KafkaInstrumentor
    Resource = importlib.import_module("opentelemetry.sdk.resources").Resource
    TracerProvider = importlib.import_module("opentelemetry.sdk.trace").TracerProvider
    BatchSpanProcessor = importlib.import_module(
        "opentelemetry.sdk.trace.export"
    ).BatchSpanProcessor
    _trace_api = importlib.import_module("opentelemetry.trace")
    Status = _trace_api.Status
    StatusCode = _trace_api.StatusCode
    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - default in minimal test environments
    context = None  # type: ignore[assignment]
    propagate = None  # type: ignore[assignment]
    trace = None  # type: ignore[assignment]
    Context = Any  # type: ignore[misc,assignment]
    FastAPIInstrumentor = None  # type: ignore[assignment]
    KafkaInstrumentor = None  # type: ignore[assignment]
    OTLPSpanExporter = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    TracerProvider = None  # type: ignore[assignment]
    BatchSpanProcessor = None  # type: ignore[assignment]
    Span = Any  # type: ignore[misc,assignment]
    Status = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]
    OTEL_AVAILABLE = False

_SETUP_LOCK = threading.Lock()
_TRACING_INITIALIZED = False
_KAFKA_INSTRUMENTED = False
_FASTAPI_INSTRUMENTED_ATTR = "_logfilter_otel_instrumented"


class _NoopSpan:
    def set_attribute(self, key: str, value: Any) -> None:
        del key, value

    def set_attributes(self, attributes: Mapping[str, Any] | None) -> None:
        del attributes

    def record_exception(self, exception: BaseException) -> None:
        del exception

    def set_status(self, status: Any) -> None:
        del status

    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback


def setup_tracing(service_name: str | None = None, endpoint: str | None = None) -> bool:
    """Initialise the global tracer provider once with an OTLP exporter."""
    global _TRACING_INITIALIZED

    if not OTEL_AVAILABLE:
        return False

    if _TRACING_INITIALIZED:
        return True

    with _SETUP_LOCK:
        if _TRACING_INITIALIZED:
            return True

        resolved_service = service_name or os.environ.get("OTEL_SERVICE_NAME", "logfilter")
        resolved_endpoint = endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        resource = Resource.create({"service.name": resolved_service})
        provider = TracerProvider(resource=resource)
        exporter = (
            OTLPSpanExporter(endpoint=resolved_endpoint)
            if resolved_endpoint
            else OTLPSpanExporter()
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        try:
            trace.set_tracer_provider(provider)
        except Exception as exc:  # noqa: BLE001
            logger.debug("OpenTelemetry tracer provider already configured", error=str(exc))
        _TRACING_INITIALIZED = True
        logger.info("OpenTelemetry tracing configured", service_name=resolved_service)
        instrument_kafka_clients()
        return True


def instrument_kafka_clients() -> bool:
    """Apply kafka-python auto-instrumentation once when available."""
    global _KAFKA_INSTRUMENTED

    if not OTEL_AVAILABLE or KafkaInstrumentor is None:
        return False
    if _KAFKA_INSTRUMENTED:
        return True
    try:
        KafkaInstrumentor().instrument()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Kafka auto-instrumentation skipped", error=str(exc))
        return False
    _KAFKA_INSTRUMENTED = True
    return True


def get_tracer(name: str = "logfilter") -> Any:
    if not OTEL_AVAILABLE:
        return None
    setup_tracing()
    return trace.get_tracer(name)


@contextmanager
def start_as_current_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
    trace_context: Context | None = None,
) -> Iterator[Span | _NoopSpan]:
    """Start a span or yield a no-op span when telemetry is unavailable."""
    if not OTEL_AVAILABLE:
        yield _NoopSpan()
        return

    tracer = get_tracer("logfilter")
    with tracer.start_as_current_span(name, context=trace_context) as span:
        set_span_attributes(span, attributes)
        try:
            yield span
        except Exception as exc:
            record_exception(span, exc)
            raise


def traced(
    name: str | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for lightweight sync or async function spans."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func_name = getattr(func, "__qualname__", getattr(func, "__name__", "callable"))
        span_name = name or f"{func.__module__}.{func_name}"

        if asyncio.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with start_as_current_span(span_name, attributes):
                    return await func(*args, **kwargs)

            return async_wrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with start_as_current_span(span_name, attributes):
                return func(*args, **kwargs)

        return wrapper

    return decorator


if OTEL_AVAILABLE:

    class _KafkaHeaderSetter:
        def set(self, carrier: list[tuple[str, bytes]], key: str, value: str) -> None:
            encoded = value.encode("utf-8")
            carrier[:] = [(k, v) for k, v in carrier if k.lower() != key.lower()]
            carrier.append((key, encoded))


    class _KafkaHeaderGetter:
        def get(self, carrier: list[tuple[str, bytes]], key: str) -> list[str] | None:
            values = [
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for header_key, value in carrier
                if header_key.lower() == key.lower()
            ]
            return values or None

        def keys(self, carrier: list[tuple[str, bytes]]) -> list[str]:
            return [key for key, _ in carrier]


    _kafka_setter = _KafkaHeaderSetter()
    _kafka_getter = _KafkaHeaderGetter()
else:
    _kafka_setter = None
    _kafka_getter = None


def inject_kafka_headers(
    headers: list[tuple[str, bytes]] | None = None,
    span: Span | None = None,
) -> list[tuple[str, bytes]]:
    """Inject W3C trace context into kafka-python message headers."""
    carrier = list(headers or [])
    if not OTEL_AVAILABLE:
        return carrier
    propagation_context = trace.set_span_in_context(span) if span is not None else None
    propagate.inject(carrier, context=propagation_context, setter=_kafka_setter)
    return carrier


def extract_kafka_context(headers: list[tuple[str, bytes]] | None) -> Context | None:
    """Extract W3C trace context from kafka-python message headers."""
    if not OTEL_AVAILABLE:
        return None
    return propagate.extract(list(headers or []), getter=_kafka_getter)


def inject_http_headers(headers: Mapping[str, str] | None = None) -> dict[str, str]:
    """Inject trace context into HTTP request headers."""
    carrier = dict(headers or {})
    if OTEL_AVAILABLE:
        propagate.inject(carrier)
    return carrier


def extract_http_context(headers: Mapping[str, str] | None) -> Context | None:
    """Extract trace context from inbound HTTP headers."""
    if not OTEL_AVAILABLE:
        return None
    return propagate.extract(headers or {})


def attach_context(trace_context: Context | None) -> object | None:
    if not OTEL_AVAILABLE or trace_context is None:
        return None
    return context.attach(trace_context)


def attach_span_context(span: Span, extracted: Context | None) -> object | None:
    if not OTEL_AVAILABLE or extracted is None:
        return None
    return context.attach(trace.set_span_in_context(span, extracted))


def detach_context(token: object | None) -> None:
    if OTEL_AVAILABLE and token is not None:
        context.detach(token)


def set_span_attributes(
    span: Span | _NoopSpan | None,
    attributes: Mapping[str, Any] | None,
) -> None:
    if span is None or not attributes:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        span.set_attribute(key, value)


def record_exception(span: Span | _NoopSpan | None, exception: BaseException) -> None:
    if span is None:
        return
    span.record_exception(exception)
    if OTEL_AVAILABLE:
        span.set_status(Status(StatusCode.ERROR, str(exception)))


def instrument_fastapi_app(app: Any) -> bool:
    """Apply FastAPI auto-instrumentation once per app instance."""
    if not OTEL_AVAILABLE or FastAPIInstrumentor is None:
        return False
    setup_tracing()
    if getattr(app.state, _FASTAPI_INSTRUMENTED_ATTR, False):
        return True
    FastAPIInstrumentor().instrument_app(app)
    setattr(app.state, _FASTAPI_INSTRUMENTED_ATTR, True)
    return True
