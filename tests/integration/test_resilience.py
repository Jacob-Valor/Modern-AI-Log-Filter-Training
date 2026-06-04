"""Integration tests for circuit breaker resilience under load.

These tests exercise the circuit breaker in realistic failure scenarios,
verifying state transitions and fast-fail behaviour without connecting to
real external services.
"""

from __future__ import annotations

import time

import pytest

from logfilter.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, State


class FlakyDependency:
    """Mock downstream service that fails a configurable number of times."""

    def __init__(self, fail_count: int = 0) -> None:
        self._calls = 0
        self._fail_count = fail_count

    def call(self) -> str:
        self._calls += 1
        if self._calls <= self._fail_count:
            raise ConnectionError(f"downstream failure #{self._calls}")
        return "ok"

    def reset(self) -> None:
        self._calls = 0


def _fail() -> None:
    raise ConnectionError("boom")


def _ok() -> str:
    return "ok"


# ── State transition tests ──────────────────────────────────────────────────


def test_closed_to_open_to_half_open_to_closed() -> None:
    """CLOSED -> OPEN -> HALF_OPEN -> CLOSED on sustained success."""
    cb = CircuitBreaker(
        "test",
        failure_threshold=2,
        recovery_timeout=0.05,
        half_open_max_calls=2,
        expected_exception=ConnectionError,
    )

    # CLOSED: first failure
    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.CLOSED

    # CLOSED: second failure -> OPEN
    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.OPEN

    # OPEN: fast-fail
    with pytest.raises(CircuitBreakerOpen):
        cb.call(_ok)

    # Wait for recovery timeout -> HALF_OPEN
    time.sleep(0.06)

    # HALF_OPEN: first probe succeeds
    assert cb.call(_ok) == "ok"
    assert cb.state is State.HALF_OPEN

    # HALF_OPEN: second probe succeeds -> CLOSED
    assert cb.call(_ok) == "ok"
    assert cb.state is State.CLOSED

    # CLOSED: normal operation resumes
    assert cb.call(_ok) == "ok"


def test_closed_to_open_to_half_open_to_open() -> None:
    """CLOSED -> OPEN -> HALF_OPEN -> OPEN when probe fails."""
    cb = CircuitBreaker(
        "test",
        failure_threshold=1,
        recovery_timeout=0.05,
        half_open_max_calls=3,
        expected_exception=ConnectionError,
    )

    # CLOSED -> OPEN
    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.OPEN

    # Wait for recovery
    time.sleep(0.06)

    # HALF_OPEN -> OPEN on probe failure
    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.OPEN

    # Still fast-fails
    with pytest.raises(CircuitBreakerOpen):
        cb.call(_ok)


def test_success_while_closed() -> None:
    """Calls succeed while the breaker is CLOSED."""
    cb = CircuitBreaker("test", expected_exception=ConnectionError)
    assert cb.call(_ok) == "ok"
    assert cb.call(_ok) == "ok"
    assert cb.state is State.CLOSED


def test_raises_circuit_breaker_open_while_open() -> None:
    """CircuitBreakerOpen is raised immediately while OPEN."""
    cb = CircuitBreaker(
        "test",
        failure_threshold=1,
        recovery_timeout=60.0,
        expected_exception=ConnectionError,
    )
    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.OPEN

    with pytest.raises(CircuitBreakerOpen):
        cb.call(_ok)


def test_half_open_allows_limited_calls() -> None:
    """HALF_OPEN allows exactly half_open_max_calls before deciding."""
    cb = CircuitBreaker(
        "test",
        failure_threshold=1,
        recovery_timeout=0.05,
        half_open_max_calls=2,
        expected_exception=ConnectionError,
    )

    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.OPEN

    time.sleep(0.06)

    # First probe allowed
    assert cb.call(_ok) == "ok"
    assert cb.state is State.HALF_OPEN

    # Second probe allowed and closes circuit
    assert cb.call(_ok) == "ok"
    assert cb.state is State.CLOSED


def test_half_open_blocks_excess_calls() -> None:
    """Additional calls in HALF_OPEN are rejected once quota is exhausted."""
    cb = CircuitBreaker(
        "test",
        failure_threshold=1,
        recovery_timeout=0.05,
        half_open_max_calls=1,
        expected_exception=ConnectionError,
    )

    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.OPEN

    time.sleep(0.06)

    # First probe succeeds and closes circuit immediately because max_calls == 1
    assert cb.call(_ok) == "ok"
    assert cb.state is State.CLOSED

    # Re-open to test blocking in half-open
    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.OPEN

    time.sleep(0.06)

    # Monkey-patch internal state to HALF_OPEN with exhausted quota
    cb._state = State.HALF_OPEN
    cb._success_count = 1

    with pytest.raises(CircuitBreakerOpen):
        cb.call(_ok)


# ── Load-style resilience test ──────────────────────────────────────────────


def test_repeated_downstream_failures_drive_breaker() -> None:
    """Simulate a flaky Elasticsearch-like dependency under load.

    The dependency fails 5 times, recovers, and the breaker should:
      - open after the threshold,
      - fast-fail subsequent attempts,
      - allow probes after recovery_timeout,
      - close once enough probes succeed.
    """
    dependency = FlakyDependency(fail_count=5)
    cb = CircuitBreaker(
        "es-archive",
        failure_threshold=3,
        recovery_timeout=0.05,
        half_open_max_calls=2,
        expected_exception=ConnectionError,
    )

    results: list[str] = []
    exceptions: list[type[Exception]] = []

    # Drive 10 calls through the breaker against the failing dependency
    for _ in range(10):
        try:
            results.append(cb.call(dependency.call))
        except Exception as exc:  # noqa: BLE001
            exceptions.append(type(exc))

    # After 3 failures the breaker opens; remaining 7 calls fast-fail
    assert cb.state is State.OPEN
    assert exceptions.count(ConnectionError) == 3
    assert exceptions.count(CircuitBreakerOpen) == 7
    assert results == []

    # Wait for recovery window
    time.sleep(0.06)

    healthy = FlakyDependency(fail_count=0)

    # First probe succeeds -> HALF_OPEN
    assert cb.call(healthy.call) == "ok"
    assert cb.state is State.HALF_OPEN

    # Second probe succeeds -> CLOSED
    assert cb.call(healthy.call) == "ok"
    assert cb.state is State.CLOSED

    # Normal operation resumes
    for _ in range(5):
        assert cb.call(healthy.call) == "ok"


def test_decorator_usage_under_load() -> None:
    """Circuit breaker as a decorator around a flaky function."""
    cb = CircuitBreaker(
        "decorator-test",
        failure_threshold=2,
        recovery_timeout=0.05,
        half_open_max_calls=1,
        expected_exception=ConnectionError,
    )

    call_count = 0

    @cb
    def flaky() -> str:
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectionError("fail")
        return "ok"

    # Two failures -> OPEN
    with pytest.raises(ConnectionError):
        flaky()
    with pytest.raises(ConnectionError):
        flaky()
    assert cb.state is State.OPEN

    # Fast-fail
    with pytest.raises(CircuitBreakerOpen):
        flaky()

    time.sleep(0.06)

    # Probe succeeds -> CLOSED (half_open_max_calls == 1)
    assert flaky() == "ok"
    assert cb.state is State.CLOSED

    # Continue normally
    assert flaky() == "ok"


# ── Context-manager usage ───────────────────────────────────────────────────


def test_context_manager_usage() -> None:
    """Circuit breaker works via its internal ``_protect`` context manager."""
    cb = CircuitBreaker(
        "ctx-test",
        failure_threshold=1,
        recovery_timeout=0.05,
        half_open_max_calls=1,
        expected_exception=ConnectionError,
    )

    with pytest.raises(ConnectionError):
        with cb._protect():
            raise ConnectionError("fail")
    assert cb.state is State.OPEN

    time.sleep(0.06)

    with cb._protect():
        result = "ok"
    assert result == "ok"
    assert cb.state is State.CLOSED


# ── Thread-safety smoke test ────────────────────────────────────────────────


def test_concurrent_fast_fails_do_not_race() -> None:
    """Many rapid calls while OPEN all raise CircuitBreakerOpen cleanly."""
    cb = CircuitBreaker(
        "race-test",
        failure_threshold=1,
        recovery_timeout=60.0,
        expected_exception=ConnectionError,
    )

    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.OPEN

    errors = 0
    for _ in range(100):
        try:
            cb.call(_ok)
        except CircuitBreakerOpen:
            errors += 1

    assert errors == 100
    assert cb.state is State.OPEN


def test_expected_exception_filtering() -> None:
    """Only configured exceptions count as failures; others propagate untouched."""
    cb = CircuitBreaker(
        "filter-test",
        failure_threshold=1,
        recovery_timeout=60.0,
        expected_exception=ConnectionError,
    )

    # ValueError is NOT counted as a failure
    with pytest.raises(ValueError):
        cb.call(lambda: (_ for _ in ()).throw(ValueError("unexpected")))
    assert cb.state is State.CLOSED

    # ConnectionError IS counted
    with pytest.raises(ConnectionError):
        cb.call(_fail)
    assert cb.state is State.OPEN
