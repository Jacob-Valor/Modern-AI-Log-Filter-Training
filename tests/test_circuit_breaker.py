"""Tests for the circuit breaker utility."""

from __future__ import annotations

import time

import pytest

from logfilter.utils.circuit_breaker import CircuitBreaker, CircuitBreakerOpen, State


def test_cb_starts_closed() -> None:
    cb = CircuitBreaker("test")
    assert cb.state is State.CLOSED


def test_cb_opens_after_failures() -> None:
    cb = CircuitBreaker("test", failure_threshold=2)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state is State.CLOSED
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state is State.OPEN


def test_cb_fails_fast_when_open() -> None:
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60.0)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state is State.OPEN
    with pytest.raises(CircuitBreakerOpen):
        cb.call(lambda: 42)


def test_cb_half_open_then_closes() -> None:
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.0, half_open_max_calls=2)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state is State.OPEN
    time.sleep(0.01)
    cb.call(lambda: 42)
    assert cb.state is State.HALF_OPEN
    cb.call(lambda: 42)
    assert cb.state is State.CLOSED


def test_cb_half_open_reopens_on_failure() -> None:
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.0, half_open_max_calls=3)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    time.sleep(0.01)
    cb.call(lambda: 42)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state is State.OPEN


def test_cb_success_reduces_failure_count() -> None:
    cb = CircuitBreaker("test", failure_threshold=3)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb._failure_count == 1
    cb.call(lambda: 42)
    assert cb._failure_count == 0


def test_cb_to_dict() -> None:
    cb = CircuitBreaker("test", failure_threshold=5)
    d = cb.to_dict()
    assert d["name"] == "test"
    assert d["state"] == "closed"
    assert d["failure_threshold"] == 5


def test_cb_open_no_reset_fails_fast() -> None:
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60.0)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state is State.OPEN
    with pytest.raises(CircuitBreakerOpen):
        cb.call(lambda: 42)


def test_cb_half_open_max_calls_reached() -> None:
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=0.0, half_open_max_calls=1)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state is State.OPEN
    time.sleep(0.01)
    cb.call(lambda: 42)
    assert cb.state is State.CLOSED


def test_cb_success_in_closed_reduces_failure_count() -> None:
    cb = CircuitBreaker("test", failure_threshold=3)
    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb._failure_count == 1
    cb.call(lambda: 42)
    assert cb._failure_count == 0
    cb.call(lambda: 42)
    assert cb._failure_count == 0


def test_cb_decorator_usage() -> None:
    cb = CircuitBreaker("test", failure_threshold=1, recovery_timeout=60.0)

    @cb
    def failing() -> None:
        raise RuntimeError("fail")

    with pytest.raises(RuntimeError):
        failing()
    assert cb.state is State.OPEN
    with pytest.raises(CircuitBreakerOpen):
        failing()


def test_cb_call_with_args_and_kwargs() -> None:
    cb = CircuitBreaker("test")

    def add(a, b, c=0):
        return a + b + c

    result = cb.call(add, 1, 2, c=3)
    assert result == 6


def test_cb_expected_exception_tuple() -> None:
    cb = CircuitBreaker("test", failure_threshold=1, expected_exception=(RuntimeError, ValueError))

    with pytest.raises(RuntimeError):
        cb.call(lambda: (_ for _ in ()).throw(RuntimeError("fail")))
    assert cb.state is State.OPEN


def test_cb_unexpected_exception_propagates() -> None:
    cb = CircuitBreaker("test", failure_threshold=1, expected_exception=RuntimeError)

    with pytest.raises(ValueError):
        cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
    assert cb.state is State.CLOSED
