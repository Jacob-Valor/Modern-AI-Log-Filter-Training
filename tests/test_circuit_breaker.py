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
