"""Circuit breaker pattern for resilient external service calls.

States:
  CLOSED   — requests pass through normally
  OPEN     — requests fail fast immediately
  HALF_OPEN — one probe request allowed to test if service recovered

Usage:
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30)
    with breaker:
        result = es.bulk(body=docs)
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when a call is rejected because the circuit is OPEN."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Circuit breaker '{name}' is OPEN")
        self.name = name


class CircuitBreaker:
    """Thread-safe circuit breaker for external service resilience.

    Parameters
    ----------
    name : str
        Identifier for logging.
    failure_threshold : int
        Number of consecutive failures before opening the circuit.
    recovery_timeout : float
        Seconds to wait before allowing a probe request (HALF_OPEN).
    half_open_max_calls : int
        Number of successful probe calls required to close the circuit.
    expected_exception : type[Exception] | tuple[type[Exception], ...]
        Exception types that count as failures. Other exceptions propagate.
    """

    def __init__(
        self,
        name: str = "default",
        *,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        expected_exception: type[Exception] | tuple[type[Exception], ...] = Exception,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.expected_exception = expected_exception

        self._state = State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.RLock()

    @property
    def state(self) -> State:
        with self._lock:
            return self._state

    def _open(self) -> None:
        self._state = State.OPEN
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = time.monotonic()
        logger.warning(
            "Circuit breaker opened",
            name=self.name,
            threshold=self.failure_threshold,
        )

    def _close(self) -> None:
        self._state = State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        logger.info("Circuit breaker closed", name=self.name)

    def _half_open(self) -> None:
        self._state = State.HALF_OPEN
        self._success_count = 0
        logger.info("Circuit breaker half-open (probing)", name=self.name)

    def _should_attempt_reset(self) -> bool:
        return time.monotonic() - self._last_failure_time >= self.recovery_timeout

    def call(self, fn, *args, **kwargs):  # noqa: ANN001, ANN202
        """Call ``fn`` through the circuit breaker, returning its result."""
        with self._protect():
            return fn(*args, **kwargs)

    @contextmanager
    def _protect(self):
        with self._lock:
            if self._state is State.OPEN:
                if self._should_attempt_reset():
                    self._half_open()
                else:
                    raise CircuitBreakerOpen(self.name)

            if self._state is State.HALF_OPEN and self._success_count >= self.half_open_max_calls:
                raise CircuitBreakerOpen(self.name)

        try:
            yield
        except self.expected_exception:
            with self._lock:
                self._failure_count += 1
                self._last_failure_time = time.monotonic()
                if self._state is State.HALF_OPEN:
                    self._open()
                elif self._failure_count >= self.failure_threshold:
                    self._open()
            raise
        else:
            with self._lock:
                if self._state is State.HALF_OPEN:
                    self._success_count += 1
                    if self._success_count >= self.half_open_max_calls:
                        self._close()
                elif self._state is State.CLOSED:
                    self._failure_count = max(0, self._failure_count - 1)

    def __call__(self, fn):  # noqa: ANN001, ANN204
        """Decorator usage: ``@breaker`` on a function."""

        def wrapper(*args, **kwargs):  # noqa: ANN001, ANN002, ANN202
            with self._protect():
                return fn(*args, **kwargs)

        return wrapper

    def to_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
                "last_failure_time": self._last_failure_time,
            }
