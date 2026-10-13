"""T13 regression: lifecycle and shutdown cleanup.

Verifies:
- Consumers use threading.Event for shutdown (no signal.signal inside run())
- API lifespan closes redis client and archiver on shutdown
- Collector joins its worker threads before exiting
"""

from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).parent.parent
_CONSUMER_SRC = (_ROOT / "src" / "logfilter" / "kafka" / "consumer.py").read_text()
_COLLECTOR_SRC = (_ROOT / "src" / "logfilter" / "collector.py").read_text()
_APP_SRC = (_ROOT / "src" / "logfilter" / "api" / "app.py").read_text()


class TestConsumerSignalSafety:
    """Consumer run() must not call signal.signal() — it may run in a thread."""

    def test_archive_consumer_no_signal_in_run(self) -> None:
        run_block = _extract_run_body(_CONSUMER_SRC, "ArchiveConsumer")
        assert "signal.signal" not in run_block, (
            "ArchiveConsumer.run() must not install signal handlers — "
            "use threading.Event for thread-safe shutdown"
        )

    def test_scorer_consumer_no_signal_in_run(self) -> None:
        run_block = _extract_run_body(_CONSUMER_SRC, "ScorerConsumer")
        assert "signal.signal" not in run_block, (
            "ScorerConsumer.run() must not install signal handlers — "
            "use threading.Event for thread-safe shutdown"
        )

    def test_archive_consumer_stop_method_exists(self) -> None:
        assert "def stop(" in _CONSUMER_SRC or "def shutdown(" in _CONSUMER_SRC, (
            "Consumer must expose stop() or shutdown() for external signal handling"
        )

    def test_archive_consumer_uses_stop_event(self) -> None:
        assert "stop_event" in _CONSUMER_SRC or "_shutdown_event" in _CONSUMER_SRC, (
            "Consumer must use a threading.Event for cooperative shutdown"
        )


class TestApiLifespanCleanup:
    """API lifespan shutdown must close redis client and archiver."""

    def test_lifespan_closes_redis(self) -> None:
        lifespan_block = _extract_lifespan_shutdown(_APP_SRC)
        assert "redis" in lifespan_block.lower() and (
            "close" in lifespan_block.lower() or "aclose" in lifespan_block.lower()
        ), "Lifespan shutdown must close the redis client to release connections"

    def test_lifespan_closes_archiver(self) -> None:
        lifespan_block = _extract_lifespan_shutdown(_APP_SRC)
        assert "archiver" in lifespan_block.lower() and (
            "close" in lifespan_block.lower() or "aclose" in lifespan_block.lower()
        ), "Lifespan shutdown must close the ES archiver to release connections"


class TestCollectorThreadJoin:
    """Collector must join its worker threads on shutdown."""

    def test_collector_joins_threads(self) -> None:
        run_block = _extract_collector_run(_COLLECTOR_SRC)
        assert ".join(" in run_block, (
            "Collector.run() must join worker threads before exiting — "
            "daemon threads alone risk losing in-flight data"
        )


# ── helpers ──────────────────────────────────────────────────────────────


def _extract_run_body(source: str, class_name: str) -> str:
    """Return the body of ``<class_name>.run``.

    Raises instead of returning "" on a miss. An empty string trivially satisfies
    every ``"x" not in block`` assertion below, so a silent miss turned these
    regression tests into no-ops that passed regardless of the implementation.
    """
    pattern = rf"class {class_name}.*?def run\(self\) -> None:(.*?)(?=\n    def |\nclass |\Z)"
    m = re.search(pattern, source, re.DOTALL)
    if m is None:
        raise AssertionError(
            f"could not locate {class_name}.run() — the source shape changed, "
            "so the signal-safety regression check below would silently pass"
        )
    return m.group(1)


def _extract_lifespan_shutdown(source: str) -> str:
    idx = source.find("yield")
    if idx == -1:
        raise AssertionError(
            "no yield found in the lifespan function — the shutdown assertions would silently pass"
        )
    return source[idx:]


def _extract_collector_run(source: str) -> str:
    pattern = r"def run\(self\) -> None.*?(?=\ndef |\Z)"
    m = re.search(pattern, source, re.DOTALL)
    if m is None:
        raise AssertionError(
            "could not locate Collector.run() — the thread-join regression check "
            "would silently pass"
        )
    return m.group(0)


class TestExtractorsActuallyMatch:
    """Guard the guard: every extractor must return a non-empty slice of real code.

    These checks assert on source text, so their only real failure mode is the
    extractor silently matching nothing while the ``"x" not in block`` assertions
    pass anyway. Asserting the extraction is non-empty — and contains the anchor
    it is supposed to key on — closes that hole without pretending the test
    exercises runtime behaviour.
    """

    def test_run_body_extractors_match_real_code(self) -> None:
        for class_name in ("ArchiveConsumer", "ScorerConsumer"):
            body = _extract_run_body(_CONSUMER_SRC, class_name)
            assert body.strip(), f"{class_name}.run() body extracted empty"
            assert "def " in body or "self." in body, (
                f"{class_name}.run() extraction looks like it captured the wrong "
                f"region: {body[:120]!r}"
            )

    def test_collector_run_extractor_matches_real_code(self) -> None:
        run_block = _extract_collector_run(_COLLECTOR_SRC)
        assert "def run(self) -> None" in run_block
        assert len(run_block) > 200, (
            f"Collector.run() extraction suspiciously short: {len(run_block)} chars"
        )

    def test_lifespan_extractor_matches_real_code(self) -> None:
        shutdown = _extract_lifespan_shutdown(_APP_SRC)
        assert shutdown.strip()
        assert "redis" in shutdown.lower() or "archiver" in shutdown.lower(), (
            "lifespan shutdown extraction did not reach the cleanup code"
        )
