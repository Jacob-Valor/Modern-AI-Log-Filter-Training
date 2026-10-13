from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent

_FATAL_INTERP = re.compile(r"\$\{[A-Z0-9_]+:\?")


def _base_compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text())


def _env_values(env: object) -> list[str]:
    if isinstance(env, dict):
        return [str(v) for v in env.values()]
    if isinstance(env, list):
        return [str(v) for v in env]
    return []


def test_dev_profile_services_do_not_block_base_stack_startup() -> None:
    # Compose interpolates the ENTIRE file regardless of the active profile, so a
    # dev-only service using ${VAR:?...} for a var absent from .env aborts every
    # command including the base `docker compose up` (regression: KIBANA_SERVICE_TOKEN).
    services = _base_compose()["services"]
    offenders = {
        name: fatal
        for name, spec in services.items()
        if "dev" in (spec.get("profiles") or [])
        and (fatal := [v for v in _env_values(spec.get("environment")) if _FATAL_INTERP.search(v)])
    }
    assert not offenders, (
        f"dev-profile services use fail-if-unset ${{VAR:?}} interpolation that blocks "
        f"the base 'docker compose up'; use ${{VAR:-}} instead: {offenders}"
    )


def test_base_compose_dev_profile_services_exist() -> None:
    """Guard the guard above: with no dev services it asserts nothing.

    The interpolation bug this file protects against only reaches a dev-only
    service, so a future edit that drops the ``dev`` profile would silently
    retire the whole check.
    """
    services = _base_compose()["services"]
    dev_services = sorted(
        name for name, spec in services.items() if "dev" in (spec.get("profiles") or [])
    )

    assert services, "docker-compose.yml defines no services — the parser is broken"
    assert dev_services, (
        "no service declares the 'dev' profile; test_dev_profile_services_do_not_"
        "block_base_stack_startup can no longer fire"
    )
