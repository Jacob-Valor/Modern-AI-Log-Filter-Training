# Small sidecar or init probe script for worker pods that do not expose
# an HTTP server.  Mount this as a ConfigMap volume under /app/probes/ and
# reference it in livenessProbe / readinessProbe exec commands.
#
# Usage in a container spec:
#   livenessProbe:
#     exec:
#       command: ["python", "/app/probes/health_probe.py", "--liveness"]
#     initialDelaySeconds: 30
#     periodSeconds: 15
#     timeoutSeconds: 5
#     failureThreshold: 3

import argparse
import os
import sys
import time


def _touch(path: str) -> None:
    """Update mtime of a file (create if missing)."""
    with open(path, "a"):
        os.utime(path, None)


def _check_liveness(marker: str, max_age_seconds: float) -> bool:
    """Pass if the marker file exists and is newer than max_age."""
    if not os.path.exists(marker):
        return False
    return time.time() - os.path.getmtime(marker) < max_age_seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--liveness", action="store_true")
    parser.add_argument("--readiness", action="store_true")
    parser.add_argument("--marker", default="/tmp/healthz")
    parser.add_argument("--max-age", type=float, default=60.0)
    args = parser.parse_args()

    if args.liveness:
        ok = _check_liveness(args.marker, args.max_age)
        if ok:
            print("alive")
            return 0
        print("stale marker", file=sys.stderr)
        return 1

    if args.readiness:
        # For workers readiness == liveness (process is running and making
        # progress).  In the future this can be extended to check Kafka
        # consumer group membership or ES connectivity.
        ok = _check_liveness(args.marker, args.max_age)
        if ok:
            print("ready")
            return 0
        print("stale marker", file=sys.stderr)
        return 1

    # Default: just touch the marker so the main loop keeps it fresh
    _touch(args.marker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
