# Security Policy

## Supported Scope

This repository is a local/development deployment template for the LogFilter
pipeline. Production deployments must place public traffic behind the owning
organization's ingress, firewall, TLS, identity, and monitoring controls.

## Security Defaults

- Required secrets have no unsafe defaults.
- Scoring endpoints require `X-API-Token`.
- Admin endpoints require `X-Admin-Token`.
- API docs and OpenAPI JSON are disabled unless `LOGFILTER_ENABLE_DOCS=1`.
- Kafka, API, Elasticsearch, Kibana, Prometheus, Grafana, and collector ports
  are bound to `127.0.0.1` in the default Compose stack.
- The collector accepts only `SYSLOG_ALLOWED_CIDRS`; the default is loopback.
- Python service containers run as non-root users, drop Linux capabilities,
  prevent privilege escalation, and use read-only root filesystems.
- Runtime scaler artifacts are JSON-only; pickle/joblib scaler loading is
  rejected.

## Production Checklist

Before exposing this system outside a developer machine:

1. Terminate TLS at a trusted ingress or service mesh.
2. Restrict collector access to trusted log forwarder CIDRs.
3. Keep Elasticsearch, Prometheus, Grafana, Kafka, and Kibana private.
4. Rotate `LOGFILTER_API_TOKEN`, `LOGFILTER_ADMIN_TOKEN`, `ES_PASSWORD`, and
   `GRAFANA_ADMIN_PASSWORD` through your secret manager.
5. Run `make verify` and `make audit` in CI before deploys.
6. Build images in CI from a clean checkout and scan them with your container
   scanner.
7. Treat model artifacts as deployable code; only promote artifacts from trusted
   training jobs.

## Reporting

Do not put credentials, production logs, or sensitive customer data in issues.
Report vulnerabilities privately through your normal internal security channel.

## Vulnerability Exceptions

### CVE-2025-69872 — `diskcache` (transitive via `pysigma`)

**Status**: Documented exception. `make audit` ignores this CVE with
`--ignore-vuln CVE-2025-69872`.

**What it is**: `diskcache 5.6.3` (the latest available release) uses Python
`pickle` for serialization by default. If an attacker can write to the cache
directory, they can achieve arbitrary code execution when the application
reads from the cache. CVSS 3.1 base score: 9.8 (Critical).

**Why we accept it**:

1. `diskcache` is a transitive dependency of `pysigma` (required for Sigma
   rule loading). It is not a direct dependency.
2. `pysigma` is used in this project **only** for `sigma.collection`
   (Sigma rule parsing and matching). We do **not** import
   `sigma.data.mitre_attack` or `sigma.data.mitre_d3fend` — the two modules
   that use `diskcache`. MITRE ATT&CK data is loaded from
   `config/mitre_techniques.json` (project-owned, version-controlled), not
   from pySigma's MITRE STIX cache.
3. A defense-in-depth test (`tests/test_sigma_cache_path_unused.py`) asserts
   the vulnerable modules are never imported in the runtime code path.
4. Even if the vulnerable path were reached, our container hardening
   narrows the attack surface: services run as non-root, use read-only root
   filesystems, and drop Linux capabilities. The default `diskcache` cache
   directory is created via `tempfile.mkdtemp` (mode 0700) inside the
   container's writable layer.

**No upstream fix exists** as of audit date. Snyk and OSV both list "no
fixed version" for `diskcache` regarding CVE-2025-69872. The diskcache
maintainer's position is that the default `tempfile.mkdtemp(0o700)` directory
is the trust boundary and that the CVE is a documentation concern, not a
code defect (see grantjenks/python-diskcache issue #357).

**Re-review trigger**: When `diskcache` ships a release that addresses
CVE-2025-69872, remove the `--ignore-vuln CVE-2025-69872` flag from
`make audit` (Makefile line 88+) and re-run the gate. The defense-in-depth
test in `tests/test_sigma_cache_path_unused.py` should remain as a guard
against accidental introduction of the vulnerable code path.

**Alternatives considered**:

- Pin `pysigma` to a version that does not require `diskcache`. Rejected:
  pySigma has required diskcache for MITRE data caching since 0.10; older
  versions have other transitive issues.
- Switch to a Sigma rule engine that does not require pySigma. Rejected:
  rewrites the Sigma matching layer; out of scope for a security patch.
- Override `pysigma`'s MITRE cache with a JSON-only backend. Possible but
  requires vendoring a fork; deferred to upstream fix.

This exception is reviewed on every dependency bump.
