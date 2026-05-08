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
