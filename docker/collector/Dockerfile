FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

LABEL description="AI Log Filter — Log Collector + Normalizer + Kafka Producer"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /usr/local/bin/uv
COPY uv.lock pyproject.toml ./
RUN uv pip install --system --no-cache kafka-python lz4 structlog tenacity PyYAML prometheus-client

COPY src/ ./src/
COPY config/ ./config/
RUN uv pip install --system --no-cache -e . --no-deps

RUN useradd -m -u 1000 logfilter && chown -R logfilter:logfilter /app
USER logfilter

# The collector entrypoint: listens on syslog port 5140 (unprivileged)
# and publishes received events to Kafka
EXPOSE 5140/udp
EXPOSE 5140/tcp

# Prometheus metrics endpoint (SYSLOG_METRICS_PORT, default 9100)
EXPOSE 9100/tcp

CMD ["python", "-m", "logfilter.collector"]
