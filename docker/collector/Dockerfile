FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

LABEL description="AI Log Filter — Log Collector + Normalizer + Kafka Producer"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.4@sha256:5164bf84e7b4e2e08ce0b4c66b4a8c996a286e6959f72ac5c6e0a3c80e8cb04a /uv /usr/local/bin/uv
COPY uv.lock pyproject.toml ./
RUN uv export --frozen --no-dev --only-group collector --no-emit-project \
        --output-file /tmp/requirements.lock \
    && uv pip install --system --no-cache -r /tmp/requirements.lock

COPY src/ ./src/
COPY config/ ./config/
RUN uv pip install --system --no-cache -e . --no-deps

# /app/spool is the SYSLOG_SPOOL_PATH mount point. Creating it in the image, owned by
# the runtime user, makes the fresh named volume inherit that ownership so the
# non-root collector can write to it despite `read_only: true`.
RUN useradd -m -u 1000 logfilter \
    && mkdir -p /app/spool \
    && chown -R logfilter:logfilter /app
USER logfilter

# The collector entrypoint: listens on syslog port 5140 (unprivileged)
# and publishes received events to Kafka
EXPOSE 5140/udp
EXPOSE 5140/tcp

# Prometheus metrics endpoint (SYSLOG_METRICS_PORT, default 9100)
EXPOSE 9100/tcp

CMD ["python", "-m", "logfilter.collector"]
