FROM python:3.11-slim

LABEL description="AI Log Filter — Log Collector + Normalizer + Kafka Producer"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN sed -i 's/\[.*\]//g' requirements.txt && \
    pip install --no-cache-dir -c requirements.txt kafka-python structlog tenacity PyYAML

COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . --no-deps

RUN useradd -m -u 1000 logfilter && chown -R logfilter:logfilter /app
USER logfilter

# The collector entrypoint: listens on syslog port 5140 (unprivileged)
# and publishes received events to Kafka
EXPOSE 5140/udp
EXPOSE 5140/tcp

# Prometheus metrics endpoint (SYSLOG_METRICS_PORT, default 9100)
EXPOSE 9100/tcp

CMD ["python", "-m", "logfilter.collector"]
