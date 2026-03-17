FROM python:3.11-slim

LABEL description="AI Log Filter — Log Collector + Normalizer + Kafka Producer"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir kafka-python structlog tenacity pyyaml

COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

RUN useradd -m -u 1000 logfilter && chown -R logfilter:logfilter /app
USER logfilter

# The collector entrypoint: listens on syslog port 5140 (unprivileged)
# and publishes received events to Kafka
EXPOSE 5140/udp
EXPOSE 5140/tcp

CMD ["python", "-m", "logfilter.collector"]
