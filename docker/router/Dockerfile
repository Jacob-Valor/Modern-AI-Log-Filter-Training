FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

LABEL description="AI Log Filter — Kafka Consumer + LEEF Router → QRadar"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.11.4 /uv /usr/local/bin/uv
COPY uv.lock pyproject.toml ./
RUN uv pip install --system --locked --no-cache kafka-python structlog tenacity PyYAML elasticsearch httpx

COPY src/ ./src/
COPY config/ ./config/
RUN uv pip install --system --no-cache -e . --no-deps

RUN useradd -m -u 1000 logfilter && chown -R logfilter:logfilter /app
USER logfilter

CMD ["python", "-m", "logfilter.kafka_router"]
