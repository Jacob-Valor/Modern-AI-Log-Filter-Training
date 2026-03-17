FROM python:3.11-slim

LABEL description="AI Log Filter — Kafka Consumer + LEEF Router → QRadar"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir kafka-python structlog tenacity pyyaml elasticsearch

COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

RUN useradd -m -u 1000 logfilter && chown -R logfilter:logfilter /app
USER logfilter

CMD ["python", "-m", "logfilter.router"]
