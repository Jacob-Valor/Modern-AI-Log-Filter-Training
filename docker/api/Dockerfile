FROM python:3.11-slim

LABEL maintainer="logfilter-ai"
LABEL description="AI Log Filter — FastAPI Scoring Service"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/
COPY config/ ./config/
COPY models/ ./models/
COPY pyproject.toml .

# Install the logfilter package
RUN pip install --no-cache-dir -e .

# Non-root user
RUN useradd -m -u 1000 logfilter && chown -R logfilter:logfilter /app
USER logfilter

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["uvicorn", "logfilter.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "4", \
     "--log-config", "/dev/null"]
