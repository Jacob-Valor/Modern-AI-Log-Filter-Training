FROM python:3.11-slim@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0

LABEL maintainer="logfilter-ai"
LABEL description="AI Log Filter — FastAPI Scoring Service"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for reproducible, lockfile-driven installs
COPY --from=ghcr.io/astral-sh/uv:0.11.4@sha256:5164bf84e7b4e2e08ce0b4c66b4a8c996a286e6959f72ac5c6e0a3c80e8cb04a /uv /usr/local/bin/uv

# Install the API's lockfile-resolved runtime dependency group.
COPY uv.lock pyproject.toml ./
RUN uv export --frozen --no-dev --only-group api --no-emit-project \
        --output-file /tmp/requirements.lock \
    && uv pip install --system --no-cache \
        --index https://download.pytorch.org/whl/cpu \
        --index-strategy unsafe-best-match \
        -r /tmp/requirements.lock

# Copy source
COPY src/ ./src/
COPY config/ ./config/
COPY pyproject.toml .
COPY scripts/download_hf_models.py ./scripts/

# Install the logfilter package
RUN uv pip install --system --no-cache -e . --no-deps

# Pre-download HuggingFace models so the image is self-contained.
# Set empty revisions to download the default branch; override with build args
# to pin exact commits for reproducible builds.
ARG HF_TOKEN=""
ARG HF_CACHE_DIR="/app/hf-cache"
RUN mkdir -p "${HF_CACHE_DIR}" \
    && HF_HOME="${HF_CACHE_DIR}" python scripts/download_hf_models.py \
        --config config/config.yaml \
        --cache-dir "${HF_CACHE_DIR}" \
        $( [ -n "${HF_TOKEN}" ] && echo "--token ${HF_TOKEN}" )

# Non-root user
RUN useradd -m -u 1000 logfilter \
    && mkdir -p /home/logfilter/.cache/huggingface \
    && chown -R logfilter:logfilter /app /home/logfilter/.cache
USER logfilter

ENV HF_HOME=/app/hf-cache

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

CMD ["sh", "-c", "uvicorn logfilter.api.app:app --host 0.0.0.0 --port 8080 --workers ${LOGFILTER_API_WORKERS:-1}"]
