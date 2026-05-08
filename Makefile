# ── AI Log Filter — Makefile ───────────────────────────────────────────────────
# Usage: make <target>

.PHONY: help install dev-install train evaluate test test-coverage lint format \
        verify audit security-check up up-dev down logs logs-api logs-router \
        build clean models-dir smoke-test smoke-pipeline

PYTHON     ?= python3
PIP        ?= pip3
COMPOSE    ?= docker compose
SAMPLE_N   ?= 50000   # subsample for quick training (default 50K normal)

help:
	@echo ""
	@echo "AI Log Filter — Available targets"
	@echo "─────────────────────────────────"
	@echo "  install        Install production dependencies"
	@echo "  dev-install    Install all deps including dev/test tools"
	@echo ""
	@echo "  train          Train XGBoost classifier on HDFS TraceBench data"
	@echo "                 (set SAMPLE_N=0 to use all ~226K samples)"
	@echo "  evaluate       Evaluate saved model on test set"
	@echo ""
	@echo "  test           Run all unit tests (pytest)"
	@echo "  verify         Run lint, tests, and local pipeline smoke test"
	@echo "  audit          Run dependency vulnerability audit"
	@echo "  security-check Run verify and dependency audit"
	@echo "  lint           Lint with ruff"
	@echo "  format         Format with ruff"
	@echo ""
	@echo "  up             Start full Docker stack (detached)"
	@echo "  up-dev         Start stack + Kibana dev services"
	@echo "  down           Stop Docker stack"
	@echo "  logs           Tail all service logs"
	@echo "  build          Build Docker images"
	@echo ""
	@echo "  clean          Remove __pycache__, .pytest_cache, coverage reports"
	@echo ""

# ── Install ────────────────────────────────────────────────────────────────────

install:
	$(PIP) install --no-cache-dir -r requirements.txt
	$(PIP) install --no-cache-dir -e .

dev-install: install
	$(PIP) install --no-cache-dir -e ".[dev]"

# ── Training ───────────────────────────────────────────────────────────────────

models-dir:
	mkdir -p models

train: models-dir
	@echo "Training classifier (SAMPLE_N=$(SAMPLE_N)) …"
	@if [ "$(SAMPLE_N)" = "0" ]; then \
		$(PYTHON) training/train.py; \
	else \
		$(PYTHON) training/train.py --sample-normal $(SAMPLE_N) --sample-failure 10000; \
	fi

evaluate:
	$(PYTHON) training/evaluate.py

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-coverage:
	$(PYTHON) -m pytest tests/ --cov=src/logfilter --cov-report=term-missing --cov-report=xml --cov-report=html

smoke-pipeline:
	$(PYTHON) scripts/smoke_test_pipeline.py

verify: lint test-coverage smoke-pipeline

audit:
	$(PYTHON) -m pip_audit -r requirements.txt --progress-spinner off

security-check: verify audit

# ── Linting ────────────────────────────────────────────────────────────────────

lint:
	$(PYTHON) -m ruff check src/ tests/ training/

format:
	$(PYTHON) -m ruff format src/ tests/ training/
	$(PYTHON) -m ruff check --fix src/ tests/ training/

# ── Docker ─────────────────────────────────────────────────────────────────────

build:
	$(COMPOSE) build

up: models-dir
	@test -f .env || (echo "ERROR: .env missing. Copy .env.example to .env and set required values." && exit 1)
	$(COMPOSE) up -d
	@echo ""
	@echo "Services starting …"
	@echo "  API:       http://localhost:8080/health"
	@echo "  API docs:  disabled unless LOGFILTER_ENABLE_DOCS=1"
	@echo "  Grafana:   http://localhost:3000  (credentials from .env: GRAFANA_ADMIN_USER/PASSWORD)"
	@echo "  Prometheus:http://localhost:9090"
	@echo "  Kafka:     localhost:9092"
	@echo "  ES:        http://localhost:9200  (credentials from .env: ES_USER/ES_PASSWORD)"
	@echo ""

up-dev: models-dir
	$(COMPOSE) --profile dev up -d
	@echo "  Kibana:    http://localhost:5601"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

logs-api:
	$(COMPOSE) logs -f logfilter-api

logs-router:
	$(COMPOSE) logs -f router

# ── Smoke test against live API ────────────────────────────────────────────────

smoke-test:
	@echo "Smoke-testing API …"
	curl -s http://localhost:8080/health | python3 -m json.tool
	@echo ""
	curl -s -X POST http://localhost:8080/score \
	  -H "Content-Type: application/json" \
	  -H "X-API-Token: $$LOGFILTER_API_TOKEN" \
	  -d '{"raw": "Jan 15 11:07:53 prod-srv01 sshd[123]: Failed password for root from 10.0.0.5 port 44382 ssh2", "source_type": "syslog"}' \
	  | python3 -m json.tool

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned."
