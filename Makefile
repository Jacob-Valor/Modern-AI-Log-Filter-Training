# ── AI Log Filter — Makefile ───────────────────────────────────────────────────
# Usage: make <target>

.PHONY: help install dev-install train evaluate test test-coverage lint format \
         verify audit security-check up up-dev down logs logs-api logs-router \
         build clean models-dir smoke-test smoke-pipeline benchmark \
         k8s-apply k8s-delete k8s-secrets certs

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
	@echo "  benchmark      Benchmark live API with Locust (set BENCHMARK_ARGS='...')"
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
		PYTHONPATH=. $(PYTHON) training/train.py; \
	else \
		PYTHONPATH=. $(PYTHON) training/train.py --sample-normal $(SAMPLE_N) --sample-failure 10000; \
	fi

evaluate:
	PYTHONPATH=. $(PYTHON) training/evaluate.py

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-coverage:
	$(PYTHON) -m pytest tests/ --cov=src/logfilter --cov-report=term-missing --cov-report=xml --cov-report=html

integration-test:
	@echo "Starting integration test stack …"
	$(COMPOSE) -f docker-compose.test.yml up -d --wait
	@echo "Running integration tests …"
	$(PYTHON) -m pytest tests/integration/ -v -m integration --tb=short
	@echo "Tearing down integration stack …"
	$(COMPOSE) -f docker-compose.test.yml down -v

smoke-pipeline:
	$(PYTHON) scripts/smoke_test_pipeline.py

benchmark-standalone:
	$(PYTHON) scripts/throughput_benchmark.py $(BENCHMARK_ARGS)

model-manifest-generate:
	$(PYTHON) scripts/model_manifest.py generate

model-manifest-validate:
	$(PYTHON) scripts/model_manifest.py validate

verify: lint test-coverage smoke-pipeline

audit:
	# CVE-2025-69872 (diskcache): vulnerability exception — see SECURITY.md.
	# diskcache is a transitive dep of pysigma. pysigma is used ONLY for
	# Sigma rule loading (sigma.collection). The vulnerable code path
	# (sigma.data.mitre_attack) is never imported in this project — MITRE
	# data comes from config/mitre_techniques.json. No upstream fix exists
	# (Snyk: "no fixed version" as of audit date). Re-review when diskcache
	# ships a patched release.
	$(PYTHON) -m pip_audit -r requirements.txt --progress-spinner off \
		--ignore-vuln CVE-2025-69872

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
	@if [ ! -f .env ]; then \
		echo "ERROR: .env missing. Copy .env.example to .env and set required values."; \
		exit 1; \
	fi
	@set -a; . ./.env; set +a; \
	curl -fsS http://localhost:8080/health | python3 -m json.tool
	@echo ""
	@set -a; . ./.env; set +a; \
	curl -fsS -X POST http://localhost:8080/score \
	  -H "Content-Type: application/json" \
	  -H "X-API-Token: $$LOGFILTER_API_TOKEN" \
	  -d '{"raw": "Jan 15 11:07:53 prod-srv01 sshd[123]: Failed password for root from 10.0.0.5 port 44382 ssh2", "source_type": "syslog"}' \
	  | python3 -m json.tool

benchmark:
	$(PYTHON) scripts/benchmark.py $(BENCHMARK_ARGS)

# ── Kubernetes ─────────────────────────────────────────────────────────────────

k8s-secrets:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/secret.yaml

k8s-apply:
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/secret.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/pvc.yaml
	kubectl apply -f k8s/network-policies.yaml
	kubectl apply -f k8s/pdb.yaml
	kubectl apply -f k8s/api.yaml
	kubectl apply -f k8s/router.yaml
	kubectl apply -f k8s/collector.yaml
	kubectl apply -f k8s/archive.yaml
	kubectl apply -f k8s/ingress.yaml

k8s-delete:
	kubectl delete -f k8s/ --ignore-not-found=true

# ── TLS Certificates ───────────────────────────────────────────────────────────

certs:
	bash scripts/certs/generate.sh

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned."
