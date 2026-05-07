.PHONY: help install lint format format-check typecheck test test-all test-cov ci clean docker-build docker-up docker-down

# ──────────────────────────────────────────────
# Variables
# ──────────────────────────────────────────────
UV := uv
PYTHON := $(UV) run python
PYTEST := $(UV) run pytest
RUFF := $(UV) run ruff
BLACK := $(UV) run black
MYPY := $(UV) run mypy
DOCKER_COMPOSE := docker compose -f docker/docker-compose.yml

# ──────────────────────────────────────────────
# Help
# ──────────────────────────────────────────────
help: ## Show this help message
	@echo "SarathiAgentInspect - Enterprise AI Evaluation Framework"
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
install: ## Install all dependencies (runtime + dev)
	$(UV) sync --all-extras
	@echo "✅ Dependencies installed successfully"

# ──────────────────────────────────────────────
# Code Quality
# ──────────────────────────────────────────────
lint: ## Run ruff linter with auto-fix
	$(RUFF) check src/ tests/ --fix
	@echo "✅ Linting passed"

format: ## Format code with black and ruff
	$(BLACK) src/ tests/
	$(RUFF) format src/ tests/
	@echo "✅ Formatting complete"

format-check: ## Check formatting without changes
	$(BLACK) --check src/ tests/
	$(RUFF) format --check src/ tests/
	@echo "✅ Format check passed"

typecheck: ## Run mypy type checking
	$(MYPY) src/sarathi_agent_inspect/
	@echo "✅ Type checking passed"

# ──────────────────────────────────────────────
# Testing
# ──────────────────────────────────────────────
test: ## Run unit tests only
	$(PYTEST) tests/ -m "not integration and not slow"
	@echo "✅ Unit tests passed"

test-all: ## Run all tests including integration
	$(PYTEST) tests/ -m ""
	@echo "✅ All tests passed"

test-cov: ## Run tests with coverage report
	$(PYTEST) tests/ -m "not integration and not slow" --cov --cov-report=html --cov-report=term
	@echo "✅ Coverage report generated in htmlcov/"

test-smoke: ## Run smoke tests only
	$(PYTEST) tests/ -m "smoke"
	@echo "✅ Smoke tests passed"

# ──────────────────────────────────────────────
# CI Pipeline (mirrors GitHub Actions)
# ──────────────────────────────────────────────
ci: lint typecheck test ## Run full CI pipeline locally (lint + typecheck + test)
	@echo ""
	@echo "🚀 CI pipeline passed successfully!"

# ──────────────────────────────────────────────
# Docker
# ──────────────────────────────────────────────
docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t sarathi-agent-inspect:latest .
	@echo "✅ Docker image built"

docker-up: ## Start docker-compose stack
	$(DOCKER_COMPOSE) up -d
	@echo "✅ Stack started"

docker-down: ## Stop docker-compose stack
	$(DOCKER_COMPOSE) down
	@echo "✅ Stack stopped"

docker-logs: ## Tail docker-compose logs
	$(DOCKER_COMPOSE) logs -f

# ──────────────────────────────────────────────
# Cleanup
# ──────────────────────────────────────────────
clean: ## Clean caches and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "✅ Cleaned"
