# ─────────────────────────────────────────────────────────────────────────────
# DigitVision — Makefile
#
# Provides short, memorable commands for every common developer task.
# Run `make help` to see all available commands.
#
# Usage:
#   make install       → Install runtime dependencies
#   make install-dev   → Install all dependencies (including dev tools)
#   make lint          → Run Ruff linter
#   make format        → Auto-format code
#   make test          → Run all tests with coverage
#   make train         → Train all three models
#   make run           → Launch the Streamlit app
#   make docker-build  → Build Docker image
#   make docker-run    → Run app in Docker
# ─────────────────────────────────────────────────────────────────────────────

.PHONY: help install install-dev lint format test \
        train train-dense train-lenet train-cnn \
        evaluate run \
        docker-build docker-run docker-stop \
        clean

PYTHON     := python
STREAMLIT  := streamlit
SRC_DIRS   := src/ tests/ config/
APP        := streamlit_app/app.py

# ── Default target ────────────────────────────────────────────────────────────
help: ## Show this help message
	@echo ""
	@echo "  DigitVision — Available Commands"
	@echo "  ─────────────────────────────────"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ── Environment ───────────────────────────────────────────────────────────────
install: ## Install runtime dependencies
	pip install -r requirements.txt

install-dev: ## Install all dependencies including dev tools
	pip install -r requirements-dev.txt

# ── Code Quality ──────────────────────────────────────────────────────────────
lint: ## Run Ruff linter
	ruff check $(SRC_DIRS)

format: ## Auto-format with Black and fix Ruff issues
	black $(SRC_DIRS)
	ruff check --fix $(SRC_DIRS)

# ── Testing ───────────────────────────────────────────────────────────────────
test: ## Run full pytest suite with coverage
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast: ## Run only fast unit tests (skip slow training tests)
	pytest tests/ -v -m "not slow"

# ── Training ──────────────────────────────────────────────────────────────────
train: ## Train all three models sequentially
	$(PYTHON) -m src.train --all

train-dense: ## Train Dense Neural Network only
	$(PYTHON) -m src.train --model dense_nn

train-lenet: ## Train LeNet-5 only
	$(PYTHON) -m src.train --model lenet5

train-cnn: ## Train Custom CNN only
	$(PYTHON) -m src.train --model custom_cnn

# ── Evaluation ────────────────────────────────────────────────────────────────
evaluate: ## Evaluate all models and generate performance plots
	$(PYTHON) -m src.evaluate --all

# ── Application ───────────────────────────────────────────────────────────────
run: ## Launch the Streamlit app locally
	$(STREAMLIT) run $(APP)

# ── Docker ────────────────────────────────────────────────────────────────────
docker-build: ## Build Docker image
	docker compose build

docker-run: ## Build and run app in Docker
	docker compose up --build

docker-stop: ## Stop running Docker containers
	docker compose down

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove all generated files (cache, logs, plots, coverage)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf logs/ .pytest_cache/ .coverage htmlcov/ .ruff_cache/
	@echo "✓ Cleaned generated files."
