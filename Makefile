# Requires Python 3.12+. Override with: make setup PYTHON=python3.13
PYTHON ?= $(shell command -v python3.12 || command -v python3)
VENV   := .venv
BIN    := $(VENV)/bin
export DEEPEVAL_TELEMETRY_OPT_OUT := 1

.PHONY: setup test lint format typecheck check run report drift drift-check \
        dashboard screenshots integration validate docker clean

setup: ## Create .venv and install the package with dev tools
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"
	@test -f .env || cp .env.example .env
	@echo "Setup done. Add API keys to .env for real runs."

test: ## Unit tests with coverage (no network)
	$(BIN)/pytest --cov --cov-report=term-missing --cov-fail-under=80

lint: ## Ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

typecheck:
	$(BIN)/mypy

check: lint typecheck test ## Everything CI runs

validate: ## Validate the golden dataset
	$(BIN)/llmdrift validate-dataset

run: ## Evaluate the real models in config/models.yaml (needs keys in .env)
	$(BIN)/llmdrift run

report:
	$(BIN)/llmdrift report

drift: ## Drift check on the latest run; exits 1 if drift is found
	$(BIN)/llmdrift drift

dashboard: ## Dashboard on data/results.db (real results from `make run`)
	$(BIN)/streamlit run dashboard/app.py

drift-check: ## Real drift check: re-run with a harmful system prompt, then compare (exits 1)
	$(BIN)/llmdrift run --fresh --no-cache --runs 1 --inject-degradation --label "injected degradation"
	$(BIN)/llmdrift drift

screenshots: ## Regenerate docs/screenshots (needs Chrome and the docs extra)
	$(BIN)/pip install -e ".[docs]"
	$(BIN)/streamlit run dashboard/app.py --server.port 8599 & \
		PID=$$!; sleep 8; $(BIN)/python scripts/screenshots.py --url http://localhost:8599; \
		kill $$PID

integration: ## Optional tests against real APIs (needs keys)
	$(BIN)/pytest -m integration -v

docker: ## Build and start the dashboard container
	docker compose up --build

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
