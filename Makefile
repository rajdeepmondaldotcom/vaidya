.PHONY: install dev lint type-check test test-unit test-integration coverage check eval eval-all eval-live rehearse-twilio run docker seed clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/ eval/ scripts/
	ruff format --check src/ tests/ eval/ scripts/

format:
	ruff check --fix src/ tests/ eval/ scripts/
	ruff format src/ tests/ eval/ scripts/

type-check:
	mypy src/vaidya/

test:
	pytest tests/ -v --tb=short -p no:logfire

test-unit:
	pytest tests/unit/ -v --tb=short -p no:logfire

test-integration:
	pytest tests/integration/ -v --tb=short -p no:logfire

coverage:
	COVERAGE_FILE=$${COVERAGE_FILE:-/tmp/vaidya.coverage} pytest tests/ --cov=src/vaidya --cov-report=term-missing:skip-covered --cov-fail-under=$${COVERAGE_FAIL_UNDER:-80} -q -p no:logfire

check: lint type-check coverage

eval:
	mkdir -p reports
	python -m eval --scenarios quick --output reports/eval_report.md

eval-all:
	mkdir -p reports
	python -m eval --scenarios all --output reports/eval_report.md

eval-live:
	mkdir -p reports
	python -m eval --base-url "$${BASE_URL:-http://localhost:8000}" --scenarios "$${SCENARIOS:-quick}" --output "$${EVAL_OUTPUT:-reports/eval_report.md}"

rehearse-twilio:
	python scripts/rehearse_twilio.py --base-url "$$BASE_URL"

run:
	uvicorn vaidya.app:create_app --factory --reload --host 0.0.0.0 --port 8000

docker:
	docker compose up --build

seed:
	python scripts/seed_knowledge.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf dist/ build/ *.egg-info/ chroma_data/
