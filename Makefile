.PHONY: install dev lint type-check test test-unit test-integration eval run docker seed clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/ eval/
	ruff format --check src/ tests/ eval/

format:
	ruff check --fix src/ tests/ eval/
	ruff format src/ tests/ eval/

type-check:
	mypy src/vaidya/

test:
	pytest tests/ -v --tb=short -p no:logfire

test-unit:
	pytest tests/unit/ -v --tb=short -p no:logfire

test-integration:
	pytest tests/integration/ -v --tb=short -p no:logfire

eval:
	python -m eval --scenarios quick

eval-all:
	python -m eval --scenarios all

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
