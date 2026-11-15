export PYTHONPATH := src

.PHONY: install lint test cov
install:
	uv sync
lint:
	uv run ruff check src tests benchmark
	uv run ruff format --check src tests benchmark
test:
	uv run pytest -q -m "not docker"
cov:
	uv run pytest -q -m "not docker" --cov=holdout_vault --cov-report=term-missing --cov-fail-under=85
