export PYTHONPATH := src

.PHONY: install lint test cov
install:
	uv sync
lint:
	uv run ruff check src tests
	uv run ruff format --check src tests
test:
	uv run pytest -q
cov:
	uv run pytest -q --cov=holdout_vault --cov-report=term-missing --cov-fail-under=85
