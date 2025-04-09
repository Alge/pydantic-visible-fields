.PHONY: setup dev test lint format update-deps clean

setup:
	pip install -e ".[dev]"

dev:
	pip install -r requirements-dev.txt

test-setup:
	pip install -r requirements-test.txt

test:
	pytest tests/ -v

lint:
	flake8 src/pydantic_visible_fields tests
	mypy src/pydantic_visible_fields

format:
	isort src/pydantic_visible_fields tests
	black src/pydantic_visible_fields tests

update-deps:
	pip-compile requirements.in --upgrade
	pip-compile requirements-dev.in --upgrade
	pip-compile requirements-test.in --upgrade

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -name __pycache__ -exec rm -rf {} +
	find . -name *.pyc -delete
