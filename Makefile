.PHONY: setup dev test lint format typecheck update-deps build clean publish publish-test bump-patch bump-minor bump-major

setup:
	pip install -e ".[dev]"

dev:
	pip install -r requirements-dev.txt

test:
	pytest tests/ -v

lint:
	flake8 src/pydantic_visible_fields tests

format:
	isort src/pydantic_visible_fields tests
	black src/pydantic_visible_fields tests

typecheck:
	mypy src/pydantic_visible_fields

update-deps:
	pip-compile requirements.in --upgrade
	pip-compile requirements-dev.in --upgrade

build:
	python3 -m build

clean:
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/
	find . -name __pycache__ -exec rm -rf {} +
	find . -name *.pyc -delete

publish-test:
	python -m build
	python -m twine upload --repository-url https://test.pypi.org/legacy/ dist/* --verbose

publish:
	python -m build
	python -m twine upload dist/*


# Bump patch version (e.g., 1.2.3 -> 1.2.4) and create git tag
bump-patch:
	bump2version patch --verbose --tag
	git push --tags

# Bump minor version (e.g., 1.2.3 -> 1.3.0) and create git tag
bump-minor:
	bump2version minor --verbose --tag
	git push --tags

# Bump major version (e.g., 1.2.3 -> 2.0.0) and create git tag
bump-major:
	bump2version major --verbose --tag
	git push --tags
