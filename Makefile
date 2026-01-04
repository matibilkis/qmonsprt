.PHONY: help install test clean lint format

help:
	@echo "Available commands:"
	@echo "  make install    - Install package and dependencies"
	@echo "  make test       - Run test suite"
	@echo "  make clean      - Remove build artifacts and cache files"
	@echo "  make lint       - Run linting checks (if configured)"

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .coverage htmlcov/

lint:
	@echo "Linting not configured. Consider adding flake8, black, or pylint."

format:
	@echo "Formatting not configured. Consider adding black or autopep8."

