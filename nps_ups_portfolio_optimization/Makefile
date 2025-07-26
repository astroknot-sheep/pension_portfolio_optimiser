.PHONY: help install install-dev test test-fast lint format type-check clean docs serve-docs build run

# Default target
help:
	@echo "NPS vs UPS Portfolio Optimization - Development Commands"
	@echo "======================================================"
	@echo ""
	@echo "Setup:"
	@echo "  install         Install package and dependencies"
	@echo "  install-dev     Install package with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run full test suite with coverage"
	@echo "  test-fast      Run fast tests (skip slow tests)"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run linting (ruff)"
	@echo "  format         Format code (black)"
	@echo "  type-check     Run type checking (mypy)"
	@echo "  quality        Run all quality checks"
	@echo ""
	@echo "Analysis:"
	@echo "  run            Run complete NPS vs UPS analysis"
	@echo "  run-fast       Run analysis with reduced simulations"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           Build documentation"
	@echo "  serve-docs     Serve documentation locally"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Clean build artifacts and cache"
	@echo "  build          Build package for distribution"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=nps_ups --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow" --cov=nps_ups --cov-report=term-missing

# Code quality
lint:
	ruff check nps_ups/ tests/

format:
	black nps_ups/ tests/ notebooks/
	ruff check nps_ups/ tests/ --fix

type-check:
	mypy nps_ups/ --ignore-missing-imports

quality: lint type-check
	@echo "✅ All quality checks passed"

# Analysis
run:
	python -m nps_ups run-analysis --n-simulations 10000

run-fast:
	python -m nps_ups run-analysis --n-simulations 1000 --scenarios base

# Documentation
docs:
	mkdocs build

serve-docs:
	mkdocs serve

# Maintenance
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf output/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

# Development workflow
dev-setup: install-dev
	@echo "✅ Development environment set up"
	@echo "Run 'make run-fast' to test the analysis pipeline"

# CI simulation
ci: quality test
	@echo "✅ CI checks passed" 