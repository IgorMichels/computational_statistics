.PHONY: help local check format test clean

help:
	@echo "🚀 Computational Statistics - Available Commands"
	@echo ""
	@echo "📦 Setup:"
	@echo "  make local     - Set up local virtual environment"
	@echo ""
	@echo "🔧 Code Quality:"
	@echo "  make check     - Check code (ruff + mypy + pylint)"
	@echo "  make format    - Format code (ruff format)"
	@echo "  make test      - Run hooks (pre-commit)"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean     - Clean temporary files"

local:
	@echo "🚀 Setting up local virtual environment..."
	@start_time=$$(date +%s); \
	if [ ! -d ".venv" ]; then \
		echo "📦 Creating virtual environment..."; \
		python3 -m venv .venv; \
	fi; \
	echo "🔧 Upgrading pip..."; \
	.venv/bin/pip install --upgrade pip; \
	echo "📦 Installing dependencies..."; \
	.venv/bin/pip install -e ".[dev]"; \
	echo "🪝 Setting up pre-commit hooks..."; \
	.venv/bin/pre-commit install; \
	end_time=$$(date +%s); \
	elapsed=$$((end_time - start_time)); \
	echo "✅ Virtual environment configured!"; \
	echo "⏱️  Time elapsed: $$elapsed seconds"; \
	echo "🎯 To use: source .venv/bin/activate"

check:
	@echo "🔍 Checking code quality..."
	@.venv/bin/python -m ruff check scripts/
	@.venv/bin/python -m mypy scripts/
	@.venv/bin/python -m pylint scripts/*.py

format:
	@echo "✨ Formatting code..."
	@.venv/bin/python -m ruff format scripts/

test:
	@echo "🧪 Running hooks..."
	@.venv/bin/python -m pre-commit run --all-files

clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -delete
	@find . -type d -name ".ruff_cache" -delete
	@find . -type d -name ".mypy_cache" -delete
	@rm -rf build/ dist/ *.egg-info/
	@rm -rf .venv/
	@echo "✅ Cleanup completed!"
