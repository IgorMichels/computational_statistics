.PHONY: help local check format test clean

help:
	@echo "🚀 Computational Statistics - Available Commands"
	@echo ""
	@echo "📦 Setup:"
	@echo "  make local     - Set up local virtual environment with uv"
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
  if ! command -v uv >/dev/null 2>&1; then \
		echo "❌ uv not found. Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "✅ uv installed successfully!"; \
	fi; \
	echo "📦 Creating virtual environment and installing dependencies..."; \
	uv sync --dev; \
	echo "🪝 Setting up pre-commit hooks..."; \
	uv run pre-commit install; \
  end_time=$$(date +%s); \
	elapsed=$$((end_time - start_time)); \
	echo "✅ Virtual environment configured!"; \
	echo "⏱️  Time elapsed: $$elapsed seconds"; \
  echo "🎯 To use: uv run <command> or activate with: source .venv/bin/activate"

check:
	@echo "🔍 Checking code quality..."
	@uv run ruff check scripts/
	@uv run mypy scripts/
	@uv run pylint scripts/*.py

format:
	@echo "✨ Formatting code..."
	@uv run ruff format scripts/

test:
	@echo "🧪 Running hooks..."
	@uv run pre-commit run --all-files

clean:
	@echo "🧹 Cleaning temporary files..."
	@find . -type f -name "*.pyc" -delete
	@find . -type d -name "__pycache__" -delete
	@find . -type d -name ".pytest_cache" -delete
	@find . -type d -name ".ruff_cache" -delete
	@find . -type d -name ".mypy_cache" -delete
	@rm -rf build/ dist/ *.egg-info/
	@rm -f uv.lock
	@rm -rf .venv/
	@echo "✅ Cleanup completed!"
