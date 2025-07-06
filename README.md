# Computational Statistics - FGV/EMAp

Final project for the Computational Statistics class of the FGV/EMAp PhD program

## 🚀 Installation

### Opção 1: Using pip
```bash
# Install main dependencies
pip install -r requirements.txt

# Install only development dependencies
pip install -r requirements-dev.txt
```

### Option 2: Using pip with pyproject.toml
```bash
# Install main dependencies
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"
```

## 🔧 Pre-commit Setup

### Initial Installation
```bash
# Install pre-commit hooks
pre-commit install
```

### Usage
```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files file.py

# Run only a specific tool
pre-commit run black
pre-commit run pylint
```

## 🛠️ Configured Linting Tools

- **Black**: Automatic code formatting
- **isort**: Import sorting
- **flake8**: Style checking (PEP 8)
- **pylint**: Static code analysis
- **mypy**: Type checking

## 📁 Project Structure

```
.
├── generate_data.py        # Data generation
├── .pre-commit-config.yaml # Pre-commit configuration
├── .pylintrc               # Pylint configuration
├── mypy.ini                # Mypy configuration
├── tox.ini                 # Flake8 configuration
├── pyproject.toml          # Project configuration
└── requirements*.txt       # Dependencies
```

## 🔍 Useful Commands

```bash
# Check code quality
pre-commit run --all-files

# Format code automatically
black .

# Check types
mypy .

# Complete analysis with pylint
pylint *.py
```
