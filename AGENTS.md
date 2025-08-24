# Agent Guidelines for Chaos vs Noise Analysis Project

## Build/Lint/Test Commands

### Testing
- **Run all tests**: `python -m pytest` (if test files exist)
- **Run single test**: `python -m pytest tests/test_file.py::TestClass::test_method`
- **Run with coverage**: `python -m pytest --cov=./ --cov-report=html`

### Linting & Code Quality
- **Lint with Ruff**: `ruff check .`
- **Fix linting issues**: `ruff check . --fix`
- **Format code**: `ruff format .`
- **Type checking**: `python -m mypy .` (if mypy configured)

### Dependencies
- **Install dependencies**: `pip install -r requirements.txt`
- **Install in development mode**: `pip install -e .`
- **Project configuration**: See `pyproject.toml` for build system and tool configurations

## Code Style Guidelines

### Python Standards
- **PEP 8**: Follow Python style guidelines
- **Type hints**: Use typing module for function parameters and return types
- **Docstrings**: Include comprehensive docstrings for all classes and functions
- **Imports**: Group standard library, third-party, then local imports with blank lines

### Naming Conventions
- **Classes**: PascalCase (e.g., `BandtPompeAnalysis`, `ExchangeRateAnalyzer`)
- **Functions/Methods**: snake_case (e.g., `ordinal_patterns`, `shannon_entropy`)
- **Variables**: snake_case, descriptive names (e.g., `embedding_dimension`, `probability_distribution`)
- **Constants**: UPPER_SNAKE_CASE

### Error Handling
- Use try/except blocks for external operations (file I/O, network calls)
- Raise ValueError for invalid parameters
- Include descriptive error messages

### Code Organization
- **Classes**: Group related functionality in classes with clear hierarchies
- **Methods**: Keep methods focused on single responsibilities
- **Documentation**: Update README.md for major changes
- **Dependencies**: Check existing imports before adding new libraries

### Scientific Computing Best Practices
- **NumPy arrays**: Use vectorized operations over loops
- **Memory efficiency**: Handle large datasets with appropriate chunking
- **Reproducibility**: Set random seeds for stochastic processes
- **Validation**: Check input data types and ranges