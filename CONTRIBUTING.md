# Contributing to CI-Lib

Thank you for your interest in contributing! CI-Lib is a research-grade computational intelligence library, and we welcome contributions of all kinds.

## How to Contribute

### Reporting Bugs

Open an issue on GitHub with:
- A minimal reproduction script
- Expected vs actual behavior
- Python version and OS

### Suggesting Features

Open an issue describing the algorithm, use case, and how it fits within CI-Lib's seven domains (neural, evolutionary, swarm, fuzzy, clustering, optimisation, utils).

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-feature`)
3. Install development dependencies: `pip install -e ".[dev]"`
4. Write tests in `tests/` following existing patterns
5. Run tests: `python -m pytest tests/ -v`
6. Lint with ruff: `ruff check .`
7. Open a PR against `master`

### Code Style

- Pure NumPy only — no external ML frameworks
- Type hints on all public functions
- Follow existing docstring conventions (NumPy style)
- Keep dependencies minimal
- Include random seed parameter for reproducibility

## Development Setup

```bash
git clone https://github.com/WilliamMajanja/Partial-Realpart-Analysis-.git
cd Partial-Realpart-Analysis-
python -m venv venv && source venv/bin/activate
pip install -e ".[all]"
```

## Code of Conduct

Be respectful, constructive, and inclusive. Harassment and discriminatory behaviour will not be tolerated.
