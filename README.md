docc
====

[![documentation badge][docs-badge]][docs]

The documentation compiler.

## Python Quickstart

### Installing

```bash
pip install docc
```

### Configuring

Add the following to your [`pyproject.toml`]:

```toml
[tool.docc.plugins."docc.python.discover"]
paths = [ "<path to Python source>" ]

[tool.docc.output]
path = "<where to put rendered documentation>"
```

### Building

Finally, to generate the documentation:

```bash
docc
```

## Development

### Setting Up

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/SamWilsn/docc.git
cd docc
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

Install in development mode to run tests and lint:

```bash
pip install -e ".[test,lint]"
```

### Code Style

This project uses:

- **black** for code formatting (line length: 79).
- **isort** for import sorting (black profile).
- **flake8** for linting.
- **pyre** for type checking.

Format code before committing:

```bash
black src tests
isort src tests
```

### Running Tests

```bash
pytest
```

Tests require 80% code coverage to pass. For a detailed coverage report:

```bash
pytest --cov-report=html
```

The HTML report will be generated in `htmlcov/`.

### Using Tox

Run the full test suite with linting:

```bash
tox
```

Run only type checking:

```bash
tox -e type
```

[docs-badge]: https://github.com/SamWilsn/docc/actions/workflows/gh-pages.yaml/badge.svg?branch=master
[docs]: https://samwilsn.github.io/docc/
[`pyproject.toml`]: https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#declaring-project-metadata
