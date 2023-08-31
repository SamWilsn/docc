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

[docs-badge]: https://github.com/SamWilsn/docc/actions/workflows/gh-pages.yaml/badge.svg?branch=master
[docs]: https://samwilsn.github.io/docc/
[`pyproject.toml`]: https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#declaring-project-metadata
