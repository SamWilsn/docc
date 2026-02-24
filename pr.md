## What changed

Cache the `entry_points(group="docc.plugins.html")` call at module level so it
is computed once per process instead of once per `HTMLVisitor` instance. A new
module-level variable `_HTML_ENTRY_POINTS` and a lazy-initializing helper
`_get_html_entry_points()` replace the per-instance discovery in
`HTMLVisitor.__init__`.

## Why

`HTMLVisitor` is instantiated once per document. On large projects (e.g.
execution-specs with 2371 documents), the repeated `entry_points()` calls
dominated the transform phase, accounting for the majority of a ~60 s runtime.
Caching the result eliminates redundant package-metadata discovery and yields a
significant speedup on the transform phase.

## Test coverage

Existing tests covered `HTMLVisitor` construction, rendering, and error paths
but did not verify caching behavior. Four new tests were added to
`tests/test_html.py` in the `TestEntryPointsCache` class:

- **Behavioral test:** Two `HTMLVisitor` instances share the same (identity-equal)
  `entry_points` dict and can both resolve known entry point names.
- **Call-count / spy test:** Patches `entry_points()` and verifies it is called
  exactly once across three `HTMLVisitor` instantiations.
- **Cache-keying test:** Verifies the cached dict maps string keys to
  `EntryPoint` objects and contains the expected `docc.document:BlankNode` key.
