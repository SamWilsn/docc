## What changed

Moved the `renderers` dictionary from a per-instance attribute on `HTMLVisitor` to a module-level shared cache (`_LOADED_RENDERERS`). Each `HTMLVisitor` instance now references the same dictionary, so `EntryPoint.load()` is called at most once per node type across all visitors rather than once per visitor per node type.

## Why

`HTMLVisitor` is instantiated once per document during the HTML transform phase. Previously, each instance maintained its own `renderers` dict and independently called `EntryPoint.load()` for every node type it encountered. For projects with many documents, this meant redundant entry point loading on every single document, making the transform phase unnecessarily slow.

## Test coverage

- **Existing tests:** `test_html.py` covered `HTMLVisitor.enter()` behavior, renderer error paths, and stack management, but did not verify cross-instance renderer sharing.
- **What was missing:** No tests confirmed that renderer loading was cached across multiple `HTMLVisitor` instances, that the cache was keyed correctly by `Type[Node]`, or that `EntryPoint.load()` was only called once per node type.
- **What was added:** `tests/test_html_renderer_cache.py` with three test classes:
  - **Behavioral tests:** Verify that visiting a `BlankNode` resolves the correct renderer and produces expected stack output.
  - **Call-count tests:** Use mock entry points to assert `EntryPoint.load()` is called exactly once when two separate visitors visit the same node type.
  - **Cache-keying tests:** Verify that `_LOADED_RENDERERS` entries are keyed by `Type[Node]` subclasses with callable values, and that all `HTMLVisitor` instances reference the same shared dictionary object.
