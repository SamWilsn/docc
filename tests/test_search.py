# Copyright (C) 2026 Ethereum Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import json
from io import StringIO
from pathlib import Path, PurePath
from typing import Any, Dict, List, Optional, Set

import pytest

from docc.context import Context
from docc.document import BlankNode, Document
from docc.plugins.references import Definition, Index, ReferenceError
from docc.plugins.search import (
    ByReference,
    BySource,
    Item,
    Search,
    Searchable,
    SearchBuilder,
    SearchContext,
    SearchDiscover,
    SearchNode,
    SearchSource,
    SearchTransform,
    _SearchVisitor,
)
from docc.settings import PluginSettings, Settings
from docc.source import Source


@pytest.fixture
def plugin_settings() -> PluginSettings:
    settings = Settings(Path("."), {"tool": {"docc": {}}})
    return settings.for_plugin("docc.search")


class MockSource(Source):
    _path: PurePath

    def __init__(self, path: Optional[PurePath] = None) -> None:
        self._path = path if path is not None else PurePath("mock.py")

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._path

    @property
    def output_path(self) -> PurePath:
        return self._path


class TestBySource:
    def test_create(self) -> None:
        source = MockSource()
        location = BySource(source=source)
        assert location.source is source

    def test_frozen(self) -> None:
        source = MockSource()
        location = BySource(source=source)
        with pytest.raises(AttributeError):
            location.source = MockSource()  # pyre-ignore[41]

    def test_equality(self) -> None:
        source = MockSource()
        first_location = BySource(source=source)
        second_location = BySource(source=source)
        assert first_location == second_location


class TestByReference:
    def test_create(self) -> None:
        location = ByReference(identifier="test.func", specifier=0)
        assert location.identifier == "test.func"
        assert location.specifier == 0

    def test_create_without_specifier(self) -> None:
        location = ByReference(identifier="test.func", specifier=None)
        assert location.specifier is None

    def test_frozen(self) -> None:
        location = ByReference(identifier="test", specifier=0)
        with pytest.raises(AttributeError):
            location.identifier = "changed"  # pyre-ignore[41]


class TestItem:
    def test_create_with_string_content(self) -> None:
        source = MockSource()
        location = BySource(source=source)
        item = Item(location=location, content="test content")

        assert item.location is location
        assert item.content == "test content"

    def test_create_with_dict_content(self) -> None:
        source = MockSource()
        location = BySource(source=source)
        item = Item(
            location=location, content={"type": "module", "name": ["test"]}
        )

        assert isinstance(item.content, dict)
        assert item.content["type"] == "module"


class TestSearch:
    def test_init(self) -> None:
        search = Search()
        assert len(search._items) == 0

    def test_add_string_content(self) -> None:
        search = Search()
        source = MockSource()
        location = BySource(source=source)
        item = Item(location=location, content="test content")

        search.add(item)

        # Note: Accessing _items directly as Search has no public query API
        assert location in search._items
        assert "text" in search._items[location]
        assert "test content" in search._items[location]["text"]

    def test_add_dict_content(self) -> None:
        search = Search()
        source = MockSource()
        location = BySource(source=source)
        item = Item(
            location=location, content={"type": "module", "name": ["test"]}
        )

        search.add(item)

        assert location in search._items
        assert "type" in search._items[location]
        assert "name" in search._items[location]

    def test_add_multiple_items_same_location(self) -> None:
        search = Search()
        source = MockSource()
        location = BySource(source=source)

        search.add(Item(location=location, content="first"))
        search.add(Item(location=location, content="second"))

        assert "first" in search._items[location]["text"]
        assert "second" in search._items[location]["text"]


class TestSearchSource:
    def test_relative_path_is_none(self) -> None:
        source = SearchSource()
        assert source.relative_path is None

    def test_output_path(self) -> None:
        source = SearchSource()
        assert source.output_path == PurePath("search")


def test_search_node_extension() -> None:
    node = SearchNode()
    assert node.extension == ".js"


class TestSearchBuilder:
    def test_build_processes_search_sources(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = SearchSource()
        unprocessed: Set[Source] = {source}
        processed: Dict[Source, Document] = {}

        builder = SearchBuilder(plugin_settings)
        builder.build(unprocessed, processed)

        assert len(unprocessed) == 0
        assert source in processed
        assert isinstance(processed[source].root, SearchNode)

    def test_build_ignores_non_search_sources(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = MockSource()
        unprocessed: Set[Source] = {source}
        processed: Dict[Source, Document] = {}

        builder = SearchBuilder(plugin_settings)
        builder.build(unprocessed, processed)

        assert source in unprocessed
        assert len(processed) == 0


def test_search_discover_yields_source(
    plugin_settings: PluginSettings,
) -> None:
    discover = SearchDiscover(plugin_settings)
    sources = list(discover.discover(frozenset()))

    assert len(sources) == 1
    assert isinstance(sources[0], SearchSource)


class TestSearchContext:
    def test_provides(self) -> None:
        assert SearchContext.provides() == Search

    def test_init(self, plugin_settings: PluginSettings) -> None:
        ctx = SearchContext(plugin_settings)
        assert isinstance(ctx.search, Search)

    def test_provide(self, plugin_settings: PluginSettings) -> None:
        ctx = SearchContext(plugin_settings)
        provided = ctx.provide()
        assert provided is ctx.search


_PREFIX = "this.SEARCH_INDEX = "
_SUFFIX = "; Object.freeze(this.SEARCH_INDEX);"


def _parse_search_output(raw: str) -> List[Dict[str, Any]]:
    """Extract and parse the JSON payload from SearchNode output."""
    assert raw.startswith(_PREFIX)
    assert raw.endswith(_SUFFIX)
    return json.loads(raw[len(_PREFIX) : -len(_SUFFIX)])


class TestSearchNodeOutput:
    def test_output_by_source(self) -> None:
        """
        Test SearchNode.output() serializes search index to
        JavaScript JSON with items indexed by BySource.
        """
        source = MockSource(PurePath("module.py"))
        search = Search()
        search.add(
            Item(location=BySource(source=source), content="hello world")
        )
        index = Index()
        context = Context({Search: search, Index: index})

        node = SearchNode()
        dest = StringIO()
        node.output(context, dest)

        data = _parse_search_output(dest.getvalue())
        assert len(data) == 1
        assert data[0]["source"]["path"] == "module.py"
        assert "hello world" in data[0]["content"]["text"]

    def test_output_by_reference_without_specifier(self) -> None:
        """Test SearchNode.output() resolves ByReference location via Index."""
        source = MockSource(PurePath("ref_module.py"))
        search = Search()
        location = ByReference(identifier="my.module.func", specifier=None)
        search.add(Item(location=location, content="func docs"))

        index = Index()
        index.define(source, "my.module.func")

        context = Context({Search: search, Index: index})
        node = SearchNode()
        dest = StringIO()
        node.output(context, dest)

        data = _parse_search_output(dest.getvalue())
        assert len(data) == 1
        assert data[0]["source"]["identifier"] == "my.module.func"
        assert data[0]["source"]["path"] == "ref_module.py"

    def test_output_by_reference_with_specifier(self) -> None:
        """Test SearchNode.output() resolves ByReference with specifier."""
        source = MockSource(PurePath("spec_module.py"))
        search = Search()
        location = ByReference(identifier="my.ident", specifier=0)
        search.add(Item(location=location, content="spec docs"))

        index = Index()
        index.define(source, "my.ident")  # specifier=0

        context = Context({Search: search, Index: index})
        node = SearchNode()
        dest = StringIO()
        node.output(context, dest)

        data = _parse_search_output(dest.getvalue())
        assert len(data) == 1
        assert data[0]["source"]["specifier"] == 0
        assert data[0]["source"]["path"] == "spec_module.py"

    def test_output_by_reference_specifier_not_found_raises(self) -> None:
        """
        Test ByReference with specifier not found raises
        ReferenceError.
        """
        source = MockSource(PurePath("err_module.py"))
        search = Search()
        # Use specifier=99 which won't match any definition
        location = ByReference(identifier="my.missing", specifier=99)
        search.add(Item(location=location, content="should fail"))

        index = Index()
        index.define(source, "my.missing")  # specifier=0

        context = Context({Search: search, Index: index})
        node = SearchNode()
        dest = StringIO()
        with pytest.raises(ReferenceError):
            node.output(context, dest)

    def test_output_mixed_sources_and_references(self) -> None:
        """
        Test SearchNode.output() with both BySource and
        ByReference items.
        """
        source = MockSource(PurePath("mixed.py"))
        search = Search()
        search.add(
            Item(location=BySource(source=source), content="source item")
        )
        ref_location = ByReference(identifier="mixed.func", specifier=None)
        search.add(Item(location=ref_location, content="ref item"))

        index = Index()
        index.define(source, "mixed.func")

        context = Context({Search: search, Index: index})
        node = SearchNode()
        dest = StringIO()
        node.output(context, dest)

        data = _parse_search_output(dest.getvalue())
        assert len(data) == 2


def test_search_transform_indexes_searchable(
    plugin_settings: PluginSettings,
) -> None:
    """
    Test that transform() indexes Searchable nodes wrapped in
    Definitions as ByReference with the correct identifier.
    """
    source = MockSource(PurePath("test_transform.py"))
    search = Search()

    searchable = MockSearchable("indexed content")
    definition = Definition(identifier="my.module.MyClass")
    definition.specifier = 0
    definition.child = searchable

    document = Document(definition)
    context = Context({Source: source, Search: search, Document: document})

    transform = SearchTransform(plugin_settings)
    transform.transform(context)

    # The searchable should be indexed as ByReference
    by_ref = ByReference(identifier="my.module.MyClass", specifier=0)
    assert by_ref in search._items
    assert "indexed content" in search._items[by_ref]["text"]


class MockSearchable(BlankNode, Searchable):
    def __init__(
        self, content: str = "test content", search_children_val: bool = True
    ) -> None:
        self._content = content
        self._search_children = search_children_val

    def to_search(self) -> str:
        return self._content

    def search_children(self) -> bool:
        return self._search_children


class TestSearchVisitor:
    def test_adds_searchable_content(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = MockSource()
        search = Search()
        document = Document(MockSearchable("searchable content"))
        context = Context({Source: source, Search: search, Document: document})

        visitor = _SearchVisitor(context)
        document.root.visit(visitor)

        by_source = BySource(source=source)
        assert by_source in search._items
        assert "searchable content" in search._items[by_source]["text"]

    def test_respects_search_children(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = MockSource()
        search = Search()
        inner = MockSearchable("inner")

        class ParentNode(MockSearchable):
            def __init__(self):
                super().__init__("parent", search_children_val=False)
                self._children = [inner]

            @property
            def children(self):
                return self._children

        document = Document(ParentNode())
        context = Context({Source: source, Search: search, Document: document})

        visitor = _SearchVisitor(context)
        document.root.visit(visitor)

        # Note: Accessing _items is necessary as Search has no public query API
        by_source = BySource(source=source)
        assert "parent" in search._items[by_source]["text"]
        # Verify children were NOT indexed when search_children=False
        assert (
            "inner" not in search._items[by_source]["text"]
        ), "Children should be skipped when search_children() returns False"
