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

from pathlib import Path, PurePath
from typing import Optional

import pytest

from docc.context import Context
from docc.document import BlankNode, Document, ListNode
from docc.plugins.references import (
    Definition,
    Index,
    IndexContext,
    IndexTransform,
    Location,
    Reference,
    ReferenceError,
)
from docc.settings import PluginSettings, Settings
from docc.source import Source


@pytest.fixture
def basic_settings(tmp_path: Path) -> Settings:
    return Settings(tmp_path, {"tool": {"docc": {}}})


@pytest.fixture
def plugin_settings(basic_settings: Settings) -> PluginSettings:
    return basic_settings.for_plugin("docc.references")


_UNSET = object()


class MockSource(Source):
    _output_path: PurePath

    def __init__(
        self,
        relative_path: object = _UNSET,
        output_path: Optional[PurePath] = None,
    ) -> None:
        if relative_path is _UNSET:
            self._relative_path: Optional[PurePath] = PurePath("test.py")
        elif relative_path is None or isinstance(relative_path, PurePath):
            self._relative_path = relative_path
        else:
            raise TypeError(f"unexpected type: {type(relative_path)}")
        self._output_path = (
            output_path or self._relative_path or PurePath("test.py")
        )

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._relative_path

    @property
    def output_path(self) -> PurePath:
        return self._output_path


class TestLocation:
    def test_create_location(self) -> None:
        source = MockSource()
        location = Location(source=source, identifier="test.func", specifier=0)

        assert location.source is source
        assert location.identifier == "test.func"
        assert location.specifier == 0

    def test_location_is_frozen(self) -> None:
        source = MockSource()
        location = Location(source=source, identifier="test", specifier=0)

        with pytest.raises(AttributeError):
            location.identifier = "changed"  # pyre-ignore[41]

    def test_location_equality(self) -> None:
        source = MockSource()
        first_location = Location(
            source=source, identifier="test", specifier=0
        )
        second_location = Location(
            source=source, identifier="test", specifier=0
        )
        third_location = Location(
            source=source, identifier="test", specifier=1
        )

        assert first_location == second_location
        assert first_location != third_location

    def test_location_is_hashable(self) -> None:
        source = MockSource()
        location_1 = Location(source=source, identifier="test", specifier=0)
        location_2 = Location(source=source, identifier="test", specifier=0)
        location_3 = Location(source=source, identifier="test", specifier=1)
        location_4 = Location(source=source, identifier="missing", specifier=1)
        location_set = {location_1, location_2, location_3}
        assert location_1 in location_set
        assert location_2 in location_set
        assert location_3 in location_set
        assert location_4 not in location_set
        assert len(location_set) == 2


class TestIndex:
    def test_create_index(self) -> None:
        index = Index()
        assert isinstance(index._index, dict)
        assert len(index._index) == 0

    def test_define_creates_location(self) -> None:
        index = Index()
        source = MockSource()

        location = index.define(source, "test.module.func")

        assert location.source is source
        assert location.identifier == "test.module.func"
        assert location.specifier == 0

    def test_define_increments_specifier(self) -> None:
        index = Index()
        source = MockSource()

        first_location = index.define(source, "test.func")
        second_location = index.define(source, "test.func")
        third_location = index.define(source, "test.func")

        assert first_location.specifier == 0
        assert second_location.specifier == 1
        assert third_location.specifier == 2

    def test_define_different_identifiers(self) -> None:
        index = Index()
        source = MockSource()

        first_location = index.define(source, "func_a")
        second_location = index.define(source, "func_b")
        third_location = index.define(source, "func_a")

        assert first_location.specifier == 0
        assert second_location.specifier == 0
        assert third_location.specifier == 1

    def test_lookup_existing(self) -> None:
        index = Index()
        source = MockSource()
        expected = index.define(source, "test.func")

        result = list(index.lookup("test.func"))

        assert len(result) == 1
        assert result[0] == expected

    def test_lookup_multiple(self) -> None:
        index = Index()
        source = MockSource()
        first_location = index.define(source, "test.func")
        second_location = index.define(source, "test.func")

        result = list(index.lookup("test.func"))

        assert len(result) == 2
        assert first_location in result
        assert second_location in result

    def test_lookup_nonexistent_raises(self) -> None:
        index = Index()

        with pytest.raises(ReferenceError):
            index.lookup("nonexistent")


class TestReferenceError:
    def test_basic_error(self) -> None:
        error = ReferenceError("undefined_func")
        assert "undefined_func" in str(error)
        assert error.identifier == "undefined_func"
        assert error.context is None

    def test_error_with_context_source(self) -> None:
        source = MockSource(relative_path=PurePath("src/module.py"))
        context = Context({Source: source})
        error = ReferenceError("missing_ref", context=context)

        assert "missing_ref" in str(error)
        assert "src/module.py" in str(error)
        assert error.context is context

    def test_error_with_context_no_relative_path(self) -> None:
        source = MockSource(
            relative_path=None, output_path=PurePath("output.html")
        )
        context = Context({Source: source})
        error = ReferenceError("missing_ref", context=context)

        assert "missing_ref" in str(error)
        assert "output.html" in str(error)


class TestBase:
    def test_children_returns_tuple(self) -> None:
        child = BlankNode()
        base = Definition(identifier="test", child=child)

        assert base.children == (child,)

    def test_default_child_is_blank(self) -> None:
        base = Definition(identifier="test")

        assert isinstance(base.child, BlankNode)

    def test_replace_child(self) -> None:
        old_child = BlankNode()
        new_child = BlankNode()
        base = Definition(identifier="test", child=old_child)

        base.replace_child(old_child, new_child)

        assert base.child is new_child

    def test_replace_child_no_match(self) -> None:
        child = BlankNode()
        other = BlankNode()
        new_child = BlankNode()
        base = Definition(identifier="test", child=child)

        base.replace_child(other, new_child)

        assert base.child is child


class TestDefinition:
    def test_create_definition(self) -> None:
        child = BlankNode()
        definition = Definition(identifier="test.func", child=child)

        assert definition.identifier == "test.func"
        assert definition.child is child
        assert definition.specifier is None

    def test_specifier_can_be_set(self) -> None:
        definition = Definition(identifier="test", specifier=5)
        assert definition.specifier == 5


def test_reference_create() -> None:
    child = BlankNode()
    reference = Reference(identifier="test.func", child=child)

    assert reference.identifier == "test.func"
    assert reference.child is child


class TestIndexContext:
    def test_provides_index(self) -> None:
        assert IndexContext.provides() == Index

    def test_init_creates_index(self, plugin_settings: PluginSettings) -> None:
        ctx = IndexContext(plugin_settings)
        assert isinstance(ctx.index, Index)

    def test_provide_returns_index(
        self, plugin_settings: PluginSettings
    ) -> None:
        ctx = IndexContext(plugin_settings)
        provided = ctx.provide()

        assert provided is ctx.index


class TestIndexTransform:
    def test_transform_indexes_definitions(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = MockSource()
        index = Index()

        definition = Definition(identifier="test.func")
        root = ListNode([definition])
        document = Document(root)

        context = Context({Document: document, Source: source, Index: index})

        transform = IndexTransform(plugin_settings)
        transform.transform(context)

        assert definition.specifier == 0

        locations = list(index.lookup("test.func"))
        assert len(locations) == 1
        assert locations[0].identifier == "test.func"

    def test_transform_nested_definitions(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = MockSource()
        index = Index()

        inner_def = Definition(identifier="inner")
        outer_def = Definition(identifier="outer", child=inner_def)
        root = ListNode([outer_def])
        document = Document(root)

        context = Context({Document: document, Source: source, Index: index})

        transform = IndexTransform(plugin_settings)
        transform.transform(context)

        assert outer_def.specifier == 0
        assert inner_def.specifier == 0

        outer_locations = list(index.lookup("outer"))
        inner_locations = list(index.lookup("inner"))
        assert len(outer_locations) == 1
        assert len(inner_locations) == 1

    def test_transform_multiple_definitions_same_id(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = MockSource()
        index = Index()

        first_definition = Definition(identifier="same_id")
        second_definition = Definition(identifier="same_id")
        root = ListNode([first_definition, second_definition])
        document = Document(root)

        context = Context({Document: document, Source: source, Index: index})

        transform = IndexTransform(plugin_settings)
        transform.transform(context)

        assert first_definition.specifier == 0
        assert second_definition.specifier == 1

    def test_transform_ignores_references(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = MockSource()
        index = Index()

        reference = Reference(identifier="some_ref")
        root = ListNode([reference])
        document = Document(root)

        context = Context({Document: document, Source: source, Index: index})

        transform = IndexTransform(plugin_settings)
        transform.transform(context)

        with pytest.raises(ReferenceError):
            index.lookup("some_ref")


class TestDefinitionReferenceInteraction:
    def test_definition_child_is_reference(self) -> None:
        ref = Reference(identifier="other")
        definition = Definition(identifier="test", child=ref)

        assert definition.child is ref
        assert definition.children == (ref,)

    def test_reference_child_is_definition(self) -> None:
        definition = Definition(identifier="inner")
        reference = Reference(identifier="test", child=definition)

        assert reference.child is definition
        assert reference.children == (definition,)
