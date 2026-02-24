# Copyright (C) 2025 Ethereum Foundation
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

"""Tests for the shared renderer cache in the HTML plugin."""

from unittest.mock import MagicMock

import pytest

from docc.context import Context
from docc.document import BlankNode, Node, Visit
from docc.plugins.html import (
    _LOADED_RENDERERS,
    HTMLVisitor,
)


@pytest.fixture(autouse=True)  # noqa: SC200
def _clear_renderer_cache() -> None:
    _LOADED_RENDERERS.clear()


class TestRendererCacheBehavioral:
    """Verify that HTMLVisitor resolves renderers and produces output."""

    def test_visitor_resolves_blank_node_renderer(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)
        blank = BlankNode()

        result = visitor.enter(blank)

        assert result == Visit.SkipChildren

    def test_visitor_produces_expected_output_for_blank_node(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)
        blank = BlankNode()

        initial_stack_len = len(visitor.stack)
        visitor.enter(blank)

        assert len(visitor.stack) == initial_stack_len + 1
        assert isinstance(visitor.stack[-1], BlankNode)


class TestRendererCacheCallCount:
    """Verify that EntryPoint.load() is called at most once per node type."""

    def test_load_called_once_for_two_visitors(self) -> None:
        mock_renderer = MagicMock(return_value=None)
        mock_entry_point = MagicMock()
        mock_entry_point.load.return_value = mock_renderer

        context = Context({})
        first_visitor = HTMLVisitor(context)
        second_visitor = HTMLVisitor(context)

        key = "docc.document:BlankNode"

        # Inject the mock entry point into both visitors.
        first_visitor.entry_points[key] = mock_entry_point
        second_visitor.entry_points[key] = mock_entry_point

        # First visitor triggers load.
        first_visitor.enter(BlankNode())
        assert mock_entry_point.load.call_count == 1

        # Second visitor reuses the cache; load is not called again.
        second_visitor.enter(BlankNode())
        assert mock_entry_point.load.call_count == 1

    def test_load_called_once_for_same_visitor_twice(self) -> None:
        mock_renderer = MagicMock(return_value=None)
        mock_entry_point = MagicMock()
        mock_entry_point.load.return_value = mock_renderer

        context = Context({})
        visitor = HTMLVisitor(context)

        key = "docc.document:BlankNode"
        visitor.entry_points[key] = mock_entry_point

        visitor.enter(BlankNode())
        # Pop the stack entry added by enter() so we can call enter() again.
        visitor.stack.pop()
        visitor.enter(BlankNode())

        assert mock_entry_point.load.call_count == 1


class TestRendererCacheKeying:
    """Verify that the shared cache is keyed by Type[Node] subclasses."""

    def test_cache_contains_blank_node_after_visit(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)
        visitor.enter(BlankNode())

        assert BlankNode in _LOADED_RENDERERS

    def test_cache_values_are_callable(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)
        visitor.enter(BlankNode())

        for key, value in _LOADED_RENDERERS.items():
            assert issubclass(key, Node), f"Key {key} is not a Node subclass."
            assert callable(value), f"Value for {key} is not callable."

    def test_visitors_share_same_renderers_dict(self) -> None:
        context = Context({})
        first_visitor = HTMLVisitor(context)
        second_visitor = HTMLVisitor(context)

        assert first_visitor.renderers is second_visitor.renderers
        assert first_visitor.renderers is _LOADED_RENDERERS
