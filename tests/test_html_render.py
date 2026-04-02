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

"""
Tests for HTML render functions, HTMLVisitor, FindVisitor, and
render_reference.
"""

from pathlib import PurePath
from typing import Optional
from unittest.mock import patch

import pytest

from docc.context import Context
from docc.document import BlankNode, ListNode, Node, Visit
from docc.plugins.html import (
    HTMLRoot,
    HTMLTag,
    HTMLVisitor,
    TextNode,
    _find_filter,
    _FindVisitor,
    blank_node,
    html_tag,
    list_node,
    render_reference,
    text_node,
)
from docc.plugins.loader import PluginError
from docc.plugins.references import Definition, Index, Reference
from docc.source import Source


class MockSource(Source):
    _output_path: PurePath

    def __init__(self, output_path: Optional[PurePath] = None) -> None:
        self._output_path = (
            output_path if output_path is not None else PurePath("test.py")
        )

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._output_path

    @property
    def output_path(self) -> PurePath:
        return self._output_path


# ---------------------------------------------------------------------------
# Render functions (blank_node, list_node, html_tag, text_node)
# ---------------------------------------------------------------------------


def test_render_blank_node_returns_none() -> None:
    context = Context({})
    parent = HTMLRoot(context)
    blank = BlankNode()

    result = blank_node(context, parent, blank)
    assert result is None


def test_render_list_node_returns_parent() -> None:
    context = Context({})
    parent = HTMLRoot(context)
    node = ListNode()

    result = list_node(context, parent, node)
    assert result is parent


def test_render_html_tag_appends_to_parent() -> None:
    context = Context({})
    parent = HTMLRoot(context)
    tag = HTMLTag("div")

    result = html_tag(context, parent, tag)
    assert result is None
    assert tag in parent.children


def test_render_text_node_appends_to_parent() -> None:
    context = Context({})
    parent = HTMLRoot(context)
    text = TextNode("hello")

    result = text_node(context, parent, text)
    assert result is None
    assert text in parent.children


# ---------------------------------------------------------------------------
# HTMLVisitor
# ---------------------------------------------------------------------------


def test_visitor_enter_no_renderer() -> None:
    context = Context({})
    visitor = HTMLVisitor(context)

    class _UnregisteredNode(Node):
        @property
        def children(self):  # type: ignore[override]
            return ()

        def replace_child(self, old, new):  # type: ignore[override]
            pass

    node = _UnregisteredNode()
    with pytest.raises(PluginError, match="no renderer found"):
        visitor.enter(node)


def test_visitor_enter_renderer_not_callable() -> None:
    context = Context({})
    visitor = HTMLVisitor(context)

    not_callable = "I am not callable"
    key = "docc.document:BlankNode"

    with patch.dict(
        visitor.entry_points,
        {
            key: type(
                "FakeEP",
                (),
                {"name": key, "load": lambda self: not_callable},
            )()
        },
    ):
        with pytest.raises(PluginError, match="not callable"):
            visitor.enter(BlankNode())


def test_visitor_enter_pushes_to_stack() -> None:
    context = Context({})
    visitor = HTMLVisitor(context)

    blank = BlankNode()
    initial_stack_len = len(visitor.stack)

    result = visitor.enter(blank)

    assert result == Visit.SkipChildren
    assert len(visitor.stack) == initial_stack_len + 1


def test_visitor_enter_returning_tag_traverses() -> None:
    context = Context({})
    visitor = HTMLVisitor(context)

    parent_tag = HTMLTag("section")
    visitor.root.append(parent_tag)
    visitor.stack.append(parent_tag)

    result_tag: HTMLTag = HTMLTag("span")

    def fake_renderer(ctx: object, parent: object, node: object) -> HTMLTag:
        return result_tag

    visitor.renderers[ListNode] = fake_renderer

    node = ListNode()
    result = visitor.enter(node)

    assert result == Visit.TraverseChildren
    assert visitor.stack[-1] is result_tag


def test_visitor_enter_and_exit() -> None:
    context = Context({})
    visitor = HTMLVisitor(context)

    blank = BlankNode()
    visitor.enter(blank)
    assert len(visitor.stack) == 2
    visitor.exit(blank)
    assert len(visitor.stack) == 1


def test_visitor_traversal_with_list_node() -> None:
    context = Context({})
    visitor = HTMLVisitor(context)

    node = ListNode([BlankNode(), BlankNode()])
    node.visit(visitor)

    assert len(visitor.stack) == 1
    assert visitor.stack[0] is visitor.root


def test_visitor_traversal_blank_node() -> None:
    context = Context({})
    visitor = HTMLVisitor(context)

    blank = BlankNode()
    blank.visit(visitor)

    assert len(visitor.stack) == 1
    assert visitor.stack[0] is visitor.root


# ---------------------------------------------------------------------------
# _FindVisitor / _find_filter
# ---------------------------------------------------------------------------


def test_find_visitor_matching_class() -> None:
    definition = Definition(identifier="test", child=BlankNode())
    definition.specifier = 0

    visitor = _FindVisitor("docc.document:BlankNode")
    definition.visit(visitor)

    assert len(visitor.found) == 1
    assert visitor.found[0][0] is definition


def test_find_visitor_no_match() -> None:
    definition = Definition(identifier="test", child=BlankNode())
    definition.specifier = 0

    visitor = _FindVisitor("some.other:Class")
    definition.visit(visitor)

    assert len(visitor.found) == 0


def test_find_visitor_max_depth() -> None:
    inner_def = Definition(identifier="inner", child=BlankNode())
    inner_def.specifier = 0
    outer_def = Definition(identifier="outer", child=inner_def)
    outer_def.specifier = 0

    visitor = _FindVisitor("docc.document:BlankNode", max_depth=1)
    outer_def.visit(visitor)

    assert len(visitor.found) == 0


def test_find_visitor_without_definition() -> None:
    blank = BlankNode()

    visitor = _FindVisitor("docc.document:BlankNode")
    blank.visit(visitor)

    assert len(visitor.found) == 0


def test_find_visitor_exit_pops_definition() -> None:
    definition = Definition(identifier="test", child=BlankNode())
    definition.specifier = 0

    visitor = _FindVisitor("docc.document:BlankNode")
    definition.visit(visitor)

    assert len(visitor._definitions) == 0
    assert len(visitor.found) == 1


def test_find_filter_basic() -> None:
    definition = Definition(identifier="test", child=BlankNode())
    definition.specifier = 0

    result = _find_filter(definition, "docc.document:BlankNode")

    assert len(result) == 1
    assert result[0][0] is definition
    assert isinstance(result[0][1], BlankNode)


# ---------------------------------------------------------------------------
# render_reference
# ---------------------------------------------------------------------------


def test_render_reference_single_definition() -> None:
    source = MockSource(PurePath("docs/page.py"))
    def_source = MockSource(PurePath("docs/target.py"))

    index = Index()
    index.define(def_source, "my.func")

    context = Context({Source: source, Index: index})
    ref = Reference(identifier="my.func")

    anchor = render_reference(context, ref)

    assert isinstance(anchor, HTMLTag)
    assert anchor.tag_name == "a"
    assert "href" in anchor.attributes
    href = anchor.attributes["href"] or ""
    assert "my.func:0" in href
    assert "target.py.html" in href


def test_render_reference_multiple_definitions() -> None:
    source = MockSource(PurePath("docs/page.py"))
    def_source1 = MockSource(PurePath("docs/target1.py"))
    def_source2 = MockSource(PurePath("docs/target2.py"))

    index = Index()
    index.define(def_source1, "my.func")
    index.define(def_source2, "my.func")

    context = Context({Source: source, Index: index})
    ref = Reference(identifier="my.func")

    container = render_reference(context, ref)

    assert isinstance(container, HTMLTag)
    assert container.tag_name == "div"
    assert container.attributes.get("class") == "tooltip"

    tooltip_div = list(container.children)[0]
    assert isinstance(tooltip_div, HTMLTag)
    assert tooltip_div.attributes.get("class") == "tooltip-content"

    anchors = list(tooltip_div.children)
    assert len(anchors) == 2
    for a in anchors:
        assert isinstance(a, HTMLTag)
        assert a.tag_name == "a"
        assert "href" in a.attributes
        children = list(a.children)
        assert len(children) == 1
        assert isinstance(children[0], TextNode)


def test_render_reference_no_definitions_raises() -> None:
    source = MockSource(PurePath("docs/page.py"))
    index = Index()

    context = Context({Source: source, Index: index})
    ref = Reference(identifier="nonexistent")

    from docc.plugins.references import ReferenceError

    with pytest.raises(ReferenceError):
        render_reference(context, ref)
