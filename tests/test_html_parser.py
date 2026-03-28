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

"""Tests for HTMLParser, _ElementTreeVisitor, and _make_relative."""

from pathlib import PurePath

import pytest

from docc.context import Context
from docc.document import BlankNode
from docc.plugins.html import (
    HTMLParser,
    HTMLTag,
    TextNode,
    _ElementTreeVisitor,
    _make_relative,
)

# ---------------------------------------------------------------------------
# HTMLParser
# ---------------------------------------------------------------------------


def test_parser_simple_tag() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed("<div>hello</div>")

    children = list(parser.root.children)
    assert len(children) == 1
    child = children[0]
    assert isinstance(child, HTMLTag)
    assert child.tag_name == "div"


def test_parser_nested_tags() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed("<div><span>text</span></div>")

    children = list(parser.root.children)
    assert len(children) == 1
    div = children[0]
    assert isinstance(div, HTMLTag)

    span = list(div.children)[0]
    assert isinstance(span, HTMLTag)
    assert span.tag_name == "span"


def test_parser_with_attributes() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed('<a href="/test" class="link">click</a>')

    children = list(parser.root.children)
    anchor = children[0]
    assert isinstance(anchor, HTMLTag)
    assert anchor.attributes["href"] == "/test"
    assert anchor.attributes["class"] == "link"


def test_parser_text_content() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed("<p>hello world</p>")

    children = list(parser.root.children)
    p = children[0]
    assert isinstance(p, HTMLTag)
    text_children = list(p.children)
    assert len(text_children) == 1
    text_child = text_children[0]
    assert isinstance(text_child, TextNode)
    assert text_child._value == "hello world"


def test_parser_multiple_elements() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed("<p>one</p><p>two</p>")

    children = list(parser.root.children)
    assert len(children) == 2


def test_parser_empty_string() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed("")

    assert list(parser.root.children) == []


def test_parser_self_closing_tag() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed("<br>")

    children = list(parser.root.children)
    assert len(children) == 1
    assert isinstance(children[0], HTMLTag)


def test_parser_self_closing_tag_with_slash() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed("<br/>")

    children = list(parser.root.children)
    assert len(children) == 1
    child = children[0]
    assert isinstance(child, HTMLTag)
    assert child.tag_name == "br"


def test_parser_boolean_attribute() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed('<input disabled type="text">')

    children = list(parser.root.children)
    input_elem = children[0]
    assert isinstance(input_elem, HTMLTag)
    assert "disabled" in input_elem.attributes


def test_parser_mixed_content() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed("<p>Text <strong>bold</strong> more text</p>")

    children = list(parser.root.children)
    p = children[0]
    assert isinstance(p, HTMLTag)
    p_children = list(p.children)
    assert len(p_children) == 3


def test_parser_deeply_nested() -> None:
    context = Context({})
    parser = HTMLParser(context)
    html = "<div><div><div><div><span>deep</span></div></div></div></div>"
    parser.feed(html)

    children = list(parser.root.children)
    assert len(children) == 1


def test_parser_special_characters() -> None:
    context = Context({})
    parser = HTMLParser(context)
    parser.feed("<p>&lt;script&gt;</p>")

    children = list(parser.root.children)
    p = children[0]
    assert isinstance(p, HTMLTag)
    text_children = list(p.children)
    text_child = text_children[0]
    assert isinstance(text_child, TextNode)
    assert "<script>" in text_child._value


def test_parser_handle_comment_raises() -> None:
    context = Context({})
    parser = HTMLParser(context)

    with pytest.raises(NotImplementedError, match="comments"):
        parser.handle_comment("comment")


def test_parser_handle_decl_raises() -> None:
    context = Context({})
    parser = HTMLParser(context)

    with pytest.raises(NotImplementedError, match="doctype"):
        parser.handle_decl("DOCTYPE html")


def test_parser_handle_pi_raises() -> None:
    context = Context({})
    parser = HTMLParser(context)

    with pytest.raises(NotImplementedError, match="processing instruction"):
        parser.handle_pi("xml version='1.0'")


def test_parser_unknown_decl_raises() -> None:
    context = Context({})
    parser = HTMLParser(context)

    with pytest.raises(NotImplementedError, match="unknown"):
        parser.unknown_decl("something")


# ---------------------------------------------------------------------------
# _ElementTreeVisitor
# ---------------------------------------------------------------------------


def test_element_tree_visitor_basic_tag() -> None:
    tag = HTMLTag("div")
    visitor = _ElementTreeVisitor()
    tag.visit(visitor)
    element = visitor.builder.close()

    assert element.tag == "div"


def test_element_tree_visitor_with_attributes() -> None:
    tag = HTMLTag("a", {"href": "/test"})
    visitor = _ElementTreeVisitor()
    tag.visit(visitor)
    element = visitor.builder.close()

    assert element.attrib["href"] == "/test"


def test_element_tree_visitor_nested_tags() -> None:
    parent = HTMLTag("div")
    child = HTMLTag("span")
    parent.append(child)

    visitor = _ElementTreeVisitor()
    parent.visit(visitor)
    element = visitor.builder.close()

    assert element.tag == "div"
    assert len(list(element)) == 1
    assert list(element)[0].tag == "span"


def test_element_tree_visitor_text_node() -> None:
    tag = HTMLTag("p")
    tag.append(TextNode("hello"))

    visitor = _ElementTreeVisitor()
    tag.visit(visitor)
    element = visitor.builder.close()

    assert element.text == "hello"


def test_element_tree_visitor_unsupported_enter() -> None:
    visitor = _ElementTreeVisitor()

    with pytest.raises(TypeError, match="unsupported node"):
        visitor.enter(BlankNode())


def test_element_tree_visitor_unsupported_exit() -> None:
    visitor = _ElementTreeVisitor()

    with pytest.raises(TypeError, match="unsupported node"):
        visitor.exit(BlankNode())


# ---------------------------------------------------------------------------
# _make_relative
# ---------------------------------------------------------------------------


def test_make_relative_same_path() -> None:
    path = PurePath("a/b/c")
    result = _make_relative(path, path)
    assert result is None


def test_make_relative_sibling() -> None:
    from_path = PurePath("a/b/c.html")
    to_path = PurePath("a/b/d.html")
    result = _make_relative(from_path, to_path)
    assert result == PurePath("d.html")


def test_make_relative_parent_directory() -> None:
    from_path = PurePath("a/b/c.html")
    to_path = PurePath("a/d.html")
    result = _make_relative(from_path, to_path)
    assert result == PurePath("../d.html")


def test_make_relative_deeper_directory() -> None:
    from_path = PurePath("a/b.html")
    to_path = PurePath("a/c/d.html")
    result = _make_relative(from_path, to_path)
    assert result == PurePath("c/d.html")


def test_make_relative_deeply_nested() -> None:
    from_path = PurePath("a/b/c/d/e.html")
    to_path = PurePath("a/b/f.html")
    result = _make_relative(from_path, to_path)

    assert result == PurePath("../../f.html")


def test_make_relative_different_subtrees_forward() -> None:
    from_path = PurePath("root/x/y/z.html")
    to_path = PurePath("root/a/b/c.html")
    result = _make_relative(from_path, to_path)

    assert result == PurePath("../../a/b/c.html")


def test_make_relative_different_subtrees_abc() -> None:
    from_path = PurePath("root/a/b/c.html")
    to_path = PurePath("root/x/y/z.html")
    result = _make_relative(from_path, to_path)
    assert result == PurePath("../../x/y/z.html")


def test_make_relative_root_to_subdirectory() -> None:
    from_path = PurePath("index.html")
    to_path = PurePath("sub/page.html")
    result = _make_relative(from_path, to_path)
    assert result == PurePath("sub/page.html")
