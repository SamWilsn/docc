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

"""Tests for HTML node types: TextNode, HTMLTag, HTMLRoot."""

import pytest

from docc.context import Context
from docc.document import BlankNode
from docc.plugins.html import HTMLRoot, HTMLTag, TextNode

# ---------------------------------------------------------------------------
# TextNode
# ---------------------------------------------------------------------------


def test_text_node_init() -> None:
    node = TextNode("hello")
    assert node._value == "hello"


def test_text_node_children_empty() -> None:
    node = TextNode("test")
    assert tuple(node.children) == ()


def test_text_node_replace_child_raises() -> None:
    node = TextNode("test")
    with pytest.raises(TypeError, match="text nodes have no children"):
        node.replace_child(BlankNode(), BlankNode())


def test_text_node_repr() -> None:
    node = TextNode("hello world")
    assert repr(node) == "'hello world'"


def test_text_node_repr_double_quote() -> None:
    node = TextNode('say "hello"')
    assert repr(node) == "'say \"hello\"'"


def test_text_node_repr_single_quote() -> None:
    node = TextNode("it's")
    assert repr(node) == '"it\'s"'


def test_text_node_html_entities() -> None:
    text = TextNode("&lt;script&gt;")
    assert text._value == "&lt;script&gt;"


def test_text_node_unicode() -> None:
    text = TextNode("Hello \u00e9\u00e8\u00ea")
    assert "\u00e9" in text._value


def test_text_node_special_characters() -> None:
    text = TextNode("<script>alert('xss')</script>")
    assert text._value == "<script>alert('xss')</script>"


# ---------------------------------------------------------------------------
# HTMLTag
# ---------------------------------------------------------------------------


def test_html_tag_init() -> None:
    tag = HTMLTag("div")
    assert tag.tag_name == "div"
    assert tag.attributes == {}
    assert list(tag.children) == []


def test_html_tag_init_with_attributes() -> None:
    tag = HTMLTag("a", {"href": "/test", "class": "link"})
    assert tag.attributes["href"] == "/test"
    assert tag.attributes["class"] == "link"


def test_html_tag_append_child() -> None:
    parent = HTMLTag("div")
    existing = HTMLTag("p")
    parent.append(existing)
    child = HTMLTag("span")
    parent.append(child)
    children = list(parent.children)
    assert children[-1] is child


def test_html_tag_append_text() -> None:
    tag = HTMLTag("p")
    existing = HTMLTag("span")
    tag.append(existing)
    text = TextNode("hello")
    tag.append(text)
    children = list(tag.children)
    assert children[-1] is text


def test_html_tag_replace_child() -> None:
    parent = HTMLTag("div")
    before = HTMLTag("header")
    old = HTMLTag("span")
    after = HTMLTag("footer")
    parent.append(before)
    parent.append(old)
    parent.append(after)

    parent.replace_child(old, new := HTMLTag("p"))
    children = list(parent.children)
    assert old not in children
    assert children.index(new) == 1


def test_html_tag_replace_child_duplicate() -> None:
    parent = HTMLTag("div")
    child = HTMLTag("a")
    parent.append(child)
    parent.append(child)

    new = HTMLTag("em")
    parent.replace_child(child, new)

    children = list(parent.children)
    assert child not in children
    assert children.count(new) == 2


def test_html_tag_multiple_children() -> None:
    parent = HTMLTag("div")
    first_child = HTMLTag("span")
    second_child = HTMLTag("p")
    text = TextNode("text")

    parent.append(first_child)
    parent.append(second_child)
    parent.append(text)

    assert len(list(parent.children)) == 3


def test_html_tag_repr() -> None:
    tag = HTMLTag("div")
    assert repr(tag) == "<div>"


def test_html_tag_repr_with_attributes() -> None:
    tag = HTMLTag("a", {"href": "/test"})
    result = repr(tag)
    assert "<a" in result
    assert 'href="/test"' in result


def test_html_tag_repr_escapes_quotes() -> None:
    tag = HTMLTag("div", {"data-value": 'test"quote'})
    result = repr(tag)
    assert "&quot;" in result


def test_html_tag_repr_none_attribute() -> None:
    tag = HTMLTag("input", {"disabled": None})
    result = repr(tag)
    assert "disabled" in result


def test_html_tag_repr_none_and_string_attributes() -> None:
    tag = HTMLTag("input", {"disabled": None, "type": "text"})
    result = repr(tag)
    assert "disabled" in result
    assert 'type="text"' in result


def test_html_tag_repr_empty_attribute() -> None:
    tag = HTMLTag("div", {"data-empty": ""})
    result = repr(tag)
    assert 'data-empty=""' in result


def test_html_tag_repr_multiple_attributes() -> None:
    tag = HTMLTag(
        "div",
        {"id": "main", "class": "container", "data-value": "test"},
    )
    result = repr(tag)
    assert "id=" in result
    assert "class=" in result
    assert "data-value=" in result


# ---------------------------------------------------------------------------
# HTMLRoot
# ---------------------------------------------------------------------------


def test_html_root_init() -> None:
    context = Context({})
    root = HTMLRoot(context)
    assert list(root.children) == []


def test_html_root_append() -> None:
    context = Context({})
    root = HTMLRoot(context)
    tag = HTMLTag("div")
    root.append(tag)
    assert tag in root.children


def test_html_root_replace_child() -> None:
    context = Context({})
    root = HTMLRoot(context)
    old = HTMLTag("div")
    new = HTMLTag("span")
    root.append(old)

    root.replace_child(old, new)
    assert old not in root.children
    assert new in root.children


def test_html_root_extension() -> None:
    context = Context({})
    root = HTMLRoot(context)
    assert root.extension == ".html"


def test_html_root_multiple_children() -> None:
    context = Context({})
    root = HTMLRoot(context)

    first_tag = HTMLTag("div")
    second_tag = HTMLTag("span")
    text = TextNode("hello")

    root.append(first_tag)
    root.append(second_tag)
    root.append(text)

    children = list(root.children)
    assert len(children) == 3
    assert children[0] is first_tag
    assert children[1] is second_tag
    assert children[2] is text


# ---------------------------------------------------------------------------
# HTMLTag._to_element
# ---------------------------------------------------------------------------


def test_html_tag_to_element_simple() -> None:
    tag = HTMLTag("div")
    element = tag._to_element()
    assert element.tag == "div"


def test_html_tag_to_element_children() -> None:
    parent = HTMLTag("div")
    child = HTMLTag("span")
    parent.append(child)

    element = parent._to_element()
    assert element.tag == "div"
    children = list(element)
    assert len(children) == 1
    assert children[0].tag == "span"


def test_html_tag_to_element_text() -> None:
    tag = HTMLTag("p")
    tag.append(TextNode("hello"))

    element = tag._to_element()
    assert element.tag == "p"
    assert element.text == "hello"


def test_html_tag_to_element_nested_text() -> None:
    parent = HTMLTag("p")
    parent.append(TextNode("start "))
    child = HTMLTag("strong")
    child.append(TextNode("bold"))
    parent.append(child)
    parent.append(TextNode(" end"))

    element = parent._to_element()
    assert element.tag == "p"
    assert element.text == "start "
    children = list(element)
    assert len(children) == 1
    assert children[0].tag == "strong"
    assert children[0].text == "bold"


def test_html_tag_to_element_deep() -> None:
    outer = HTMLTag("div")
    middle = HTMLTag("section")
    inner = HTMLTag("article")
    inner.append(TextNode("content"))
    middle.append(inner)
    outer.append(middle)

    element = outer._to_element()
    assert element.tag == "div"
    section = list(element)[0]
    assert section.tag == "section"
    article = list(section)[0]
    assert article.tag == "article"
    assert article.text == "content"


def test_html_tag_to_element_with_attributes() -> None:
    tag = HTMLTag("a", {"href": "/link", "class": "nav", "id": "main"})
    tag.append(TextNode("click"))

    element = tag._to_element()
    assert element.tag == "a"
    assert element.attrib["href"] == "/link"
    assert element.attrib["class"] == "nav"
    assert element.attrib["id"] == "main"
    assert element.text == "click"
