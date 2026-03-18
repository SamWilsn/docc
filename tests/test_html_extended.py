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

import tempfile
from pathlib import Path, PurePath
from typing import Iterator, Optional

import pytest

from docc.context import Context
from docc.document import BlankNode
from docc.plugins.html import (
    HTML,
    HTMLContext,
    HTMLParser,
    HTMLRoot,
    HTMLTag,
    HTMLVisitor,
    TextNode,
    _ElementTreeVisitor,
    _make_relative,
)
from docc.settings import Settings
from docc.source import Source


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


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


class TestHTMLTagToElement:
    def test_simple_tag_to_element(self) -> None:
        tag = HTMLTag("div")
        element = tag._to_element()
        assert element.tag == "div"

    def test_tag_with_children_to_element(self) -> None:
        parent = HTMLTag("div")
        child = HTMLTag("span")
        parent.append(child)

        element = parent._to_element()
        assert element.tag == "div"
        children = list(element)
        assert len(children) == 1
        assert children[0].tag == "span"

    def test_tag_with_text_to_element(self) -> None:
        tag = HTMLTag("p")
        tag.append(TextNode("hello"))

        element = tag._to_element()
        assert element.tag == "p"
        assert element.text == "hello"

    def test_tag_with_attributes_to_element(self) -> None:
        tag = HTMLTag("a", {"href": "/link", "class": "nav"})
        element = tag._to_element()

        assert element.attrib["href"] == "/link"
        assert element.attrib["class"] == "nav"


class TestHTMLRootWithSource:
    def test_root_with_html_context(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.html.context": {
                                "extra_css": ["custom.css"],
                                "breadcrumbs": False,
                            }
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.html.context")
        html_ctx = HTMLContext(plugin_settings)
        html = html_ctx.provide()

        context = Context({HTML: html})
        root = HTMLRoot(context)

        assert root.extra_css == ["custom.css"]
        assert root.breadcrumbs is False

    def test_root_without_html_context(self) -> None:
        context = Context({})
        root = HTMLRoot(context)

        assert root.extra_css == []
        assert root.breadcrumbs is True


class TestHTMLParserExtended:
    def test_parse_empty_string(self) -> None:
        context = Context({})
        parser = HTMLParser(context)
        parser.feed("")

        assert list(parser.root.children) == []

    def test_parse_self_closing_tag(self) -> None:
        context = Context({})
        parser = HTMLParser(context)
        parser.feed("<br>")

        children = list(parser.root.children)
        assert len(children) == 1
        assert isinstance(children[0], HTMLTag)

    def test_parse_self_closing_tag_with_slash(self) -> None:
        context = Context({})
        parser = HTMLParser(context)
        parser.feed("<br/>")

        children = list(parser.root.children)
        assert len(children) == 1
        child = children[0]
        assert isinstance(child, HTMLTag)
        assert child.tag_name == "br"

    def test_parse_with_boolean_attribute(self) -> None:
        context = Context({})
        parser = HTMLParser(context)
        parser.feed('<input disabled type="text">')

        children = list(parser.root.children)
        input_elem = children[0]
        assert isinstance(input_elem, HTMLTag)
        assert "disabled" in input_elem.attributes

    def test_parse_mixed_content(self) -> None:
        context = Context({})
        parser = HTMLParser(context)
        parser.feed("<p>Text <strong>bold</strong> more text</p>")

        children = list(parser.root.children)
        p = children[0]
        assert isinstance(p, HTMLTag)
        p_children = list(p.children)
        assert len(p_children) == 3


class TestElementTreeVisitorEdgeCases:
    def test_visitor_with_unsupported_node_enter(self) -> None:
        visitor = _ElementTreeVisitor()

        with pytest.raises(TypeError, match="unsupported node"):
            visitor.enter(BlankNode())

    def test_visitor_with_unsupported_node_exit(self) -> None:
        visitor = _ElementTreeVisitor()

        with pytest.raises(TypeError, match="unsupported node"):
            visitor.exit(BlankNode())


class TestMakeRelativeEdgeCases:
    def test_different_subtrees(self) -> None:
        from_path = PurePath("root/a/b/c.html")
        to_path = PurePath("root/x/y/z.html")
        result = _make_relative(from_path, to_path)
        assert result == PurePath("../../x/y/z.html")

    def test_root_to_subdirectory(self) -> None:
        from_path = PurePath("index.html")
        to_path = PurePath("sub/page.html")
        result = _make_relative(from_path, to_path)
        assert result == PurePath("sub/page.html")


class TestHTMLTagAttributes:
    def test_empty_attribute_value(self) -> None:
        tag = HTMLTag("div", {"data-empty": ""})
        result = repr(tag)
        assert 'data-empty=""' in result

    def test_multiple_attributes(self) -> None:
        tag = HTMLTag(
            "div",
            {"id": "main", "class": "container", "data-value": "test"},
        )
        result = repr(tag)
        assert "id=" in result
        assert "class=" in result
        assert "data-value=" in result


class TestHTMLRootChildren:
    def test_children_property(self) -> None:
        context = Context({})
        root = HTMLRoot(context)

        first_tag = HTMLTag("div")
        second_tag = HTMLTag("span")
        root.append(first_tag)
        root.append(second_tag)

        children = list(root.children)
        assert len(children) == 2
        assert first_tag in children
        assert second_tag in children


class TestTextNodeValue:
    def test_text_node_stores_value(self) -> None:
        text = TextNode("Hello World")
        assert text._value == "Hello World"

    def test_text_node_special_characters(self) -> None:
        text = TextNode("<script>alert('xss')</script>")
        assert text._value == "<script>alert('xss')</script>"


class TestHTMLVisitorTraversal:
    def test_visitor_traversal_blank_node(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)

        blank = BlankNode()
        blank.visit(visitor)

        assert len(visitor.stack) == 1
        assert visitor.stack[0] is visitor.root
