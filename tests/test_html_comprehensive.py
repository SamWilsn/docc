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
from docc.document import BlankNode, Document, ListNode
from docc.plugins.html import (
    HTMLContext,
    HTMLParser,
    HTMLRoot,
    HTMLTag,
    HTMLTransform,
    HTMLVisitor,
    TextNode,
    _find_filter,
    _FindVisitor,
    _make_relative,
)
from docc.plugins.references import Definition
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


class TestFindVisitor:
    def test_find_matching_class(self) -> None:
        definition = Definition(identifier="test", child=BlankNode())
        definition.specifier = 0

        visitor = _FindVisitor("docc.document:BlankNode")
        definition.visit(visitor)

        assert len(visitor.found) == 1
        assert visitor.found[0][0] is definition

    def test_find_no_match(self) -> None:
        definition = Definition(identifier="test", child=BlankNode())
        definition.specifier = 0

        visitor = _FindVisitor("some.other:Class")
        definition.visit(visitor)

        assert len(visitor.found) == 0

    def test_max_depth_limits_search(self) -> None:
        inner_def = Definition(identifier="inner", child=BlankNode())
        inner_def.specifier = 0
        outer_def = Definition(identifier="outer", child=inner_def)
        outer_def.specifier = 0

        visitor = _FindVisitor("docc.document:BlankNode", max_depth=1)
        outer_def.visit(visitor)

        assert len(visitor.found) == 0

    def test_without_definition(self) -> None:
        blank = BlankNode()

        visitor = _FindVisitor("docc.document:BlankNode")
        blank.visit(visitor)

        assert len(visitor.found) == 0

    def test_exit_pops_definition_via_traversal(self) -> None:
        definition = Definition(identifier="test", child=BlankNode())
        definition.specifier = 0

        visitor = _FindVisitor("docc.document:BlankNode")
        definition.visit(visitor)

        assert len(visitor._definitions) == 0
        assert len(visitor.found) == 1


class TestFindFilter:
    def test_find_filter_basic(self) -> None:
        definition = Definition(identifier="test", child=BlankNode())
        definition.specifier = 0

        result = _find_filter(definition, "docc.document:BlankNode")

        assert len(result) == 1
        assert result[0][0] is definition
        assert isinstance(result[0][1], BlankNode)


class TestHTMLParserEdgeCases:
    def test_handle_comment_raises(self) -> None:
        context = Context({})
        parser = HTMLParser(context)

        with pytest.raises(NotImplementedError, match="comments"):
            parser.handle_comment("comment")

    def test_handle_decl_raises(self) -> None:
        context = Context({})
        parser = HTMLParser(context)

        with pytest.raises(NotImplementedError, match="doctype"):
            parser.handle_decl("DOCTYPE html")

    def test_handle_pi_raises(self) -> None:
        context = Context({})
        parser = HTMLParser(context)

        with pytest.raises(
            NotImplementedError, match="processing instruction"
        ):
            parser.handle_pi("xml version='1.0'")

    def test_unknown_decl_raises(self) -> None:
        context = Context({})
        parser = HTMLParser(context)

        with pytest.raises(NotImplementedError, match="unknown"):
            parser.unknown_decl("something")


class TestMakeRelativeComprehensive:
    def test_deeply_nested_paths(self) -> None:
        from_path = PurePath("a/b/c/d/e.html")
        to_path = PurePath("a/b/f.html")
        result = _make_relative(from_path, to_path)

        assert result == PurePath("../../f.html")

    def test_different_subtrees(self) -> None:
        from_path = PurePath("root/x/y/z.html")
        to_path = PurePath("root/a/b/c.html")
        result = _make_relative(from_path, to_path)

        assert result == PurePath("../../a/b/c.html")


class TestHTMLVisitorTraversal:
    def test_visitor_enter_and_exit_via_traversal(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)

        blank = BlankNode()
        visitor.enter(blank)
        assert len(visitor.stack) == 2
        visitor.exit(blank)
        assert len(visitor.stack) == 1

    def test_visitor_traversal_with_list_node(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)

        node = ListNode([BlankNode(), BlankNode()])
        node.visit(visitor)

        assert len(visitor.stack) == 1
        assert visitor.stack[0] is visitor.root


class TestHTMLRootChildren:
    def test_root_with_multiple_children(self) -> None:
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


class TestHTMLTagComprehensive:
    def test_tag_with_none_attribute(self) -> None:
        tag = HTMLTag("input", {"disabled": None, "type": "text"})

        result = repr(tag)
        assert "disabled" in result
        assert 'type="text"' in result

    def test_tag_multiple_children(self) -> None:
        parent = HTMLTag("div")
        first_child = HTMLTag("span")
        second_child = HTMLTag("p")
        text = TextNode("text")

        parent.append(first_child)
        parent.append(second_child)
        parent.append(text)

        assert len(list(parent.children)) == 3

    def test_replace_multiple_occurrences(self) -> None:
        parent = HTMLTag("div")
        old = HTMLTag("span")
        new = HTMLTag("p")

        parent.append(old)
        parent.append(HTMLTag("br"))
        parent.append(old)

        parent.replace_child(old, new)

        children = list(parent.children)
        assert children.count(new) == 2
        assert children.count(old) == 0


class TestTextNodeComprehensive:
    def test_text_with_html_entities(self) -> None:
        text = TextNode("&lt;script&gt;")
        assert text._value == "&lt;script&gt;"

    def test_text_with_unicode(self) -> None:
        text = TextNode("Hello \u00e9\u00e8\u00ea")
        assert "\u00e9" in text._value


class TestHTMLTransformComprehensive:
    def test_transform_blank_node(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = settings.for_plugin("docc.html.transform")

        blank = BlankNode()
        document = Document(blank)
        context = Context({Document: document})

        transform = HTMLTransform(plugin_settings)
        transform.transform(context)

        assert isinstance(document.root, HTMLRoot)
        assert document.root.extension == ".html"

    def test_transform_list_node(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = settings.for_plugin("docc.html.transform")

        list_node = ListNode([BlankNode(), BlankNode()])
        document = Document(list_node)
        context = Context({Document: document})

        transform = HTMLTransform(plugin_settings)
        transform.transform(context)

        assert isinstance(document.root, HTMLRoot)
        assert document.root.extension == ".html"


class TestHTMLContextComprehensive:
    def test_html_context_with_all_options(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.html.context": {
                                "extra_css": ["style1.css", "style2.css"],
                                "breadcrumbs": False,
                            }
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.html.context")

        ctx = HTMLContext(plugin_settings)
        html = ctx.provide()

        assert html.extra_css == ["style1.css", "style2.css"]
        assert html.breadcrumbs is False


class TestElementConversion:
    def test_tag_to_element_with_nested_text(self) -> None:
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

    def test_tag_to_element_deep_nesting(self) -> None:
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

    def test_tag_to_element_with_attributes(self) -> None:
        tag = HTMLTag("a", {"href": "/link", "class": "nav", "id": "main"})
        tag.append(TextNode("click"))

        element = tag._to_element()
        assert element.tag == "a"
        assert element.attrib["href"] == "/link"
        assert element.attrib["class"] == "nav"
        assert element.attrib["id"] == "main"
        assert element.text == "click"
