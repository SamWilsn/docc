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
from io import StringIO
from pathlib import Path, PurePath
from typing import Iterator, Optional
from unittest.mock import patch

import pytest

from docc.context import Context
from docc.document import BlankNode, Document, ListNode, Node, Visit
from docc.plugins.html import (
    HTML,
    HTMLContext,
    HTMLDiscover,
    HTMLParser,
    HTMLRoot,
    HTMLTag,
    HTMLTransform,
    HTMLVisitor,
    TextNode,
    _ElementTreeVisitor,
    _make_relative,
    blank_node,
    html_tag,
    list_node,
    render_reference,
    text_node,
)
from docc.plugins.loader import PluginError
from docc.plugins.references import Index, Reference
from docc.settings import PluginSettings, Settings, SettingsError
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


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


@pytest.fixture
def basic_settings(temp_dir: Path) -> Settings:
    return Settings(temp_dir, {"tool": {"docc": {}}})


@pytest.fixture
def plugin_settings(basic_settings: Settings) -> PluginSettings:
    return basic_settings.for_plugin("docc.html")


class TestTextNode:
    def test_init_with_value(self) -> None:
        node = TextNode("hello")
        assert node._value == "hello"

    def test_children_empty(self) -> None:
        node = TextNode("test")
        assert tuple(node.children) == ()

    def test_replace_child_raises(self) -> None:
        node = TextNode("test")
        with pytest.raises(TypeError, match="text nodes have no children"):
            node.replace_child(BlankNode(), BlankNode())

    def test_repr(self) -> None:
        node = TextNode("hello world")
        assert repr(node) == "'hello world'"

    def test_repr_with_double_quote(self) -> None:
        node = TextNode('say "hello"')
        assert repr(node) == "'say \"hello\"'"

    def test_repr_with_single_quote(self) -> None:
        node = TextNode("it's")
        assert repr(node) == '"it\'s"'


class TestHTMLTag:
    def test_init_basic(self) -> None:
        tag = HTMLTag("div")
        assert tag.tag_name == "div"
        assert tag.attributes == {}
        assert list(tag.children) == []

    def test_init_with_attributes(self) -> None:
        tag = HTMLTag("a", {"href": "/test", "class": "link"})
        assert tag.attributes["href"] == "/test"
        assert tag.attributes["class"] == "link"

    def test_append_child(self) -> None:
        parent = HTMLTag("div")
        existing = HTMLTag("p")
        parent.append(existing)
        child = HTMLTag("span")
        parent.append(child)
        children = list(parent.children)
        assert children[-1] is child

    def test_append_text(self) -> None:
        tag = HTMLTag("p")
        existing = HTMLTag("span")
        tag.append(existing)
        text = TextNode("hello")
        tag.append(text)
        children = list(tag.children)
        assert children[-1] is text

    def test_replace_child(self) -> None:
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

    def test_replace_child_with_duplicate_children(self) -> None:
        parent = HTMLTag("div")
        child = HTMLTag("a")
        parent.append(child)
        parent.append(child)

        new = HTMLTag("em")
        parent.replace_child(child, new)

        children = list(parent.children)
        assert child not in children
        assert children.count(new) == 2

    def test_repr_basic(self) -> None:
        tag = HTMLTag("div")
        assert repr(tag) == "<div>"

    def test_repr_with_attributes(self) -> None:
        tag = HTMLTag("a", {"href": "/test"})
        result = repr(tag)
        assert "<a" in result
        assert 'href="/test"' in result

    def test_repr_escapes_attribute_values(self) -> None:
        tag = HTMLTag("div", {"data-value": 'test"quote'})
        result = repr(tag)
        assert "&quot;" in result

    def test_repr_attribute_without_value(self) -> None:
        tag = HTMLTag("input", {"disabled": None})
        result = repr(tag)
        assert "disabled" in result


class TestHTMLRoot:
    def test_init(self) -> None:
        context = Context({})
        root = HTMLRoot(context)
        assert list(root.children) == []

    def test_append(self) -> None:
        context = Context({})
        root = HTMLRoot(context)
        tag = HTMLTag("div")
        root.append(tag)
        assert tag in root.children

    def test_replace_child(self) -> None:
        context = Context({})
        root = HTMLRoot(context)
        old = HTMLTag("div")
        new = HTMLTag("span")
        root.append(old)

        root.replace_child(old, new)
        assert old not in root.children
        assert new in root.children

    def test_extension(self) -> None:
        context = Context({})
        root = HTMLRoot(context)
        assert root.extension == ".html"


class TestHTMLContext:
    def test_provides_html_type(self) -> None:
        assert HTMLContext.provides() == HTML

    def test_init_default_values(
        self, plugin_settings: PluginSettings
    ) -> None:
        ctx = HTMLContext(plugin_settings)
        html = ctx.provide()
        assert html.extra_css == []
        assert html.breadcrumbs is True

    def test_init_with_extra_css(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.html.context": {
                                "extra_css": ["custom.css", "theme.css"]
                            }
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.html.context")
        ctx = HTMLContext(plugin_settings)
        html = ctx.provide()
        assert html.extra_css == ["custom.css", "theme.css"]

    def test_init_invalid_extra_css_raises(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {
                "tool": {
                    "docc": {
                        "plugins": {"docc.html.context": {"extra_css": [123]}}
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.html.context")
        with pytest.raises(SettingsError, match="extra_css"):
            HTMLContext(plugin_settings)

    def test_init_breadcrumbs_false(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.html.context": {"breadcrumbs": False}
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.html.context")
        ctx = HTMLContext(plugin_settings)
        html = ctx.provide()
        assert html.breadcrumbs is False

    def test_init_invalid_breadcrumbs_raises(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.html.context": {"breadcrumbs": "yes"}
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.html.context")
        with pytest.raises(SettingsError, match="breadcrumbs"):
            HTMLContext(plugin_settings)


class TestHTMLDiscover:
    def test_discover_yields_static_resources(
        self, plugin_settings: PluginSettings
    ) -> None:
        discover = HTMLDiscover(plugin_settings)
        sources = list(discover.discover(frozenset()))

        assert len(sources) == 4

        output_paths = [str(s.output_path) for s in sources]
        assert any("chota" in p for p in output_paths)
        assert any("docc" in p for p in output_paths)
        assert any("fuse" in p for p in output_paths)
        assert any("search" in p for p in output_paths)


class TestHTMLParser:
    def test_parse_simple_tag(self) -> None:
        context = Context({})
        parser = HTMLParser(context)
        parser.feed("<div>hello</div>")

        children = list(parser.root.children)
        assert len(children) == 1
        child = children[0]
        assert isinstance(child, HTMLTag)
        assert child.tag_name == "div"

    def test_parse_nested_tags(self) -> None:
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

    def test_parse_with_attributes(self) -> None:
        context = Context({})
        parser = HTMLParser(context)
        parser.feed('<a href="/test" class="link">click</a>')

        children = list(parser.root.children)
        anchor = children[0]
        assert isinstance(anchor, HTMLTag)
        assert anchor.attributes["href"] == "/test"
        assert anchor.attributes["class"] == "link"

    def test_parse_text_content(self) -> None:
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

    def test_parse_multiple_elements(self) -> None:
        context = Context({})
        parser = HTMLParser(context)
        parser.feed("<p>one</p><p>two</p>")

        children = list(parser.root.children)
        assert len(children) == 2


class TestElementTreeVisitor:
    def test_basic_tag(self) -> None:
        tag = HTMLTag("div")
        visitor = _ElementTreeVisitor()
        tag.visit(visitor)
        element = visitor.builder.close()

        assert element.tag == "div"

    def test_tag_with_attributes(self) -> None:
        tag = HTMLTag("a", {"href": "/test"})
        visitor = _ElementTreeVisitor()
        tag.visit(visitor)
        element = visitor.builder.close()

        assert element.attrib["href"] == "/test"

    def test_nested_tags(self) -> None:
        parent = HTMLTag("div")
        child = HTMLTag("span")
        parent.append(child)

        visitor = _ElementTreeVisitor()
        parent.visit(visitor)
        element = visitor.builder.close()

        assert element.tag == "div"
        assert len(list(element)) == 1
        assert list(element)[0].tag == "span"

    def test_text_node(self) -> None:
        tag = HTMLTag("p")
        tag.append(TextNode("hello"))

        visitor = _ElementTreeVisitor()
        tag.visit(visitor)
        element = visitor.builder.close()

        assert element.text == "hello"


class TestMakeRelative:
    def test_same_path_returns_none(self) -> None:
        path = PurePath("a/b/c")
        result = _make_relative(path, path)
        assert result is None

    def test_sibling_file(self) -> None:
        from_path = PurePath("a/b/c.html")
        to_path = PurePath("a/b/d.html")
        result = _make_relative(from_path, to_path)
        assert result == PurePath("d.html")

    def test_parent_directory(self) -> None:
        from_path = PurePath("a/b/c.html")
        to_path = PurePath("a/d.html")
        result = _make_relative(from_path, to_path)
        assert result == PurePath("../d.html")

    def test_deeper_directory(self) -> None:
        from_path = PurePath("a/b.html")
        to_path = PurePath("a/c/d.html")
        result = _make_relative(from_path, to_path)
        assert result == PurePath("c/d.html")


class TestRenderFunctions:
    def test_blank_node_returns_none(self) -> None:
        context = Context({})
        parent = HTMLRoot(context)
        blank = BlankNode()

        result = blank_node(context, parent, blank)
        assert result is None

    def test_list_node_returns_parent(self) -> None:
        context = Context({})
        parent = HTMLRoot(context)
        node = ListNode()

        result = list_node(context, parent, node)
        assert result is parent

    def test_html_tag_appends_to_parent(self) -> None:
        context = Context({})
        parent = HTMLRoot(context)
        tag = HTMLTag("div")

        result = html_tag(context, parent, tag)
        assert result is None
        assert tag in parent.children

    def test_text_node_appends_to_parent(self) -> None:
        context = Context({})
        parent = HTMLRoot(context)
        text = TextNode("hello")

        result = text_node(context, parent, text)
        assert result is None
        assert text in parent.children


class TestHTMLTransform:
    def test_transform_skips_output_nodes(
        self, plugin_settings: PluginSettings
    ) -> None:
        context_obj = Context({})
        root = HTMLRoot(context_obj)
        document = Document(root)
        context = Context({Document: document})

        transform = HTMLTransform(plugin_settings)
        transform.transform(context)

        assert context[Document].root is root


class TestHTMLEdgeCases:
    def test_html_tag_none_attribute_value(self) -> None:
        tag = HTMLTag("input", {"disabled": None, "type": "text"})
        result = repr(tag)
        assert "disabled" in result
        assert 'type="text"' in result

    def test_deeply_nested_html(self) -> None:
        context = Context({})
        parser = HTMLParser(context)
        html = "<div><div><div><div><span>deep</span></div></div></div></div>"
        parser.feed(html)

        children = list(parser.root.children)
        assert len(children) == 1

    def test_html_with_special_characters(self) -> None:
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


class TestHTMLRootOutput:
    def test_output_renders_html_document(self) -> None:
        source = MockSource(PurePath("docs/page.html"))
        context = Context({Source: source})
        root = HTMLRoot(context)

        div = HTMLTag("div", {"class": "content"})
        div.append(TextNode("Hello World"))
        root.append(div)

        dest = StringIO()
        root.output(context, dest)
        output = dest.getvalue()

        assert "<!DOCTYPE html>" in output
        assert "<html>" in output
        assert "</html>" in output
        assert "Hello World" in output
        assert "<div" in output
        assert "content" in output
        assert "<head>" in output
        assert "<body>" in output

    def test_output_with_text_node_child(self) -> None:
        source = MockSource(PurePath("index.html"))
        context = Context({Source: source})
        root = HTMLRoot(context)

        root.append(TextNode("raw text"))

        dest = StringIO()
        root.output(context, dest)
        output = dest.getvalue()

        assert "raw text" in output
        assert "<!DOCTYPE html>" in output

    def test_output_breadcrumbs_for_nested_path(self) -> None:
        source = MockSource(PurePath("a/b/page.html"))
        context = Context({Source: source})
        root = HTMLRoot(context)
        root.append(HTMLTag("p"))

        dest = StringIO()
        root.output(context, dest)
        output = dest.getvalue()

        assert "breadcrumbs" in output
        assert "page.html" in output


class TestHTMLVisitorEnter:
    def test_enter_no_renderer_found(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)

        class _UnregisteredNode(Node):
            @property
            def children(self):
                return ()

            def replace_child(self, old, new):
                pass

        node = _UnregisteredNode()
        with pytest.raises(PluginError, match="no renderer found"):
            visitor.enter(node)

    def test_enter_renderer_not_callable(self) -> None:
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

    def test_enter_successful_render_pushes_to_stack(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)

        blank = BlankNode()
        initial_stack_len = len(visitor.stack)

        result = visitor.enter(blank)

        assert result == Visit.SkipChildren
        assert len(visitor.stack) == initial_stack_len + 1

    def test_enter_returning_tag_pushes_and_traverses(self) -> None:
        context = Context({})
        visitor = HTMLVisitor(context)

        parent_tag = HTMLTag("section")
        visitor.root.append(parent_tag)
        visitor.stack.append(parent_tag)

        result_tag: HTMLTag = HTMLTag("span")

        def fake_renderer(
            ctx: object, parent: object, node: object
        ) -> HTMLTag:
            return result_tag

        visitor.renderers[ListNode] = fake_renderer

        node = ListNode()
        result = visitor.enter(node)

        assert result == Visit.TraverseChildren
        assert visitor.stack[-1] is result_tag


class TestRenderReference:
    def test_single_definition(self) -> None:
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

    def test_multiple_definitions(self) -> None:
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

    def test_no_definitions_raises(self) -> None:
        source = MockSource(PurePath("docs/page.py"))
        index = Index()

        context = Context({Source: source, Index: index})
        ref = Reference(identifier="nonexistent")

        from docc.plugins.references import ReferenceError

        with pytest.raises(ReferenceError):
            render_reference(context, ref)
