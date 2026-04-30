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
End-to-end tests for the HTML rendering pipeline.

These tests exercise the full docc pipeline path through HTML rendering:
discover -> build -> context -> transform -> HTML render -> file output.
"""

from io import StringIO
from pathlib import PurePath
from typing import Optional

from docc.context import Context
from docc.document import BlankNode, Document, ListNode
from docc.plugins.html import (
    HTML,
    HTMLRoot,
    HTMLTag,
    HTMLTransform,
    TextNode,
    blank_node,
    html_tag,
    list_node,
    references_definition,
    references_reference,
    render_reference,
    text_node,
)
from docc.plugins.references import (
    Definition,
    Index,
    IndexTransform,
    Reference,
)
from docc.settings import PluginSettings
from docc.source import Source, StringSource


class _MockSource(StringSource):
    """A minimal Source implementation for testing."""

    def __init__(
        self,
        output_path: Optional[PurePath] = None,
        relative_path: Optional[PurePath] = None,
    ) -> None:
        if output_path is None:
            output_path = PurePath("test.py")
        if relative_path is None:
            relative_path = output_path

        super().__init__("", output_path, relative_path)


class TestHTMLTransformProducesHTMLRoot:
    """
    Test that HTMLTransform replaces a document tree with an HTMLRoot.

    This exercises the pipeline step where the document tree (containing
    Definition, BlankNode, ListNode, etc.) is converted into an HTML tree
    via HTMLVisitor and the entry-point-based renderer lookup.
    """

    def test_definition_with_blank_becomes_html_root(
        self, plugin_settings: PluginSettings
    ) -> None:
        """
        A document containing a Definition wrapping a BlankNode is
        transformed into an HTMLRoot.

        The Definition renderer (references_definition) internally creates
        its own HTMLVisitor to render children, so this exercises the
        nested visitor path through the entry point system.
        """
        source = _MockSource(PurePath("docs/module.py"))
        index = Index()

        definition = Definition(
            identifier="my_module.MyClass", child=BlankNode()
        )

        document = Document(definition)
        context = Context({Document: document, Source: source, Index: index})

        # First apply IndexTransform to assign specifiers.
        index_transform = IndexTransform(plugin_settings)
        index_transform.transform(context)
        assert definition.specifier == 0

        # Then apply HTMLTransform.
        html_transform = HTMLTransform(plugin_settings)
        html_transform.transform(context)

        assert isinstance(document.root, HTMLRoot)

    def test_transform_skips_already_output_node(
        self, plugin_settings: PluginSettings
    ) -> None:
        """
        If the document root is already an OutputNode (e.g. HTMLRoot),
        HTMLTransform leaves it unchanged.
        """
        context = Context({})
        existing_root = HTMLRoot(context)
        existing_root.append(HTMLTag("div"))
        document = Document(existing_root)
        context = Context({Document: document})

        transform = HTMLTransform(plugin_settings)
        transform.transform(context)

        assert document.root is existing_root


class TestHTMLRootOutputProducesValidHTML:
    """
    Test that HTMLRoot.output() renders a full HTML document string
    containing the expected structural elements from the Jinja2 template.
    """

    def test_output_includes_static_css_links(self) -> None:
        """The rendered HTML includes links to chota and docc CSS."""
        source = _MockSource(PurePath("docs/page.py"))
        context = Context({Source: source})
        root = HTMLRoot(context)
        root.append(HTMLTag("p"))

        buffer = StringIO()
        root.output(context, buffer)
        output = buffer.getvalue()

        assert "chota.min.css" in output
        assert "docc.css" in output

    def test_output_with_extra_css(self) -> None:
        """Extra CSS files configured via HTML context appear in output."""
        source = _MockSource(PurePath("index.py"))
        html_config = HTML(extra_css=["custom.css"], breadcrumbs=True)
        context = Context({Source: source, HTML: html_config})
        root = HTMLRoot(context)
        root.append(HTMLTag("p"))

        buffer = StringIO()
        root.output(context, buffer)
        output = buffer.getvalue()

        assert "custom.css" in output

    def test_output_breadcrumbs_for_nested_path(self) -> None:
        """Breadcrumbs are rendered for documents in nested directories."""
        source = _MockSource(PurePath("a/b/c/page.py"))
        context = Context({Source: source})
        root = HTMLRoot(context)
        root.append(HTMLTag("section"))

        buffer = StringIO()
        root.output(context, buffer)
        output = buffer.getvalue()

        assert "breadcrumbs" in output
        assert "page.py" in output

    def test_output_no_breadcrumbs_when_disabled(self) -> None:
        """Breadcrumbs section is absent when breadcrumbs=False."""
        source = _MockSource(PurePath("a/b/page.py"))
        html_config = HTML(extra_css=[], breadcrumbs=False)
        context = Context({Source: source, HTML: html_config})
        root = HTMLRoot(context)
        root.append(HTMLTag("div"))

        buffer = StringIO()
        root.output(context, buffer)
        output = buffer.getvalue()

        assert "breadcrumbs" not in output

    def test_output_nested_html_tags_produce_markup(self) -> None:
        """Nested HTMLTag structures are correctly serialized."""
        source = _MockSource(PurePath("page.py"))
        context = Context({Source: source})
        root = HTMLRoot(context)

        section = HTMLTag("section", {"id": "main"})
        heading = HTMLTag("h1")
        heading.append(TextNode("Title"))
        section.append(heading)
        paragraph = HTMLTag("p")
        paragraph.append(TextNode("Body text"))
        section.append(paragraph)
        root.append(section)

        buffer = StringIO()
        root.output(context, buffer)
        output = buffer.getvalue()

        assert "<section" in output
        assert 'id="main"' in output
        assert "<h1>" in output
        assert "Title" in output
        assert "<p>" in output
        assert "Body text" in output


class TestDefinitionRendersWithId:
    """
    Test that rendering a Definition node produces HTML with an id
    attribute matching the definition's identifier and specifier.
    """

    def test_definition_blank_child_gets_id(self) -> None:
        """
        Calling references_definition directly with a Definition whose
        child is a BlankNode produces a span with the correct id.
        """
        source = _MockSource(PurePath("docs/module.py"))
        index = Index()
        location = index.define(source, "my_func")

        definition = Definition(
            identifier="my_func",
            child=BlankNode(),
            specifier=location.specifier,
        )

        context = Context({Source: source, Index: index})
        parent = HTMLRoot(context)

        result = references_definition(context, parent, definition)
        # references_definition appends to parent and returns None.
        assert result is None

        children = list(parent.children)
        assert len(children) >= 1

        first = children[0]
        assert isinstance(first, HTMLTag)
        assert first.tag_name == "span"
        assert first.attributes.get("id") == "my_func:0"

    def test_definition_with_list_child_gets_id(self) -> None:
        """
        A Definition wrapping a ListNode containing an HTMLTag gets
        the id applied to the first rendered child element.
        """
        source = _MockSource(PurePath("docs/module.py"))
        index = Index()
        location = index.define(source, "MyClass")

        # The child is an HTMLTag so that after HTMLVisitor renders
        # the ListNode's children, the first child is already an HTMLTag.
        inner_tag = HTMLTag("div", {"class": "class-def"})
        inner_tag.append(TextNode("MyClass"))
        child_list = ListNode([inner_tag])

        definition = Definition(
            identifier="MyClass",
            child=child_list,
            specifier=location.specifier,
        )

        context = Context({Source: source, Index: index})
        parent = HTMLRoot(context)

        references_definition(context, parent, definition)

        children = list(parent.children)
        assert len(children) >= 1

        first = children[0]
        assert isinstance(first, HTMLTag)
        assert first.attributes.get("id") == "MyClass:0"

    def test_definition_text_child_wrapped_in_span(self) -> None:
        """
        When the first rendered child is a TextNode, it gets wrapped
        in a <span> and the id is set on the span.
        """
        source = _MockSource(PurePath("docs/module.py"))
        index = Index()
        location = index.define(source, "some_var")

        text = TextNode("some_var")
        inner_tag = HTMLTag("p")
        inner_tag.append(text)
        child_list = ListNode([inner_tag])

        definition = Definition(
            identifier="some_var",
            child=child_list,
            specifier=location.specifier,
        )

        context = Context({Source: source, Index: index})
        parent = HTMLRoot(context)

        references_definition(context, parent, definition)

        children = list(parent.children)
        assert len(children) >= 1

        first = children[0]
        assert isinstance(first, HTMLTag)
        assert first.attributes.get("id") == "some_var:0"


class TestReferenceRendersAsLink:
    """
    Test that rendering a Reference produces an <a> tag linking to the
    definition's location.
    """

    def test_same_file_reference_has_fragment_only(self) -> None:
        """
        When the reference and definition are in the same source file,
        the href is just a fragment (no file path component).
        """
        source = _MockSource(PurePath("docs/module.py"))

        index = Index()
        index.define(source, "local_func")

        context = Context({Source: source, Index: index})
        ref = Reference(identifier="local_func")

        anchor = render_reference(context, ref)

        assert isinstance(anchor, HTMLTag)
        href = anchor.attributes.get("href", "")
        assert href is not None
        assert "local_func:0" in href

    def test_references_reference_appends_anchor(self) -> None:
        """
        The references_reference render function appends an anchor to the
        parent and returns it for child traversal when a child is present.
        """
        source = _MockSource(PurePath("docs/page.py"))
        def_source = _MockSource(PurePath("docs/target.py"))

        index = Index()
        index.define(def_source, "func")

        context = Context({Source: source, Index: index})
        parent = HTMLRoot(context)

        child_node = HTMLTag("code")
        child_node.append(TextNode("func"))
        ref = Reference(identifier="func", child=child_node)

        result = references_reference(context, parent, ref)

        # When the reference has a child, the anchor is returned so
        # the visitor can traverse into it.
        assert isinstance(result, HTMLTag)
        assert result.tag_name == "a"
        assert result in parent.children

    def test_references_reference_no_child_appends_text(self) -> None:
        """
        When a Reference has no meaningful child (BlankNode), the anchor
        gets a TextNode with the identifier and returns None.
        """
        source = _MockSource(PurePath("docs/page.py"))
        def_source = _MockSource(PurePath("docs/target.py"))

        index = Index()
        index.define(def_source, "some_id")

        context = Context({Source: source, Index: index})
        parent = HTMLRoot(context)

        ref = Reference(identifier="some_id")

        result = references_reference(context, parent, ref)

        assert result is None
        children = list(parent.children)
        assert len(children) == 1
        anchor = children[0]
        assert isinstance(anchor, HTMLTag)
        assert anchor.tag_name == "a"

        anchor_children = list(anchor.children)
        assert len(anchor_children) == 1
        text_child = anchor_children[0]
        assert isinstance(text_child, TextNode)
        assert text_child._value == "some_id"


class TestFullPipelineHTMLOutput:
    """
    Integration test: build a document tree, apply transforms, render
    to an HTML string, and verify the output contains all expected
    structural and content elements.
    """

    def test_definition_to_html_output(
        self, plugin_settings: PluginSettings
    ) -> None:
        """
        Full pipeline: Definition with BlankNode -> IndexTransform ->
        HTMLTransform -> HTMLRoot.output() -> HTML string.
        """
        source = _MockSource(PurePath("docs/example.py"))
        index = Index()

        definition = Definition(
            identifier="example.hello",
            child=BlankNode(),
        )
        tree = ListNode([definition])
        document = Document(tree)

        context = Context({Document: document, Source: source, Index: index})

        # Step 1: IndexTransform assigns specifiers.
        index_transform = IndexTransform(plugin_settings)
        index_transform.transform(context)
        assert definition.specifier == 0

        # Step 2: HTMLTransform converts the tree to HTML.
        html_transform = HTMLTransform(plugin_settings)
        html_transform.transform(context)
        root = document.root
        assert isinstance(root, HTMLRoot)

        # Step 3: Output the HTML to a string.
        buffer = StringIO()
        root.output(context, buffer)
        output = buffer.getvalue()

        # Verify full HTML document structure.
        assert "<!DOCTYPE html>" in output
        assert "<html>" in output
        assert "</html>" in output
        assert "<head>" in output
        assert "<body>" in output
        assert "<main" in output

        # Verify the definition id made it into the output.
        assert "example.hello:0" in output

    def test_render_helpers_produce_expected_results(self) -> None:
        """
        Test the individual render helper functions (blank_node,
        list_node, html_tag, text_node) that form the foundation of
        the HTML rendering pipeline.
        """
        context = Context({})
        root = HTMLRoot(context)

        # blank_node returns None and does not modify parent.
        assert blank_node(context, root, BlankNode()) is None
        assert len(list(root.children)) == 0

        # list_node returns the parent (for child traversal).
        ln = ListNode()
        assert list_node(context, root, ln) is root

        # html_tag appends the tag and returns None.
        tag = HTMLTag("div", {"class": "test"})
        assert html_tag(context, root, tag) is None
        assert tag in root.children

        # text_node appends the text and returns None.
        tn = TextNode("hello")
        assert text_node(context, root, tn) is None
        assert tn in root.children
