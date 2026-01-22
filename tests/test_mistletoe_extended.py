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

import typing
from pathlib import Path

import mistletoe as md
import pytest
from conftest import ReferenceChecker
from mistletoe.token import Token as MarkdownToken

from docc.context import Context
from docc.document import BlankNode, Document, ListNode
from docc.plugins.mistletoe import (
    DocstringTransform,
    MarkdownNode,
    ReferenceTransform,
    _DocstringVisitor,
    _ReferenceVisitor,
    _SearchVisitor,
)
from docc.plugins.python import nodes
from docc.plugins.references import Reference
from docc.settings import PluginSettings, Settings


@pytest.fixture
def plugin_settings() -> PluginSettings:
    settings = Settings(Path("."), {"tool": {"docc": {}}})
    return settings.for_plugin("docc.mistletoe.transform")


class TestDocstringVisitor:
    def test_enter_non_docstring_node(self) -> None:
        visitor = _DocstringVisitor()
        blank = BlankNode()

        visitor.enter(blank)
        assert visitor.root is blank

    def test_exit_non_docstring_node(self) -> None:
        visitor = _DocstringVisitor()
        blank = BlankNode()

        visitor.enter(blank)
        visitor.exit(blank)

        assert len(visitor.stack) == 0

    def test_transforms_docstring_to_markdown(self) -> None:
        visitor = _DocstringVisitor()
        docstring = nodes.Docstring("Test **bold**")
        parent = ListNode([docstring])

        parent.visit(visitor)

        children = list(parent.children)
        assert len(children) == 1
        assert not isinstance(children[0], nodes.Docstring)


class TestReferenceVisitor:
    def test_enter_non_markdown_node(self) -> None:
        visitor = _ReferenceVisitor()
        blank = BlankNode()

        visitor.enter(blank)
        assert visitor.root is blank

    def test_exit_non_markdown_node(self) -> None:
        visitor = _ReferenceVisitor()
        blank = BlankNode()

        visitor.enter(blank)
        visitor.exit(blank)

        assert len(visitor.stack) == 0

    def test_transforms_ref_link_to_reference(self) -> None:
        visitor = _ReferenceVisitor()
        markdown = "[text](ref:identifier)"
        root = MarkdownNode(md.Document(markdown))

        root.visit(visitor)

        checker = ReferenceChecker()
        assert visitor.root is not None
        visitor.root.visit(checker)
        assert (
            checker.found
        ), "ref: link should be transformed to Reference node"

    def test_ignores_non_ref_links(self) -> None:
        visitor = _ReferenceVisitor()
        markdown = "[text](https://example.com)"
        root = MarkdownNode(md.Document(markdown))

        root.visit(visitor)

        checker = ReferenceChecker()
        assert visitor.root is not None
        visitor.root.visit(checker)
        assert (
            not checker.found
        ), "Regular links should not become Reference nodes"

    def test_ref_link_becomes_root_when_only_element(self) -> None:
        visitor = _ReferenceVisitor()
        markdown = "[ref](ref:test)"
        token = md.Document(markdown)
        paragraph = token.children[0]
        link_token = paragraph.children[0]
        link_node = MarkdownNode(link_token)
        link_node.visit(visitor)

        assert isinstance(
            visitor.root, Reference
        ), "Single ref link should become Reference root"


class TestSearchVisitor:
    def test_collect_empty(self) -> None:
        result = _SearchVisitor.collect([])
        assert result == []

    def test_collect_blank_node(self) -> None:
        blank = BlankNode()
        result = _SearchVisitor.collect(blank)
        assert result == []

    def test_collect_single_node(self) -> None:
        markdown = "Hello world"
        node = MarkdownNode(md.Document(markdown))
        result = _SearchVisitor.collect(node)
        assert "Hello world" in " ".join(result)

    def test_collect_multiple_nodes(self) -> None:
        nodes_list = [
            MarkdownNode(md.Document("First")),
            MarkdownNode(md.Document("Second")),
        ]
        result = _SearchVisitor.collect(nodes_list)
        combined = " ".join(result)
        assert "First" in combined
        assert "Second" in combined


class TestDocstringTransform:
    def test_transform_simple_docstring(
        self, plugin_settings: PluginSettings
    ) -> None:
        docstring = nodes.Docstring("A simple docstring")
        root = ListNode([docstring])
        document = Document(root)
        context = Context({Document: document})

        transform = DocstringTransform(plugin_settings)
        transform.transform(context)

        children = list(context[Document].root.children)
        assert not isinstance(children[0], nodes.Docstring)


class TestReferenceTransform:
    def test_transform_creates_references(
        self, plugin_settings: PluginSettings
    ) -> None:
        markdown = "[link](ref:test.module)"
        root = MarkdownNode(md.Document(markdown))
        document = Document(root)
        context = Context({Document: document})

        transform = ReferenceTransform(plugin_settings)
        transform.transform(context)

        checker = ReferenceChecker()
        context[Document].root.visit(checker)
        assert (
            checker.found
        ), "Transform should create Reference nodes from ref: links"


class TestMarkdownNodeChildren:
    def test_children_with_token_children_none(self) -> None:
        class MockToken:
            children = None

        node = MarkdownNode(typing.cast(MarkdownToken, MockToken()))
        children = list(node.children)
        assert children == []

    def test_children_lazy_evaluation(self) -> None:
        markdown = "Test **bold**"
        node = MarkdownNode(md.Document(markdown))

        assert node._children is None

        children = list(node.children)

        assert node._children is not None
        assert len(children) > 0


class TestMarkdownFormats:
    def test_strong_text(self) -> None:
        markdown = "**strong**"
        node = MarkdownNode(md.Document(markdown))
        result = node.to_search()
        assert "strong" in result

    def test_emphasis_text(self) -> None:
        markdown = "*emphasis*"
        node = MarkdownNode(md.Document(markdown))
        result = node.to_search()
        assert "emphasis" in result

    def test_code_block(self) -> None:
        markdown = "```python\ncode\n```"
        node = MarkdownNode(md.Document(markdown))
        result = node.to_search()
        assert "code" in result, "Code block content should be searchable"

    def test_list_items(self) -> None:
        markdown = "- item 1\n- item 2"
        node = MarkdownNode(md.Document(markdown))
        result = node.to_search()
        assert "item 1" in result
        assert "item 2" in result

    def test_heading(self) -> None:
        markdown = "# Heading"
        node = MarkdownNode(md.Document(markdown))
        result = node.to_search()
        assert "Heading" in result

    def test_link_with_text(self) -> None:
        markdown = "[link text](http://example.com)"
        node = MarkdownNode(md.Document(markdown))
        result = node.to_search()
        assert "link text" in result

    def test_mixed_content(self) -> None:
        markdown = """
# Title

Paragraph with **bold** and *italic*.

- List item
- Another item

[Link](http://example.com)
"""
        node = MarkdownNode(md.Document(markdown))
        result = node.to_search()
        assert "Title" in result
        assert "bold" in result
        assert "List item" in result
