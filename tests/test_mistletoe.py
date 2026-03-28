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
Tests for MarkdownNode, mistletoe visitors, and transforms.
"""

import typing
from pathlib import Path
from typing import Any

import mistletoe as md
import pytest
from conftest import ReferenceChecker
from mistletoe.token import Token as MarkdownToken

from docc.context import Context
from docc.document import BlankNode, Document, ListNode, Visit
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


# ---------------------------------------------------------------------------
# MarkdownNode
# ---------------------------------------------------------------------------


def test_markdown_node_repr() -> None:
    node = MarkdownNode(md.Document("test"))
    result = repr(node)
    assert "MarkdownNode" in result
    assert "Document" in result


def test_markdown_node_replace_child() -> None:
    node = MarkdownNode(md.Document("**bold**"))

    children = list(node.children)
    assert len(children) == 1
    old = children[0]
    new = BlankNode()
    node.replace_child(old, new)

    new_children = list(node.children)
    assert len(new_children) == 1
    assert new_children[0] is new


def test_markdown_node_search_children() -> None:
    node = MarkdownNode(md.Document("test"))
    assert node.search_children() is False


def test_markdown_node_to_search() -> None:
    node = MarkdownNode(md.Document("hello world"))
    result = node.to_search()
    assert "hello world" in result


def test_markdown_node_children_none_token() -> None:
    class MockToken:
        children = None

    node = MarkdownNode(typing.cast(MarkdownToken, MockToken()))
    children = list(node.children)
    assert children == []


def test_markdown_node_to_search_joins_fragments() -> None:
    node = MarkdownNode(md.Document("**bold**\n\nnew paragraph"))
    result = node.to_search()
    assert "bold new" in result


# ---------------------------------------------------------------------------
# MarkdownNode search formats
# ---------------------------------------------------------------------------


def test_markdown_search_strong() -> None:
    node = MarkdownNode(md.Document("**strong**"))
    assert "strong" in node.to_search()


def test_markdown_search_emphasis() -> None:
    node = MarkdownNode(md.Document("*emphasis*"))
    assert "emphasis" in node.to_search()


def test_markdown_search_code_block() -> None:
    node = MarkdownNode(md.Document("```python\ncode\n```"))
    assert "code" in node.to_search()


def test_markdown_search_list_items() -> None:
    result = MarkdownNode(md.Document("- item 1\n- item 2")).to_search()
    assert "item 1" in result
    assert "item 2" in result


def test_markdown_search_heading() -> None:
    assert "Heading" in MarkdownNode(md.Document("# Heading")).to_search()


def test_markdown_search_link_text() -> None:
    result = MarkdownNode(
        md.Document("[link text](http://example.com)")
    ).to_search()
    assert "link text" in result


def test_markdown_search_mixed_content() -> None:
    markdown = """
# Title

Paragraph with **bold** and *italic*.

- List item
- Another item

[Link](http://example.com)
"""
    result = MarkdownNode(md.Document(markdown)).to_search()
    assert "Title" in result
    assert "bold" in result
    assert "List item" in result


# ---------------------------------------------------------------------------
# _DocstringVisitor
# ---------------------------------------------------------------------------


def test_docstring_visitor_enter_non_docstring() -> None:
    visitor = _DocstringVisitor()
    blank = BlankNode()

    visitor.enter(blank)
    assert visitor.root is blank


def test_docstring_visitor_exit_non_docstring() -> None:
    visitor = _DocstringVisitor()
    blank = BlankNode()

    visitor.enter(blank)
    visitor.exit(blank)

    assert len(visitor.stack) == 0


def test_docstring_visitor_transforms_to_markdown() -> None:
    visitor = _DocstringVisitor()
    docstring = nodes.Docstring("Test **bold**")
    parent = ListNode([docstring])

    parent.visit(visitor)

    children = list(parent.children)
    assert len(children) == 1
    assert not isinstance(children[0], nodes.Docstring)


def test_docstring_visitor_at_root_becomes_markdown() -> None:
    visitor = _DocstringVisitor()
    docstring = nodes.Docstring("**bold** text")

    docstring.visit(visitor)

    assert visitor.root is not None
    assert isinstance(visitor.root, MarkdownNode)


def test_docstring_visitor_nested() -> None:
    visitor = _DocstringVisitor()
    inner_doc = nodes.Docstring("inner")
    outer = ListNode([inner_doc])

    outer.visit(visitor)

    children = list(outer.children)
    assert len(children) == 1
    assert isinstance(children[0], MarkdownNode)


# ---------------------------------------------------------------------------
# _ReferenceVisitor
# ---------------------------------------------------------------------------


def test_reference_visitor_enter_non_markdown() -> None:
    visitor = _ReferenceVisitor()
    blank = BlankNode()

    visitor.enter(blank)
    assert visitor.root is blank


def test_reference_visitor_exit_non_markdown() -> None:
    visitor = _ReferenceVisitor()
    blank = BlankNode()

    visitor.enter(blank)
    visitor.exit(blank)

    assert len(visitor.stack) == 0


def test_reference_visitor_transforms_ref_link() -> None:
    visitor = _ReferenceVisitor()
    root = MarkdownNode(md.Document("[text](ref:identifier)"))
    root.visit(visitor)

    checker = ReferenceChecker()
    assert visitor.root is not None
    visitor.root.visit(checker)
    assert checker.found


def test_reference_visitor_ignores_non_ref() -> None:
    visitor = _ReferenceVisitor()
    root = MarkdownNode(md.Document("[text](https://example.com)"))
    root.visit(visitor)

    checker = ReferenceChecker()
    assert visitor.root is not None
    visitor.root.visit(checker)
    assert not checker.found


def test_reference_visitor_ref_link_becomes_root() -> None:
    visitor = _ReferenceVisitor()
    doc = md.Document("[ref](ref:test)")
    link_node = MarkdownNode(doc.children[0].children[0])
    link_node.visit(visitor)

    assert isinstance(visitor.root, Reference)


def test_reference_visitor_autolink_ref() -> None:
    visitor = _ReferenceVisitor()
    root = MarkdownNode(md.Document("<ref:test.module>"))
    root.visit(visitor)

    checker = ReferenceChecker()
    assert visitor.root is not None
    visitor.root.visit(checker)
    assert checker.found


def test_reference_visitor_link_multiple_children() -> None:
    visitor = _ReferenceVisitor()
    root = MarkdownNode(md.Document("[**bold** text](ref:test)"))
    root.visit(visitor)

    checker = ReferenceChecker()
    assert visitor.root is not None
    visitor.root.visit(checker)
    assert checker.found


def test_reference_visitor_non_ref_unchanged() -> None:
    visitor = _ReferenceVisitor()
    root = MarkdownNode(md.Document("[link](https://example.com)"))
    root.visit(visitor)

    assert isinstance(visitor.root, MarkdownNode)
    checker = ReferenceChecker()
    assert visitor.root is not None
    visitor.root.visit(checker)
    assert not checker.found


# ---------------------------------------------------------------------------
# _SearchVisitor
# ---------------------------------------------------------------------------


def test_search_visitor_enter_non_markdown() -> None:
    visitor = _SearchVisitor()
    result = visitor.enter(BlankNode())
    assert result == Visit.TraverseChildren


def test_search_visitor_exit_does_nothing() -> None:
    visitor = _SearchVisitor()
    visitor.exit(BlankNode())


def test_search_visitor_raw_text_extraction() -> None:
    visitor = _SearchVisitor()
    node = MarkdownNode(md.Document("plain text content"))
    node.visit(visitor)
    assert "plain text content" in visitor.texts


def test_search_visitor_collect_empty() -> None:
    assert _SearchVisitor.collect([]) == []


def test_search_visitor_collect_blank_node() -> None:
    assert _SearchVisitor.collect(BlankNode()) == []


def test_search_visitor_collect_single_node() -> None:
    node = MarkdownNode(md.Document("Hello world"))
    result = _SearchVisitor.collect(node)
    assert "Hello world" in " ".join(result)


def test_search_visitor_collect_multiple_nodes() -> None:
    nodes_list = [
        MarkdownNode(md.Document("First")),
        MarkdownNode(md.Document("Second")),
    ]
    result = _SearchVisitor.collect(nodes_list)
    combined = " ".join(result)
    assert "First" in combined
    assert "Second" in combined


def test_search_visitor_collect_formatted_text() -> None:
    root = MarkdownNode(md.Document("Hello **world** and *everyone*"))
    texts = _SearchVisitor.collect(root)
    combined = " ".join(texts)
    assert "Hello" in combined
    assert "world" in combined
    assert "everyone" in combined


# ---------------------------------------------------------------------------
# DocstringTransform
# ---------------------------------------------------------------------------


def test_docstring_transform(
    plugin_settings: PluginSettings,
) -> None:
    docstring = nodes.Docstring("A simple docstring")
    root = ListNode([docstring])
    document = Document(root)
    context = Context({Document: document})

    transform = DocstringTransform(plugin_settings)
    transform.transform(context)

    children = list(context[Document].root.children)
    assert not isinstance(children[0], nodes.Docstring)


# ---------------------------------------------------------------------------
# ReferenceTransform
# ---------------------------------------------------------------------------


def test_reference_transform_creates_references(
    plugin_settings: PluginSettings,
) -> None:
    root = MarkdownNode(md.Document("[link](ref:test.module)"))
    document = Document(root)
    context = Context({Document: document})

    transform = ReferenceTransform(plugin_settings)
    transform.transform(context)

    checker = ReferenceChecker()
    context[Document].root.visit(checker)
    assert checker.found


def test_reference_transform_to_reference(
    assert_in: Any,
    make_context: Any,
    plugin_settings: PluginSettings,
) -> None:
    root = MarkdownNode(md.Document("[a reference](ref:hello-world)"))
    context = make_context(root)

    transform = ReferenceTransform(plugin_settings)
    transform.transform(context)

    from docc.document import Node

    def matcher(node: Node) -> bool:
        if not isinstance(node, Reference):
            return False
        return node.identifier == "hello-world"

    assert_in(root, matcher)


def test_reference_transform_no_reference(
    assert_not_in: Any,
    make_context: Any,
    plugin_settings: PluginSettings,
) -> None:
    root = MarkdownNode(md.Document("[a reference](http://hello-world)"))
    context = make_context(root)

    transform = ReferenceTransform(plugin_settings)
    transform.transform(context)

    from docc.document import Node

    def matcher(node: Node) -> bool:
        if isinstance(node, Reference):
            return True
        return False

    assert_not_in(root, matcher)
