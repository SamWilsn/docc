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

"""Tests for mistletoe HTML render functions."""

from unittest.mock import MagicMock

import mistletoe as md
import pytest
from mistletoe import block_token as blocks
from mistletoe import span_token as spans

from docc.context import Context
from docc.plugins.html import HTMLRoot, HTMLTag, TextNode
from docc.plugins.mistletoe import (
    MarkdownNode,
    _render_auto_link,
    _render_block_code,
    _render_document,
    _render_emphasis,
    _render_escape_sequence,
    _render_heading,
    _render_image,
    _render_inline_code,
    _render_line_break,
    _render_link,
    _render_list,
    _render_list_item,
    _render_paragraph,
    _render_quote,
    _render_raw_text,
    _render_strikethrough,
    _render_strong,
    _render_table,
    _render_table_cell,
    _render_table_row,
    _render_thematic_break,
    render_html,
)


@pytest.fixture
def context() -> Context:
    return Context({})


@pytest.fixture
def html_root(context: Context) -> HTMLRoot:
    return HTMLRoot(context)


# ---------------------------------------------------------------------------
# render_html dispatch
# ---------------------------------------------------------------------------


def test_render_html_dispatches_to_document(
    context: Context, html_root: HTMLRoot
) -> None:
    node = MarkdownNode(md.Document("plain text"))
    result = render_html(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "div"


def test_render_html_dispatches_to_paragraph(
    context: Context, html_root: HTMLRoot
) -> None:
    doc = md.Document("paragraph")
    node = MarkdownNode(doc.children[0])
    result = render_html(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "p"


# ---------------------------------------------------------------------------
# Inline renderers
# ---------------------------------------------------------------------------


def test_render_strong(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("**bold**")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_strong(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "strong"
    assert result in html_root.children


def test_render_emphasis(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("*italic*")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_emphasis(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "em"


def test_render_inline_code(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("`code`")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_inline_code(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "code"


def test_render_raw_text(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("plain text")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_raw_text(context, html_root, node)

    assert result is None
    children = list(html_root.children)
    text_nodes = [c for c in children if isinstance(c, TextNode)]
    assert len(text_nodes) == 1
    assert text_nodes[0]._value == "plain text"


def test_render_strikethrough(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("~~strikethrough~~")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_strikethrough(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "del"


def test_render_escape_sequence_raises(
    context: Context, html_root: HTMLRoot
) -> None:
    node = MarkdownNode(md.Document("\\\\*"))
    paragraph = next(iter(node.children))
    escape = next(iter(paragraph.children))
    assert isinstance(escape, MarkdownNode)
    assert isinstance(escape.token, md.span_token.EscapeSequence)

    with pytest.raises(NotImplementedError):
        _render_escape_sequence(context, html_root, escape)


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------


def test_render_image(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("![alt text](image.png)")
    node = MarkdownNode(doc.children[0].children[0])

    result = _render_image(context, html_root, node)

    assert result is None
    children = list(html_root.children)
    img = next(
        c for c in children if isinstance(c, HTMLTag) and c.tag_name == "img"
    )
    assert img.attributes["src"] == "image.png"
    assert img.attributes["alt"] == "alt text"


def test_render_image_with_title(
    context: Context, html_root: HTMLRoot
) -> None:
    doc = md.Document('![alt text](image.png "title")')
    node = MarkdownNode(doc.children[0].children[0])

    _render_image(context, html_root, node)

    children = list(html_root.children)
    img = next(
        c for c in children if isinstance(c, HTMLTag) and c.tag_name == "img"
    )
    assert img.attributes.get("title") == "title"


# ---------------------------------------------------------------------------
# Link / auto-link
# ---------------------------------------------------------------------------


def test_render_link(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("[link](http://example.com)")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_link(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "a"
    assert result.attributes.get("href") == "http://example.com"


def test_render_link_with_title(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document('[link](http://example.com "title")')
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_link(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.attributes.get("title") == "title"


def test_render_auto_link_url(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("<http://example.com>")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_auto_link(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "a"


def test_render_auto_link_email(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("<test@example.com>")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_auto_link(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert "mailto:" in (result.attributes.get("href") or "")


# ---------------------------------------------------------------------------
# Block-level renderers
# ---------------------------------------------------------------------------


def test_render_heading_h1(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("# Heading 1")
    node = MarkdownNode(doc.children[0])
    result = _render_heading(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "h1"


def test_render_heading_h2(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("## Heading 2")
    node = MarkdownNode(doc.children[0])
    result = _render_heading(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "h2"


def test_render_heading_h3(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("### Heading 3")
    node = MarkdownNode(doc.children[0])
    result = _render_heading(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "h3"


def test_render_quote(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("> quoted text")
    node = MarkdownNode(doc.children[0])
    result = _render_quote(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "blockquote"


def test_render_paragraph(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("paragraph text")
    node = MarkdownNode(doc.children[0])
    result = _render_paragraph(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "p"


def test_render_block_code(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("```\ncode block\n```")
    node = MarkdownNode(doc.children[0])
    result = _render_block_code(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "code"
    children = list(html_root.children)
    pre = children[0]
    assert isinstance(pre, HTMLTag)
    assert pre.tag_name == "pre"


def test_render_document(context: Context, html_root: HTMLRoot) -> None:
    node = MarkdownNode(md.Document("document content"))
    result = _render_document(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "div"
    assert result.attributes.get("class") == "markdown"


def test_render_thematic_break(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("---")
    node = MarkdownNode(doc.children[0])
    result = _render_thematic_break(context, html_root, node)

    assert result is None
    children = list(html_root.children)
    assert any(isinstance(c, HTMLTag) and c.tag_name == "hr" for c in children)


# ---------------------------------------------------------------------------
# Line break
# ---------------------------------------------------------------------------


def test_render_line_break_hard(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("line1  \nline2")
    para = doc.children[0]

    break_token = None
    for child in para.children:
        if isinstance(child, spans.LineBreak):
            break_token = child
            break

    assert break_token is not None
    assert not break_token.soft

    node = MarkdownNode(break_token)
    result = _render_line_break(context, html_root, node)

    assert result is None
    br_tags = [
        c
        for c in html_root.children
        if isinstance(c, HTMLTag) and c.tag_name == "br"
    ]
    assert len(br_tags) == 1


def test_render_line_break_soft(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("line1\nline2")
    para = doc.children[0]

    break_token = None
    for child in para.children:
        if isinstance(child, spans.LineBreak):
            break_token = child
            break

    assert break_token is not None
    assert break_token.soft

    node = MarkdownNode(break_token)
    result = _render_line_break(context, html_root, node)

    assert result is None
    text_nodes = [c for c in html_root.children if isinstance(c, TextNode)]
    assert len(text_nodes) == 1
    assert text_nodes[0]._value == "\n"


# ---------------------------------------------------------------------------
# List / list item
# ---------------------------------------------------------------------------


def test_render_list_unordered(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("- item 1\n- item 2")
    node = MarkdownNode(doc.children[0])
    result = _render_list(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "ul"


def test_render_list_ordered(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("1. item 1\n2. item 2")
    node = MarkdownNode(doc.children[0])
    result = _render_list(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "ol"


def test_render_list_ordered_custom_start(
    context: Context, html_root: HTMLRoot
) -> None:
    doc = md.Document("5. item 1\n6. item 2")
    node = MarkdownNode(doc.children[0])
    result = _render_list(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.attributes.get("start") == 5


def test_render_list_item(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("- item")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_list_item(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "li"


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------


def test_render_table(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("| A | B |\n|---|---|\n| 1 | 2 |")
    node = MarkdownNode(doc.children[0])
    result = _render_table(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "table"


def test_render_table_with_thead(
    context: Context, html_root: HTMLRoot
) -> None:
    doc = md.Document("| A | B |\n|---|---|\n| 1 | 2 |")
    node = MarkdownNode(doc.children[0])
    result = _render_table(context, html_root, node)

    assert isinstance(result, HTMLTag)
    thead_children = [
        c
        for c in result.children
        if isinstance(c, HTMLTag) and c.tag_name == "thead"
    ]
    assert len(thead_children) == 1


def test_render_table_row(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("| A | B |\n|---|---|\n| 1 | 2 |")
    node = MarkdownNode(doc.children[0].children[0])
    result = _render_table_row(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "tr"


def test_render_table_cell(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("| A | B |\n|---|---|\n| 1 | 2 |")
    node = MarkdownNode(doc.children[0].children[0].children[0])
    result = _render_table_cell(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.tag_name == "td"


def test_render_table_cell_default_left(
    context: Context, html_root: HTMLRoot
) -> None:
    doc = md.Document("| A |\n|---|\n| 1 |")
    node = MarkdownNode(doc.children[0].children[0].children[0])
    result = _render_table_cell(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.attributes.get("align") == "left"


def test_render_table_cell_center(
    context: Context, html_root: HTMLRoot
) -> None:
    doc = md.Document("| A |\n|:---:|\n| 1 |")
    node = MarkdownNode(doc.children[0].children[0].children[0])
    result = _render_table_cell(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.attributes.get("align") == "center"


def test_render_table_cell_left(context: Context, html_root: HTMLRoot) -> None:
    doc = md.Document("| A |\n|:--- |\n| 1 |")
    node = MarkdownNode(doc.children[0].children[0].children[0])
    result = _render_table_cell(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.attributes.get("align") == "left"


def test_render_table_cell_right(
    context: Context, html_root: HTMLRoot
) -> None:
    doc = md.Document("| A |\n| ---:|\n| 1 |")
    node = MarkdownNode(doc.children[0].children[0].children[0])
    result = _render_table_cell(context, html_root, node)

    assert isinstance(result, HTMLTag)
    assert result.attributes.get("align") == "right"


def test_render_table_cell_unknown_alignment_raises(
    context: Context, html_root: HTMLRoot
) -> None:
    mock_token = MagicMock(spec=blocks.TableCell)
    mock_token.align = 99
    node = MarkdownNode(mock_token)

    with pytest.raises(NotImplementedError, match="table alignment 99"):
        _render_table_cell(context, html_root, node)
