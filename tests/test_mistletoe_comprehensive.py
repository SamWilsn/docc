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

from unittest.mock import MagicMock

import mistletoe as md
import pytest
from conftest import ReferenceChecker
from mistletoe import block_token as blocks
from mistletoe import span_token as spans

from docc.context import Context
from docc.document import BlankNode, ListNode, Visit
from docc.plugins.html import HTMLRoot, HTMLTag, TextNode
from docc.plugins.mistletoe import (
    MarkdownNode,
    _DocstringVisitor,
    _ReferenceVisitor,
    _render_auto_link,
    _render_block_code,
    _render_document,
    _render_emphasis,
    _render_escape_sequence,
    _render_heading,
    _render_html_block,
    _render_html_span,
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
    _SearchVisitor,
    render_html,
)
from docc.plugins.python import nodes


@pytest.fixture
def context() -> Context:
    return Context({})


@pytest.fixture
def html_root(context: Context) -> HTMLRoot:
    return HTMLRoot(context)


class TestMarkdownNodeComprehensive:
    def test_repr(self) -> None:
        markdown = "test"
        node = MarkdownNode(md.Document(markdown))
        result = repr(node)
        assert "MarkdownNode" in result
        assert "Document" in result

    def test_replace_child(self) -> None:
        markdown = "**bold**"
        node = MarkdownNode(md.Document(markdown))

        children = list(node.children)
        # "**bold**" produces a Document with one Paragraph child
        assert len(children) == 1
        old = children[0]
        new = BlankNode()
        node.replace_child(old, new)

        new_children = list(node.children)
        assert len(new_children) == 1
        assert new_children[0] is new

    def test_search_children_returns_false(self) -> None:
        node = MarkdownNode(md.Document("test"))
        assert node.search_children() is False

    def test_to_search_returns_text(self) -> None:
        node = MarkdownNode(md.Document("hello world"))
        result = node.to_search()
        assert "hello world" in result


class TestRenderStrong:
    def test_creates_strong_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "**bold**"
        doc = md.Document(markdown)
        para = doc.children[0]
        strong_token = para.children[0]
        node = MarkdownNode(strong_token)

        result = _render_strong(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "strong"
        children = list(html_root.children)
        assert result in children


class TestRenderEmphasis:
    def test_creates_em_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "*italic*"
        doc = md.Document(markdown)
        para = doc.children[0]
        em_token = para.children[0]
        node = MarkdownNode(em_token)

        result = _render_emphasis(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "em"


class TestRenderInlineCode:
    def test_creates_code_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "`code`"
        doc = md.Document(markdown)
        para = doc.children[0]
        code_token = para.children[0]
        node = MarkdownNode(code_token)

        result = _render_inline_code(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "code"


class TestRenderRawText:
    def test_creates_text_node(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "plain text"
        doc = md.Document(markdown)
        para = doc.children[0]
        text_token = para.children[0]
        node = MarkdownNode(text_token)

        result = _render_raw_text(context, html_root, node)

        assert result is None
        children = list(html_root.children)
        text_nodes = [c for c in children if isinstance(c, TextNode)]
        assert len(text_nodes) == 1
        assert text_nodes[0]._value == "plain text"


class TestRenderStrikethrough:
    def test_creates_del_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "~~strikethrough~~"
        doc = md.Document(markdown)
        para = doc.children[0]
        strike_token = para.children[0]
        node = MarkdownNode(strike_token)

        result = _render_strikethrough(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "del"


class TestRenderImage:
    def test_creates_img_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        mock_token = MagicMock(spec=spans.Image)
        mock_token.src = "image.png"
        mock_token.content = "alt text"
        mock_token.title = ""
        node = MarkdownNode(mock_token)

        result = _render_image(context, html_root, node)

        assert result is None
        children = list(html_root.children)
        img = next(
            c
            for c in children
            if isinstance(c, HTMLTag) and c.tag_name == "img"
        )
        assert img.attributes["src"] == "image.png"
        assert img.attributes["alt"] == "alt text"

    def test_img_with_title(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        mock_token = MagicMock(spec=spans.Image)
        mock_token.src = "image.png"
        mock_token.content = "alt"
        mock_token.title = "title"
        node = MarkdownNode(mock_token)

        _render_image(context, html_root, node)

        children = list(html_root.children)
        img = next(
            c
            for c in children
            if isinstance(c, HTMLTag) and c.tag_name == "img"
        )
        assert img.attributes.get("title") == "title"


class TestRenderLink:
    def test_creates_anchor_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "[link](http://example.com)"
        doc = md.Document(markdown)
        para = doc.children[0]
        link_token = para.children[0]
        node = MarkdownNode(link_token)

        result = _render_link(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "a"
        assert result.attributes.get("href") == "http://example.com"

    def test_link_with_title(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = '[link](http://example.com "title")'
        doc = md.Document(markdown)
        para = doc.children[0]
        link_token = para.children[0]
        node = MarkdownNode(link_token)

        result = _render_link(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.attributes.get("title") == "title"


class TestRenderAutoLink:
    def test_creates_anchor_for_url(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "<http://example.com>"
        doc = md.Document(markdown)
        para = doc.children[0]
        auto_token = para.children[0]
        node = MarkdownNode(auto_token)

        result = _render_auto_link(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "a"

    def test_mailto_prefix_for_email(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "<test@example.com>"
        doc = md.Document(markdown)
        para = doc.children[0]
        auto_token = para.children[0]
        node = MarkdownNode(auto_token)

        result = _render_auto_link(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert "mailto:" in (result.attributes.get("href") or "")


class TestRenderEscapeSequence:
    def test_raises_not_implemented(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        node = MarkdownNode(md.Document("test"))

        with pytest.raises(NotImplementedError):
            _render_escape_sequence(context, html_root, node)


class TestRenderHeading:
    def test_creates_h1_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "# Heading 1"
        doc = md.Document(markdown)
        heading_token = doc.children[0]
        node = MarkdownNode(heading_token)

        result = _render_heading(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "h1"

    def test_creates_h2_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "## Heading 2"
        doc = md.Document(markdown)
        heading_token = doc.children[0]
        node = MarkdownNode(heading_token)

        result = _render_heading(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "h2"

    def test_creates_h3_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "### Heading 3"
        doc = md.Document(markdown)
        heading_token = doc.children[0]
        node = MarkdownNode(heading_token)

        result = _render_heading(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "h3"


class TestRenderQuote:
    def test_creates_blockquote_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "> quoted text"
        doc = md.Document(markdown)
        quote_token = doc.children[0]
        node = MarkdownNode(quote_token)

        result = _render_quote(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "blockquote"


class TestRenderParagraph:
    def test_creates_p_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "paragraph text"
        doc = md.Document(markdown)
        para_token = doc.children[0]
        node = MarkdownNode(para_token)

        result = _render_paragraph(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "p"


class TestRenderBlockCode:
    def test_creates_pre_code_tags(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "```\ncode block\n```"
        doc = md.Document(markdown)
        code_token = doc.children[0]
        node = MarkdownNode(code_token)

        result = _render_block_code(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "code"
        children = list(html_root.children)
        pre = children[0]
        assert isinstance(pre, HTMLTag)
        assert pre.tag_name == "pre"


class TestRenderList:
    def test_creates_ul_for_unordered(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "- item 1\n- item 2"
        doc = md.Document(markdown)
        list_token = doc.children[0]
        node = MarkdownNode(list_token)

        result = _render_list(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "ul"

    def test_creates_ol_for_ordered(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "1. item 1\n2. item 2"
        doc = md.Document(markdown)
        list_token = doc.children[0]
        node = MarkdownNode(list_token)

        result = _render_list(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "ol"

    def test_ol_with_custom_start(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "5. item 1\n6. item 2"
        doc = md.Document(markdown)
        list_token = doc.children[0]
        node = MarkdownNode(list_token)

        result = _render_list(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.attributes.get("start") == 5


class TestRenderListItem:
    def test_creates_li_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "- item"
        doc = md.Document(markdown)
        list_token = doc.children[0]
        item_token = list_token.children[0]
        node = MarkdownNode(item_token)

        result = _render_list_item(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "li"


class TestRenderTable:
    def test_creates_table_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
        doc = md.Document(markdown)
        table_token = doc.children[0]
        node = MarkdownNode(table_token)

        result = _render_table(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "table"

    def test_table_with_header_creates_thead(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
        doc = md.Document(markdown)
        table_token = doc.children[0]
        node = MarkdownNode(table_token)

        result = _render_table(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "table"
        # Table with header row should have a <thead> child
        thead_children = [
            c
            for c in result.children
            if isinstance(c, HTMLTag) and c.tag_name == "thead"
        ]
        assert (
            len(thead_children) == 1
        ), "Table with header row should have a <thead> child element"


class TestRenderTableRow:
    def test_creates_tr_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
        doc = md.Document(markdown)
        table_token = doc.children[0]
        row_token = table_token.children[0]
        node = MarkdownNode(row_token)

        result = _render_table_row(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "tr"


class TestRenderTableCell:
    def test_creates_td_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "| A | B |\n|---|---|\n| 1 | 2 |"
        doc = md.Document(markdown)
        table_token = doc.children[0]
        row_token = table_token.children[0]
        cell_token = row_token.children[0]
        node = MarkdownNode(cell_token)

        result = _render_table_cell(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "td"

    def test_default_alignment_is_left(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "| A |\n|---|\n| 1 |"
        doc = md.Document(markdown)
        table_token = doc.children[0]
        row_token = table_token.children[0]
        cell_token = row_token.children[0]
        node = MarkdownNode(cell_token)

        result = _render_table_cell(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.attributes.get("align") == "left"

    def test_center_alignment(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "| A |\n|:---:|\n| 1 |"
        doc = md.Document(markdown)
        table_token = doc.children[0]
        row_token = table_token.children[0]
        cell_token = row_token.children[0]
        node = MarkdownNode(cell_token)

        result = _render_table_cell(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.attributes.get("align") == "center"

    def test_right_alignment(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        mock_token = MagicMock(spec=blocks.TableCell)
        mock_token.align = 2
        node = MarkdownNode(mock_token)

        result = _render_table_cell(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.attributes.get("align") == "right"

    def test_unknown_alignment_raises_not_implemented(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        mock_token = MagicMock(spec=blocks.TableCell)
        mock_token.align = 99
        node = MarkdownNode(mock_token)

        with pytest.raises(NotImplementedError, match="table alignment 99"):
            _render_table_cell(context, html_root, node)


class TestRenderThematicBreak:
    def test_creates_hr_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "---"
        doc = md.Document(markdown)
        hr_token = doc.children[0]
        node = MarkdownNode(hr_token)

        result = _render_thematic_break(context, html_root, node)

        assert result is None
        children = list(html_root.children)
        assert any(
            isinstance(c, HTMLTag) and c.tag_name == "hr" for c in children
        )


class TestRenderLineBreak:
    def test_hard_break_creates_br_tag(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        # Two trailing spaces before newline creates a hard break
        markdown = "line1  \nline2"
        doc = md.Document(markdown)
        para = doc.children[0]

        break_token = None
        for child in para.children:
            if isinstance(child, spans.LineBreak):
                break_token = child
                break

        assert break_token is not None, "Expected a LineBreak token"
        assert (
            not break_token.soft
        ), "Two trailing spaces should produce a hard break"

        node = MarkdownNode(break_token)
        result = _render_line_break(context, html_root, node)

        assert result is None
        children = list(html_root.children)
        br_tags = [
            c
            for c in children
            if isinstance(c, HTMLTag) and c.tag_name == "br"
        ]
        assert len(br_tags) == 1, "Hard break should append an HTMLTag('br')"

    def test_soft_break_creates_text_newline(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        # No trailing spaces before newline creates a soft break
        markdown = "line1\nline2"
        doc = md.Document(markdown)
        para = doc.children[0]

        break_token = None
        for child in para.children:
            if isinstance(child, spans.LineBreak):
                break_token = child
                break

        assert break_token is not None, "Expected a LineBreak token"
        assert (
            break_token.soft
        ), "No trailing spaces should produce a soft break"

        node = MarkdownNode(break_token)
        result = _render_line_break(context, html_root, node)

        assert result is None
        children = list(html_root.children)
        text_nodes = [c for c in children if isinstance(c, TextNode)]
        assert (
            len(text_nodes) == 1
        ), "Soft break should append a TextNode('\\n')"
        assert text_nodes[0]._value == "\n"


class TestRenderHtmlSpan:
    def test_parses_inline_html(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        mock_token = MagicMock(spec=spans.HTMLSpan)
        mock_token.content = "<em>hello</em>"
        node = MarkdownNode(mock_token)

        result = _render_html_span(context, html_root, node)

        assert result is None
        children = list(html_root.children)
        # "<em>hello</em>" produces exactly one <em> tag
        assert len(children) == 1
        em_tag = children[0]
        assert isinstance(em_tag, HTMLTag)
        assert em_tag.tag_name == "em"


class TestRenderHtmlBlock:
    def test_parses_block_html(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        mock_token = MagicMock(spec=blocks.HTMLBlock)
        mock_token.content = "<div>block content</div>"
        node = MarkdownNode(mock_token)

        result = _render_html_block(context, html_root, node)

        assert result is None
        children = list(html_root.children)
        # "<div>block content</div>" produces exactly one <div> tag
        assert len(children) == 1
        div_tag = children[0]
        assert isinstance(div_tag, HTMLTag)
        assert div_tag.tag_name == "div"


class TestRenderDocument:
    def test_creates_div_with_class(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "document content"
        doc = md.Document(markdown)
        node = MarkdownNode(doc)

        result = _render_document(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "div"
        assert result.attributes.get("class") == "markdown"


class TestRenderHtml:
    def test_dispatches_to_correct_renderer(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "plain text"
        doc = md.Document(markdown)
        node = MarkdownNode(doc)

        result = render_html(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "div"

    def test_renders_paragraph(
        self, context: Context, html_root: HTMLRoot
    ) -> None:
        markdown = "paragraph"
        doc = md.Document(markdown)
        para = doc.children[0]
        node = MarkdownNode(para)

        result = render_html(context, html_root, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "p"


class TestDocstringVisitorEdgeCases:
    def test_docstring_at_root_becomes_markdown(self) -> None:
        visitor = _DocstringVisitor()
        docstring = nodes.Docstring("**bold** text")

        docstring.visit(visitor)

        assert visitor.root is not None
        assert isinstance(visitor.root, MarkdownNode)

    def test_nested_docstrings(self) -> None:
        visitor = _DocstringVisitor()
        inner_doc = nodes.Docstring("inner")
        outer = ListNode([inner_doc])

        outer.visit(visitor)

        children = list(outer.children)
        assert len(children) == 1
        assert isinstance(children[0], MarkdownNode)


class TestReferenceVisitorEdgeCases:
    def test_autolink_with_ref_becomes_reference(self) -> None:
        visitor = _ReferenceVisitor()
        markdown = "<ref:test.module>"
        root = MarkdownNode(md.Document(markdown))

        root.visit(visitor)

        checker = ReferenceChecker()
        assert visitor.root is not None
        visitor.root.visit(checker)
        assert (
            checker.found
        ), "Autolink with ref: prefix should become Reference"

    def test_link_with_multiple_children(self) -> None:
        visitor = _ReferenceVisitor()
        markdown = "[**bold** text](ref:test)"
        root = MarkdownNode(md.Document(markdown))

        root.visit(visitor)

        checker = ReferenceChecker()
        assert visitor.root is not None
        visitor.root.visit(checker)
        assert (
            checker.found
        ), "Link with multiple children should become Reference"

    def test_link_without_ref_prefix_unchanged(self) -> None:
        visitor = _ReferenceVisitor()
        markdown = "[link](https://example.com)"
        root = MarkdownNode(md.Document(markdown))

        root.visit(visitor)

        assert isinstance(
            visitor.root, MarkdownNode
        ), "Root should remain MarkdownNode"
        checker = ReferenceChecker()
        assert visitor.root is not None
        visitor.root.visit(checker)
        assert (
            not checker.found
        ), "Regular links should not become Reference nodes"


class TestSearchVisitorEdgeCases:
    def test_enter_returns_traverse_children_for_non_markdown(self) -> None:
        visitor = _SearchVisitor()
        blank = BlankNode()

        result = visitor.enter(blank)

        assert result == Visit.TraverseChildren

    def test_exit_does_nothing(self) -> None:
        visitor = _SearchVisitor()
        blank = BlankNode()

        visitor.exit(blank)

    def test_raw_text_extraction(self) -> None:
        visitor = _SearchVisitor()
        markdown = "plain text content"
        node = MarkdownNode(md.Document(markdown))

        node.visit(visitor)

        assert "plain text content" in visitor.texts

    def test_to_search_joins_multiple_fragments_with_spaces(self) -> None:
        # Two paragraphs produce separate RawText tokens: "bold" and
        # "new paragraph". With " ".join() these become
        # "bold new paragraph"; with "".join() they would become
        # "boldnew paragraph".
        node = MarkdownNode(md.Document("**bold**\n\nnew paragraph"))
        result = node.to_search()

        assert "bold new" in result
