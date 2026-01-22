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

from io import StringIO
from pathlib import Path, PurePath
from typing import Any, List, Optional, Sequence, TextIO

import pytest

from docc.context import Context
from docc.document import BlankNode, Document
from docc.plugins.html import HTMLTag, TextNode
from docc.plugins.verbatim import (
    Fragment,
    Highlight,
    Line,
    Pos,
    Text,
    Transcribe,
    Transcribed,
    Verbatim,
    VerbatimVisitor,
    _BoundsVisitor,
)
from docc.settings import PluginSettings, Settings
from docc.source import TextSource


class MockTextSource(TextSource):
    _path: PurePath

    def __init__(self, content: str, path: Optional[PurePath] = None) -> None:
        self._content = content
        self._path = path if path is not None else PurePath("test.py")

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._path

    @property
    def output_path(self) -> PurePath:
        return self._path

    def open(self) -> TextIO:
        return StringIO(self._content)


@pytest.fixture
def plugin_settings() -> PluginSettings:
    settings = Settings(Path("."), {"tool": {"docc": {}}})
    return settings.for_plugin("docc.verbatim.transform")


class TestPos:
    def test_create(self) -> None:
        pos = Pos(line=1, column=5)
        assert pos.line == 1
        assert pos.column == 5

    def test_frozen(self) -> None:
        pos = Pos(line=1, column=5)
        with pytest.raises(AttributeError):
            pos.line = 2  # pyre-ignore[41]

    def test_repr(self) -> None:
        pos = Pos(line=10, column=20)
        assert repr(pos) == "10:20"

    def test_ordering(self) -> None:
        first_pos = Pos(line=1, column=0)
        second_pos = Pos(line=1, column=5)
        third_pos = Pos(line=2, column=0)

        assert first_pos < second_pos < third_pos

    def test_equality(self) -> None:
        first_pos = Pos(line=1, column=5)
        second_pos = Pos(line=1, column=5)
        assert first_pos == second_pos


class TestText:
    def test_create(self) -> None:
        text = Text(text="hello")
        assert text.text == "hello"

    def test_children_empty(self) -> None:
        text = Text(text="hello")
        assert text.children == ()

    def test_replace_child_raises(self) -> None:
        text = Text(text="hello")
        with pytest.raises(TypeError):
            text.replace_child(BlankNode(), BlankNode())


class TestLine:
    def test_create(self) -> None:
        line = Line(number=1)
        assert line.number == 1
        assert list(line.children) == []

    def test_children_with_content(self) -> None:
        line = Line(number=1, _children=[Text("hello")])
        children = list(line.children)
        assert len(children) == 1
        assert isinstance(children[0], Text)

    def test_replace_child(self) -> None:
        old = Text("old")
        new = Text("new")
        line = Line(number=1, _children=[old])

        line.replace_child(old, new)

        children = list(line.children)
        assert new in children
        assert old not in children

    def test_repr(self) -> None:
        line = Line(number=5)
        assert "Line" in repr(line)
        assert "5" in repr(line)


class TestHighlight:
    def test_create(self) -> None:
        highlight = Highlight(highlights=["keyword"])
        assert highlight.highlights == ["keyword"]

    def test_children_empty(self) -> None:
        highlight = Highlight()
        assert list(highlight.children) == []

    def test_children_with_content(self) -> None:
        text = Text("highlighted")
        highlight = Highlight(_children=[text])
        assert text in highlight.children

    def test_replace_child(self) -> None:
        old = Text("old")
        new = Text("new")
        highlight = Highlight(_children=[old])

        highlight.replace_child(old, new)

        assert new in highlight.children
        assert old not in highlight.children

    def test_repr(self) -> None:
        highlight = Highlight(highlights=["keyword", "function"])
        result = repr(highlight)
        assert "Highlight" in result
        assert "keyword" in result


class TestTranscribed:
    def test_create(self) -> None:
        transcribed = Transcribed()
        assert list(transcribed.children) == []

    def test_children(self) -> None:
        line = Line(number=1)
        transcribed = Transcribed(_children=[line])
        assert line in transcribed.children

    def test_replace_child(self) -> None:
        old = Line(number=1)
        new = Line(number=2)
        transcribed = Transcribed(_children=[old])

        transcribed.replace_child(old, new)

        assert new in transcribed.children
        assert old not in transcribed.children

    def test_repr(self) -> None:
        transcribed = Transcribed()
        assert repr(transcribed) == "Transcribed(...)"


class TestFragment:
    def test_create(self) -> None:
        start = Pos(line=1, column=0)
        end = Pos(line=1, column=10)
        fragment = Fragment(start, end)

        assert fragment.start == start
        assert fragment.end == end
        assert fragment.highlights == []

    def test_create_with_highlights(self) -> None:
        start = Pos(line=1, column=0)
        end = Pos(line=1, column=10)
        fragment = Fragment(start, end, highlights=["keyword"])

        assert fragment.highlights == ["keyword"]

    def test_repr(self) -> None:
        start = Pos(line=1, column=0)
        end = Pos(line=1, column=10)
        fragment = Fragment(start, end, highlights=["test"])

        result = repr(fragment)
        assert "Fragment" in result
        assert "1:0" in result
        assert "1:10" in result


class TestVerbatim:
    def test_create(self) -> None:
        source = MockTextSource("content")
        verbatim = Verbatim(source)

        assert verbatim.source is source
        assert list(verbatim.children) == []

    def test_repr(self) -> None:
        source = MockTextSource("content")
        verbatim = Verbatim(source)

        result = repr(verbatim)
        assert "Verbatim" in result


class TestVerbatimNode:
    def test_append(self) -> None:
        source = MockTextSource("content")
        verbatim = Verbatim(source)
        fragment = Fragment(Pos(1, 0), Pos(1, 5))

        verbatim.append(fragment)

        assert fragment in verbatim.children

    def test_append_nested_verbatim_raises(self) -> None:
        source = MockTextSource("content")
        outer = Verbatim(source)
        inner = Verbatim(source)

        with pytest.raises(ValueError, match="cannot nest"):
            outer.append(inner)

    def test_replace_child(self) -> None:
        source = MockTextSource("content")
        verbatim = Verbatim(source)
        old = Fragment(Pos(1, 0), Pos(1, 5))
        new = Fragment(Pos(1, 0), Pos(1, 10))
        verbatim.append(old)

        verbatim.replace_child(old, new)

        assert new in verbatim.children
        assert old not in verbatim.children


class TestBoundsVisitor:
    def test_finds_start_end(self) -> None:
        first_fragment = Fragment(Pos(1, 5), Pos(1, 10))
        second_fragment = Fragment(Pos(2, 0), Pos(2, 15))

        source = MockTextSource("line1\nline2")
        verbatim = Verbatim(source)
        verbatim.append(first_fragment)
        verbatim.append(second_fragment)

        visitor = _BoundsVisitor()
        verbatim.visit(visitor)

        assert visitor.start == Pos(1, 5)
        assert visitor.end == Pos(2, 15)

    def test_no_fragments(self) -> None:
        visitor = _BoundsVisitor()
        blank = BlankNode()
        blank.visit(visitor)

        assert visitor.start is None
        assert visitor.end is None


class ConcreteVerbatimVisitor(VerbatimVisitor):
    lines: List[int]
    texts: List[str]
    highlights: List[Any]

    def __init__(self) -> None:
        super().__init__()
        self.lines = []
        self.texts = []
        self.highlights = []

    def line(self, source: TextSource, line: int) -> None:
        self.lines.append(line)

    def text(self, text: str) -> None:
        self.texts.append(text)

    def begin_highlight(self, highlights: Sequence[str]) -> None:
        self.highlights.append(("begin", list(highlights)))

    def end_highlight(self) -> None:
        self.highlights.append(("end", None))


class TestVerbatimVisitor:
    def test_visit_verbatim_with_fragment(self) -> None:
        source = MockTextSource("hello world")
        verbatim = Verbatim(source)
        fragment = Fragment(Pos(1, 0), Pos(1, 5), highlights=["keyword"])
        verbatim.append(fragment)

        visitor = ConcreteVerbatimVisitor()
        verbatim.visit(visitor)

        assert 1 in visitor.lines
        assert ("begin", ["keyword"]) in visitor.highlights
        assert ("end", None) in visitor.highlights

    def test_visit_verbatim_multi_line_fragment(self) -> None:
        source = MockTextSource("line one\nline two\nline three")
        verbatim = Verbatim(source)
        fragment = Fragment(Pos(1, 0), Pos(3, 10))
        verbatim.append(fragment)

        visitor = ConcreteVerbatimVisitor()
        verbatim.visit(visitor)

        assert 1 in visitor.lines
        assert 2 in visitor.lines
        assert 3 in visitor.lines

        joined = "".join(visitor.texts)
        assert "line one" in joined
        assert "line two" in joined
        assert "line three" in joined

    def test_visit_verbatim_multiple_fragments(self) -> None:
        source = MockTextSource("alpha\nbeta\ngamma\ndelta")
        verbatim = Verbatim(source)
        first_fragment = Fragment(Pos(1, 0), Pos(2, 4))
        second_fragment = Fragment(Pos(3, 0), Pos(4, 5))
        verbatim.append(first_fragment)
        verbatim.append(second_fragment)

        visitor = ConcreteVerbatimVisitor()
        verbatim.visit(visitor)

        joined = "".join(visitor.texts)
        assert "alpha" in joined
        assert "beta" in joined
        assert "gamma" in joined
        assert "delta" in joined

    def test_nested_verbatim_raises(self) -> None:
        # This test verifies the visitor's safety check against nested
        # Verbatim. While Verbatim.append() prevents nesting at construction
        # time, the visitor has an additional runtime check as defense-in-
        # depth. We simulate already being inside a Verbatim by setting _depth.
        source = MockTextSource("content")
        outer = Verbatim(source)

        visitor = ConcreteVerbatimVisitor()
        visitor._depth = 1  # Simulate already inside a Verbatim

        with pytest.raises(Exception, match="cannot be nested"):
            visitor._enter_verbatim(outer)

    def test_fragment_outside_verbatim_raises(self) -> None:
        fragment = Fragment(Pos(1, 0), Pos(1, 5))
        visitor = ConcreteVerbatimVisitor()

        with pytest.raises(Exception, match="must appear inside"):
            visitor._enter_fragment(fragment)


class TestTranscribe:
    def test_transform_simple(self, plugin_settings: PluginSettings) -> None:
        source = MockTextSource("hello world")
        verbatim = Verbatim(source)
        fragment = Fragment(Pos(1, 0), Pos(1, 5))
        verbatim.append(fragment)

        document = Document(verbatim)
        context = Context({Document: document})

        transform = Transcribe(plugin_settings)
        transform.transform(context)

        assert isinstance(document.root, Transcribed)

    def test_transcribe_multi_line(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = MockTextSource("line one\nline two\nline three")
        verbatim = Verbatim(source)
        fragment = Fragment(Pos(1, 0), Pos(3, 10))
        verbatim.append(fragment)

        document = Document(verbatim)
        context = Context({Document: document})

        transform = Transcribe(plugin_settings)
        transform.transform(context)

        assert isinstance(document.root, Transcribed)
        lines = [
            child
            for child in document.root.children
            if isinstance(child, Line)
        ]
        assert len(lines) == 3
        assert lines[0].number == 1
        assert lines[1].number == 2
        assert lines[2].number == 3

        def _find_text(node: object) -> List[Text]:
            found: List[Text] = []
            if isinstance(node, Text):
                found.append(node)
            if hasattr(node, "children"):
                for child in node.children:  # type: ignore[union-attr]
                    found.extend(_find_text(child))
            return found

        for line_node in lines:
            text_nodes = _find_text(line_node)
            assert len(text_nodes) > 0

    def test_transform_no_verbatim(
        self, plugin_settings: PluginSettings
    ) -> None:
        blank = BlankNode()
        document = Document(blank)
        context = Context({Document: document})

        transform = Transcribe(plugin_settings)
        transform.transform(context)

        assert document.root is blank


class TestVerbatimHtmlRendering:
    """Tests for src/docc/plugins/verbatim/html.py render functions."""

    def test_render_transcribed(self) -> None:
        """render_transcribed produces an HTML table with class 'verbatim'."""
        from docc.plugins.verbatim.html import render_transcribed

        context = Context({})
        parent = HTMLTag("div")
        node = Transcribed()

        result = render_transcribed(context, parent, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "table"
        assert result.attributes.get("class") == "verbatim"
        # The table should be appended to parent
        assert result in parent.children

    def test_render_line_inside_table(self) -> None:
        """
        render_line produces a tr with th (line number) and td>pre
        (code) when parent is a table.
        """
        from docc.plugins.verbatim.html import render_line

        context = Context({})
        parent = HTMLTag("table")
        node = Line(number=42)

        result = render_line(context, parent, node)

        # result should be the <pre> inside the <td>
        assert isinstance(result, HTMLTag)
        assert result.tag_name == "pre"

        # Parent (table) should have a <tbody> child
        assert len(list(parent.children)) == 1
        tbody = list(parent.children)[0]
        assert isinstance(tbody, HTMLTag)
        assert tbody.tag_name == "tbody"

        # tbody should contain a <tr>
        tr = list(tbody.children)[0]
        assert isinstance(tr, HTMLTag)
        assert tr.tag_name == "tr"

        # <tr> should contain <th> and <td>
        tr_children = list(tr.children)
        assert len(tr_children) == 2
        th, td = tr_children
        assert isinstance(th, HTMLTag)
        assert th.tag_name == "th"
        assert isinstance(td, HTMLTag)
        assert td.tag_name == "td"

        # <th> should contain TextNode with line number
        th_children = list(th.children)
        assert len(th_children) == 1
        th_child = th_children[0]
        assert isinstance(th_child, HTMLTag)
        assert th_child.tag_name == "a"

        a_children = list(th_child.children)
        assert len(a_children) == 1
        a_child = a_children[0]
        assert isinstance(a_child, TextNode)
        assert a_child._value == "42"

    def test_render_line_outside_table(self) -> None:
        """render_line appends <tr> directly to non-table parents."""
        from docc.plugins.verbatim.html import render_line

        context = Context({})
        parent = HTMLTag("div")
        node = Line(number=1)

        result = render_line(context, parent, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "pre"
        # Parent (div) should have <tr> directly (no <tbody>)
        assert len(list(parent.children)) == 1
        tr = list(parent.children)[0]
        assert isinstance(tr, HTMLTag)
        assert tr.tag_name == "tr"

    def test_render_text(self) -> None:
        """render_text appends a TextNode with correct content."""
        from docc.plugins.verbatim.html import render_text

        context = Context({})
        parent = HTMLTag("pre")
        node = Text(text="hello world")

        result = render_text(context, parent, node)

        assert result is None
        children = list(parent.children)
        assert len(children) == 1
        child = children[0]
        assert isinstance(child, TextNode)
        assert child._value == "hello world"

    def test_render_highlight(self) -> None:
        """render_highlight produces <span> with hi-{name} hi classes."""
        from docc.plugins.verbatim.html import render_highlight

        context = Context({})
        parent = HTMLTag("pre")
        node = Highlight(highlights=["keyword", "function"])

        result = render_highlight(context, parent, node)

        assert isinstance(result, HTMLTag)
        assert result.tag_name == "span"
        classes = result.attributes.get("class") or ""
        assert "hi-keyword" in classes
        assert "hi-function" in classes
        assert "hi" in classes.split()
        # The span should be appended to parent
        assert result in parent.children
