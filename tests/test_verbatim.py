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
from pathlib import PurePath
from typing import Any, List, Optional, TextIO

from typing_extensions import override

from docc.document import Document, Node
from docc.plugins.verbatim import (
    Fragment,
    Hidden,
    Highlight,
    Line,
    Pos,
    Text,
    Transcribe,
    Transcribed,
    Verbatim,
)
from docc.settings import PluginSettings
from docc.source import TextSource


class _Source(TextSource):
    _text: str

    def __init__(self, text: str) -> None:
        self._text = text

    @override
    def open(self) -> TextIO:
        return StringIO(self._text)

    @property
    @override
    def output_path(self) -> PurePath:
        return PurePath("source")

    @property
    @override
    def relative_path(self) -> Optional[PurePath]:
        return None


def _line_numbers(root: Node) -> List[int]:
    return [c.number for c in root.children if isinstance(c, Line)]


def _line_text(line: Line) -> str:
    out: List[str] = []

    def walk(node: Node) -> None:
        if isinstance(node, Text):
            out.append(node.text)
            return
        if isinstance(node, (Line, Highlight)):
            for c in node.children:
                walk(c)

    walk(line)
    return "".join(out)


def test_hidden_skips_lines(
    make_context: Any, plugin_settings: PluginSettings
) -> None:
    text = "def foo():\n    A\n    B\n    C\n    D\n"
    source = _Source(text)

    verbatim = Verbatim(source)
    indented = Fragment(
        start=Pos(2, 4), end=Pos(5, 5), highlights=["indented-block"]
    )
    indented.append(Hidden(start=Pos(2, 4), end=Pos(4, 5)))
    indented.append(
        Fragment(start=Pos(5, 4), end=Pos(5, 5), highlights=["last"])
    )
    verbatim.append(indented)

    ctx = make_context(verbatim)
    Transcribe(plugin_settings).transform(ctx)

    root = ctx[Document].root
    assert isinstance(root, Transcribed)

    # Lines 3 and 4 should be elided, but line 2 (where the hidden block
    # starts) is still opened by the parent fragment's leading copy.
    assert _line_numbers(root) == [2, 5]

    line_2 = next(c for c in root.children if isinstance(c, Line))
    assert _line_text(line_2) == "    <snip>"

    line_5 = [c for c in root.children if isinstance(c, Line)][1]
    assert _line_text(line_5) == "    D"
