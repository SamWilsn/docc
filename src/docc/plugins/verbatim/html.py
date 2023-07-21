# Copyright (C) 2022-2023 Ethereum Foundation
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
Rendering functions to transform verbatim nodes into HTML.
"""

from typing import Final, List, Sequence, Union

from docc.context import Context
from docc.document import Document, Node, Visit
from docc.plugins import references
from docc.plugins.html import (
    HTMLRoot,
    HTMLTag,
    RenderResult,
    TextNode,
    render_reference,
)
from docc.source import TextSource

from . import Verbatim, VerbatimVisitor


class _VerbatimVisitor(VerbatimVisitor):
    context: Final[Context]
    document: Final[Document]
    root: HTMLTag
    body: HTMLTag
    output_stack: List[Node]
    input_stack: List[Union[Sequence[str], references.Reference]]

    def __init__(self, context: Context) -> None:
        super().__init__()
        self.context = context
        self.document = context[Document]

        self.body = HTMLTag("tbody")

        self.root = HTMLTag("table", attributes={"class": "verbatim"})
        self.root.append(self.body)

        self.output_stack = []
        self.input_stack = []

    def line(self, source: TextSource, line: int) -> None:
        line_text = TextNode(str(line))

        line_cell = HTMLTag("th")
        line_cell.append(line_text)

        code_pre = HTMLTag("pre")

        code_cell = HTMLTag("td")
        code_cell.append(code_pre)

        row = HTMLTag("tr")
        row.append(line_cell)
        row.append(code_cell)

        self.body.append(row)

        self.output_stack = [code_pre]
        self._highlight(self.input_stack)

    def _highlight(
        self,
        highlight_groups: Sequence[Union[Sequence[str], references.Reference]],
    ) -> None:
        for item in highlight_groups:
            top = self.output_stack[-1]
            assert isinstance(top, HTMLTag)

            if isinstance(item, references.Reference):
                new_node = render_reference(self.context, item)
            else:
                classes = [f"hi-{h}" for h in item] + ["hi"]
                new_node = HTMLTag(
                    "span",
                    attributes={
                        "class": " ".join(classes),
                    },
                )

            top.append(new_node)
            self.output_stack.append(new_node)

    def text(self, text: str) -> None:
        top = self.output_stack[-1]
        assert isinstance(top, HTMLTag)
        top.append(TextNode(text))

    def begin_highlight(self, highlights: Sequence[str]) -> None:
        self.input_stack.append(highlights)
        self._highlight([highlights])

    def end_highlight(self) -> None:
        self.input_stack.pop()
        popped_node = self.output_stack.pop()
        assert isinstance(popped_node, HTMLTag)
        assert (
            popped_node.tag_name == "span"
        ), f"expected span, got `{popped_node.tag_name}`"

    def enter_node(self, node: Node) -> Visit:
        """
        Visit a non-verbatim Node.
        """
        if isinstance(node, references.Reference):
            if "<" in node.identifier:
                # TODO: Create definitions for local variables.
                return Visit.TraverseChildren
            self.input_stack.append(node)
            if self.output_stack:
                self._highlight([node])
            return Visit.TraverseChildren
        else:
            return super().enter_node(node)

    def exit_node(self, node: Node) -> None:
        """
        Leave a non-verbatim Node.
        """
        if isinstance(node, references.Reference):
            if "<" in node.identifier:
                # TODO: Create definitions for local variables.
                return

            popped = self.input_stack.pop()
            assert popped == node

            popped_output = self.output_stack.pop()
            assert isinstance(popped_output, HTMLTag)
        else:
            return super().exit_node(node)


def render_verbatim(
    context: Context,
    parent: object,
    node: Verbatim,
) -> RenderResult:
    """
    Render a verbatim block as HTML.
    """
    assert isinstance(context, Context)
    assert isinstance(parent, (HTMLRoot, HTMLTag))
    assert isinstance(node, Verbatim)

    visitor = _VerbatimVisitor(context)
    node.visit(visitor)
    parent.append(visitor.root)
    return None
