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
Support for copying verbatim text from a Source into the output, with syntax
highlighting support.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from docc.document import Node, Visit, Visitor
from docc.source import TextSource


class VerbatimNode(Node):
    """
    Base class for verbatim nodes.
    """

    _children: List[Node]

    def __init__(self) -> None:
        self._children = []

    @property
    def children(self) -> Iterable[Node]:
        """
        Child nodes belonging to this node.
        """
        return self._children

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        self._check(new)
        for index, child in enumerate(self._children):
            if child == old:
                self._children[index] = new

    def append(self, new: Node) -> None:
        """
        Append a new child node.
        """
        self._check(new)
        self._children.append(new)

    def _check(self, new: Node) -> None:
        if isinstance(new, Verbatim):
            # TODO: is it worth checking children of children?
            class_ = new.__class__.__name__
            raise ValueError(f"cannot nest a {class_} in a verbatim node")


@dataclass(order=True)
class _Pos:
    line: int
    column: int


@dataclass(order=True, frozen=True, repr=False)
class Pos:
    """
    Position in a Source.
    """

    line: int
    column: int

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        return f"{self.line}:{self.column}"


class Verbatim(VerbatimNode):
    """
    A block of lines of text from one Source.
    """

    source: TextSource

    def __init__(self, source: TextSource) -> None:
        super().__init__()
        self.source = source

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        return f"Verbatim({self.source!r})"


class Fragment(VerbatimNode):
    """
    A snippet of text from a Source.
    """

    start: Pos
    end: Pos

    highlights: List[str]

    def __init__(
        self, start: Pos, end: Pos, highlights: Optional[List[str]] = None
    ) -> None:
        super().__init__()
        self.start = start
        self.end = end
        self.highlights = [] if highlights is None else highlights

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        return f"Fragment({self.start}, {self.end}, {self.highlights!r})"


@dataclass
class _VerbatimContext:
    node: Verbatim
    written: Optional[_Pos] = None

    @property
    def source(self) -> TextSource:
        return self.node.source


class VerbatimVisitor(Visitor):
    """
    Visitor for verbatim node trees that emits a series of events.
    """

    # TODO: I wonder if there's a way to visit all nodes, put them in a heap,
    #       and read the text that way.

    _depth: Optional[int]
    _verbatim: Optional[_VerbatimContext]

    def __init__(self) -> None:
        super().__init__()
        self._depth = None
        self._verbatim = None

    #
    # Override in Subclasses:
    #

    @abstractmethod
    def line(self, source: TextSource, line: int) -> None:
        """
        Marks the start of a new line.
        """

    @abstractmethod
    def text(self, text: str) -> None:
        """
        String copied from the Source.
        """

    @abstractmethod
    def begin_highlight(self, highlights: Sequence[str]) -> None:
        """
        Marks the start of a highlighted section.
        """

    @abstractmethod
    def end_highlight(self) -> None:
        """
        Marks the end of a highlighted section.
        """

    def enter_node(self, node: Node) -> Visit:
        """
        Visit a non-verbatim Node.
        """
        if node:
            logging.warning(
                "`%s` entered non-verbatim node `%s`",
                self.__class__.__name__,
                node.__class__.__name__,
            )
        return Visit.TraverseChildren

    def exit_node(self, node: Node) -> None:
        """
        Leave a non-verbatim Node.
        """
        if node:
            logging.warning(
                "`%s` exited non-verbatim node `%s`",
                self.__class__.__name__,
                node.__class__.__name__,
            )

    #
    # Implementation Details:
    #

    def _copy(self, node: Fragment, until: Pos) -> None:
        verbatim = self._verbatim

        assert verbatim is not None
        assert until.line > 0
        assert until.column >= 0

        if verbatim.written is None:
            verbatim.written = _Pos(
                line=node.start.line,
                column=0,
            )
            self.line(verbatim.source, verbatim.written.line)

        written = verbatim.written
        assert written is not None

        while written.line < until.line:
            text = verbatim.source.line(written.line)
            self.text(text[written.column :])

            written.line += 1
            written.column = 0

            self.line(verbatim.source, written.line)

        if written.line > until.line:
            return

        if written.column >= until.column:
            return

        text = verbatim.source.line(written.line)
        self.text(text[written.column : until.column])

        written.column = until.column

    def _enter_fragment(self, node: Fragment) -> Visit:
        depth = self._depth

        if depth is None or depth < 1:
            raise Exception("Fragment nodes must appear inside Varbatim")

        self._copy(node, node.start)

        self._depth = depth + 1
        self.begin_highlight(node.highlights)

        return Visit.TraverseChildren

    def _exit_fragment(self, node: Fragment) -> None:
        depth = self._depth
        assert depth is not None
        assert depth > 1

        self._copy(node, node.end)
        self.end_highlight()

        self._depth = depth - 1

    def _enter_verbatim(self, node: Verbatim) -> Visit:
        if self._depth is not None:
            raise Exception("Verbatim nodes cannot be nested")
        self._depth = 1
        self._verbatim = _VerbatimContext(node)
        return Visit.TraverseChildren

    def _exit_verbatim(self, node: Verbatim) -> None:
        assert self._depth == 1
        assert self._verbatim is not None
        assert self._verbatim.node == node
        self._verbatim = None
        self._depth = None

    def enter(self, node: Node) -> Visit:
        """
        Visit a node.
        """
        if isinstance(node, Fragment):
            return self._enter_fragment(node)
        elif isinstance(node, Verbatim):
            return self._enter_verbatim(node)
        else:
            return self.enter_node(node)

    def exit(self, node: Node) -> None:
        """
        Depart a node.
        """
        if isinstance(node, Fragment):
            return self._exit_fragment(node)
        elif isinstance(node, Verbatim):
            return self._exit_verbatim(node)
        else:
            return self.exit_node(node)
