# Copyright (C) 2022 Ethereum Foundation
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
Documentation plugin for Python.
"""

from typing import Dict, List, Optional, Sequence, Set

import libcst as cst
from inflection import dasherize, underscore

from docc.build import Builder
from docc.document import BlankNode, Document, Node, Visit, Visitor
from docc.languages.verbatim import Fragment, Pos, Stanza, Verbatim
from docc.references import Index
from docc.settings import PluginSettings
from docc.source import Source, TextSource
from docc.transform import Transform


class CstBuilder(Builder):
    def __init__(self, config: PluginSettings) -> None:
        """
        Create a CstBuilder with the given configuration.
        """

    def build(
        self,
        index: Index,
        all_sources: Sequence[Source],
        unprocessed: Set[Source],
        processed: Dict[Source, Document],
    ) -> None:
        """
        Consume unprocessed Sources and insert their Documents into processed.
        """

        source_set = set(
            s
            for s in unprocessed
            if isinstance(s, TextSource)
            and s.relative_path
            and s.relative_path.suffix == ".py"
        )
        unprocessed -= source_set

        for source in source_set:
            with source.open() as f:
                text = f.read()

            tree = cst.parse_module(text)

            visitor = _CstVisitor()
            cst.metadata.MetadataWrapper(tree).visit(visitor)
            assert visitor.root is not None

            document = Document(
                all_sources,
                index,
                source,
                visitor.root,
            )

            processed[source] = document


class CstNode(Node):
    cst_node: cst.CSTNode
    _children: List[Node]
    start: Pos
    end: Pos

    def __init__(
        self, cst_node: cst.CSTNode, start: Pos, end: Pos, children: List[Node]
    ) -> None:
        self.cst_node = cst_node
        self._children = children
        self.start = start
        self.end = end

    @property
    def children(self) -> Sequence[Node]:
        return self._children

    def replace_child(self, old: Node, new: Node) -> None:
        self._children = [new if c == old else c for c in self.children]

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        cst_node = self.cst_node
        text = f"{self.__class__.__name__}({cst_node.__class__.__name__}(...)"
        text += f", start={self.start}"
        text += f", end={self.end}"
        return text + ")"


class _CstVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (cst.metadata.PositionProvider,)

    stack: List[CstNode]
    root: Optional[CstNode]

    def __init__(self) -> None:
        super().__init__()
        self.stack = []
        self.root = None

    def on_visit(self, node: cst.CSTNode) -> bool:
        position = self.get_metadata(cst.metadata.PositionProvider, node)
        start = Pos(
            line=position.start.line,
            column=position.start.column,
        )
        end = Pos(
            line=position.end.line,
            column=position.end.column,
        )
        new = CstNode(node, start, end, [])

        if self.stack:
            self.stack[-1]._children.append(new)
        else:
            assert self.root is None

        if self.root is None:
            self.root = new

        self.stack.append(new)
        return True

    def on_leave(self, node: cst.CSTNode) -> None:
        self.stack.pop()


class CstTransform(Transform):
    """
    Transforms CstNode instances into Python language nodes.
    """

    def __init__(self, config: PluginSettings) -> None:
        """
        Create a Transform with the given configuration.
        """

    def transform(self, document: Document) -> None:
        """
        Apply the transformation to the given document.
        """
        verbatim_visitor = _VerbatimTransform()
        document.root.visit(verbatim_visitor)
        assert verbatim_visitor.root is not None
        stanza = Stanza(document.source)
        stanza.append(verbatim_visitor.root)
        verbatim = Verbatim()
        verbatim.append(stanza)
        document.root = verbatim


class _VerbatimTransform(Visitor):
    root: Optional[Node]
    stack: List[Node]

    def __init__(self) -> None:
        self.stack = []
        self.root = None

    def enter(self, node: Node) -> Visit:
        if self.root is None:
            assert 0 == len(self.stack)
            self.root = node
        else:
            assert 0 < len(self.stack)

        self.stack.append(node)
        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        popped = self.stack.pop()
        assert popped == node

        if not isinstance(node, CstNode):
            return

        name = dasherize(underscore(node.cst_node.__class__.__name__))

        assert node.start is not None, f"{node} has no `start`"
        assert node.end is not None, f"{node} has no `end`"

        new = Fragment(
            start=node.start,
            end=node.end,
            highlights=[name],
        )

        for child in node.children:
            new.append(child)

        if self.stack:
            self.stack[-1].replace_child(node, new)
        else:
            assert self.root == node
            self.root = new
