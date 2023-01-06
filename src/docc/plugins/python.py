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

import ast
from typing import Dict, List, Optional, Sequence, Set

from inflection import dasherize, underscore

from docc.build import Builder
from docc.document import BlankNode, Document, Node, Visit, Visitor
from docc.languages.verbatim import Fragment, Pos
from docc.references import Index
from docc.settings import PluginSettings
from docc.source import Source, TextSource
from docc.transform import Transform


class PythonBuilder(Builder):
    def __init__(self, config: PluginSettings) -> None:
        """
        Create a PythonBuilder with the given configuration.
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

            assert source.relative_path is not None
            tree = ast.parse(text, filename=str(source.relative_path))
            ast.fix_missing_locations(tree)

            document = Document(
                all_sources,
                index,
                source,
                AstNode(tree),
            )

            # Propagate source locations from children to parents.
            document.root.visit(_ChildPositionTransform())

            document.root.dump()
            # Assume a preceding node ends at the start of the next one.
            sibling_transform = _SiblingPositionTransform()
            document.root.visit(sibling_transform)
            last_exited = sibling_transform.last_exited

            # Assume the last exited node ends at the file boundary.
            if last_exited and not last_exited.end:
                lines = text.split("\n")
                column = len(lines[-1]) if lines else 0

                last_exited.end = Pos(
                    line=len(lines),
                    column=column,
                )

            # Assume the first/last child starts/ends at the start/end of its
            # parent.
            document.root.visit(_FirstOrLastChildPositionTransform())

            processed[source] = document


class AstNode(Node):
    ast_node: ast.AST
    _start: Optional[Pos]
    _end: Optional[Pos]

    _children: Optional[List[Node]]

    def __init__(self, ast_node: ast.AST):
        self.ast_node = ast_node
        self._start = None
        self._end = None
        self._children = None

    @property
    def start(self) -> Optional[Pos]:
        return self._start

    @start.setter
    def start(self, value: Optional[Pos]) -> None:
        if value and self._end and value > self._end:
            raise ValueError(f"start `{value}` out of range for `{self}`")

        self._start = value

    @property
    def end(self) -> Optional[Pos]:
        return self._end

    @end.setter
    def end(self, value: Optional[Pos]) -> None:
        if value and self._start and value < self._start:
            raise ValueError(f"end `{value}` out of range for `{self}`")

        self._end = value

    @property
    def children(self) -> Sequence[Node]:
        if self._children is None:
            children = ast.iter_child_nodes(self.ast_node)
            self._children = [AstNode(n) for n in children]
        return self._children

    def replace_child(self, old: Node, new: Node) -> None:
        self._children = [new if c == old else c for c in self.children]

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        ast_node = self.ast_node
        text = f"{self.__class__.__name__}({ast_node.__class__.__name__}(...)"
        if self.start is not None:
            text += f", start={self.start}"
        if self.end is not None:
            text += f", end={self.end}"
        return text + ")"


class PythonTransform(Transform):
    """
    Transforms AstNode instances into Python language nodes.
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
        document.root = verbatim_visitor.root
        document.root.dump()


class _SiblingPositionTransform(Visitor):
    last_exited: Optional[AstNode]

    def __init__(self) -> None:
        self.last_exited = None

    def enter(self, node: Node) -> Visit:
        if not isinstance(node, AstNode) or self.last_exited is None:
            return Visit.TraverseChildren

        if node.start and self.last_exited.end is None:
            self.last_exited.end = node.start

        self.last_exited = None

        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        if isinstance(node, AstNode):
            self.last_exited = node


class _FirstOrLastChildPositionTransform(Visitor):
    stack: List[AstNode]

    def __init__(self) -> None:
        self.stack = []

    def enter(self, node: Node) -> Visit:
        if not isinstance(node, AstNode):
            return Visit.TraverseChildren

        if not self.stack:
            self.stack.append(node)
            return Visit.TraverseChildren

        parent = self.stack[-1]
        siblings = list(parent.children)

        if node.start is None and siblings[0] == node:
            node.start = parent.start

        if node.end is None and siblings[-1] == node:
            node.end = parent.end

        self.stack.append(node)
        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        if isinstance(node, AstNode):
            popped = self.stack.pop()
            assert popped == node


class _ChildPositionTransform(Visitor):
    stack: List[AstNode]

    def __init__(self) -> None:
        self.stack = []

    def enter(self, node: Node) -> Visit:
        assert isinstance(node, AstNode)
        self.stack.append(node)

        ast_node = node.ast_node
        if hasattr(ast_node, "lineno") and hasattr(ast_node, "col_offset"):
            node.start = Pos(
                line=ast_node.lineno,
                column=ast_node.col_offset,
            )

        end_lineno = getattr(ast_node, "end_lineno", None)
        end_col_offset = getattr(ast_node, "end_col_offset", None)

        if end_lineno is not None and end_col_offset is not None:
            self.end = Pos(
                line=end_lineno,
                column=end_col_offset,
            )

        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        popped = self.stack.pop()
        assert popped == node

        if not self.stack:
            return

        parent = self.stack[-1]

        if popped.start:
            if not parent.start or popped.start < parent.start:
                parent.start = popped.start

        if popped.end:
            if not parent.end or popped.end > parent.end:
                parent.end = popped.end


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

        if not isinstance(node, AstNode):
            return

        name = dasherize(underscore(node.ast_node.__class__.__name__))

        if not node.start or not node.end:
            assert self.root is not None
            self.root.dump()

        assert node.start is not None, f"{node} has no `start`"
        assert node.end is not None, f"{node} has no `end`"

        new = Fragment(
            start=node.start,
            end=node.end,
            highlights=[f"hi-{name}"],
        )

        for child in node.children:
            new.append(child)

        if self.stack:
            self.stack[-1].replace_child(node, new)
        else:
            assert self.root == node
            self.root = new
