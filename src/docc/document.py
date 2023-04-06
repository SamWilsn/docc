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
Documents are the in-flight representation of a Source.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum, auto
from io import StringIO, TextIOBase
from typing import IO, Iterable, List, Optional, Sequence, Union

import rich.markup
import rich.tree
from rich.console import Console

from .references import Index
from .source import Source


class Node(ABC):
    """
    Representation of a node in a Document.
    """

    @property
    @abstractmethod
    def children(self) -> Iterable["Node"]:
        """
        Child nodes belonging to this node.
        """

    @abstractmethod
    def replace_child(self, old: "Node", new: "Node") -> None:
        """
        Replace the old node with the given new node.
        """

    def visit(self, visitor: "Visitor") -> None:
        """
        Visit, in depth-first order, this node and its children.
        """
        if visitor.enter(self) == Visit.SkipChildren:
            visitor.exit(self)
            return

        stack = [(self, iter(self.children))]

        while stack:
            node, children = stack.pop()

            try:
                child = next(children)
            except StopIteration:
                visitor.exit(node)
                continue

            stack.append((node, children))

            if visitor.enter(child) == Visit.SkipChildren:
                visitor.exit(child)
            else:
                stack.append((child, iter(child.children)))

    def dump(self, file: Optional[IO[str]] = None) -> None:
        """
        Render the tree to the console or given file.
        """
        visitor = _StrVisitor()
        self.visit(visitor)

        console = Console(file=file)
        console.print(visitor.root)

    def dumps(self) -> str:
        """
        Render the tree to a str.
        """
        io = StringIO()
        self.dump(file=io)
        return io.getvalue()


class BlankNode(Node):
    """
    A placeholder node with no conent and no children.
    """

    __slots__: Iterable[str] = tuple()

    @property
    def children(self) -> Iterable[Node]:
        """
        Child nodes belonging to this node.
        """
        return tuple()

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        raise TypeError()

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        return "<blank>"

    def __bool__(self) -> bool:
        """
        Cast this instance to a bool.
        """
        return False


class OutputNode(Node):
    """
    A Node that understands how to write to a file.
    """

    @property
    @abstractmethod
    def extension(self) -> str:
        """
        The preferred file extension for this node.
        """

    @abstractmethod
    def output(self, document: "Document", destination: TextIOBase) -> None:
        """
        Write this Node to destination.
        """


class Visit(Enum):
    """
    How to proceed after visiting a Node.
    """

    TraverseChildren = auto()
    SkipChildren = auto()


class Visitor(ABC):
    """
    Base class for visitors.
    """

    @abstractmethod
    def enter(self, node: Node) -> Visit:
        """
        Called when visiting the given node, before any children (if any) are
        visited.
        """

    @abstractmethod
    def exit(self, node: Node) -> None:
        """
        Called after visiting the last child of the given node (or immediately
        if the node has no children.)
        """


class _StrVisitor(Visitor):
    root: Union[None, rich.tree.Tree]
    stack: List[rich.tree.Tree]

    def __init__(self) -> None:
        self.stack = []
        self.root = None

    def enter(self, node: Node) -> Visit:
        tree = rich.tree.Tree(rich.markup.escape(repr(node)))
        if self.root is None:
            assert 0 == len(self.stack)
            self.root = tree
        else:
            self.stack[-1].add(tree)
        self.stack.append(tree)
        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        self.stack.pop()


class _OutputVisitor(Visitor):
    destination: TextIOBase
    document: "Document"

    def __init__(self, document: "Document", destination: TextIOBase) -> None:
        self.document = document
        self.destination = destination

    def enter(self, node: Node) -> Visit:
        if isinstance(node, OutputNode):
            node.output(self.document, self.destination)
            return Visit.SkipChildren
        else:
            return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        pass


class _ExtensionVisitor(Visitor):
    extension: Optional[str]

    def __init__(self) -> None:
        self.extension = None

    def enter(self, node: Node) -> Visit:
        if isinstance(node, OutputNode):
            extension = node.extension
            if self.extension is not None and self.extension != extension:
                logging.warning(
                    "document has extension `%s` but node wants `%s`",
                    self.extension,
                    extension,
                )
            else:
                self.extension = extension
        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        pass


class Document:
    """
    In-flight representation of a Source.
    """

    all_sources: Sequence[Source]
    source: Source
    root: Node
    index: Index

    def __init__(
        self,
        all_sources: Sequence[Source],
        index: Index,
        source: Source,
        root: Node,
    ) -> None:
        self.all_sources = all_sources
        self.index = index
        self.source = source
        self.root = root

    def output(self, destination: TextIOBase) -> None:
        """
        Attempt to write this document to destination.
        """
        self.root.visit(_OutputVisitor(self, destination))

    def extension(self) -> Optional[str]:
        """
        Find the file extension for this document.
        """
        visitor = _ExtensionVisitor()
        self.root.visit(visitor)
        return visitor.extension
