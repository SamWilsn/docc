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

import itertools
import glob
import logging
import os.path
import subprocess
import sys
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import PurePath
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
)

import libcst as cst
from inflection import dasherize, underscore
from libcst.metadata import MetadataWrapper

from docc.build import Builder
from docc.discover import Discover, T
from docc.document import Document, Node, Visit, Visitor
from docc.languages import python
from docc.languages.verbatim import Fragment, Pos
from docc.references import Index
from docc.settings import PluginSettings
from docc.source import Source, TextSource
from docc.transform import Transform


class PythonDiscover(Discover):
    """
    Find Python source files.
    """

    paths: Sequence[str]
    settings: PluginSettings

    def __init__(self, config: PluginSettings) -> None:
        super().__init__(config)
        self.settings = config

        paths = config.get("paths", [])
        if not isinstance(paths, Sequence):
            raise TypeError("python paths must be a list")

        if any(not isinstance(path, str) for path in paths):
            raise TypeError("every python path must be a string")

        if not paths:
            raise ValueError("python needs at least one path")

        self.paths = [str(config.resolve_path(path)) for path in paths]

    def discover(self, known: FrozenSet[T]) -> Iterator[Source]:
        """
        Find sources.
        """
        escaped = (glob.escape(path) for path in self.paths)
        joined = (os.path.join(path, "**", "*.py") for path in escaped)
        globbed = (glob.glob(path, recursive=True) for path in joined)

        for absolute_path_text in itertools.chain.from_iterable(globbed):
            absolute_path = PurePath(absolute_path_text)
            relative_path = self.settings.unresolve_path(absolute_path)

            yield PythonSource(relative_path, absolute_path)


class PythonSource(TextSource):
    """
    A Source representing a Python file.
    """

    absolute_path: PurePath
    _relative_path: PurePath

    def __init__(
        self, relative_path: PurePath, absolute_path: PurePath
    ) -> None:
        self._relative_path = relative_path
        self.absolute_path = absolute_path

    @property
    def relative_path(self) -> Optional[PurePath]:
        """
        The relative path to the Source.
        """
        return self._relative_path

    @property
    def output_path(self) -> PurePath:
        """
        Where to put the output derived from this source.
        """
        return self._relative_path

    def open(self) -> TextIO:
        """
        Open the source for reading.
        """
        return open(self.absolute_path, "r")


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

            tree = cst.parse_module(text)

            visitor = _CstVisitor(source)
            MetadataWrapper(tree).visit(visitor)
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
    source: Source
    _children: List[Node]
    start: Pos
    end: Pos

    def __init__(
        self,
        cst_node: cst.CSTNode,
        source: Source,
        start: Pos,
        end: Pos,
        children: List[Node],
    ) -> None:
        self.cst_node = cst_node
        self.source = source
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
    source: Source

    def __init__(self, source: Source) -> None:
        super().__init__()
        self.stack = []
        self.root = None
        self.source = source

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
        new = CstNode(node, self.source, start, end, [])

        if self.stack:
            self.stack[-1]._children.append(new)
        else:
            assert self.root is None

        if self.root is None:
            self.root = new

        self.stack.append(new)
        return True

    def on_leave(self, original_node: cst.CSTNode) -> None:
        self.stack.pop()


class PythonTransform(Transform):
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
        sources: Set[str] = set()  # TODO
        visitor = _TransformVisitor(document, sources)
        document.root.visit(visitor)
        assert visitor.root is not None
        document.root = visitor.root


@dataclass
class _TransformContext:
    node: "CstNode"
    child_offset: int = 0


class _TransformVisitor(Visitor):
    root: Optional[Node]
    old_stack: List[_TransformContext]
    new_stack: List[Node]
    document: Document
    source_paths: Set[str]

    def __init__(self, document: Document, source_paths: Set[str]) -> None:
        self.root = None
        self.old_stack = []
        self.new_stack = []
        self.document = document
        self.source_paths = source_paths

    def push_new(self, node: Node) -> None:
        if self.root is None:
            assert 0 == len(self.new_stack)
            self.root = node
        self.new_stack.append(node)

    def enter_module(self, node: Node, cst_node: cst.Module) -> Visit:
        assert 0 == len(self.new_stack)
        module = python.Module()
        self.push_new(module)

        docstring = cst_node.get_docstring(True)
        if docstring is not None:
            module.docstring = python.Docstring(docstring)

        return Visit.TraverseChildren

    def exit_module(self) -> None:
        self.new_stack.pop()

    def enter_class_def(self, node: Node, cst_node: cst.ClassDef) -> Visit:
        assert 0 < len(self.new_stack)

        class_def = python.Class()
        self.push_new(class_def)

        docstring = cst_node.get_docstring(True)
        if docstring is not None:
            class_def.docstring = python.Docstring(docstring)

        # TODO: decorators
        # TODO: members
        # TODO: base classes
        # TODO: metaclass

        return Visit.SkipChildren

    def exit_class_def(self) -> None:
        self.new_stack.pop()

    def enter_function_def(
        self, node: Node, cst_node: cst.FunctionDef
    ) -> Visit:
        # TODO
        assert 0 < len(self.new_stack)

        function_def = python.Function()
        self.push_new(function_def)

        docstring = cst_node.get_docstring(True)
        if docstring is not None:
            function_def.docstring = python.Docstring(docstring)

        # TODO: decorators
        # TODO: name
        # TODO: arguments
        # TODO: return type
        # TODO: body

        return Visit.SkipChildren

    def exit_function_def(self) -> None:
        self.new_stack.pop()

    def enter(self, node: Node) -> Visit:
        if not isinstance(node, CstNode):
            raise ValueError(
                "expected `"
                + CstNode.__name__
                + "` but got `"
                + node.__class__.__name__
                + "`"
            )

        cst_node = node.cst_node
        try:
            parent = self.old_stack[-1].node.cst_node
        except IndexError:
            parent = None

        module_member = isinstance(parent, cst.Module)

        visit: Visit

        if isinstance(cst_node, cst.Module):
            visit = self.enter_module(node, cst_node)
        elif isinstance(cst_node, cst.ClassDef):
            visit = self.enter_class_def(node, cst_node)
        elif isinstance(cst_node, cst.FunctionDef):
            visit = self.enter_function_def(node, cst_node)
        elif module_member and isinstance(cst_node, cst.CSTNode):
            logging.warning("skipping module member node %s", node)
            visit = Visit.SkipChildren
        else:
            raise Exception(f"unknown node type {node}")

        self.old_stack.append(_TransformContext(node=node))

        return visit

    def exit(self, node: Node) -> None:
        module_member = False

        self.old_stack.pop()
        if self.old_stack:
            self.old_stack[-1].child_offset += 1
            parent = self.old_stack[-1].node.cst_node
            module_member = isinstance(parent, cst.Module)

        assert isinstance(node, CstNode)
        cst_node = node.cst_node

        if isinstance(cst_node, cst.Module):
            self.exit_module()
        elif isinstance(cst_node, cst.ClassDef):
            self.exit_class_def()
        elif isinstance(cst_node, cst.FunctionDef):
            self.exit_function_def()
        elif module_member and isinstance(cst_node, cst.CSTNode):
            pass
        else:
            raise Exception(f"unknown node type {cst_node}")


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
