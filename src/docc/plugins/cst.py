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
from mypy.find_sources import create_source_list
from mypy.modulefinder import BuildSource
from mypy.options import Options

from docc.build import Builder
from docc.discover import Discover, T
from docc.document import Document, Node, Visit, Visitor
from docc.languages import python
from docc.languages.verbatim import Fragment, Pos
from docc.references import Index
from docc.settings import PluginSettings
from docc.source import Source, TextSource
from docc.transform import Transform


class MyPyDiscover(Discover):
    """
    Source discovery based on mypy.
    """

    paths: Sequence[str]
    settings: PluginSettings

    def __init__(self, config: PluginSettings) -> None:
        super().__init__(config)
        self.settings = config

        paths = config.get("paths", [])
        if not isinstance(paths, Sequence):
            raise TypeError("mypy paths must be a list")

        if any(not isinstance(path, str) for path in paths):
            raise TypeError("every mypy path must be a string")

        if not paths:
            raise ValueError("mypy needs at least one path")

        self.paths = [str(config.resolve_path(path)) for path in paths]

    def discover(self, known: FrozenSet[T]) -> Iterator[Source]:
        """
        Uses mypy's `create_source_list` to find sources.
        """
        options = Options()
        options.python_version = (
            sys.version_info.major,
            sys.version_info.minor,
        )
        options.export_types = True
        options.preserve_asts = True
        options.show_column_numbers = True
        for build_source in create_source_list(self.paths, options):
            relative_path = None
            assert build_source.path is not None
            path = PurePath(build_source.path)
            relative_path = self.settings.unresolve_path(path)
            yield MyPySource(relative_path, build_source)


class MyPySource(TextSource):
    """
    A Source based on mypy.
    """

    build_source: BuildSource
    _relative_path: PurePath

    def __init__(
        self, relative_path: PurePath, build_source: BuildSource
    ) -> None:
        self.build_source = build_source
        self._relative_path = relative_path

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
        assert self.build_source.path is not None
        return open(self.build_source.path, "r")


class MyPyBuilder(Builder):
    _exit_stack: Optional[ExitStack]
    _process: Optional[subprocess.Popen]
    _status_directory: Optional[TemporaryDirectory]
    settings: PluginSettings

    def __init__(self, config: PluginSettings) -> None:
        """
        Create a MyPyBuilder with the given configuration.
        """
        self._exit_stack = None
        self._process = None
        self._status_directory = None
        self.settings = config

    def build(
        self,
        index: Index,
        all_sources: Sequence[Source],
        unprocessed: Set[Source],
        processed: Dict[Source, Document],
    ) -> None:
        """
        Passes collected sources to dmypy.
        """
        paths = set()
        for source, document in processed.items():
            if not source.relative_path:
                continue

            if source.relative_path.suffix != ".py":
                continue

            paths.add(str(self.settings.resolve_path(source.relative_path)))
            root = MyPyDaemonNode(self, document.root)
            document.root = root

        if paths:
            self._dmypy(["check", "--export-types"] + list(paths), check=False)

    def attrs(self, source: Source, start: Pos, end: Pos) -> str:
        assert source.relative_path is not None
        return self._dmypy(
            [
                "inspect",
                "--show",
                "attrs",
                "--include-span",
                f"{source.relative_path}:{start}:{end}",
            ]
        )

    def _dmypy(self, args: Sequence[str], check: bool = True) -> str:
        assert self._status_directory is not None
        assert self._process is not None

        polled = self._process.poll()
        if polled is not None:
            raise Exception(f"dmypy exited unexpectedly: {polled}")

        cmd = [
            "dmypy",
            "--status-file",
            os.path.join(self._status_directory.name, ".dmypy.json"),
        ] + list(args)

        result = subprocess.run(
            cmd,
            check=check,
            stdout=subprocess.PIPE,
            text=True,
        )

        try:
            result.check_returncode()
        except subprocess.CalledProcessError:
            logging.warning(
                "mypy command exited unsuccessfully: %r",
                cmd,
                exc_info=True,
            )

        return result.stdout

    def __enter__(self) -> "MyPyBuilder":
        assert self._exit_stack is None
        assert self._process is None
        assert self._status_directory is None

        with ExitStack() as exit_stack:
            self._status_directory = TemporaryDirectory()
            status_file = os.path.join(
                exit_stack.enter_context(self._status_directory), ".dmypy.json"
            )

            self._process = subprocess.Popen(
                [
                    "dmypy",
                    "--status-file",
                    status_file,
                    "daemon",
                    "--",
                    "--ignore-missing-imports",
                ]
            )

            exit_stack.enter_context(self._process)

            while not os.path.exists(status_file):
                time.sleep(0.1)
                result = self._process.poll()
                if result is not None:
                    raise Exception(f"dmypy exited early: {result}")

            self._exit_stack = exit_stack.pop_all()

        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        assert self._exit_stack is not None
        assert self._process is not None
        assert self._status_directory is not None

        try:
            self._dmypy(["stop"])
        finally:
            self._exit_stack.close()
            self._exit_stack = None
            self._process = None
            self._status_directory = None


class MyPyDaemonNode(Node):
    child: Node
    _build: MyPyBuilder

    def __init__(self, build: MyPyBuilder, child: Node) -> None:
        self.child = child
        self._build = build

    @property
    def children(self) -> Tuple[Node]:
        return (self.child,)

    def replace_child(self, old: Node, new: Node) -> None:
        if self.child == old:
            self.child = new


class _AnnotateVisitor(Visitor):
    build: MyPyBuilder

    def __init__(self, build: MyPyBuilder) -> None:
        self.build = build

    def enter(self, node: Node) -> Visit:
        if not isinstance(node, CstNode):
            return Visit.TraverseChildren

        # TODO

        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        pass  # TODO


class MyPyTransform(Transform):
    """
    Converts mypy nodes into Python language nodes.
    """

    def __init__(self, config: PluginSettings) -> None:
        pass

    def transform(self, document: Document) -> None:
        """
        Apply the transformation to the given document.
        """
        root = document.root
        if not isinstance(root, MyPyDaemonNode):
            return
        document.root = root.child

        visitor = _AnnotateVisitor(root._build)
        document.root.visit(visitor)


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

    def enter(self, any_node: Node) -> Visit:
        if not isinstance(any_node, CstNode):
            raise ValueError(
                "expected `"
                + CstNode.__name__
                + "` but got `"
                + any_node.__class__.__name__
                + "`"
            )

        node = any_node.cst_node
        try:
            parent = self.old_stack[-1].node.cst_node
        except IndexError:
            parent = None

        module_member = isinstance(parent, cst.Module)

        visit: Visit

        if isinstance(node, cst.Module):
            visit = self.enter_module(any_node, node)
        elif isinstance(node, cst.ClassDef):
            visit = self.enter_class_def(any_node, node)
        elif isinstance(node, cst.FunctionDef):
            visit = self.enter_function_def(any_node, node)
        elif module_member and isinstance(node, cst.CSTNode):
            logging.warning("skipping module member node %s", any_node)
            visit = Visit.SkipChildren
        else:
            raise Exception(f"unknown node type {node}")

        self.old_stack.append(_TransformContext(node=any_node))

        return visit

    def exit(self, any_node: Node) -> None:
        module_member = False

        self.old_stack.pop()
        if self.old_stack:
            self.old_stack[-1].child_offset += 1
            parent = self.old_stack[-1].node.cst_node
            module_member = isinstance(parent, cst.Module)

        assert isinstance(any_node, CstNode)
        node = any_node.cst_node

        if isinstance(node, cst.Module):
            self.exit_module()
        elif isinstance(node, cst.ClassDef):
            self.exit_class_def()
        elif isinstance(node, cst.FunctionDef):
            self.exit_function_def()
        elif module_member and isinstance(node, cst.CSTNode):
            pass
        else:
            raise Exception(f"unknown node type {node}")


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
