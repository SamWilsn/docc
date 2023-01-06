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
Source discovery based on mypy.
"""

import os.path
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import PurePath
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
)

from mypy.find_sources import create_source_list
from mypy.modulefinder import BuildSource
from mypy.options import Options

from docc.build import Builder
from docc.discover import Discover, T
from docc.document import Document, Node, Visit, Visitor
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

    def _dmypy(self, args: Sequence[str], check=True) -> str:
        assert self._status_directory is not None
        assert self._process is not None

        polled = self._process.poll()
        if polled is not None:
            raise Exception(f"dmypy exited unexpectedly: {polled}")

        result = subprocess.run(
            [
                "dmypy",
                "--status-file",
                os.path.join(self._status_directory.name, ".dmypy.json"),
            ]
            + list(args),
            check=check,
            stdout=subprocess.PIPE,
            text=True,
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


class _Visitor(Visitor):
    build: MyPyBuilder

    def __init__(self, build: MyPyBuilder) -> None:
        self.build = build

    def enter(self, node: Node) -> Visit:
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

        visitor = _Visitor(root._build)
        document.root.visit(visitor)
