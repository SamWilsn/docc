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

import logging
from io import StringIO, TextIOBase
from pathlib import Path
from typing import Tuple

import pytest

from docc.cli import _OutputVisitor, main
from docc.context import Context
from docc.document import (
    BlankNode,
    ListNode,
    Node,
    OutputNode,
    Visit,
)


class MockOutputNode(OutputNode):
    def __init__(
        self, content: str = "output content", ext: str = ".html"
    ) -> None:
        self._content = content
        self._ext = ext

    @property
    def children(self) -> Tuple[()]:
        return ()

    def replace_child(self, old: Node, new: Node) -> None:
        pass

    @property
    def extension(self) -> str:
        return self._ext

    def output(self, context: Context, destination: TextIOBase) -> None:
        destination.write(self._content)


class TestOutputVisitor:
    def test_enter_output_node_calls_output(self) -> None:
        output_node = MockOutputNode("test output")
        context = Context({})
        destination = StringIO()

        visitor = _OutputVisitor(context, destination)
        result = visitor.enter(output_node)

        assert result == Visit.SkipChildren
        assert destination.getvalue() == "test output"

    def test_enter_non_output_node_traverses(self) -> None:
        node = BlankNode()
        context = Context({})
        destination = StringIO()

        visitor = _OutputVisitor(context, destination)
        result = visitor.enter(node)

        assert result == Visit.TraverseChildren

    def test_enter_list_node_traverses(self) -> None:
        node = ListNode([BlankNode()])
        context = Context({})
        destination = StringIO()

        visitor = _OutputVisitor(context, destination)
        result = visitor.enter(node)

        assert result == Visit.TraverseChildren

    def test_exit_does_nothing(self) -> None:
        node = BlankNode()
        context = Context({})
        destination = StringIO()

        visitor = _OutputVisitor(context, destination)
        result = visitor.exit(node)

        assert result is None, "exit() should return None"
        assert (
            destination.getvalue() == ""
        ), "exit() should not write to destination"


class TestMainFunction:
    def test_main_requires_output_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text("[tool.docc]\n")

        monkeypatch.chdir(tmp_path)
        with caplog.at_level(logging.CRITICAL):
            with pytest.raises(SystemExit) as exc_info:
                main([])
        assert exc_info.value.code == 1
        assert any(
            "Output path is required" in r.message for r in caplog.records
        )

    def test_main_with_output_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        py_file = src_dir / "example.py"
        py_file.write_text('"""Module docstring."""\n')

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            f"""
[tool.docc]
discovery = ["docc.python.discover"]
build = ["docc.python.build"]
transform = ["docc.python.transform", "docc.html.transform"]
context = ["docc.html.context"]

[tool.docc.plugins."docc.python.discover"]
paths = ["{src_dir}"]

[tool.docc.output]
path = "docs"
"""
        )

        output_dir = tmp_path / "output"

        monkeypatch.chdir(tmp_path)
        main(["--output", str(output_dir)])

        assert output_dir.exists(), "Output directory should be created"

    def test_main_uses_settings_output_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        py_file = src_dir / "example.py"
        py_file.write_text('"""Module docstring."""\n')

        output_dir = tmp_path / "docs"
        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            f"""
[tool.docc]
discovery = ["docc.python.discover"]
build = ["docc.python.build"]
transform = ["docc.python.transform", "docc.html.transform"]
context = ["docc.html.context"]

[tool.docc.plugins."docc.python.discover"]
paths = ["{src_dir}"]

[tool.docc.output]
path = "{output_dir}"
"""
        )

        monkeypatch.chdir(tmp_path)
        main([])

        assert (
            output_dir.exists()
        ), "Output directory should be created from settings"


class TestOutputVisitorWithNestedNodes:
    def test_nested_output_nodes(self) -> None:
        inner = MockOutputNode("inner")
        outer_content = ListNode([inner])

        class ContainerNode(OutputNode):
            @property
            def children(self) -> Tuple[ListNode]:
                return (outer_content,)

            def replace_child(self, old: Node, new: Node) -> None:
                raise NotImplementedError

            @property
            def extension(self) -> str:
                return ".html"

            def output(
                self, context: Context, destination: TextIOBase
            ) -> None:
                destination.write("outer")

        container = ContainerNode()
        context = Context({})
        destination = StringIO()

        visitor = _OutputVisitor(context, destination)
        container.visit(visitor)

        assert destination.getvalue() == "outer"

    def test_multiple_output_nodes(self) -> None:
        first_node = MockOutputNode("first")
        second_node = MockOutputNode("second")
        root = ListNode([first_node, second_node])

        context = Context({})
        destination = StringIO()

        visitor = _OutputVisitor(context, destination)
        root.visit(visitor)

        assert destination.getvalue() == "firstsecond"


def test_main_processes_python_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()

    py_file = src_dir / "example.py"
    py_file.write_text('"""Module docstring."""\n\ndef hello():\n    pass\n')

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        f"""
[tool.docc]
discovery = ["docc.python.discover"]
build = ["docc.python.build"]
transform = [
    "docc.python.transform",
    "docc.verbatim.transform",
    "docc.references.index",
    "docc.html.transform",
]
context = ["docc.html.context", "docc.references.context"]

[tool.docc.plugins."docc.python.discover"]
paths = ["{src_dir}"]

[tool.docc.output]
path = "docs"
"""
    )

    output_dir = tmp_path / "docs"

    monkeypatch.chdir(tmp_path)
    main(["--output", str(output_dir)])

    assert output_dir.exists(), "Output directory should be created"
    html_files = list(output_dir.rglob("*.html"))
    assert len(html_files) >= 1, "Should produce at least one HTML file"
    content = html_files[0].read_text()
    assert "Module docstring" in content


def test_main_empty_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.docc]
discovery = []
build = []
transform = []
context = []

[tool.docc.output]
path = "docs"
"""
    )

    output_dir = tmp_path / "docs"

    monkeypatch.chdir(tmp_path)
    main(["--output", str(output_dir)])

    assert not output_dir.exists(), "No output should be created"


def test_main_duplicate_context_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    When two context plugins provide the same type, main() raises
    an Exception about the conflict.
    """
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        """
[tool.docc]
discovery = []
build = []
transform = []
context = ["docc.references.context", "docc.references.context"]

[tool.docc.output]
path = "docs"
"""
    )

    monkeypatch.chdir(tmp_path)
    with pytest.raises(Exception, match="conflicts with"):
        main(["--output", str(tmp_path / "docs")])


def test_main_document_without_extension_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    When a document has no extension (no OutputNode), the write
    phase logs an error and skips it.
    """
    pyproject = tmp_path / "pyproject.toml"
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    py_file = src_dir / "example.py"
    py_file.write_text('"""Module docstring."""\n')

    pyproject.write_text(
        f"""
[tool.docc]
discovery = ["docc.python.discover"]
build = ["docc.python.build"]
transform = []
context = []

[tool.docc.plugins."docc.python.discover"]
paths = ["{src_dir}"]

[tool.docc.output]
path = "docs"
"""
    )

    output_dir = tmp_path / "docs"

    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.ERROR):
        main(["--output", str(output_dir)])

    assert any(
        "does not specify a file extension" in r.message
        for r in caplog.records
    )
