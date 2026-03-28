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

import tempfile
from io import StringIO
from pathlib import Path, PurePath
from typing import Optional, TextIO

import pytest

from docc.source import Source, TextSource


class ConcreteSource(Source):
    _output: PurePath

    def __init__(
        self,
        relative: Optional[PurePath] = None,
        output: Optional[PurePath] = None,
    ) -> None:
        self._relative = relative
        self._output = output or PurePath("output.html")

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._relative

    @property
    def output_path(self) -> PurePath:
        return self._output


class ConcreteTextSource(TextSource):
    _relative: PurePath
    _output: PurePath

    def __init__(
        self,
        content: str,
        relative: Optional[PurePath] = None,
        output: Optional[PurePath] = None,
    ) -> None:
        self._content = content
        self._relative = relative or PurePath("test.py")
        self._output = output or self._relative

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._relative

    @property
    def output_path(self) -> PurePath:
        return self._output

    def open(self) -> TextIO:
        return StringIO(self._content)


class TestSource:
    def test_repr_with_relative_path(self) -> None:
        source = ConcreteSource(relative=PurePath("src/module.py"))
        result = repr(source)

        assert "src/module.py" in result
        assert "ConcreteSource" in result

    def test_repr_without_relative_path(self) -> None:
        source = ConcreteSource(relative=None)
        result = repr(source)

        assert "ConcreteSource" in result

    def test_output_path(self) -> None:
        source = ConcreteSource(output=PurePath("docs/output.html"))
        assert source.output_path == PurePath("docs/output.html")


class TestTextSource:
    def test_open_returns_text_io(self) -> None:
        source = ConcreteTextSource("content")
        with source.open() as f:
            assert f.read() == "content"

    def test_line_returns_correct_line(self) -> None:
        content = "line1\nline2\nline3"
        source = ConcreteTextSource(content)

        assert source.line(1) == "line1"
        assert source.line(2) == "line2"
        assert source.line(3) == "line3"

    def test_line_out_of_range_raises(self) -> None:
        content = "line1\nline2"
        source = ConcreteTextSource(content)

        with pytest.raises(IndexError, match="line 10 out of range"):
            source.line(10)

    def test_line_single_line(self) -> None:
        content = "single line"
        source = ConcreteTextSource(content)

        assert source.line(1) == "single line"

    def test_line_empty_content(self) -> None:
        content = ""
        source = ConcreteTextSource(content)

        assert source.line(1) == ""

    def test_line_with_empty_lines(self) -> None:
        content = "first\n\nthird"
        source = ConcreteTextSource(content)

        assert source.line(1) == "first"
        assert source.line(2) == ""
        assert source.line(3) == "third"


class TestTextSourceBoundary:
    def test_line_zero_returns_last_line(self) -> None:
        """
        line(0) computes lines[0 - 1] = lines[-1], which silently
        returns the last line due to Python negative indexing.
        """
        content = "first\nsecond\nthird"
        source = ConcreteTextSource(content)
        # line(0) accesses lines[-1] which is "third"
        assert source.line(0) == "third"

    def test_line_negative_one_returns_second_to_last(self) -> None:
        """
        line(-1) computes lines[-1 - 1] = lines[-2], which silently
        returns the second-to-last line due to Python negative indexing.
        """
        content = "first\nsecond\nthird"
        source = ConcreteTextSource(content)
        # line(-1) accesses lines[-2] which is "second"
        assert source.line(-1) == "second"


def test_text_source_line_from_real_file() -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False
    ) as f:
        f.write("# Line 1\n# Line 2\n# Line 3\n")
        f.flush()
        path = Path(f.name)

    class FileTextSource(TextSource):
        def __init__(self, file_path: Path):
            self._path = file_path

        @property
        def relative_path(self) -> Optional[PurePath]:
            return PurePath(self._path.name)

        @property
        def output_path(self) -> PurePath:
            return PurePath(self._path.name)

        def open(self) -> TextIO:
            return open(self._path, "r")

    try:
        source = FileTextSource(path)
        assert source.line(1) == "# Line 1"
        assert source.line(2) == "# Line 2"
    finally:
        path.unlink()
