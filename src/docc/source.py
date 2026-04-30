# Copyright (C) 2022-2023,2026 Ethereum Foundation
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
Sources are the inputs for documentation generation.
"""

from abc import ABC, abstractmethod
from io import StringIO
from pathlib import PurePath
from typing import Final, Optional, Sequence, TextIO

from typing_extensions import override


class Source(ABC):
    """
    An input to generate documentation for.
    """

    @property
    @abstractmethod
    def relative_path(self) -> Optional[PurePath]:
        """
        Path to the Source (if one exists) relative to the project root.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_path(self) -> PurePath:
        """
        Where to write the output from this Source relative to the output path.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """
        String representation of the source.
        """
        if self.relative_path is None:
            return super().__repr__()
        else:
            return (
                f"<{self.__module__}."
                f"{self.__class__.__qualname__}: "
                f'"{self.relative_path}">'
            )


class TextSource(Source):
    """
    A Source that supports reading text snippets.
    """

    @abstractmethod
    def open(self) -> TextIO:
        """
        Open the source for reading.
        """

    def line(self, number: int) -> str:
        """
        Extract a line of text from the source.
        """
        # TODO: Don't reopen and reread the file every time...
        with self.open() as f:
            lines = f.read().split("\n")
            try:
                return lines[number - 1]
            except IndexError as e:
                raise IndexError(
                    f"line {number} out of range for `{self.relative_path}`"
                ) from e


class StringSource(TextSource):
    """
    A Source that reads text snippets from a `str`.
    """

    _output_path: Final[PurePath]
    _relative_path: Final[Optional[PurePath]]

    def __init__(
        self,
        text: str,
        output_path: PurePath,
        relative_path: Optional[PurePath] = None,
    ) -> None:
        self._lines: Final[Sequence[str]] = text.split("\n")
        self._text: Final[str] = text
        self._output_path = output_path
        self._relative_path = relative_path

    @override
    def open(self) -> TextIO:
        """
        Open the source for reading.
        """
        return StringIO(self._text)

    @override
    def line(self, number: int) -> str:
        """
        Extract a line of text from the source.
        """
        return self._lines[number - 1]

    @override
    @property
    def output_path(self) -> PurePath:
        """
        Where to write the output from this Source relative to the output path.
        """
        return self._output_path

    @override
    @property
    def relative_path(self) -> Optional[PurePath]:
        """
        Path to the Source (if one exists) relative to the project root.
        """
        return self._relative_path
