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
Shared index of definitions.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Set

from .source import Source


@dataclass(eq=True, frozen=True)
class Definition:
    """
    Location of a node.
    """

    source: Source
    identifier: str
    specifier: int


class ReferenceError(Exception):
    """
    Exception raised when a reference doesn't match a definition.
    """

    identifier: str

    def __init__(self, identifier: str) -> None:
        super().__init__(f"undefined identifier: `{identifier}`")
        self.identifier = identifier


class Index:
    """
    Tracks the location of definitions.
    """

    _index: Dict[str, Set[Definition]]

    def __init__(self) -> None:
        self._index = defaultdict(set)

    def define(self, source: Source, identifier: str) -> Definition:
        """
        Register a new definition in the index.
        """
        existing = self._index[identifier]
        definition = Definition(
            source=source, identifier=identifier, specifier=len(existing)
        )
        existing.add(definition)
        return definition

    def lookup(self, identifier: str) -> Iterable[Definition]:
        """
        Find a definition that was previously registered.
        """
        got = self._index[identifier]
        if not got:
            raise ReferenceError(identifier)
        return got
