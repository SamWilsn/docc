# Copyright (C) 2025 Ethereum Foundation
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

from pathlib import Path
from typing import Callable, Final, Union

import pytest
from typing_extensions import override

from docc.context import Context
from docc.document import Document, Node, Visit, Visitor
from docc.settings import PluginSettings, Settings


class ContainsVisitor(Visitor):
    found: bool
    matcher: Final[Callable[[Node], bool]]

    def __init__(self, matcher: Callable[[Node], bool]) -> None:
        self.found = False
        self.matcher = matcher

    @override
    def enter(self, node: Node) -> Visit:
        if self.found:
            return Visit.SkipChildren

        if self.matcher(node):
            self.found = True
            return Visit.SkipChildren

        return Visit.TraverseChildren

    @override
    def exit(self, node: Node) -> None:
        del node


def _assert_in(
    haystack: Union[Node, Document], matcher: Callable[[Node], bool]
) -> None:
    if isinstance(haystack, Document):
        haystack = haystack.root
    visitor = ContainsVisitor(matcher)
    haystack.visit(visitor)
    assert visitor.found


def _assert_not_in(
    haystack: Union[Node, Document], matcher: Callable[[Node], bool]
) -> None:
    if isinstance(haystack, Document):
        haystack = haystack.root
    visitor = ContainsVisitor(matcher)
    haystack.visit(visitor)
    assert not visitor.found


@pytest.fixture
def assert_in() -> (
    Callable[[Union[Node, Document], Callable[[Node], bool]], None]
):
    return _assert_in


@pytest.fixture
def assert_not_in() -> (
    Callable[[Union[Node, Document], Callable[[Node], bool]], None]
):
    return _assert_not_in


@pytest.fixture
def settings() -> Settings:
    return Settings(Path("."), {})


@pytest.fixture
def plugin_settings(settings: Settings) -> PluginSettings:
    return PluginSettings(settings, {})


def _make_context(root: Node) -> Context:
    document = Document(root)

    return Context(
        {
            Document: document,
        }
    )


@pytest.fixture
def make_context() -> Callable[[Node], Context]:
    return _make_context
