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

from io import StringIO
from pathlib import Path

import pytest

from docc.context import Context
from docc.document import BlankNode, Document, ListNode
from docc.plugins.debug import DebugNode, DebugTransform
from docc.settings import PluginSettings, Settings


@pytest.fixture
def plugin_settings() -> PluginSettings:
    settings = Settings(Path("."), {"tool": {"docc": {}}})
    return settings.for_plugin("docc.debug.transform")


class TestDebugNode:
    def test_init(self) -> None:
        child = BlankNode()
        node = DebugNode(child)
        assert node.child is child

    def test_children(self) -> None:
        child = BlankNode()
        node = DebugNode(child)
        children = tuple(node.children)
        assert children == (child,)

    def test_replace_child(self) -> None:
        old_child = BlankNode()
        new_child = BlankNode()
        node = DebugNode(old_child)

        node.replace_child(old_child, new_child)
        assert node.child is new_child

    def test_replace_child_no_match(self) -> None:
        child = BlankNode()
        other = BlankNode()
        new_child = BlankNode()
        node = DebugNode(child)

        node.replace_child(other, new_child)
        assert node.child is child

    def test_extension(self) -> None:
        node = DebugNode(BlankNode())
        assert node.extension == ".txt"

    def test_output(self) -> None:
        child = BlankNode()
        node = DebugNode(child)
        context = Context({})
        destination = StringIO()

        node.output(context, destination)

        result = destination.getvalue()
        assert "<blank>" in result

    def test_output_nested(self) -> None:
        inner = BlankNode()
        outer = ListNode([inner])
        node = DebugNode(outer)
        context = Context({})
        destination = StringIO()

        node.output(context, destination)

        result = destination.getvalue()
        assert "<list>" in result
        assert "<blank>" in result


class TestDebugTransform:
    def test_transform(self, plugin_settings: PluginSettings) -> None:
        root = BlankNode()
        document = Document(root)
        context = Context({Document: document})

        transform = DebugTransform(plugin_settings)
        transform.transform(context)

        assert isinstance(document.root, DebugNode)
        assert document.root.child is root

    def test_transform_with_nested(
        self, plugin_settings: PluginSettings
    ) -> None:
        inner = BlankNode()
        root = ListNode([inner])
        document = Document(root)
        context = Context({Document: document})

        transform = DebugTransform(plugin_settings)
        transform.transform(context)

        assert isinstance(document.root, DebugNode)
        assert document.root.child is root
