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

from typing import Any

import mistletoe as md

from docc.document import Node
from docc.plugins.mistletoe import MarkdownNode, ReferenceTransform
from docc.plugins.references import Reference
from docc.settings import PluginSettings


def test_reference_transform_to_reference(
    assert_in: Any, make_context: Any, plugin_settings: PluginSettings
) -> None:
    markdown = "[a reference](ref:hello-world)"

    root = MarkdownNode(md.Document(markdown))
    context = make_context(root)

    transform = ReferenceTransform(plugin_settings)
    transform.transform(context)

    def matcher(node: Node) -> bool:
        if not isinstance(node, Reference):
            return False

        return node.identifier == "hello-world"

    assert_in(root, matcher)


def test_reference_transform_no_reference(
    assert_not_in: Any, make_context: Any, plugin_settings: PluginSettings
) -> None:
    markdown = "[a reference](http://hello-world)"

    root = MarkdownNode(md.Document(markdown))
    context = make_context(root)

    transform = ReferenceTransform(plugin_settings)
    transform.transform(context)

    def matcher(node: Node) -> bool:
        if isinstance(node, Reference):
            return True
        return False

    assert_not_in(root, matcher)
