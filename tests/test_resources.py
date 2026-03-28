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
from pathlib import Path, PurePath
from typing import Dict, Set

import pytest

from docc.context import Context
from docc.document import BlankNode, Document
from docc.plugins.resources import (
    ResourceBuilder,
    ResourceNode,
    ResourceSource,
)
from docc.settings import PluginSettings, Settings
from docc.source import Source


@pytest.fixture
def plugin_settings() -> PluginSettings:
    settings = Settings(Path("."), {"tool": {"docc": {}}})
    return settings.for_plugin("docc.resources.build")


class TestResourceSource:
    def test_with_path_creates_source(self) -> None:
        source = ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "docc.css",
            PurePath("static") / "docc",
        )
        assert source is not None
        assert source.output_path == PurePath("static") / "docc"
        assert source.extension == ".css"

    def test_relative_path_is_none(self) -> None:
        source = ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "docc.css",
            PurePath("static") / "docc",
        )
        assert source.relative_path is None

    def test_output_path(self) -> None:
        source = ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "docc.css",
            PurePath("output") / "style",
        )
        assert source.output_path == PurePath("output") / "style"


class TestResourceNode:
    def test_children_empty(self) -> None:
        source = ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "docc.css",
            PurePath("static") / "docc",
        )
        node = ResourceNode(source.resource, source.extension)
        assert node.children == ()

    def test_extension(self) -> None:
        source = ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "docc.css",
            PurePath("static") / "docc",
        )
        node = ResourceNode(source.resource, source.extension)
        assert node.extension == ".css"

    def test_replace_child_raises(self) -> None:
        source = ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "docc.css",
            PurePath("static") / "docc",
        )
        node = ResourceNode(source.resource, source.extension)

        with pytest.raises(TypeError):
            node.replace_child(BlankNode(), BlankNode())

    def test_output(self) -> None:
        source = ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "docc.css",
            PurePath("static") / "docc",
        )
        node = ResourceNode(source.resource, source.extension)
        context = Context({})
        destination = StringIO()

        node.output(context, destination)

        result = destination.getvalue()
        assert len(result) > 0, "Output should not be empty"
        assert "{" in result, "CSS output should contain style blocks"


class TestResourceBuilder:
    def test_build_processes_resource_sources(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "docc.css",
            PurePath("static") / "docc",
        )

        unprocessed: Set[Source] = {source}
        processed: Dict[Source, Document] = {}

        builder = ResourceBuilder(plugin_settings)
        builder.build(unprocessed, processed)

        assert len(unprocessed) == 0
        assert len(processed) == 1
        assert source in processed
        assert isinstance(processed[source].root, ResourceNode)

    def test_build_ignores_non_resource_sources(
        self, plugin_settings: PluginSettings
    ) -> None:
        class OtherSource(Source):
            @property
            def relative_path(self):
                return PurePath("other.py")

            @property
            def output_path(self):
                return PurePath("other.py")

        source = OtherSource()
        unprocessed: Set[Source] = {source}
        processed: Dict[Source, Document] = {}

        builder = ResourceBuilder(plugin_settings)
        builder.build(unprocessed, processed)

        assert source in unprocessed
        assert len(processed) == 0
