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
from typing import Dict, Optional, Set

import pytest

from docc.context import Context
from docc.document import BlankNode, Document
from docc.plugins.files import (
    FileNode,
    FilesBuilder,
    FilesDiscover,
    FileSource,
)
from docc.settings import PluginSettings, Settings
from docc.source import Source


@pytest.fixture
def plugin_settings(tmp_path: Path) -> PluginSettings:
    settings = Settings(tmp_path, {"tool": {"docc": {}}})
    return settings.for_plugin("docc.files.discover")


class TestFileSource:
    def test_init(self) -> None:
        relative = PurePath("src/file.txt")
        absolute = PurePath("/path/to/src/file.txt")
        source = FileSource(relative, absolute)

        assert source.relative_path == relative
        assert source.absolute_path == absolute

    def test_output_path_removes_suffix(self) -> None:
        relative = PurePath("docs/readme.md")
        absolute = PurePath("/path/docs/readme.md")
        source = FileSource(relative, absolute)

        assert source.output_path == PurePath("docs/readme")

    def test_output_path_compound_extension(self) -> None:
        relative = PurePath("docs/archive.tar.gz")
        absolute = PurePath("/path/docs/archive.tar.gz")
        source = FileSource(relative, absolute)

        assert source.output_path == PurePath("docs/archive.tar")

    def test_output_path_no_suffix(self) -> None:
        relative = PurePath("docs/readme")
        absolute = PurePath("/path/docs/readme")
        source = FileSource(relative, absolute)

        assert source.output_path == PurePath("docs/readme")


class TestFileNode:
    def test_init(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        node = FileNode(file_path)
        assert node.path == file_path

    def test_children_empty(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        node = FileNode(file_path)
        assert node.children == ()

    def test_replace_child_raises(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        node = FileNode(file_path)
        with pytest.raises(TypeError):
            node.replace_child(BlankNode(), BlankNode())

    def test_extension(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        node = FileNode(file_path)
        assert node.extension == ".txt"

    def test_extension_multiple_suffixes(self, tmp_path: Path) -> None:
        file_path = tmp_path / "archive.tar.gz"
        file_path.write_text("content")

        node = FileNode(file_path)
        assert node.extension == ".gz"

    def test_output(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("file content here")

        node = FileNode(file_path)
        context = Context({})
        destination = StringIO()

        node.output(context, destination)

        assert destination.getvalue() == "file content here"


class TestFilesBuilder:
    def test_build_processes_file_sources(
        self, tmp_path: Path, plugin_settings: PluginSettings
    ) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")

        source = FileSource(PurePath("test.txt"), file_path)
        unprocessed: Set[Source] = {source}
        processed: Dict[Source, Document] = {}

        builder = FilesBuilder(plugin_settings)
        builder.build(unprocessed, processed)

        assert source in processed
        assert len(processed) == 1
        assert len(unprocessed) == 0
        assert isinstance(processed[source].root, FileNode)

    def test_build_ignores_non_file_sources(
        self, plugin_settings: PluginSettings
    ) -> None:
        class OtherSource(Source):
            @property
            def relative_path(self) -> Optional[PurePath]:
                return PurePath("other.py")

            @property
            def output_path(self) -> PurePath:
                return PurePath("other.py")

        source = OtherSource()
        unprocessed: Set[Source] = {source}
        processed: Dict[Source, Document] = {}

        builder = FilesBuilder(plugin_settings)
        builder.build(unprocessed, processed)

        assert source in unprocessed
        assert len(processed) == 0


class TestFilesDiscover:
    def test_init_no_files(self, tmp_path: Path) -> None:
        settings = Settings(tmp_path, {"tool": {"docc": {}}})
        plugin_settings = settings.for_plugin("docc.files.discover")

        discover = FilesDiscover(plugin_settings)
        assert discover.sources == []

    def test_init_with_files(self, tmp_path: Path) -> None:
        first_file = tmp_path / "file1.txt"
        first_file.write_text("content1")
        second_file = tmp_path / "file2.txt"
        second_file.write_text("content2")

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.files.discover": {
                                "files": ["file1.txt", "file2.txt"]
                            }
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.files.discover")

        discover = FilesDiscover(plugin_settings)
        assert len(discover.sources) == 2

    def test_discover_yields_sources(self, tmp_path: Path) -> None:
        first_file = tmp_path / "file1.txt"
        first_file.write_text("content")

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.files.discover": {"files": ["file1.txt"]}
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.files.discover")

        discover = FilesDiscover(plugin_settings)
        sources = list(discover.discover(frozenset()))

        assert len(sources) == 1
        assert isinstance(sources[0], FileSource)

    def test_discover_empty_when_no_files(self, tmp_path: Path) -> None:
        settings = Settings(tmp_path, {"tool": {"docc": {}}})
        plugin_settings = settings.for_plugin("docc.files.discover")

        discover = FilesDiscover(plugin_settings)
        sources = list(discover.discover(frozenset()))

        assert sources == []
