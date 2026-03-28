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

from pathlib import Path, PurePath
from typing import Dict, FrozenSet, Iterator, Optional, Set

from docc.build import Builder
from docc.build import load as load_builders
from docc.discover import Discover
from docc.discover import load as load_discovers
from docc.document import BlankNode, Document
from docc.settings import PluginSettings, Settings
from docc.source import Source


class MockSource(Source):
    _path: PurePath

    def __init__(self, path: Optional[PurePath] = None) -> None:
        self._path = path if path is not None else PurePath("mock.py")

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._path

    @property
    def output_path(self) -> PurePath:
        return self._path


class ConcreteDiscover(Discover):
    def __init__(self, config: PluginSettings) -> None:
        self.config = config

    def discover(self, known: FrozenSet[Source]) -> Iterator[Source]:
        yield MockSource()


class ConcreteBuilder(Builder):
    def __init__(self, config: PluginSettings) -> None:
        self.config = config

    def build(
        self,
        unprocessed: Set[Source],
        processed: Dict[Source, Document],
    ) -> None:
        for source in list(unprocessed):
            if isinstance(source, MockSource):
                unprocessed.remove(source)
                processed[source] = Document(BlankNode())


def test_discover_yields_source(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"tool": {"docc": {}}})
    plugin_settings = settings.for_plugin("test")

    discover = ConcreteDiscover(plugin_settings)
    sources = list(discover.discover(frozenset()))

    assert len(sources) == 1
    assert isinstance(sources[0], MockSource)


def test_builder_processes_source(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"tool": {"docc": {}}})
    plugin_settings = settings.for_plugin("test")

    builder = ConcreteBuilder(plugin_settings)
    unprocessed: Set[Source] = {MockSource()}
    processed: Dict[Source, Document] = {}

    builder.build(unprocessed, processed)

    assert len(unprocessed) == 0
    assert len(processed) == 1


def test_builder_context_manager(tmp_path: Path) -> None:
    """
    Builder extends AbstractContextManager, so it must support
    the with statement (enter/exit protocol).
    """
    settings = Settings(tmp_path, {"tool": {"docc": {}}})
    plugin_settings = settings.for_plugin("test")

    builder = ConcreteBuilder(plugin_settings)
    with builder as b:
        assert b is builder


class TestLoadDiscovers:
    def test_load_empty_discovery_list(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {"tool": {"docc": {"discovery": []}}},
        )

        result = list(load_discovers(settings))
        assert result == []

    def test_load_single_discover(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "discovery": ["docc.python.discover"],
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]}
                        },
                    }
                }
            },
        )

        result = list(load_discovers(settings))
        assert len(result) == 1
        assert result[0][0] == "docc.python.discover"
        assert isinstance(result[0][1], Discover)


class TestLoadBuilders:
    def test_load_empty_builder_list(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {"tool": {"docc": {"build": []}}},
        )

        result = list(load_builders(settings))
        assert result == []

    def test_load_single_builder(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "build": ["docc.python.build"],
                        "plugins": {"docc.python.build": {}},
                    }
                }
            },
        )

        result = list(load_builders(settings))
        assert len(result) == 1
        assert result[0][0] == "docc.python.build"
        assert isinstance(result[0][1], Builder)
