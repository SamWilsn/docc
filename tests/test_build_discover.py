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

from pathlib import Path

from docc.build import Builder
from docc.build import load as load_builders
from docc.discover import Discover
from docc.discover import load as load_discovers
from docc.settings import Settings


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
