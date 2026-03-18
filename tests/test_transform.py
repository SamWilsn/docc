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

import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from docc.context import Context
from docc.settings import PluginSettings, Settings
from docc.transform import Transform, load


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


class ConcreteTransform(Transform):
    def __init__(self, config: PluginSettings) -> None:
        self.config = config

    def transform(self, context: Context) -> None:
        pass


def test_transform_init(temp_dir: Path) -> None:
    settings = Settings(temp_dir, {"tool": {"docc": {}}})
    plugin_settings = settings.for_plugin("test")

    transform = ConcreteTransform(plugin_settings)
    assert transform.config is plugin_settings


class TestTransformLoad:
    def test_load_empty_transform_list(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {"tool": {"docc": {"transform": []}}},
        )

        result = load(settings)
        assert result == []

    def test_load_single_transform(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {"tool": {"docc": {"transform": ["docc.python.transform"]}}},
        )

        result = load(settings)
        assert len(result) == 1
        assert result[0][0] == "docc.python.transform"

    def test_load_multiple_transforms(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {
                "tool": {
                    "docc": {
                        "transform": [
                            "docc.python.transform",
                            "docc.mistletoe.transform",
                        ]
                    }
                }
            },
        )

        result = load(settings)
        assert len(result) == 2

    def test_load_preserves_order(self, temp_dir: Path) -> None:
        transforms = [
            "docc.python.transform",
            "docc.mistletoe.transform",
            "docc.html.transform",
        ]
        settings = Settings(
            temp_dir,
            {"tool": {"docc": {"transform": transforms}}},
        )

        result = load(settings)
        for i, (name, _) in enumerate(result):
            assert name == transforms[i]
