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

from docc.context import Context
from docc.settings import PluginSettings, Settings
from docc.transform import Transform, load


class ConcreteTransform(Transform):
    def __init__(self, config: PluginSettings) -> None:
        self.config = config

    def transform(self, context: Context) -> None:
        pass


def test_transform_init(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"tool": {"docc": {}}})
    plugin_settings = settings.for_plugin("test")

    transform = ConcreteTransform(plugin_settings)
    assert transform.config is plugin_settings


class TestTransformLoad:
    def test_load_empty_transform_list(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {"tool": {"docc": {"transform": []}}},
        )

        result = load(settings)
        assert result == []

    def test_load_single_transform(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {"tool": {"docc": {"transform": ["docc.python.transform"]}}},
        )

        result = load(settings)
        assert len(result) == 1
        assert result[0][0] == "docc.python.transform"

    def test_load_multiple_transforms(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
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

    def test_load_preserves_order(self, tmp_path: Path) -> None:
        transforms = [
            "docc.python.transform",
            "docc.mistletoe.transform",
            "docc.html.transform",
        ]
        settings = Settings(
            tmp_path,
            {"tool": {"docc": {"transform": transforms}}},
        )

        result = load(settings)
        for i, (name, _) in enumerate(result):
            assert name == transforms[i]
