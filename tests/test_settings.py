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
from pathlib import Path, PurePath
from typing import Iterator

import pytest

from docc.settings import (
    FILE_NAME,
    MAX_DEPTH,
    Output,
    PluginSettings,
    Settings,
    SettingsError,
)


@pytest.fixture
def temp_dir() -> Iterator[Path]:
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


class TestOutput:
    def test_create_with_path(self) -> None:
        output = Output(path=Path("/output/docs"))
        assert output.path == Path("/output/docs")

    def test_create_with_none(self) -> None:
        output = Output(path=None)
        assert output.path is None


class TestPluginSettings:
    def test_init(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = PluginSettings(settings, {"key": "value"})

        assert plugin_settings["key"] == "value"

    def test_len(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = PluginSettings(settings, {"a": 1, "b": 2, "c": 3})

        assert len(plugin_settings) == 3

    def test_iter(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = PluginSettings(settings, {"a": 1, "b": 2})

        keys = list(plugin_settings)
        assert "a" in keys
        assert "b" in keys

    def test_getitem(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = PluginSettings(settings, {"test_key": "test_value"})

        assert plugin_settings["test_key"] == "test_value"

    def test_getitem_missing_raises(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = PluginSettings(settings, {})

        with pytest.raises(KeyError):
            plugin_settings["missing"]

    def test_get_with_default(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = PluginSettings(settings, {})

        assert plugin_settings.get("missing", "default") == "default"

    def test_resolve_path(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = PluginSettings(settings, {})

        resolved = plugin_settings.resolve_path(PurePath("subdir"))
        assert resolved.is_absolute()
        assert str(temp_dir) in str(resolved)

    def test_unresolve_path(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = PluginSettings(settings, {})

        absolute = temp_dir / "subdir" / "file.py"
        relative = plugin_settings.unresolve_path(absolute)

        assert not relative.is_absolute()


class TestSettings:
    def test_init_with_empty_tool_docc(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        assert isinstance(settings.context, list)
        assert settings.output.path is None

    def test_init_without_tool_key(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {})
        assert isinstance(settings.context, list)
        assert settings.output.path is None

    def test_init_with_invalid_tool_type(self, temp_dir: Path) -> None:
        with pytest.raises(TypeError, match="must be a dict"):
            Settings(temp_dir, {"tool": "not_a_dict"})

    def test_output_path_from_settings(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {"tool": {"docc": {"output": {"path": "docs"}}}},
        )
        assert settings.output.path == Path("docs")

    def test_output_path_none_when_not_specified(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        assert settings.output.path is None

    def test_for_plugin(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {
                "tool": {
                    "docc": {"plugins": {"test.plugin": {"option": "value"}}}
                }
            },
        )
        plugin_settings = settings.for_plugin("test.plugin")

        assert plugin_settings["option"] == "value"

    def test_for_plugin_missing_returns_empty(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        plugin_settings = settings.for_plugin("nonexistent.plugin")

        assert len(plugin_settings) == 0

    def test_context_default(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        context = settings.context

        assert isinstance(context, list)
        assert "docc.references.context" in context
        assert "docc.search.context" in context
        assert "docc.html.context" in context

    def test_context_custom(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {"tool": {"docc": {"context": ["custom.context"]}}},
        )
        context = settings.context

        assert context == ["custom.context"]

    def test_discovery_default(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        discovery = settings.discovery

        assert isinstance(discovery, list)
        assert "docc.python.discover" in discovery
        assert "docc.html.discover" in discovery

    def test_discovery_custom(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {"tool": {"docc": {"discovery": ["custom.discover"]}}},
        )
        discovery = settings.discovery

        assert discovery == ["custom.discover"]

    def test_build_default(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        build = settings.build

        assert isinstance(build, list)
        assert "docc.python.build" in build

    def test_build_custom(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {"tool": {"docc": {"build": ["custom.build"]}}},
        )
        build = settings.build

        assert build == ["custom.build"]

    def test_transform_default(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        transform = settings.transform

        assert isinstance(transform, list)
        assert "docc.python.transform" in transform
        assert "docc.html.transform" in transform

    def test_transform_custom(self, temp_dir: Path) -> None:
        settings = Settings(
            temp_dir,
            {"tool": {"docc": {"transform": ["custom.transform"]}}},
        )
        transform = settings.transform

        assert transform == ["custom.transform"]

    def test_resolve_path(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        resolved = settings.resolve_path(PurePath("subdir"))

        assert resolved.is_absolute()
        assert str(temp_dir) in str(resolved)

    def test_resolve_path_escapes_root_raises(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})

        with pytest.raises(ValueError):
            settings.resolve_path(PurePath("../escape"))

    def test_unresolve_path(self, temp_dir: Path) -> None:
        settings = Settings(temp_dir, {"tool": {"docc": {}}})
        absolute = temp_dir / "subdir" / "file.py"
        relative = settings.unresolve_path(absolute)

        assert not relative.is_absolute()
        assert relative == PurePath("subdir") / "file.py"


class TestSettingsFromFile:
    def test_from_file_finds_pyproject_toml(self, temp_dir: Path) -> None:
        pyproject = temp_dir / "pyproject.toml"
        pyproject.write_text('[tool.docc]\noutput = { path = "docs" }\n')

        settings = Settings.from_file(temp_dir)

        assert settings.output.path == Path("docs")

    def test_from_file_searches_parent_directories(
        self, temp_dir: Path
    ) -> None:
        pyproject = temp_dir / "pyproject.toml"
        pyproject.write_text('[tool.docc]\noutput = { path = "docs" }\n')

        subdir = temp_dir / "src" / "submodule"
        subdir.mkdir(parents=True)

        settings = Settings.from_file(subdir)

        assert settings.output.path == Path("docs")

    def test_from_file_respects_max_depth(self, temp_dir: Path) -> None:
        deep_path = temp_dir
        for i in range(MAX_DEPTH + 5):
            deep_path = deep_path / f"level{i}"
        deep_path.mkdir(parents=True)

        with pytest.raises(SettingsError, match="could not find"):
            Settings.from_file(deep_path)

    def test_from_file_not_found_raises(self, temp_dir: Path) -> None:
        with pytest.raises(SettingsError, match="could not find"):
            Settings.from_file(temp_dir)

    def test_from_file_with_complete_config(self, temp_dir: Path) -> None:
        config = """
[tool.docc]
context = ["custom.context"]
discovery = ["custom.discover"]
build = ["custom.build"]
transform = ["custom.transform"]

[tool.docc.output]
path = "output"

[tool.docc.plugins."custom.plugin"]
option = "value"
"""
        pyproject = temp_dir / "pyproject.toml"
        pyproject.write_text(config)

        settings = Settings.from_file(temp_dir)

        assert settings.context == ["custom.context"]
        assert settings.discovery == ["custom.discover"]
        assert settings.build == ["custom.build"]
        assert settings.transform == ["custom.transform"]
        assert settings.output.path == Path("output")


class TestSettingsConstants:
    def test_max_depth(self) -> None:
        assert MAX_DEPTH == 10

    def test_file_name(self) -> None:
        assert FILE_NAME == "pyproject.toml"
