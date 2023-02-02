# Copyright (C) 2022 Ethereum Foundation
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

"""
Find and load configuration.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from itertools import chain, islice
from pathlib import Path, PurePath
from typing import Any, Dict, Iterator, Sequence

import tomli

MAX_DEPTH = 10
FILE_NAME = "pyproject.toml"


class SettingsError(Exception):
    """
    An error encountered while attempting to load the settings.
    """


class PluginSettings(Mapping):
    """
    Stores settings for individual plugins.
    """

    _settings: "Settings"
    _store: Dict[str, Any]

    def __init__(self, settings: "Settings", store: Dict[str, Any]) -> None:
        """
        Create an instance with the given backing store.
        """
        self._settings = settings

        # Copy the dict since we're a Mapping and not a MappingView.
        self._store = dict(store)

    def __len__(self) -> int:
        """
        Return len(self).
        """
        return len(self._store)

    def __iter__(self) -> Iterator[str]:
        """
        Implement iter(self).
        """
        return iter(self._store)

    def __getitem__(self, k: str) -> object:
        """
        Return the item with the given key.
        """
        return self._store[k]

    def unresolve_path(self, path: PurePath) -> PurePath:
        """
        Convert an absolute path to a path relative to the settings file.
        """
        return self._settings.unresolve_path(path)

    def resolve_path(self, path: PurePath) -> Path:
        """
        Convert the given path to an absolute path relative to the settings
        file.
        """
        return self._settings.resolve_path(path)


@dataclass
class Output:
    """
    Settings for the output of the documentation process.
    """

    path: Path
    extension: str


class Settings:
    """
    Handles loading settings for generating documentation.
    """

    _settings: Dict[str, Any]
    _root: Path
    output: Output

    def __init__(self, path: Path) -> None:
        """
        Load the settings in the nearest configuration file to path.
        """
        settings_bytes = None

        search_directories = islice(chain([path], path.parents), MAX_DEPTH)

        for current_directory in search_directories:
            settings_file = current_directory / FILE_NAME
            try:
                settings_bytes = settings_file.read_bytes()
                self._root = settings_file.parent
                break
            except FileNotFoundError:
                pass

        if settings_bytes is None:
            raise SettingsError(
                f"could not find {FILE_NAME} (max depth: {MAX_DEPTH})"
            )

        settings_toml = tomli.load(BytesIO(settings_bytes))

        try:
            self._settings = settings_toml["tool"]["docc"]
        except KeyError:
            # TODO: Come up with some defaults.
            self._settings = {}

        self.output = Output(
            path=Path(self._settings["output"]["path"]),
            extension=self._settings["output"]["extension"],
        )

    def for_plugin(self, name: str) -> PluginSettings:
        """
        Retrieve the settings for the given plugin.
        """
        try:
            settings = self._settings["plugins"][name]
        except KeyError:
            settings = {}

        assert isinstance(settings, dict)

        for k in settings.keys():
            assert isinstance(k, str)

        return PluginSettings(self, settings)

    @property
    def discovery(self) -> Sequence[str]:
        """
        Retrieve a list of enabled discovery plugins.
        """
        try:
            discovery = self._settings["discovery"]
        except KeyError:
            discovery = []

        assert isinstance(discovery, list)

        for item in discovery:
            assert isinstance(item, str)

        return discovery

    @property
    def build(self) -> Sequence[str]:
        """
        Retrieve a list of enabled build plugins.
        """
        try:
            build = self._settings["build"]
        except KeyError:
            build = []

        assert isinstance(build, list)

        for item in build:
            assert isinstance(item, str)

        return build

    @property
    def transform(self) -> Sequence[str]:
        """
        Retrieve a list of enabled transform plugins.
        """
        try:
            transform = self._settings["transform"]
        except KeyError:
            transform = []

        assert isinstance(transform, list)

        for item in transform:
            assert isinstance(item, str)

        return transform

    def resolve_path(self, path: PurePath) -> Path:
        """
        Convert the given path to an absolute path relative to the settings
        file.
        """
        joined = self._root / path
        resolved = joined.resolve()

        # relative_to raises an error if the argument isn't a super-path.
        resolved.relative_to(self._root.resolve())

        return resolved

    def unresolve_path(self, path: PurePath) -> PurePath:
        """
        Convert an absolute path to a path relative to the settings file.
        """
        return path.relative_to(self._root)