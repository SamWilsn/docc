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

from unittest.mock import MagicMock

import pytest

from docc.build import Builder
from docc.discover import Discover
from docc.plugins.loader import Loader, PluginError
from docc.transform import Transform


class TestLoader:
    def test_init(self) -> None:
        loader = Loader()
        assert isinstance(loader.entry_points, dict)
        assert len(loader.entry_points) > 0, "Entry points should be populated"
        assert "docc.python.discover" in loader.entry_points

    def test_entry_points_populated(self) -> None:
        loader = Loader()
        assert len(loader.entry_points) > 0
        assert "docc.python.discover" in loader.entry_points

    def test_load_discover_plugin(self) -> None:
        loader = Loader()
        cls = loader.load(Discover, "docc.python.discover")
        assert isinstance(cls, type)
        assert issubclass(cls, Discover)

    def test_load_builder_plugin(self) -> None:
        loader = Loader()
        cls = loader.load(Builder, "docc.python.build")
        assert isinstance(cls, type)
        assert issubclass(cls, Builder)

    def test_load_transform_plugin(self) -> None:
        loader = Loader()
        cls = loader.load(Transform, "docc.python.transform")
        assert isinstance(cls, type)
        assert issubclass(cls, Transform)

    def test_load_nonexistent_plugin_raises(self) -> None:
        loader = Loader()
        with pytest.raises(KeyError):
            loader.load(Discover, "nonexistent.plugin")

    def test_load_abstract_class_raises_plugin_error(self) -> None:
        loader = Loader()
        # Inject a fake entry point that loads an abstract class
        mock_ep = MagicMock()
        mock_ep.load.return_value = Discover
        loader.entry_points["fake.abstract"] = mock_ep

        with pytest.raises(PluginError, match="is abstract"):
            loader.load(Discover, "fake.abstract")

    def test_load_wrong_subclass_raises_plugin_error(self) -> None:
        loader = Loader()
        # Inject a fake entry point that loads a class not subclassing the base

        class NotADiscover:
            pass

        mock_ep = MagicMock()
        mock_ep.load.return_value = NotADiscover
        loader.entry_points["fake.wrong_type"] = mock_ep

        with pytest.raises(PluginError, match="is not a subclass of"):
            loader.load(Discover, "fake.wrong_type")


class TestPluginError:
    def test_create_error(self) -> None:
        error = PluginError("test error message")
        assert "test error message" in str(error)

    def test_error_inheritance(self) -> None:
        error = PluginError("test")
        assert isinstance(error, Exception)


class TestLoaderMultiplePlugins:
    def test_load_multiple_transforms(self) -> None:
        loader = Loader()

        transforms = [
            "docc.python.transform",
            "docc.mistletoe.transform",
            "docc.mistletoe.reference",
            "docc.html.transform",
        ]

        for name in transforms:
            cls = loader.load(Transform, name)
            assert isinstance(cls, type)
            assert issubclass(cls, Transform)

    def test_load_multiple_discovers(self) -> None:
        loader = Loader()

        discovers = [
            "docc.python.discover",
            "docc.html.discover",
        ]

        for name in discovers:
            cls = loader.load(Discover, name)
            assert isinstance(cls, type)
            assert issubclass(cls, Discover)
