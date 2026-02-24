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

import sys
from unittest.mock import MagicMock, patch

import pytest

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint
else:
    from importlib.metadata import EntryPoint

import docc.plugins.loader as loader_module
from docc.build import Builder
from docc.discover import Discover
from docc.plugins.loader import Loader, PluginError, _get_plugin_entry_points
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


class TestLoaderCacheBehavioral:
    """Behavioral tests: two Loader instances share cached entry points."""

    def test_two_loaders_share_identical_entry_points(self) -> None:
        """Two Loader instances must have identical entry_points dicts."""
        loader_a = Loader()
        loader_b = Loader()
        assert loader_a.entry_points is loader_b.entry_points

    def test_both_loaders_can_load_known_plugin(self) -> None:
        """Both Loader instances can load a known plugin after caching."""
        loader_a = Loader()
        loader_b = Loader()
        cls_a = loader_a.load(Discover, "docc.python.discover")
        cls_b = loader_b.load(Discover, "docc.python.discover")
        assert cls_a is cls_b
        assert callable(cls_a)


class TestLoaderCacheCallCount:
    """Spy tests: entry_points() called once across instances."""

    def setup_method(self) -> None:  # noqa: SC200
        """Reset the module-level cache before each test."""
        loader_module._PLUGIN_ENTRY_POINTS = None

    def teardown_method(self) -> None:  # noqa: SC200
        """Reset the module-level cache after each test."""
        loader_module._PLUGIN_ENTRY_POINTS = None

    def test_entry_points_called_once_for_multiple_loaders(
        self,
    ) -> None:
        """Multiple Loader instances call entry_points() once."""
        with patch(
            "docc.plugins.loader.entry_points",
            wraps=loader_module.entry_points,
        ) as mock_ep:
            Loader()
            Loader()
            Loader()
            mock_ep.assert_called_once_with(group="docc.plugins")

    def test_get_plugin_entry_points_calls_once(self) -> None:
        """Repeated _get_plugin_entry_points() calls entry_points() once."""
        with patch(
            "docc.plugins.loader.entry_points",
            wraps=loader_module.entry_points,
        ) as mock_ep:
            _get_plugin_entry_points()
            _get_plugin_entry_points()
            mock_ep.assert_called_once_with(group="docc.plugins")


class TestLoaderCacheKeying:
    """Cache-keying tests: the cached dict has correct structure."""

    def test_cache_maps_names_to_entry_point_objects(self) -> None:
        """The cached dict maps string names to EntryPoint instances."""
        loader = Loader()
        for name, ep in loader.entry_points.items():
            assert isinstance(name, str), "Keys must be strings."
            assert isinstance(
                ep, EntryPoint
            ), "Values must be EntryPoint instances."

    def test_cache_contains_known_entry_points(self) -> None:
        """The cached dict includes known plugin entry point names."""
        loader = Loader()
        expected_names = [
            "docc.python.discover",
            "docc.python.build",
            "docc.python.transform",
        ]
        for name in expected_names:
            assert (
                name in loader.entry_points
            ), f"Expected entry point '{name}' not found in cache."
