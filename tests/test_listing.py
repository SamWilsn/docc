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

from pathlib import PurePath

from docc.plugins.listing import (
    Listing,
    ListingDiscover,
    ListingSource,
    _display_path,
    _hierarchy_path,
)
from docc.plugins.python.cst import PythonSource
from docc.settings import PluginSettings, Settings


def _python_source(
    rel: str, *, strip_root: bool = True, root: str = "src"
) -> PythonSource:
    relative = PurePath(rel)
    absolute = PurePath("/abs") / relative
    return PythonSource(
        root_path=PurePath("/abs") / root,
        relative_path=relative,
        absolute_path=absolute,
        strip_root=strip_root,
    )


def test_hierarchy_path_uses_index_dir_for_init() -> None:
    source = _python_source("src/pkg/__init__.py")
    assert _hierarchy_path(source) == PurePath("pkg")


def test_hierarchy_path_uses_output_for_module() -> None:
    source = _python_source("src/pkg/mod.py")
    assert _hierarchy_path(source) == PurePath("pkg/mod.py")


def test_hierarchy_path_uses_index_dir_for_listing_source() -> None:
    listing = ListingSource(PurePath("pkg"), PurePath("pkg/index"))
    assert _hierarchy_path(listing) == PurePath("pkg")


def test_display_path_keeps_init_filename() -> None:
    source = _python_source("src/pkg/__init__.py")
    assert _display_path(source) == PurePath("pkg/__init__.py")


def test_display_path_for_module_is_output_path() -> None:
    source = _python_source("src/pkg/mod.py")
    assert _display_path(source) == PurePath("pkg/mod.py")


def test_listing_groups_init_and_siblings_under_same_dir() -> None:
    listing = Listing()
    init_src = _python_source("src/pkg/__init__.py")
    mod_src = _python_source("src/pkg/mod.py")

    listing.add_source(init_src)
    listing.add_source(mod_src)

    # `mod.py` lives in the `pkg/` listing.
    assert listing.sources[PurePath("pkg")] == {init_src, mod_src}
    # `__init__.py` also appears in its parent's listing as the directory.
    assert listing.sources[PurePath(".")] == {init_src}


def test_listing_descendants_and_siblings() -> None:
    listing = Listing()
    init_src = _python_source("src/pkg/__init__.py")
    mod_src = _python_source("src/pkg/mod.py")
    top = ListingSource(PurePath("."), PurePath("index"))

    listing.add_source(init_src)
    listing.add_source(mod_src)
    listing.add_source(top)

    # Descendants of the package index are the package's contents.
    assert set(listing.descendants(init_src)) == {init_src, mod_src}
    # Siblings of `mod.py` include the `__init__.py` and itself.
    assert set(listing.siblings(mod_src)) == {init_src, mod_src}


def _empty_discover() -> ListingDiscover:
    settings = PluginSettings(
        Settings(PurePath("."), {}),  # type: ignore[arg-type]
        {},
    )
    return ListingDiscover(settings)


def test_listing_discover_creates_root_listing_only() -> None:
    init_src = _python_source("src/pkg/__init__.py")
    mod_src = _python_source("src/pkg/mod.py")

    listings = list(_empty_discover().discover(frozenset({init_src, mod_src})))

    # __init__.py supplies the `pkg/` listing, so only the root is new.
    assert {ls.output_path for ls in listings} == {PurePath("index")}


def test_listing_discover_creates_intermediate_listings() -> None:
    # No __init__.py at `pkg/`, so a synthetic listing is created for it.
    mod_src = _python_source("src/pkg/mod.py")

    listings = list(_empty_discover().discover(frozenset({mod_src})))

    assert {ls.output_path for ls in listings} == {
        PurePath("pkg/index"),
        PurePath("index"),
    }
