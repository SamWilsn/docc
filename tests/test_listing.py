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

from pathlib import Path, PurePath
from typing import Dict, List, Optional, Set

import pytest

from docc.context import Context
from docc.document import BlankNode, Document
from docc.plugins.html import HTMLTag
from docc.plugins.listing import (
    Listable,
    ListingBuilder,
    ListingDiscover,
    ListingNode,
    ListingSource,
    render_html,
)
from docc.settings import PluginSettings, Settings
from docc.source import Source


@pytest.fixture
def plugin_settings(tmp_path: Path) -> PluginSettings:
    settings = Settings(tmp_path, {"tool": {"docc": {}}})
    return settings.for_plugin("docc.listing.discover")


class MockSource(Source):
    _output_path: PurePath

    def __init__(
        self,
        relative_path: Optional[PurePath] = None,
        output_path: Optional[PurePath] = None,
    ) -> None:
        self._relative_path = relative_path
        self._output_path = output_path or relative_path or PurePath("output")

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._relative_path

    @property
    def output_path(self) -> PurePath:
        return self._output_path


class ListableSource(Source, Listable):
    def __init__(
        self,
        relative_path: Optional[PurePath] = None,
        show: bool = True,
    ) -> None:
        self._relative_path = relative_path
        self._show = show

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._relative_path

    @property
    def output_path(self) -> PurePath:
        return self._relative_path or PurePath("output")

    @property
    def show_in_listing(self) -> bool:
        return self._show


class TestListingSource:
    def test_init(self) -> None:
        relative = PurePath("src")
        output = PurePath("src/index")
        sources: Set[Source] = set()

        source = ListingSource(relative, output, sources)

        assert source.relative_path == relative
        assert source.output_path == output
        assert source.sources is sources

    def test_output_path_property(self) -> None:
        source = ListingSource(
            PurePath("docs"),
            PurePath("docs/index"),
            set(),
        )
        assert source.output_path == PurePath("docs/index")

    def test_relative_path_property(self) -> None:
        source = ListingSource(
            PurePath("api"),
            PurePath("api/index"),
            set(),
        )
        assert source.relative_path == PurePath("api")


class TestListingNode:
    def test_init(self) -> None:
        sources: Set[Source] = {MockSource(PurePath("test.py"))}
        node = ListingNode(sources)
        assert node.sources is sources

    def test_children_empty(self) -> None:
        node = ListingNode(set())
        assert node.children == ()

    def test_replace_child_raises(self) -> None:
        node = ListingNode(set())
        with pytest.raises(TypeError):
            node.replace_child(BlankNode(), BlankNode())


class TestListingBuilder:
    def test_build_processes_listing_sources(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = ListingSource(PurePath("docs"), PurePath("docs/index"), set())
        unprocessed: Set[Source] = {source}
        processed: Dict[Source, Document] = {}

        builder = ListingBuilder(plugin_settings)
        builder.build(unprocessed, processed)

        assert len(unprocessed) == 0
        assert source in processed
        assert isinstance(processed[source].root, ListingNode)

    def test_build_ignores_non_listing_sources(
        self, plugin_settings: PluginSettings
    ) -> None:
        source = MockSource(PurePath("test.py"))
        unprocessed: Set[Source] = {source}
        processed: Dict[Source, Document] = {}

        builder = ListingBuilder(plugin_settings)
        builder.build(unprocessed, processed)

        assert source in unprocessed
        assert len(processed) == 0


class TestListingDiscover:
    def test_discover_empty_known(
        self, plugin_settings: PluginSettings
    ) -> None:
        discover = ListingDiscover(plugin_settings)
        sources = list(discover.discover(frozenset()))
        assert sources == []

    def test_discover_creates_listing_for_directory(
        self, plugin_settings: PluginSettings
    ) -> None:
        known_source = MockSource(PurePath("src/module.py"))
        known: frozenset[Source] = frozenset([known_source])

        discover = ListingDiscover(plugin_settings)
        sources = list(discover.discover(known))

        # "src/module.py" has parents "src" and ".", so 2 listings are created
        assert len(sources) == 2, "Should create listings for 'src' and root"
        assert all(isinstance(s, ListingSource) for s in sources)
        # Verify a listing was created for the 'src' directory
        src_listing = next(
            (s for s in sources if s.relative_path == PurePath("src")), None
        )
        assert (
            src_listing is not None
        ), "Should create listing for 'src' directory"

    def test_discover_creates_nested_listings(
        self, plugin_settings: PluginSettings
    ) -> None:
        known_source = MockSource(PurePath("a/b/c/module.py"))
        known: frozenset[Source] = frozenset([known_source])

        discover = ListingDiscover(plugin_settings)
        sources = list(discover.discover(known))

        # Should create listings for a, a/b, a/b/c (at least 3 levels)
        assert (
            len(sources) >= 3
        ), "Should create listings for each directory level"
        # Verify specific directory listings were created
        paths = {s.relative_path for s in sources}
        assert PurePath("a") in paths, "Should create listing for 'a'"
        assert PurePath("a/b") in paths, "Should create listing for 'a/b'"
        assert PurePath("a/b/c") in paths, "Should create listing for 'a/b/c'"

    def test_discover_skips_listable_hidden(
        self, plugin_settings: PluginSettings
    ) -> None:
        hidden_source = ListableSource(PurePath("hidden/file.py"), show=False)
        visible_source = MockSource(PurePath("visible/file.py"))
        known: frozenset[Source] = frozenset([hidden_source, visible_source])

        discover = ListingDiscover(plugin_settings)
        sources = list(discover.discover(known))

        listing_paths = [str(s.relative_path) for s in sources]
        assert "hidden" not in listing_paths
        assert any("visible" in p for p in listing_paths)

    def test_discover_includes_listable_shown(
        self, plugin_settings: PluginSettings
    ) -> None:
        shown_source = ListableSource(PurePath("shown/file.py"), show=True)
        known: frozenset[Source] = frozenset([shown_source])

        discover = ListingDiscover(plugin_settings)
        sources = list(discover.discover(known))

        # "shown/file.py" has parents "shown" and ".", so 2 listings
        assert len(sources) == 2, "Should create listings for 'shown' and root"
        # Verify a listing was created for the 'shown' directory
        shown_listing = next(
            (s for s in sources if s.relative_path == PurePath("shown")), None
        )
        assert (
            shown_listing is not None
        ), "Should create listing for 'shown' directory"
        # Verify the shown source is included in the listing
        assert (
            shown_source in shown_listing.sources
        ), "Shown source should be in listing"

    def test_discover_skips_source_without_path(
        self, plugin_settings: PluginSettings
    ) -> None:
        no_path_source = MockSource(
            relative_path=None, output_path=PurePath("out")
        )
        known: frozenset[Source] = frozenset([no_path_source])

        discover = ListingDiscover(plugin_settings)
        sources = list(discover.discover(known))

        assert sources == []

    def test_discover_adds_sources_to_listing(
        self, plugin_settings: PluginSettings
    ) -> None:
        first_source = MockSource(PurePath("dir/file1.py"))
        second_source = MockSource(PurePath("dir/file2.py"))
        known: frozenset[Source] = frozenset([first_source, second_source])

        discover = ListingDiscover(plugin_settings)
        sources = list(discover.discover(known))

        dir_listing = next(
            (s for s in sources if s.relative_path == PurePath("dir")), None
        )
        assert dir_listing is not None
        assert first_source in dir_listing.sources
        assert second_source in dir_listing.sources

    def test_discover_listable_with_no_relative_path_falls_back_to_output_path(
        self, plugin_settings: PluginSettings
    ) -> None:
        """
        Test that Listable sources with relative_path=None fall
        back to output_path for directory listing discovery.
        """
        source = ListableSource(relative_path=None, show=True)
        known: frozenset[Source] = frozenset([source])

        discover = ListingDiscover(plugin_settings)
        sources = list(discover.discover(known))

        # The source has output_path="output", so a listing
        # should be created for its parent directory (".")
        # output_path="output" has one parent ".", so 1 listing is created
        assert (
            len(sources) == 1
        ), "Should create listing when Listable falls back to output_path"


def test_render_html_produces_links() -> None:
    """Test that render_html produces correct HTML with relative links."""
    entry_source = MockSource(
        relative_path=PurePath("docs/api/module.py"),
        output_path=PurePath("docs/api/module"),
    )
    listing_sources: Set[Source] = {entry_source}
    listing_source = ListingSource(
        PurePath("docs"),
        PurePath("docs/index"),
        listing_sources,
    )
    node = ListingNode(listing_sources)

    context = Context({Source: listing_source})
    parent = HTMLTag("div")

    render_html(context, parent, node)

    # Parent should now contain children from the template
    children = list(parent.children)
    assert len(children) > 0, "render_html should append children to parent"

    # Walk the HTML tree to find <a> tags with href attributes
    def find_tags(node: object, tag_name: str) -> List[HTMLTag]:
        results: List[HTMLTag] = []
        if isinstance(node, HTMLTag):
            if node.tag_name == tag_name:
                results.append(node)
            for child in node.children:
                results.extend(find_tags(child, tag_name))
        return results

    links = find_tags(parent, "a")
    assert len(links) >= 1, "Should produce at least one link"
    # The link should have an href attribute ending with .html
    href = links[0].attributes.get("href") or ""
    assert href.endswith(
        ".html"
    ), f"Link href should end with .html, got: {href}"
