# Copyright (C) 2023,2026 Ethereum Foundation
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
Plugin that renders directory listings.
"""

from collections import defaultdict
from collections.abc import Iterable
from os.path import commonpath
from pathlib import PurePath
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Final,
    FrozenSet,
    Iterator,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

if TYPE_CHECKING:
    from _typeshed import SupportsDunderGT, SupportsDunderLT

from jinja2 import Environment, PackageLoader, select_autoescape

from docc.build import Builder
from docc.context import Context, Provider
from docc.discover import Discover, T
from docc.document import Document, Node
from docc.plugins import html
from docc.settings import PluginSettings
from docc.source import Source
from docc.transform import Transform


def _hierarchy_path(source: Source) -> PurePath:
    """
    Return the navigation-tree position of `source`.

    Index sources (synthetic listings, ``__init__.py``) occupy the
    directory they index; everything else occupies its `output_path`.
    """
    return Listable._index_dir(source) or source.output_path


def _display_path(source: Source) -> PurePath:
    """
    Return the path used to display `source` as a file entry.

    For a file-backed index like ``__init__.py``, this rejoins the
    original filename to its URL-relative directory so any wrapper
    prefix stripped from `output_path` is not shown.
    """
    index_dir = Listable._index_dir(source)
    if index_dir and source.relative_path:
        return index_dir / source.relative_path.name
    return source.output_path


class Listable:
    """
    Mixin to change visibility of a Source in a directory listing.
    """

    @staticmethod
    def _sorting_key(
        thing: object,
    ) -> Union["SupportsDunderGT[Any]", "SupportsDunderLT[Any]"]:
        if isinstance(thing, Listable):
            return thing.listing_order_key()
        elif isinstance(thing, Source):
            return (
                Listable._index_dir(thing) is None,
                _hierarchy_path(thing),
                None,
            )
        return (True, None, thing)

    @staticmethod
    def _show_source(source: Source) -> bool:
        if isinstance(source, Listable):
            if not source.show_in_listing:
                return False
        elif not source.relative_path:
            return False

        return True

    @staticmethod
    def _index_dir(source: Source) -> Optional[PurePath]:
        if isinstance(source, Listable):
            return source.index_dir
        return None

    @property
    def index_dir(self) -> Optional[PurePath]:
        """
        For index sources, the directory the source indexes. For other sources,
        `None`.

        For example, for an output path of `./foo/index`, this should return
        `./foo`.
        """
        return None

    @property
    def show_in_listing(self) -> bool:
        """
        `True` if this `Source` should be shown in directory listings.
        """
        return True

    def listing_order_key(
        self,
    ) -> Union["SupportsDunderGT[Any]", "SupportsDunderLT[Any]"]:
        """
        Key to use when sorting instances while rendering.
        """
        if isinstance(self, Source):
            return (self.index_dir is None, _hierarchy_path(self), None)
        return (True, None, self)


class ListingDiscover(Discover):
    """
    Creates listing sources for each directory.
    """

    def __init__(self, config: PluginSettings) -> None:
        pass

    def _index_path(self, parent: PurePath) -> PurePath:
        return parent / "index"

    def _listing_source(
        self, source: Source, parent: PurePath
    ) -> "ListingSource":
        return ListingSource(parent, self._index_path(parent))

    def discover(self, known: FrozenSet[T]) -> Iterator["ListingSource"]:
        """
        Find sources.
        """
        output_paths = {s.output_path: s for s in known}

        listings = {}

        for source in known:
            if not Listable._show_source(source):
                continue

            for parent in _hierarchy_path(source).parents:
                try:
                    listing = listings[parent]
                except KeyError:
                    index_path = self._index_path(parent)
                    try:
                        listing = output_paths[index_path]
                    except KeyError:
                        listing = self._listing_source(source, parent)
                        listings[parent] = listing
                        yield listing

                source = listing


class Listing:
    """
    Tracks listable [`Source`]s.

    [`Source`]: ref:docc.source.Source
    """

    sources: Final[Dict[PurePath, Set[Source]]]

    def __init__(self) -> None:
        self.sources = defaultdict(set)

    def add_source(self, source: Source) -> None:
        """
        Register a source.
        """
        hierarchy = _hierarchy_path(source)
        self.sources[hierarchy.parent].add(source)
        if Listable._index_dir(source) is not None:
            # Index sources also appear as the index of their own directory.
            self.sources[hierarchy].add(source)

    def descendants(self, source: Source) -> Iterable[Source]:
        """
        All children of the given source.
        """
        return self.sources[_hierarchy_path(source)]

    def siblings(self, source: Source) -> Iterable[Source]:
        """
        All sources with the same parent as the given source.

        An index source like ``__init__.py`` is treated as a member of
        the directory it indexes, so its siblings are that directory's
        entries rather than entries one level higher in the tree.
        """
        if Listable._index_dir(source) is not None:
            return self.descendants(source)
        return self.sources[_hierarchy_path(source).parent]


class ListingContext(Provider[Listing]):
    """
    Injects a [`Listing`] instance into the [`Context`].

    [`Listing`]: ref:docc.plugins.listing.Listing
    [`Context`]: ref:docc.context.Context
    """

    listing: Listing

    def __init__(self, config: PluginSettings) -> None:
        super().__init__(config)
        self.listing = Listing()

    @classmethod
    def provides(class_) -> Type[Listing]:
        """
        Return the type used as a key in the [`Context`].

        [`Context`]: ref:docc.context.Context
        """
        return Listing

    def provide(self) -> Listing:
        """
        Return the object to add to the [`Context`].

        [`Context`]: ref:docc.context.Context
        """
        return self.listing


class ListingSource(Source, Listable):
    """
    A synthetic source that describes the contents of a directory.
    """

    _relative_path: Final[PurePath]
    _output_path: Final[PurePath]

    def __init__(
        self,
        relative_path: PurePath,
        output_path: PurePath,
    ) -> None:
        self._relative_path = relative_path
        self._output_path = output_path

    @property
    def output_path(self) -> PurePath:
        """
        Where to write the output from this Source relative to the output path.
        """
        return self._output_path

    @property
    def relative_path(self) -> PurePath:
        """
        Path to the Source (if one exists) relative to the project root.
        """
        return self._relative_path

    @property
    def index_dir(self) -> PurePath:
        """
        For index sources, the directory the source indexes. For other sources,
        `None`.

        For example, for an output path of `./foo/index`, this should return
        `./foo`.
        """
        return self.output_path.parent


class ListingBuilder(Builder):
    """
    Converts ListingSource instances into Documents.
    """

    def __init__(self, config: PluginSettings) -> None:
        """
        Create a Builder with the given configuration.
        """

    def build(
        self,
        unprocessed: Set[Source],
        processed: Dict[Source, Document],
    ) -> None:
        """
        Consume unprocessed Sources and insert their Documents into processed.
        """
        to_process = set(
            x for x in unprocessed if isinstance(x, ListingSource)
        )
        unprocessed -= to_process

        for source in to_process:
            processed[source] = Document(ListingNode(leaf=False))


class ListingNode(Node):
    """
    A node representing a directory listing.
    """

    leaf: bool

    def __init__(self, leaf: bool) -> None:
        self.leaf = leaf

    @property
    def children(self) -> Tuple[()]:
        """
        Child nodes belonging to this node.
        """
        return ()

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        raise TypeError()


class ListingTransform(Transform):
    """
    Collect [`Source`]s and insert them into the [`Listing`] context.

    [`Source`]: ref:docc.source.Source
    [`Listing`]: ref:docc.plugins.listing.Listing
    """

    def __init__(self, config: PluginSettings) -> None:
        pass

    def transform(self, context: Context) -> None:
        """
        Apply the transformation to the given document.
        """
        source = context[Source]
        if not Listable._show_source(source):
            return

        listing = context[Listing]
        listing.add_source(source)


def render_html(
    context: object,
    parent: object,
    node: object,
) -> html.RenderResult:
    """
    Render a ListingNode as HTML.
    """
    assert isinstance(context, Context)
    assert isinstance(parent, (html.HTMLRoot, html.HTMLTag))
    assert isinstance(node, ListingNode)

    if node.leaf:
        sources = context[Listing].siblings(context[Source])
    else:
        sources = context[Listing].descendants(context[Source])

    sources = sorted(sources, key=Listable._sorting_key)

    output_path = context[Source].output_path
    output_parent = output_path.parent

    entries = []

    for source in sources:
        entry_path = source.output_path

        if output_path == entry_path:
            relative_path = ""
        else:
            common_path = commonpath((output_path, entry_path))

            parents = len(output_path.relative_to(common_path).parents) - 1

            relative_path = (
                str(
                    PurePath(*[".."] * parents)
                    / entry_path.relative_to(common_path)
                )
                + ".html"
            )  # TODO: Don't hardcode extension.

        active = source is context[Source]
        entry_index = Listable._index_dir(source)
        relative = source.relative_path or source.output_path

        if entry_index is None or (
            entry_index == output_parent and relative != entry_index
        ):
            # Regular source, or an index source (like `__init__.py`) appearing
            # in its own listing. Show as `<name>`.
            display = _display_path(source)
            path = display.name if node.leaf else str(display)
        else:
            # Synthetic listing, or a file-based index (like `__init__.py`)
            # appearing in its parent directory's listing. Show as `<dir>/`.
            path = entry_index.name if node.leaf else str(entry_index)
            path = path + "/"

        entries.append((path, relative_path, active))

    env = Environment(
        loader=PackageLoader("docc.plugins.listing"),
        autoescape=select_autoescape(),
    )
    template = env.get_template("listing.html")
    parser = html.HTMLParser(context)
    parser.feed(template.render(context=context, entries=entries))
    for child in parser.root._children:
        parent.append(child)
    return None
