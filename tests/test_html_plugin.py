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

"""Tests for HTMLContext, HTMLDiscover, HTMLTransform, and HTMLRoot output."""

from io import StringIO
from pathlib import Path, PurePath
from typing import Optional

import pytest

from docc.context import Context
from docc.document import BlankNode, Document, ListNode
from docc.plugins.html import (
    HTML,
    HTMLContext,
    HTMLDiscover,
    HTMLRoot,
    HTMLTag,
    HTMLTransform,
    TextNode,
)
from docc.settings import PluginSettings, Settings, SettingsError
from docc.source import Source


class MockSource(Source):
    _output_path: PurePath

    def __init__(self, output_path: Optional[PurePath] = None) -> None:
        self._output_path = (
            output_path if output_path is not None else PurePath("test.py")
        )

    @property
    def relative_path(self) -> Optional[PurePath]:
        return self._output_path

    @property
    def output_path(self) -> PurePath:
        return self._output_path


@pytest.fixture
def basic_settings(tmp_path: Path) -> Settings:
    return Settings(tmp_path, {"tool": {"docc": {}}})


@pytest.fixture
def plugin_settings(basic_settings: Settings) -> PluginSettings:
    return basic_settings.for_plugin("docc.html")


# ---------------------------------------------------------------------------
# HTMLContext
# ---------------------------------------------------------------------------


def test_context_provides_html_type() -> None:
    assert HTMLContext.provides() == HTML


def test_context_default_values(
    plugin_settings: PluginSettings,
) -> None:
    ctx = HTMLContext(plugin_settings)
    html = ctx.provide()
    assert html.extra_css == []
    assert html.breadcrumbs is True


def test_context_with_extra_css(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path,
        {
            "tool": {
                "docc": {
                    "plugins": {
                        "docc.html.context": {
                            "extra_css": ["custom.css", "theme.css"]
                        }
                    }
                }
            }
        },
    )
    plugin_settings = settings.for_plugin("docc.html.context")
    ctx = HTMLContext(plugin_settings)
    html = ctx.provide()
    assert html.extra_css == ["custom.css", "theme.css"]


def test_context_invalid_extra_css_raises(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path,
        {
            "tool": {
                "docc": {
                    "plugins": {"docc.html.context": {"extra_css": [123]}}
                }
            }
        },
    )
    plugin_settings = settings.for_plugin("docc.html.context")
    with pytest.raises(SettingsError, match="extra_css"):
        HTMLContext(plugin_settings)


def test_context_breadcrumbs_false(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path,
        {
            "tool": {
                "docc": {
                    "plugins": {"docc.html.context": {"breadcrumbs": False}}
                }
            }
        },
    )
    plugin_settings = settings.for_plugin("docc.html.context")
    ctx = HTMLContext(plugin_settings)
    html = ctx.provide()
    assert html.breadcrumbs is False


def test_context_invalid_breadcrumbs_raises(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path,
        {
            "tool": {
                "docc": {
                    "plugins": {"docc.html.context": {"breadcrumbs": "yes"}}
                }
            }
        },
    )
    plugin_settings = settings.for_plugin("docc.html.context")
    with pytest.raises(SettingsError, match="breadcrumbs"):
        HTMLContext(plugin_settings)


def test_context_with_all_options(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path,
        {
            "tool": {
                "docc": {
                    "plugins": {
                        "docc.html.context": {
                            "extra_css": ["style1.css", "style2.css"],
                            "breadcrumbs": False,
                        }
                    }
                }
            }
        },
    )
    plugin_settings = settings.for_plugin("docc.html.context")

    ctx = HTMLContext(plugin_settings)
    html = ctx.provide()

    assert html.extra_css == ["style1.css", "style2.css"]
    assert html.breadcrumbs is False


# ---------------------------------------------------------------------------
# HTMLDiscover
# ---------------------------------------------------------------------------


def test_discover_yields_static_resources(
    plugin_settings: PluginSettings,
) -> None:
    discover = HTMLDiscover(plugin_settings)
    sources = list(discover.discover(frozenset()))

    assert len(sources) == 4

    output_paths = [str(s.output_path) for s in sources]
    assert any("chota" in p for p in output_paths)
    assert any("docc" in p for p in output_paths)
    assert any("fuse" in p for p in output_paths)
    assert any("search" in p for p in output_paths)


# ---------------------------------------------------------------------------
# HTMLTransform
# ---------------------------------------------------------------------------


def test_transform_skips_output_nodes(
    plugin_settings: PluginSettings,
) -> None:
    context_obj = Context({})
    root = HTMLRoot(context_obj)
    document = Document(root)
    context = Context({Document: document})

    transform = HTMLTransform(plugin_settings)
    transform.transform(context)

    assert context[Document].root is root


def test_transform_blank_node(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"tool": {"docc": {}}})
    plugin_settings = settings.for_plugin("docc.html.transform")

    blank = BlankNode()
    document = Document(blank)
    context = Context({Document: document})

    transform = HTMLTransform(plugin_settings)
    transform.transform(context)

    assert isinstance(document.root, HTMLRoot)
    assert document.root.extension == ".html"


def test_transform_list_node(tmp_path: Path) -> None:
    settings = Settings(tmp_path, {"tool": {"docc": {}}})
    plugin_settings = settings.for_plugin("docc.html.transform")

    node = ListNode([BlankNode(), BlankNode()])
    document = Document(node)
    context = Context({Document: document})

    transform = HTMLTransform(plugin_settings)
    transform.transform(context)

    assert isinstance(document.root, HTMLRoot)
    assert document.root.extension == ".html"


# ---------------------------------------------------------------------------
# HTMLRoot.output
# ---------------------------------------------------------------------------


def test_output_renders_html_document() -> None:
    source = MockSource(PurePath("docs/page.html"))
    context = Context({Source: source})
    root = HTMLRoot(context)

    div = HTMLTag("div", {"class": "content"})
    div.append(TextNode("Hello World"))
    root.append(div)

    dest = StringIO()
    root.output(context, dest)
    output = dest.getvalue()

    assert "<!DOCTYPE html>" in output
    assert "<html>" in output
    assert "</html>" in output
    assert "Hello World" in output
    assert "<div" in output
    assert "content" in output
    assert "<head>" in output
    assert "<body>" in output


def test_output_with_text_node_child() -> None:
    source = MockSource(PurePath("index.html"))
    context = Context({Source: source})
    root = HTMLRoot(context)

    root.append(TextNode("raw text"))

    dest = StringIO()
    root.output(context, dest)
    output = dest.getvalue()

    assert "raw text" in output
    assert "<!DOCTYPE html>" in output


def test_output_breadcrumbs_for_nested_path() -> None:
    source = MockSource(PurePath("a/b/page.html"))
    context = Context({Source: source})
    root = HTMLRoot(context)
    root.append(HTMLTag("p"))

    dest = StringIO()
    root.output(context, dest)
    output = dest.getvalue()

    assert "breadcrumbs" in output
    assert "page.html" in output


# ---------------------------------------------------------------------------
# HTMLRoot with HTML context
# ---------------------------------------------------------------------------


def test_root_with_html_context(tmp_path: Path) -> None:
    settings = Settings(
        tmp_path,
        {
            "tool": {
                "docc": {
                    "plugins": {
                        "docc.html.context": {
                            "extra_css": ["custom.css"],
                            "breadcrumbs": False,
                        }
                    }
                }
            }
        },
    )
    plugin_settings = settings.for_plugin("docc.html.context")
    html_ctx = HTMLContext(plugin_settings)
    html = html_ctx.provide()

    context = Context({HTML: html})
    root = HTMLRoot(context)

    assert root.extra_css == ["custom.css"]
    assert root.breadcrumbs is False


def test_root_without_html_context() -> None:
    context = Context({})
    root = HTMLRoot(context)

    assert root.extra_css == []
    assert root.breadcrumbs is True
