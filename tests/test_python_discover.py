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

from pathlib import Path, PurePath
from typing import Tuple

import pytest

from docc.plugins.python.cst import PythonDiscover, PythonSource
from docc.settings import PluginSettings, Settings


def _settings(root: Path, paths: Tuple[str, ...]) -> PluginSettings:
    return PluginSettings(
        Settings(
            root,
            {"tool": {"docc": {"plugins": {"docc.python.discover": {}}}}},
        ),
        {"paths": list(paths)},
    )


def _write_tree(root: Path, files: Tuple[str, ...]) -> None:
    for file in files:
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")


def test_wrapper_root_is_stripped(tmp_path: Path) -> None:
    _write_tree(tmp_path, ("src/pkg/__init__.py", "src/pkg/mod.py"))

    discover = PythonDiscover(_settings(tmp_path, ("src",)))
    sources = sorted(
        discover.discover(frozenset()), key=lambda s: s.absolute_path
    )

    assert [s.output_path for s in sources] == [
        PurePath("pkg/index"),
        PurePath("pkg/mod.py"),
    ]
    # relative_path is the on-disk location and is unaffected.
    assert [s.relative_path for s in sources] == [
        PurePath("src/pkg/__init__.py"),
        PurePath("src/pkg/mod.py"),
    ]


def test_package_root_is_kept(tmp_path: Path) -> None:
    _write_tree(tmp_path, ("pkg/__init__.py", "pkg/mod.py"))

    discover = PythonDiscover(_settings(tmp_path, ("pkg",)))
    sources = sorted(
        discover.discover(frozenset()), key=lambda s: s.absolute_path
    )

    assert [s.output_path for s in sources] == [
        PurePath("pkg/index"),
        PurePath("pkg/mod.py"),
    ]


def test_namespace_wrapper_is_stripped(tmp_path: Path) -> None:
    # No __init__.py at the wrapper root: it's still treated as a wrapper.
    _write_tree(tmp_path, ("src/pkg/__init__.py", "src/loose.py"))

    discover = PythonDiscover(_settings(tmp_path, ("src",)))
    sources = sorted(
        discover.discover(frozenset()), key=lambda s: s.absolute_path
    )

    assert [s.output_path for s in sources] == [
        PurePath("loose.py"),
        PurePath("pkg/index"),
    ]


def test_index_dir_follows_stripped_output(tmp_path: Path) -> None:
    _write_tree(tmp_path, ("src/pkg/__init__.py",))

    discover = PythonDiscover(_settings(tmp_path, ("src",)))
    (source,) = list(discover.discover(frozenset()))

    assert isinstance(source, PythonSource)
    assert source.index_dir == PurePath("pkg")


def test_mixed_roots(tmp_path: Path) -> None:
    _write_tree(
        tmp_path,
        (
            "src/pkg/__init__.py",
            "lib/__init__.py",
            "lib/util.py",
        ),
    )

    discover = PythonDiscover(_settings(tmp_path, ("src", "lib")))
    sources = {s.output_path for s in discover.discover(frozenset())}

    assert sources == {
        PurePath("pkg/index"),  # src/ stripped
        PurePath("lib/index"),  # lib kept (lib has __init__.py)
        PurePath("lib/util.py"),
    }


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("__init__.py", PurePath("pkg/index")),
        ("mod.py", PurePath("pkg/mod.py")),
    ],
)
def test_python_source_strip_root_param(
    tmp_path: Path, filename: str, expected: PurePath
) -> None:
    abs_path = tmp_path / "src" / "pkg" / filename
    source = PythonSource(
        root_path=PurePath(tmp_path / "src"),
        relative_path=PurePath("src/pkg") / filename,
        absolute_path=PurePath(abs_path),
        strip_root=True,
    )
    assert source.output_path == expected


def test_python_source_default_keeps_relative(tmp_path: Path) -> None:
    abs_path = tmp_path / "src" / "pkg" / "mod.py"
    source = PythonSource(
        root_path=PurePath(tmp_path / "src"),
        relative_path=PurePath("src/pkg/mod.py"),
        absolute_path=PurePath(abs_path),
    )
    assert source.output_path == PurePath("src/pkg/mod.py")
