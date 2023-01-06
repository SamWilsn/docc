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
Command line interface to docc.
"""

import logging
from contextlib import ExitStack
from pathlib import Path
from shutil import rmtree
from typing import Dict, Set

from . import build, discover, transform
from .document import Document
from .references import Index
from .settings import Settings
from .source import Source


def main() -> None:
    """
    Entry-point for the command line tool.
    """
    settings = Settings(Path.cwd())

    discover_plugins = discover.load(settings)
    transform_plugins = transform.load(settings)

    known: Set[Source] = set()

    for name, instance in discover_plugins:
        found = instance.discover(frozenset(known))
        for item in found:
            if item.relative_path is None:
                logging.info("[%s] found source without a path", name)
            else:
                logging.info("[%s] found source: %s", name, item.relative_path)
            known.add(item)

    all_sources = list(known)
    index = Index()

    with ExitStack() as exit_stack:
        build_plugins = [
            (n, exit_stack.enter_context(c)) for (n, c) in build.load(settings)
        ]

        documents: Dict[Source, Document] = {}
        for name, build_plugin in build_plugins:
            before = len(documents)
            build_plugin.build(index, all_sources, known, documents)
            after = len(documents)
            logging.info("[%s] built %s documents", name, after - before)

        for _name, transform_plugin in transform_plugins:
            for document in documents.values():
                transform_plugin.transform(document)

        rmtree(settings.output.path, ignore_errors=True)

        for source, document in documents.items():
            output_path = settings.output.path / source.output_path
            output_path = Path(
                output_path.with_suffix(
                    output_path.suffix + settings.output.extension
                )
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as destination:
                document.output(destination)
