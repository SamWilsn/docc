# Copyright (C) 2022-2023 Ethereum Foundation
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

import argparse
import logging
import sys
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
    parser = argparse.ArgumentParser(
        description="Python documentation generator."
    )

    parser.add_argument(
        "--output", help="The directory to write documentation to."
    )

    args = parser.parse_args()
    settings = Settings(Path.cwd())

    if args.output is None:
        output_root = settings.output.path

        if output_root is None:
            logging.critical(
                "Output path is required. "
                "Either define `tool.docc.output.path` in `pyproject.toml` "
                "or use `--output foo/bar` on the command line."
            )
            sys.exit(1)
    else:
        output_root = Path(args.output)

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

        rmtree(output_root, ignore_errors=True)

        for source, document in documents.items():
            extension = document.extension()

            if extension is None:
                logging.error(
                    "document from `%s` does not specify a file extension",
                    source.relative_path,
                )
                continue

            output_path = output_root / source.output_path
            output_path = Path(
                output_path.with_suffix(output_path.suffix + extension)
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as destination:
                document.output(destination)
