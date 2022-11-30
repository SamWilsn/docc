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
Builders convert Sources into Documents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Sequence, Set, Tuple

from .document import Document
from .plugins.loader import Loader
from .settings import PluginSettings, Settings
from .source import Source


class Builder(ABC):
    """
    Consumes unprocessed Sources and creates Documents.
    """

    @abstractmethod
    def __init__(self, config: PluginSettings) -> None:
        """
        Create a Builder with the given configuration.
        """
        pass

    @abstractmethod
    def build(
        self, unprocessed: Set[Source], processed: Dict[Source, Document]
    ) -> None:
        """
        Consume unprocessed Sources and insert their Documents into processed.
        """
        pass


def load(settings: Settings) -> Sequence[Tuple[str, Builder]]:
    """
    Load the builder plugins as requested in settings.
    """
    loader = Loader()

    sources = []

    for name in settings.build:
        class_ = loader.load(Builder, name)
        plugin_settings = settings.for_plugin(name)
        sources.append((name, class_(plugin_settings)))

    return sources
