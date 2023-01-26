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
Definitions and references for interlinking documents.
"""

import dataclasses
from dataclasses import dataclass
from typing import Optional, Tuple

from docc.document import BlankNode, Document, Node, Visit, Visitor
from docc.settings import PluginSettings
from docc.transform import Transform


@dataclass(repr=False)
class Base(Node):
    """
    Node implementation for Definition and Reference.
    """

    identifier: str
    child: Node = dataclasses.field(default_factory=BlankNode)

    @property
    def children(self) -> Tuple[Node]:
        """
        Return the children of this node.
        """
        return (self.child,)

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace a child with a different node.
        """
        if old == self.child:
            self.child = new


@dataclass
class Definition(Base):
    """
    A target for a Reference.
    """

    specifier: Optional[int] = dataclasses.field(default=None)


@dataclass
class Reference(Base):
    """
    A link to a Definition.
    """


class IndexTransform(Transform):
    """
    Collect Definition nodes and insert them into the index.
    """

    def __init__(self, config: PluginSettings) -> None:
        super().__init__(config)

    def transform(self, document: Document) -> None:
        """
        Apply the transformation to the given document.
        """
        document.root.visit(_TransformVisitor(document))


class _TransformVisitor(Visitor):
    document: Document

    def __init__(self, document: Document) -> None:
        self.document = document

    def enter(self, node: Node) -> Visit:
        if isinstance(node, Definition):
            definition = self.document.index.define(
                self.document.source, node.identifier
            )
            assert node.specifier is None
            node.specifier = definition.specifier
        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        pass
