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
Python language support for docc.
"""

import dataclasses
from dataclasses import dataclass, fields
from typing import Iterable, Optional, Sequence

from ..document import BlankNode, Node


class PythonNode(Node):
    """
    Base implementation of Node operations for Python nodes.
    """

    @property
    def children(self) -> Iterable[Node]:
        """
        Child nodes belonging to this node.
        """
        for field in fields(self):
            value = getattr(self, field.name)

            if isinstance(value, Node):
                # Value is a single child.
                yield value
                continue

            # Assume value is a list of children instead.
            for item in value:
                assert isinstance(item, Node)
                yield item

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        for field in fields(self):
            value = getattr(self, field.name)
            if value == old:
                assert isinstance(new, field.type)
                setattr(self, field.name, new)
                continue

            try:
                iterator = enumerate(value)
            except TypeError:
                continue

            for index, item in iterator:
                if old == item:
                    value[index] = new

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        return self.__class__.__name__ + "(...)"


@dataclass(repr=False)
class Module(PythonNode):
    """
    A Python module.
    """

    name: Node = dataclasses.field(default_factory=BlankNode)
    docstring: Node = dataclasses.field(default_factory=BlankNode)
    members: Sequence[Node] = dataclasses.field(default_factory=list)


@dataclass(repr=False)
class Class(PythonNode):
    """
    A class declaration.
    """

    decorators: Sequence[Node] = dataclasses.field(default_factory=list)
    name: Node = dataclasses.field(default_factory=BlankNode)
    bases: Sequence[Node] = dataclasses.field(default_factory=list)
    metaclass: Node = dataclasses.field(default_factory=BlankNode)
    docstring: Node = dataclasses.field(default_factory=BlankNode)
    members: Sequence[Node] = dataclasses.field(default_factory=list)


@dataclass(repr=False)
class Function(PythonNode):
    """
    A function definition.
    """

    decorators: Sequence[Node] = dataclasses.field(default_factory=list)
    name: Node = dataclasses.field(default_factory=BlankNode)
    parameters: Sequence[Node] = dataclasses.field(default_factory=list)
    return_type: Node = dataclasses.field(default_factory=BlankNode)
    docstring: Node = dataclasses.field(default_factory=BlankNode)
    body: Node = dataclasses.field(default_factory=BlankNode)


@dataclass(repr=False)
class Type(PythonNode):
    """
    A type, usually used in a PEP 484 annotation.
    """

    name: Node = dataclasses.field(default_factory=BlankNode)
    generics: Node = dataclasses.field(default_factory=BlankNode)


@dataclass(repr=False)
class Generics(PythonNode):
    """
    The square brackets and list of types in a type annotation.
    """

    arguments: Sequence[Node] = dataclasses.field(default_factory=list)


@dataclass(repr=False)
class Parameter(PythonNode):
    """
    A parameter descriptor in a function definition.
    """

    name: Node = dataclasses.field(default_factory=BlankNode)
    type_annotation: Node = dataclasses.field(default_factory=BlankNode)
    default_value: Node = dataclasses.field(default_factory=BlankNode)


@dataclass(repr=False)
class Attribute(PythonNode):
    """
    An assignment.
    """

    names: Sequence[Node] = dataclasses.field(default_factory=list)
    body: Node = dataclasses.field(default_factory=BlankNode)
    docstring: Node = dataclasses.field(default_factory=BlankNode)


@dataclass
class Name(Node):
    """
    The name of a class, function, variable, or other Python member.
    """

    name: str
    full_name: Optional[str] = dataclasses.field(default=None)

    @property
    def children(self) -> Iterable[Node]:
        """
        Child nodes belonging to this node.
        """
        return tuple()

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        raise TypeError()


@dataclass
class Docstring(Node):
    """
    Node representing a documentation string.
    """

    text: str

    @property
    def children(self) -> Iterable[Node]:
        """
        Child nodes belonging to this node.
        """
        return tuple()

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        raise TypeError()
