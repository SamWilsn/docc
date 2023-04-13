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
Python language support for docc.
"""

import dataclasses
from dataclasses import dataclass, fields
from typing import Iterable, Literal, Optional, Sequence, Union

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

            if field.type == Node:
                # Value is a single child.
                if not isinstance(value, Node):
                    raise TypeError("child not Node")
                yield value
            elif field.type == Sequence[Node]:
                # Value is a list of children.
                for item in value:
                    if not isinstance(item, Node):
                        raise TypeError("child not Node")
                    yield item
            else:
                # Not a child, so just ignore it.
                pass

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

    asynchronous: bool
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
class List(PythonNode):
    """
    Square brackets wrapping a list of elements, usually separated by commas.
    """

    elements: Sequence[Node] = dataclasses.field(default_factory=list)


@dataclass(repr=False)
class Tuple(PythonNode):
    """
    Parentheses wrapping a list of elements, usually separated by commas.
    """

    elements: Sequence[Node] = dataclasses.field(default_factory=list)


@dataclass(repr=False)
class Parameter(PythonNode):
    """
    A parameter descriptor in a function definition.
    """

    star: Optional[Union[Literal["*"], Literal["**"]]] = None
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


@dataclass(repr=False)
class Access(PythonNode):
    """
    One level of attribute access.

    The example `foo.bar.baz` should be represented as:

    ```python
    Access(
        value=Access(
            value=Name(name="foo"),
            attribute=Name(name="bar"),
        ),
        attribute=Name(name="baz"),
    )
    ```
    """

    value: Node = dataclasses.field(default_factory=BlankNode)
    attribute: Node = dataclasses.field(default_factory=BlankNode)


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
