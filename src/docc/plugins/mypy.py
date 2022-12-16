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
Source discovery based on mypy.
"""

import logging
import sys
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import (
    Dict,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Union,
)

import mypy.build
import mypy.defaults
import mypy.nodes
import mypy.types
from mypy.find_sources import create_source_list
from mypy.modulefinder import BuildSource
from mypy.nodes import (
    AssignmentStmt,
    ClassDef,
    Decorator,
    ExpressionStmt,
    FuncDef,
    MypyFile,
    NameExpr,
    StrExpr,
)
from mypy.options import Options
from mypy.traverser import ExtendedTraverserVisitor
from mypy.type_visitor import TypeVisitor

from docc.build import Builder
from docc.discover import Discover, T
from docc.document import Document, Node, Visit, Visitor
from docc.languages import python
from docc.plugins.references import Definition, Reference
from docc.references import Index
from docc.settings import PluginSettings
from docc.source import Source
from docc.transform import Transform


class MyPyDiscover(Discover):
    """
    Source discovery based on mypy.
    """

    paths: Sequence[str]
    settings: PluginSettings

    def __init__(self, config: PluginSettings) -> None:
        super().__init__(config)
        self.settings = config

        paths = config.get("paths", [])
        if not isinstance(paths, Sequence):
            raise TypeError("mypy paths must be a list")

        if any(not isinstance(path, str) for path in paths):
            raise TypeError("every mypy path must be a string")

        if not paths:
            raise ValueError("mypy needs at least one path")

        self.paths = [str(config.resolve_path(path)) for path in paths]

    def discover(self, known: FrozenSet[T]) -> Iterator[Source]:
        """
        Uses mypy's `create_source_list` to find sources.
        """
        options = Options()
        options.python_version = mypy.defaults.PYTHON3_VERSION
        options.export_types = True
        options.preserve_asts = True
        for build_source in create_source_list(self.paths, options):
            relative_path = None
            assert build_source.path is not None
            path = PurePath(build_source.path)
            relative_path = self.settings.unresolve_path(path)
            yield MyPySource(relative_path, build_source)


class MyPySource(Source):
    """
    A Source based on mypy.
    """

    build_source: BuildSource
    _relative_path: PurePath

    def __init__(
        self, relative_path: PurePath, build_source: BuildSource
    ) -> None:
        self.build_source = build_source
        self._relative_path = relative_path

    @property
    def relative_path(self) -> Optional[PurePath]:
        """
        The relative path to the Source.
        """
        return self._relative_path

    @property
    def output_path(self) -> PurePath:
        """
        Where to put the output derived from this source.
        """
        return self._relative_path


class MyPyBuilder(Builder):
    """
    A Builder based on mypy.
    """

    settings: PluginSettings

    def __init__(self, config: PluginSettings) -> None:
        super().__init__(config)
        self.settings = config

    def build(
        self,
        index: Index,
        all_sources: Sequence[Source],
        unprocessed: Set[Source],
        processed: Dict[Source, Document],
    ) -> None:
        """
        Process MyPySources into Documents.
        """
        source_set = set(s for s in unprocessed if isinstance(s, MyPySource))
        unprocessed -= source_set
        build_sources = [x.build_source for x in source_set]

        source_map = {}
        for source in source_set:
            if source.relative_path is None:
                continue
            absolute_path = self.settings.resolve_path(source.relative_path)
            absolute_path = Path(absolute_path).resolve()
            source_map[absolute_path] = source

        options = Options()
        options.python_version = (
            sys.version_info.major,
            sys.version_info.minor,
        )
        options.export_types = True
        options.preserve_asts = True
        result = mypy.build.build(sources=build_sources, options=options)

        if result.errors:
            # TODO: Report errors better.
            raise Exception("mypy build has errors: " + str(result.errors))

        for name, state in result.graph.items():
            if state.abspath is None:
                if state.path is None:
                    continue

                absolute_path = self.settings.resolve_path(Path(state.path))
            else:
                absolute_path = Path(state.abspath)

            absolute_path = absolute_path.resolve()

            try:
                source = source_map.pop(absolute_path)
            except KeyError:
                continue

            assert source not in processed
            assert state.tree is not None, f"missing state tree for `{name}`"

            processed[source] = Document(
                all_sources, index, source, MyPyNode(result, state.tree)
            )

        assert 0 == len(source_map)


class _FuncBodyVisitor(Visitor):
    def __init__(self, func: python.Function) -> None:
        pass  # TODO

    def enter(self, any_node: Node) -> Visit:
        return Visit.SkipChildren  # TODO

    def exit(self, any_node: Node) -> None:
        pass  # TODO


class _ClassDefsVisitor(Visitor):
    def __init__(self, class_: python.Class) -> None:
        pass  # TODO

    def enter(self, any_node: Node) -> Visit:
        return Visit.SkipChildren  # TODO

    def exit(self, any_node: Node) -> None:
        pass  # TODO


@dataclass
class _TransformContext:
    node: "MyPyNode"
    child_offset: int = 0


class _TransformVisitor(Visitor):
    root: Optional[Node]
    old_stack: List[_TransformContext]
    new_stack: List[Node]
    source_paths: Set[str]

    def __init__(self, source_paths: Set[str]) -> None:
        self.root = None
        self.old_stack = []
        self.new_stack = []
        self.source_paths = source_paths

    def push_new(self, node: Node) -> None:
        if self.root is None:
            assert 0 == len(self.new_stack)
            self.root = node
        self.new_stack.append(node)

    def enter_file(self, node: Node, file: MypyFile) -> Visit:
        assert 0 == len(self.new_stack)
        self.push_new(python.Module())
        return Visit.TraverseChildren

    def exit_file(self) -> None:
        self.new_stack.pop()

    def enter_expression_stmt(
        self, node: Node, expr_stmt: ExpressionStmt
    ) -> Visit:
        old_parent = self.old_stack[-1]
        if isinstance(expr_stmt.expr, StrExpr):
            if isinstance(old_parent.node.node, MypyFile):
                if old_parent.child_offset == 0:
                    # It's a module docstring!
                    return Visit.TraverseChildren

        return Visit.SkipChildren

    def exit_expression_stmt(self) -> None:
        pass

    def enter_str_expr(self, node: Node, str_expr: StrExpr) -> Visit:
        parent = self.old_stack[-1].node.node
        grandparent = self.old_stack[-2].node.node

        if not isinstance(parent, ExpressionStmt):
            raise NotImplementedError()  # TODO

        if not isinstance(grandparent, MypyFile):
            raise NotImplementedError()  # TODO

        docstring_node = getattr(self.new_stack[-1], "docstring", None)

        if docstring_node:
            new = python.Docstring(text=str_expr.value)
            self.new_stack[-1].replace_child(docstring_node, new)

        return Visit.SkipChildren

    def exit_str_expr(self) -> None:
        pass

    def enter_decorator(self, node: "MyPyNode", decorator: Decorator) -> Visit:
        func_def = MyPyNode(node._build, decorator.func)

        visitor = _TransformVisitor(self.source_paths)
        func_def.visit(visitor)
        assert isinstance(visitor.root, python.Function)
        assert isinstance(visitor.root.decorators, list)

        # TODO: `abc.abstractmethod` is not a decorator in mypy.
        # TODO: `builtins.property` is not a decorator in mypy.

        for expression in decorator.decorators:
            # TODO: Translate the decorator into a languages.python type.
            visitor.root.decorators.append(MyPyNode(node._build, expression))

        try:
            parent = self.new_stack[-1]
        except IndexError:
            parent = None

        if parent is not None:
            if hasattr(parent, "members"):
                if isinstance(parent.members, list):
                    parent.members.append(visitor.root)

        self.push_new(visitor.root)

        return Visit.SkipChildren

    def exit_decorator(self) -> None:
        self.new_stack.pop()

    def enter_class_def(self, node: "MyPyNode", class_def: ClassDef) -> Visit:
        name = python.Name(name=class_def.name, full_name=class_def.fullname)
        class_ = python.Class(name=name)
        definition = Definition(identifier=class_def.fullname, child=class_)

        try:
            parent = self.new_stack[-1]
        except IndexError:
            parent = None

        if parent is not None:
            if hasattr(parent, "members"):
                if isinstance(parent.members, list):
                    parent.members.append(definition)

        self.push_new(class_)

        # TODO: class_def.metaclass
        # TODO: class_def.decorators
        # TODO: class_def.info (base types, etc)

        visitor = _ClassDefsVisitor(class_)
        MyPyNode(node._build, class_def.defs).visit(visitor)

        return Visit.SkipChildren

    def exit_class_def(self) -> None:
        self.new_stack.pop()

    def enter_func_def(self, node: "MyPyNode", func_def: FuncDef) -> Visit:
        name = python.Name(name=func_def.name, full_name=func_def.fullname)
        func = python.Function(name=name)
        definition = Definition(identifier=func_def.fullname, child=func)

        try:
            parent = self.new_stack[-1]
        except IndexError:
            parent = None

        if parent is not None:
            if hasattr(parent, "members"):
                if isinstance(parent.members, list):
                    parent.members.append(definition)

        self.push_new(func)

        # TODO: arguments

        if func_def.type:
            assert isinstance(func_def.type, mypy.types.CallableType)
            type_visitor = _TypeVisitor(self.source_paths, node._build)
            func_def.type.ret_type.accept(type_visitor)
            assert type_visitor.root is not None
            func.return_type = type_visitor.root

        visitor = _FuncBodyVisitor(func)
        MyPyNode(node._build, func_def.body).visit(visitor)

        return Visit.SkipChildren

    def exit_func_def(self) -> None:
        self.new_stack.pop()

    def enter_assignment_stmt(
        self, mypy: "MyPyNode", node: AssignmentStmt
    ) -> Visit:
        names = []

        for name in node.lvalues:
            if not isinstance(name, NameExpr):
                logging.debug("skipping assignment %s", node)
                return Visit.SkipChildren
            python_name = python.Name(name=name.name, full_name=name.fullname)
            names.append(python_name)

        # TODO: Convert lvalue to a non-mypy type.
        attribute = python.Attribute(
            names=names, value=MyPyNode(mypy._build, node.rvalue)
        )

        try:
            parent = self.new_stack[-1]
        except IndexError:
            parent = None

        if parent is not None:
            if hasattr(parent, "members"):
                if isinstance(parent.members, list):
                    parent.members.append(attribute)
                    return Visit.SkipChildren

        raise Exception("unexpected assignment")

    def exit_assignment_stmt(self) -> None:
        pass

    def enter(self, any_node: Node) -> Visit:
        if not isinstance(any_node, MyPyNode):
            raise ValueError(
                "expected `"
                + MyPyNode.__name__
                + "` but got `"
                + any_node.__class__.__name__
                + "`"
            )

        node = any_node.node
        try:
            parent = self.old_stack[-1].node.node
        except IndexError:
            parent = None

        module_member = isinstance(parent, MypyFile)

        visit: Visit

        if isinstance(node, MypyFile):
            visit = self.enter_file(any_node, node)
        elif isinstance(node, ExpressionStmt):
            visit = self.enter_expression_stmt(any_node, node)
        elif isinstance(node, StrExpr):
            visit = self.enter_str_expr(any_node, node)
        elif isinstance(node, Decorator):
            visit = self.enter_decorator(any_node, node)
        elif isinstance(node, ClassDef):
            visit = self.enter_class_def(any_node, node)
        elif isinstance(node, FuncDef):
            visit = self.enter_func_def(any_node, node)
        elif isinstance(node, AssignmentStmt):
            visit = self.enter_assignment_stmt(any_node, node)
        elif module_member and isinstance(node, mypy.nodes.Node):
            logging.debug("skipping module member node %s", node)
            visit = Visit.SkipChildren
        else:
            raise Exception(f"unknown node type {node}")

        self.old_stack.append(_TransformContext(node=any_node))

        return visit

    def exit(self, any_node: Node) -> None:
        module_member = False

        self.old_stack.pop()
        if self.old_stack:
            self.old_stack[-1].child_offset += 1
            module_member = isinstance(self.old_stack[-1].node.node, MypyFile)

        assert isinstance(any_node, MyPyNode)
        node = any_node.node

        if isinstance(node, MypyFile):
            self.exit_file()
        elif isinstance(node, ExpressionStmt):
            self.exit_expression_stmt()
        elif isinstance(node, StrExpr):
            self.exit_str_expr()
        elif isinstance(node, ClassDef):
            self.exit_class_def()
        elif isinstance(node, Decorator):
            self.exit_decorator()
        elif isinstance(node, FuncDef):
            self.exit_func_def()
        elif isinstance(node, AssignmentStmt):
            self.exit_assignment_stmt()
        elif module_member and isinstance(node, mypy.nodes.Node):
            pass
        else:
            raise Exception(f"unknown node type {node}")


class MyPyTransform(Transform):
    """
    Converts mypy nodes into Python language nodes.
    """

    def __init__(self, config: PluginSettings) -> None:
        super().__init__(config)
        self.settings = config

    def transform(self, document: Document) -> None:
        """
        Apply the transformation to the given document.
        """
        mypy_sources = (
            x for x in document.all_sources if isinstance(x, MyPySource)
        )

        source_paths = set(
            x.build_source.path
            for x in mypy_sources
            if x.build_source.path is not None
        )
        visitor = _TransformVisitor(source_paths)
        try:
            document.root.visit(visitor)
        except NotImplementedError:
            logging.exception("unsupported node")  # TODO: Don't catch this.
        assert visitor.root is not None
        document.root = visitor.root


class _ChildrenVisitor(ExtendedTraverserVisitor):
    # I'm using ExtendedTraverserVisitor as a base class here instead of
    # NodeVisitor because ExtendedTraverserVisitor gives us a single `visit`
    # method that's called for all node types, instead of having to override
    # NodeVisitor's concrete methods. I hope this'll be more resilient against
    # mypy's internals changing.

    depth: int
    children: List[mypy.nodes.Node]

    def __init__(self) -> None:
        self.depth = 0
        self.children = []

    def visit(self, o: mypy.nodes.Node) -> bool:
        if self.depth == 0:
            self.depth += 1
            return True

        self.children.append(o)
        return False


class _TypeVisitor(TypeVisitor[None]):
    root: Optional[python.Type]
    stack: List[python.Type]
    build: mypy.build.BuildResult
    source_paths: Set[str]

    def __init__(
        self, source_paths: Set[str], build: mypy.build.BuildResult
    ) -> None:
        self.root = None
        self.stack = []
        self.build = build

        self.source_paths = source_paths

    def push(self, type_: python.Type) -> None:
        if self.root is None:
            assert not self.stack
            self.root = type_
            self.stack.append(type_)
            return

        assert 0 < len(self.stack)
        assert isinstance(self.stack[-1].generics, python.Generics)
        assert isinstance(self.stack[-1].generics.arguments, list)
        self.stack[-1].generics.arguments.append(type_)
        self.stack.append(type_)

    def pop(self) -> None:
        self.stack.pop()

    def push_leaf(self, type_: python.Type) -> None:
        self.push(type_)
        self.pop()

    def visit_unbound_type(self, t: mypy.types.UnboundType) -> None:
        raise NotImplementedError()

    def visit_any(self, t: mypy.types.AnyType) -> None:
        # TODO: is `typing.Any` always the correct full_name?
        name = python.Name("Any", None)
        type_ = python.Type(name=name)
        self.push_leaf(type_)

    def visit_none_type(self, t: mypy.types.NoneType) -> None:
        # TODO: What is the correct full_name?
        name = python.Name("None", None)
        type_ = python.Type(name=name)
        self.push_leaf(type_)

    def visit_uninhabited_type(self, t: mypy.types.UninhabitedType) -> None:
        raise NotImplementedError(repr(t))

    def visit_erased_type(self, t: mypy.types.ErasedType) -> None:
        raise NotImplementedError(repr(t))

    def visit_deleted_type(self, t: mypy.types.DeletedType) -> None:
        raise NotImplementedError(repr(t))

    def visit_type_var(self, t: mypy.types.TypeVarType) -> None:
        raise NotImplementedError(repr(t))

    def visit_param_spec(self, t: mypy.types.ParamSpecType) -> None:
        raise NotImplementedError(repr(t))

    def visit_parameters(self, t: mypy.types.Parameters) -> None:
        raise NotImplementedError(repr(t))

    def visit_type_var_tuple(self, t: mypy.types.TypeVarTupleType) -> None:
        raise NotImplementedError(repr(t))

    def visit_instance(self, t: mypy.types.Instance) -> None:
        full_name = t.type_ref if t.type_ref is not None else t.type.fullname
        name: Node = python.Name(name=t.type.name, full_name=full_name)

        try:
            source_path = self.build.files[t.type.module_name].path
        except IndexError:
            source_path = None

        if source_path in self.source_paths:
            name = Reference(identifier=full_name, child=name)

        if not t.args:
            self.push_leaf(python.Type(name=name))
            return

        generics = python.Generics()
        type_ = python.Type(name=name, generics=generics)
        self.push(type_)

        for argument in t.args:
            argument.accept(self)

        self.pop()

    def visit_callable_type(self, t: mypy.types.CallableType) -> None:
        raise NotImplementedError(repr(t))

    def visit_overloaded(self, t: mypy.types.Overloaded) -> None:
        raise NotImplementedError(repr(t))

    def visit_tuple_type(self, t: mypy.types.TupleType) -> None:
        # TODO: is `typing.Tuple` always the correct full_name?
        name = python.Name(name="Tuple", full_name="typing.Tuple")
        generics = python.Generics()
        type_ = python.Type(name=name, generics=generics)
        self.push(type_)

        for item in t.items:
            item.accept(self)

        self.pop()

    def visit_typeddict_type(self, t: mypy.types.TypedDictType) -> None:
        raise NotImplementedError(repr(t))

    def visit_literal_type(self, t: mypy.types.LiteralType) -> None:
        raise NotImplementedError(repr(t))

    def visit_union_type(self, t: mypy.types.UnionType) -> None:
        # TODO: is `typing.Union` always the correct full_name?
        name = python.Name(name="Union", full_name="typing.Union")
        generics = python.Generics()
        type_ = python.Type(name=name, generics=generics)
        self.push(type_)

        for item in t.items:
            item.accept(self)

        self.pop()

    def visit_partial_type(self, t: mypy.types.PartialType) -> None:
        raise NotImplementedError(repr(t))

    def visit_type_type(self, t: mypy.types.TypeType) -> None:
        raise NotImplementedError(repr(t))

    def visit_type_alias_type(self, t: mypy.types.TypeAliasType) -> None:
        raise NotImplementedError(repr(t))

    def visit_unpack_type(self, t: mypy.types.UnpackType) -> None:
        raise NotImplementedError(repr(t))


class MyPyNode(Node):
    """
    A wrapper around a mypy tree node.
    """

    __slots__ = ("node", "_children")

    node: mypy.nodes.Node
    _children: Union[List[Node], Iterator[Node]]
    _build: mypy.build.BuildResult

    def __init__(self, build: mypy.build.BuildResult, node: mypy.nodes.Node):
        self.node = node

        visitor = _ChildrenVisitor()
        self.node.accept(visitor)
        self._build = build
        self._children = (MyPyNode(build, n) for n in visitor.children)

    @property
    def children(self) -> List[Node]:
        """
        Child nodes belonging to this node.
        """
        if not isinstance(self._children, list):
            self._children = list(self._children)
        return self._children

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        self._children = [
            new if child == old else child for child in self.children
        ]

    def __repr__(self) -> str:
        """
        Represent this node as a string.
        """
        arguments = "..."
        if hasattr(self.node, "name"):
            arguments = f"name={self.node.name!r}, {arguments}"

        my_name = self.__class__.__name__
        inner_name = self.node.__class__.__name__
        return f"{my_name}({inner_name}({arguments}))"
