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
Documentation plugin for Python.
"""

import glob
import logging
import os.path
from collections import defaultdict
from dataclasses import dataclass
from pathlib import PurePath
from typing import (
    Dict,
    Final,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    TextIO,
    Tuple,
    Type,
)

import libcst as cst
from inflection import dasherize, underscore
from libcst.metadata import ExpressionContext

from docc.build import Builder
from docc.discover import Discover, T
from docc.document import Document, Node, Visit, Visitor
from docc.languages import python
from docc.languages.verbatim import Fragment, Pos, Stanza, Verbatim
from docc.plugins.references import Definition, Reference
from docc.references import Index
from docc.settings import PluginSettings
from docc.source import Source, TextSource
from docc.transform import Transform

WHITESPACE: Tuple[Type[cst.CSTNode], ...] = (
    cst.TrailingWhitespace,
    cst.EmptyLine,
)


class PythonDiscover(Discover):
    """
    Find Python source files.
    """

    paths: Sequence[str]
    settings: PluginSettings

    def __init__(self, config: PluginSettings) -> None:
        super().__init__(config)
        self.settings = config

        paths = config.get("paths", [])
        if not isinstance(paths, Sequence):
            raise TypeError("python paths must be a list")

        if any(not isinstance(path, str) for path in paths):
            raise TypeError("every python path must be a string")

        if not paths:
            raise ValueError("python needs at least one path")

        self.paths = [str(config.resolve_path(path)) for path in paths]

    def discover(self, known: FrozenSet[T]) -> Iterator[Source]:
        """
        Find sources.
        """
        escaped = ((path, glob.escape(path)) for path in self.paths)
        joined = ((r, os.path.join(pat, "**", "*.py")) for r, pat in escaped)
        globbed = ((r, glob.glob(pat, recursive=True)) for r, pat in joined)

        for root_text, absolute_texts in globbed:
            root_path = PurePath(root_text)
            for absolute_text in absolute_texts:
                absolute_path = PurePath(absolute_text)
                relative_path = self.settings.unresolve_path(absolute_path)

                yield PythonSource(root_path, relative_path, absolute_path)


class PythonSource(TextSource):
    """
    A Source representing a Python file.
    """

    root_path: Final[PurePath]
    absolute_path: Final[PurePath]
    _relative_path: Final[PurePath]

    def __init__(
        self,
        root_path: PurePath,
        relative_path: PurePath,
        absolute_path: PurePath,
    ) -> None:
        self.root_path = root_path
        self._relative_path = relative_path
        self.absolute_path = absolute_path

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

    def open(self) -> TextIO:
        """
        Open the source for reading.
        """
        return open(self.absolute_path, "r")


class PythonBuilder(Builder):
    """
    Convert python source files into syntax trees.
    """

    settings: PluginSettings

    def __init__(self, config: PluginSettings) -> None:
        """
        Create a PythonBuilder with the given configuration.
        """
        self.settings = config

    def build(
        self,
        index: Index,
        all_sources: Sequence[Source],
        unprocessed: Set[Source],
        processed: Dict[Source, Document],
    ) -> None:
        """
        Consume unprocessed Sources and insert their Documents into processed.
        """
        source_set = set(s for s in unprocessed if isinstance(s, PythonSource))
        unprocessed -= source_set

        all_modules = set()

        sources_by_root = defaultdict(set)
        for source in source_set:
            root = self.settings.resolve_path(source.root_path)
            sources_by_root[root].add(source)

        for root, sources in sources_by_root.items():
            paths = set()

            for source in sources:
                if source.relative_path is None:
                    continue
                abs_path = self.settings.resolve_path(source.relative_path)
                paths.add(str(abs_path.relative_to(root)))

            repo_manager = cst.metadata.FullRepoManager(
                repo_root_dir=str(root),
                paths=list(paths),
                providers=_CstVisitor.METADATA_DEPENDENCIES,
            )

            for source in sources:
                assert source.relative_path
                abs_path = self.settings.resolve_path(source.relative_path)
                relative_path = abs_path.relative_to(root)

                visitor = _CstVisitor(all_modules, source)
                repo_manager.get_metadata_wrapper_for_path(
                    str(relative_path)
                ).visit(visitor)
                assert visitor.root is not None

                document = Document(
                    all_sources,
                    index,
                    source,
                    visitor.root,
                )

                processed[source] = document


class CstNode(Node):
    """
    A python concrete syntax tree node.
    """

    cst_node: cst.CSTNode
    source: Source
    _children: List[Node]
    start: Pos
    end: Pos
    type: Optional[str]
    global_scope: Optional[bool]
    class_scope: Optional[bool]
    expression_context: Optional[ExpressionContext]
    names: Set[str]
    all_modules: Set[str]  # TODO: Make this a FrozenSet

    def __init__(
        self,
        all_modules: Set[str],
        cst_node: cst.CSTNode,
        source: Source,
        start: Pos,
        end: Pos,
        type_: Optional[str],
        global_scope: Optional[bool],
        class_scope: Optional[bool],
        expression_context: Optional[ExpressionContext],
        names: Set[str],
        children: List[Node],
    ) -> None:
        self.all_modules = all_modules
        self.cst_node = cst_node
        self.source = source
        self._children = children
        self.start = start
        self.end = end
        self.type = type_
        self.global_scope = global_scope
        self.class_scope = class_scope
        self.expression_context = expression_context
        self.names = names

    @property
    def children(self) -> Sequence[Node]:
        """
        Child nodes belonging to this node.
        """
        return self._children

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        self._children = [new if c == old else c for c in self.children]

    def find_child(self, cst_node: cst.CSTNode) -> Node:
        """
        Given a libcst node, find the CstNode in the same position.
        """
        index = self.cst_node.children.index(cst_node)
        return self.children[index]

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        cst_node = self.cst_node
        text = f"{self.__class__.__name__}({cst_node.__class__.__name__}(...)"
        text += f", start={self.start}"
        text += f", end={self.end}"
        if self.type is not None:
            text += f", type={self.type!r}"
        if self.names:
            text += f", names={self.names!r}"
        return text + ")"


class _CstVisitor(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (
        cst.metadata.PositionProvider,
        cst.metadata.FullyQualifiedNameProvider,
        # cst.metadata.TypeInferenceProvider,
        cst.metadata.ScopeProvider,
        cst.metadata.ExpressionContextProvider,
    )

    stack: Final[List[CstNode]]
    source: Final[Source]
    all_modules: Final[Set[str]]
    root: Optional[CstNode]

    def __init__(self, all_modules: Set[str], source: Source) -> None:
        super().__init__()
        self.stack = []
        self.root = None
        self.source = source
        self.all_modules = all_modules

    def on_visit(self, node: cst.CSTNode) -> bool:
        position = self.get_metadata(cst.metadata.PositionProvider, node)
        type_ = None  # self.get_metadata(
        #    cst.metadata.TypeInferenceProvider, node, None
        # )

        scope = self.get_metadata(cst.metadata.ScopeProvider, node, None)
        global_scope = isinstance(scope, cst.metadata.GlobalScope)
        class_scope = isinstance(scope, cst.metadata.ClassScope)

        expression_context = self.get_metadata(
            cst.metadata.ExpressionContextProvider, node, None
        )

        qualified_names = self.get_metadata(
            cst.metadata.FullyQualifiedNameProvider, node, None
        )
        names: Set[str] = set()
        if qualified_names:
            names = set(n.name for n in qualified_names)

        start = Pos(
            line=position.start.line,
            column=position.start.column,
        )
        end = Pos(
            line=position.end.line,
            column=position.end.column,
        )
        new = CstNode(
            self.all_modules,
            node,
            self.source,
            start,
            end,
            type_,
            global_scope,
            class_scope,
            expression_context,
            names,
            [],
        )

        if self.stack:
            self.stack[-1]._children.append(new)
        else:
            assert self.root is None

        if self.root is None:
            self.root = new

        if isinstance(node, cst.Module):
            self.all_modules.update(names)

        self.stack.append(new)
        return True

    def on_leave(self, original_node: cst.CSTNode) -> None:
        self.stack.pop()


class PythonTransform(Transform):
    """
    Transforms CstNode instances into Python language nodes.
    """

    excluded_references: Final[FrozenSet[str]]

    def __init__(self, config: PluginSettings) -> None:
        """
        Create a Transform with the given configuration.
        """
        self.excluded_references = frozenset(
            config.get("excluded_references", [])
        )

    def transform(self, document: Document) -> None:
        """
        Apply the transformation to the given document.
        """
        document.root.visit(
            _AnnotationReferenceTransformVisitor(self.excluded_references)
        )

        sources: Set[str] = set()  # TODO
        visitor = _TransformVisitor(document, sources)
        document.root.visit(visitor)
        assert visitor.root is not None
        document.root = visitor.root

        document.root.visit(_AnnotationTransformVisitor())
        document.root.visit(_NameTransformVisitor())


@dataclass
class _TransformContext:
    node: "CstNode"
    child_offset: int = 0


class _TransformVisitor(Visitor):
    root: Optional[Node]
    old_stack: Final[List[_TransformContext]]
    new_stack: Final[List[Node]]
    document: Final[Document]
    source_paths: Final[Set[str]]

    def __init__(self, document: Document, source_paths: Set[str]) -> None:
        self.root = None
        self.old_stack = []
        self.new_stack = []
        self.document = document
        self.source_paths = source_paths

    def push_new(self, node: Node) -> None:
        if self.root is None:
            assert 0 == len(self.new_stack)
            self.root = node
        self.new_stack.append(node)

    def enter_module(self, node: CstNode, cst_node: cst.Module) -> Visit:
        assert 0 == len(self.new_stack)
        module = python.Module()

        names = sorted(node.names)

        if names:
            module.name = python.Name(names[0], names[0])

        maybe_definition: Node = module
        for name in names:
            maybe_definition = Definition(
                identifier=name, child=maybe_definition
            )
            self.push_new(maybe_definition)

        self.push_new(module)

        docstring = cst_node.get_docstring(True)
        if docstring is not None:
            module.docstring = python.Docstring(docstring)

        return Visit.TraverseChildren

    def exit_module(self, node: CstNode) -> None:
        self.new_stack.pop()
        for _name in node.names:
            self.new_stack.pop()

    def enter_class_def(self, node: CstNode, cst_node: cst.ClassDef) -> Visit:
        assert 0 < len(self.new_stack)

        class_def = python.Class()
        self.push_new(class_def)

        docstring = cst_node.get_docstring(True)
        if docstring is not None:
            class_def.docstring = python.Docstring(docstring)

        class_def.name = node.find_child(cst_node.name)

        source = self.document.source
        if isinstance(source, TextSource):
            decorators = []
            for cst_decorator in cst_node.decorators:
                decorator = node.find_child(cst_decorator)

                # Trim whitespace from each decorator.
                if isinstance(decorator, CstNode):
                    decorator = decorator.find_child(cst_decorator.decorator)

                decorators.append(_VerbatimTransform.apply(source, decorator))

            class_def.decorators = decorators

        maybe_definition: Node = class_def
        for name in node.names:
            maybe_definition = Definition(
                identifier=name, child=maybe_definition
            )

        if 1 < len(self.new_stack):
            parent = self.new_stack[-2]
            members = getattr(parent, "members", None)
            if isinstance(members, list):
                members.append(maybe_definition)

        body = node.find_child(cst_node.body)
        assert isinstance(body, CstNode)

        for cst_statement in cst_node.body.body:
            statement = body.find_child(cst_statement)
            statement.visit(self)

        # TODO: base classes
        # TODO: metaclass

        return Visit.SkipChildren

    def exit_class_def(self) -> None:
        self.new_stack.pop()

    def enter_function_def(
        self, node: CstNode, cst_node: cst.FunctionDef
    ) -> Visit:
        assert 0 < len(self.new_stack)

        function_def = python.Function(
            asynchronous=cst_node.asynchronous is not None
        )
        self.push_new(function_def)

        docstring = cst_node.get_docstring(True)
        if docstring is not None:
            function_def.docstring = python.Docstring(docstring)

        function_def.name = node.find_child(cst_node.name)

        if cst_node.returns is not None:
            function_def.return_type = node.find_child(cst_node.returns)

        source = self.document.source
        if isinstance(source, TextSource):
            body = node.find_child(cst_node.body)
            function_def.body = _VerbatimTransform.apply(source, body)

            decorators = []
            for cst_decorator in cst_node.decorators:
                decorator = node.find_child(cst_decorator)

                # Trim whitespace from each decorator.
                if isinstance(decorator, CstNode):
                    decorator = decorator.find_child(cst_decorator.decorator)

                decorators.append(_VerbatimTransform.apply(source, decorator))

            function_def.decorators = decorators

        maybe_definition: Node = function_def
        for name in node.names:
            maybe_definition = Definition(
                identifier=name, child=maybe_definition
            )

        if 1 < len(self.new_stack):
            parent = self.new_stack[-2]
            members = getattr(parent, "members", None)
            if isinstance(members, list):
                members.append(maybe_definition)

        # TODO: arguments
        # TODO: argument defaults

        return Visit.SkipChildren

    def exit_function_def(self) -> None:
        self.new_stack.pop()

    def _enter_assignment(
        self,
        node: CstNode,
        targets: Sequence[Node],
        value: Optional[Node],
    ) -> Visit:
        if not node.global_scope and not node.class_scope:
            return Visit.SkipChildren

        if not self.new_stack:
            return Visit.SkipChildren

        parent = self.new_stack[-1]
        members = getattr(parent, "members", None)
        if not isinstance(members, list):
            return Visit.SkipChildren

        names: List[Node] = []
        attribute = python.Attribute(names=names)

        # TODO: Docstring

        if isinstance(self.document.source, TextSource):
            attribute.body = _VerbatimTransform.apply(
                self.document.source, node
            )

        for target in targets:
            names.append(target)

        maybe_definition = attribute
        for name_node in names:
            if not isinstance(name_node, CstNode):
                continue

            for name in name_node.names:
                maybe_definition = Definition(
                    identifier=name, child=maybe_definition
                )

        members.append(maybe_definition)
        return Visit.SkipChildren

    def enter_ann_assign(
        self, node: CstNode, cst_node: cst.AnnAssign
    ) -> Visit:
        value = None
        if cst_node.value is not None:
            value = node.find_child(cst_node.value)
        return self._enter_assignment(
            node,
            [node.find_child(cst_node.target)],
            value,
        )

    def exit_ann_assign(self) -> None:
        pass

    def enter_assign(self, node: CstNode, cst_node: cst.Assign) -> Visit:
        targets = []
        for cst_target in cst_node.targets:
            target = node.find_child(cst_target)
            assert isinstance(target, CstNode)  # TODO: Assumes only CstNodes.
            targets.append(target.find_child(cst_target.target))

        return self._enter_assignment(
            node,
            targets,
            node.find_child(cst_node.value),
        )

    def exit_assign(self) -> None:
        pass

    def enter(self, node: Node) -> Visit:
        if not isinstance(node, CstNode):
            raise ValueError(
                "expected `"
                + CstNode.__name__
                + "` but got `"
                + node.__class__.__name__
                + "`"
            )

        cst_node = node.cst_node
        try:
            parent = self.old_stack[-1].node.cst_node
        except IndexError:
            parent = None

        module_member = isinstance(parent, cst.Module)

        visit: Visit

        if isinstance(cst_node, cst.Module):
            visit = self.enter_module(node, cst_node)
        elif isinstance(cst_node, cst.ClassDef):
            visit = self.enter_class_def(node, cst_node)
        elif isinstance(cst_node, cst.FunctionDef):
            visit = self.enter_function_def(node, cst_node)
        elif isinstance(cst_node, cst.AnnAssign):
            visit = self.enter_ann_assign(node, cst_node)
        elif isinstance(cst_node, cst.Assign):
            visit = self.enter_assign(node, cst_node)
        elif isinstance(cst_node, cst.SimpleStatementLine):
            visit = Visit.TraverseChildren
        elif isinstance(cst_node, cst.Expr):
            visit = Visit.SkipChildren
        elif isinstance(cst_node, (cst.Import, cst.ImportFrom)):
            visit = Visit.SkipChildren
        elif isinstance(cst_node, cst.Pass):
            visit = Visit.SkipChildren
        elif isinstance(cst_node, WHITESPACE):
            visit = Visit.SkipChildren
        elif module_member and isinstance(cst_node, cst.CSTNode):
            logging.debug("skipping module member node %s", node)
            visit = Visit.SkipChildren
        else:
            raise Exception(f"unknown node type {node}")

        self.old_stack.append(_TransformContext(node=node))

        return visit

    def exit(self, node: Node) -> None:
        module_member = False

        self.old_stack.pop()
        if self.old_stack:
            self.old_stack[-1].child_offset += 1
            parent = self.old_stack[-1].node.cst_node
            module_member = isinstance(parent, cst.Module)

        assert isinstance(node, CstNode)
        cst_node = node.cst_node

        if isinstance(cst_node, cst.Module):
            self.exit_module(node)
        elif isinstance(cst_node, cst.ClassDef):
            self.exit_class_def()
        elif isinstance(cst_node, cst.FunctionDef):
            self.exit_function_def()
        elif isinstance(cst_node, cst.AnnAssign):
            self.exit_ann_assign()
        elif isinstance(cst_node, cst.Assign):
            self.exit_assign()
        elif isinstance(cst_node, cst.SimpleStatementLine):
            pass
        elif isinstance(cst_node, cst.Expr):
            pass
        elif isinstance(cst_node, (cst.ImportFrom, cst.Import)):
            pass
        elif isinstance(cst_node, cst.Pass):
            pass
        elif isinstance(cst_node, WHITESPACE):
            pass
        elif module_member and isinstance(cst_node, cst.CSTNode):
            pass
        else:
            raise Exception(f"unknown node type {cst_node}")


class _AnnotationReferenceTransformVisitor(Visitor):
    root: Optional[Node]
    stack: Final[List[Node]]
    excluded_references: Final[FrozenSet[str]]

    def __init__(self, excluded_references: FrozenSet[str]) -> None:
        self.stack = []
        self.root = None
        self.excluded_references = excluded_references

    def enter(self, node: Node) -> Visit:
        if self.root is None:
            assert not self.stack
            self.root = node

        self.stack.append(node)

        if not isinstance(node, CstNode):
            return Visit.TraverseChildren

        cst_node = node.cst_node

        if isinstance(cst_node, cst.Attribute):
            # TODO: Need to figure out how to detect if a method is defined in
            #       a superclass (ex. `Foo.__name__`.)
            pass
        elif isinstance(cst_node, (cst.Name, cst.SimpleString)):
            if node.expression_context != ExpressionContext.STORE:
                self._make_reference(node)

        return Visit.TraverseChildren

    def _make_reference(self, node: CstNode) -> None:
        for name in node.names:
            if name in self.excluded_references:
                continue

            # TODO: This incorrectly matches `foobar.do_thing` if `foo` is in
            #       `node.all_modules`.
            if any(name.startswith(m) for m in node.all_modules):
                reference = Reference(
                    identifier=name,
                    child=node,
                )
                if len(self.stack) > 1:
                    self.stack[-2].replace_child(node, reference)
                else:
                    self.root = reference

    def exit(self, node: Node) -> None:
        self.stack.pop()


class _TypeVisitor(Visitor):
    root: Final[Node]
    type: Final[python.Type]

    def __init__(self, root: Node, type_: python.Type) -> None:
        self.root = root
        self.type = type_

    def enter(self, node: Node) -> Visit:
        if not isinstance(node, CstNode):
            return Visit.TraverseChildren

        cst_node = node.cst_node
        if isinstance(cst_node, cst.Name):
            self.type.name = self.root
        elif isinstance(cst_node, cst.SimpleString):
            self.type.name = self.root
        elif isinstance(cst_node, cst.Ellipsis):
            # TODO: check this.
            self.type.name = python.Name(name="...")
        elif isinstance(cst_node, cst.Attribute):
            if not node.names:
                raise NotImplementedError("attributes without full names")
            names = sorted(node.names)
            # TODO: While accurate, this doesn't match the exact text from the
            #       source file.
            self.type.name = python.Name(names[0], names[0])
        elif isinstance(cst_node, cst.Subscript):
            arguments = []
            generics = python.Generics(arguments=arguments)
            self.type.generics = generics
            type_ = python.Type()
            value = node.find_child(cst_node.value)
            value.visit(_TypeVisitor(value, type_))
            self.type.name = type_

            for cst_element in cst_node.slice:
                # TODO: This traversal assumes no new nodes were added between
                #       the Subscript and the Index's contents.
                assert isinstance(cst_element, cst.SubscriptElement)
                element = node.find_child(cst_element)

                cst_index = cst_element.slice
                assert isinstance(cst_index, cst.Index)

                assert isinstance(element, CstNode)
                index = element.find_child(cst_index)
                generic = python.Type()

                cst_index_value = cst_index.value
                assert cst_index.star is None
                assert isinstance(index, CstNode)
                index_value = index.find_child(cst_index_value)

                index_value.visit(_TypeVisitor(index_value, generic))

                arguments.append(generic)
        else:
            raise Exception(str(node))

        return Visit.SkipChildren

    def exit(self, node: Node) -> None:
        pass


class _AnnotationTransformVisitor(Visitor):
    stack: Final[List[Node]]

    def __init__(self) -> None:
        self.stack = []

    def enter(self, node: Node) -> Visit:
        self.stack.append(node)

        if not isinstance(node, CstNode):
            return Visit.TraverseChildren

        cst_node = node.cst_node

        if not isinstance(cst_node, cst.Annotation):
            return Visit.TraverseChildren

        type_ = python.Type()

        self.stack[-2].replace_child(node, type_)
        self.stack[-1] = type_

        annotation = node.find_child(cst_node.annotation)

        annotation.visit(_TypeVisitor(annotation, type_))

        return Visit.SkipChildren

    def exit(self, node: Node) -> None:
        self.stack.pop()


class _NameTransformVisitor(Visitor):
    stack: Final[List[Node]]

    def __init__(self) -> None:
        self.stack = []

    def enter(self, node: Node) -> Visit:
        if isinstance(node, CstNode):
            cst_node = node.cst_node
            if isinstance(cst_node, (cst.Name, cst.SimpleString)):
                names = sorted(node.names)
                try:
                    full_name = names[0]
                except IndexError:
                    full_name = None

                name = python.Name(cst_node.value, full_name)
                self.stack[-1].replace_child(node, name)

        self.stack.append(node)
        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        self.stack.pop()


class _VerbatimTransform(Visitor):
    root: Optional[Node]
    stack: Final[List[Node]]

    @staticmethod
    def apply(source: TextSource, node: Node) -> Node:
        transform = _VerbatimTransform()

        node.visit(transform)

        verbatim = Verbatim()
        stanza = Stanza(source)
        assert transform.root is not None
        stanza.append(transform.root)
        verbatim.append(stanza)
        return verbatim

    def __init__(self) -> None:
        self.stack = []
        self.root = None

    def enter(self, node: Node) -> Visit:
        if self.root is None:
            assert 0 == len(self.stack)
            self.root = node
        else:
            assert 0 < len(self.stack)

        self.stack.append(node)
        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        popped = self.stack.pop()
        assert popped == node

        if not isinstance(node, CstNode):
            return

        name = dasherize(underscore(node.cst_node.__class__.__name__))

        new = Fragment(
            start=node.start,
            end=node.end,
            highlights=[name],
        )

        for child in node.children:
            new.append(child)

        if self.stack:
            self.stack[-1].replace_child(node, new)
        else:
            assert self.root == node
            self.root = new
