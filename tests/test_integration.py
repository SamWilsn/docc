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
from typing import Dict, List, Optional, Set, Tuple, Type

import libcst
import pytest

from docc.context import Context
from docc.document import Document, ListNode, Node, Visit, Visitor
from docc.plugins.mistletoe import (
    DocstringTransform,
    MarkdownNode,
    ReferenceTransform,
)
from docc.plugins.python import nodes
from docc.plugins.python.cst import (
    PythonBuilder,
    PythonDiscover,
    PythonTransform,
)
from docc.plugins.references import (
    Definition,
    Index,
    IndexTransform,
    Reference,
)
from docc.settings import Settings
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


class NodeCollector(Visitor):
    """Visitor that collects all nodes in the tree for easy assertion."""

    def __init__(self) -> None:
        self.all_nodes: List[Node] = []
        self.by_type: Dict[Type[Node], List[Node]] = {}

    def enter(self, node: Node) -> Visit:
        self.all_nodes.append(node)
        node_type = type(node)
        if node_type not in self.by_type:
            self.by_type[node_type] = []
        self.by_type[node_type].append(node)
        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        pass

    def get(self, node_type: Type[Node]) -> List[Node]:
        return self.by_type.get(node_type, [])

    def get_names(self, node_type: Type[Node]) -> List[str]:
        """Extract name strings from nodes that have a name attribute."""
        result = []
        for node in self.get(node_type):
            name_node = getattr(node, "name", None)
            if isinstance(name_node, nodes.Name):
                result.append(name_node.name)
        return result


def _run_pipeline(
    tmp_path: Path,
    py_content: str,
    filename: str = "example.py",
) -> Tuple[Document, Source, Context]:
    """Helper: write source, discover, build, and transform."""
    (tmp_path / filename).write_text(py_content)

    settings = Settings(
        tmp_path,
        {
            "tool": {
                "docc": {
                    "plugins": {
                        "docc.python.discover": {"paths": [str(tmp_path)]},
                        "docc.python.transform": {},
                    }
                }
            }
        },
    )

    discover_settings = settings.for_plugin("docc.python.discover")
    discover = PythonDiscover(discover_settings)
    sources: Set[Source] = set(discover.discover(frozenset()))

    builder = PythonBuilder(discover_settings)
    documents: Dict[Source, Document] = {}
    builder.build(sources, documents)

    document = list(documents.values())[0]
    source = list(documents.keys())[0]
    index = Index()

    context = Context({Document: document, Source: source, Index: index})

    transform_settings = settings.for_plugin("docc.python.transform")
    transform = PythonTransform(transform_settings)
    transform.transform(context)

    return document, source, context


class TestPythonPipeline:
    def test_discover_build_transform_pipeline(self, tmp_path: Path) -> None:
        py_content = '''"""Module docstring."""

class MyClass:
    """A test class."""

    def method(self, x: int) -> str:
        """A method."""
        return str(x)


def standalone_func(arg: str) -> None:
    """Standalone function."""
    pass
'''
        (tmp_path / "example.py").write_text(py_content)

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]},
                            "docc.python.transform": {},
                        }
                    }
                }
            },
        )

        discover_settings = settings.for_plugin("docc.python.discover")
        discover = PythonDiscover(discover_settings)
        sources: Set[Source] = set(discover.discover(frozenset()))
        assert len(sources) == 1

        builder = PythonBuilder(discover_settings)
        documents: Dict[Source, Document] = {}
        builder.build(sources, documents)
        assert len(documents) == 1

        document = list(documents.values())[0]
        source = list(documents.keys())[0]
        index = Index()

        context = Context({Document: document, Source: source, Index: index})

        transform_settings = settings.for_plugin("docc.python.transform")
        transform = PythonTransform(transform_settings)
        transform.transform(context)

        # Verify the tree contains Module, Class, Function nodes
        class NodeTypeChecker(Visitor):
            found_module = False
            found_class = False
            found_function = False

            def enter(self, node: Node) -> Visit:
                if isinstance(node, nodes.Module):
                    self.found_module = True
                elif isinstance(node, nodes.Class):
                    self.found_class = True
                elif isinstance(node, nodes.Function):
                    self.found_function = True
                return Visit.TraverseChildren

            def exit(self, node: Node) -> None:
                pass

        checker = NodeTypeChecker()
        document.root.visit(checker)
        assert checker.found_module, "Should contain a Module node"
        assert checker.found_class, "Should contain a Class node"
        assert checker.found_function, "Should contain a Function"

        # Strengthened assertions: verify names and structure
        collector = NodeCollector()
        document.root.visit(collector)

        # Module should have a name derived from the filename
        modules = collector.get(nodes.Module)
        assert len(modules) == 1
        module = modules[0]
        assert isinstance(module, nodes.Module)
        assert isinstance(module.name, nodes.Name)

        # Class should be named "MyClass"
        class_names = collector.get_names(nodes.Class)
        assert (
            "MyClass" in class_names
        ), f"Expected 'MyClass' in class names, got {class_names}"

        # Functions should include "method" and "standalone_func"
        func_names = collector.get_names(nodes.Function)
        assert (
            "method" in func_names
        ), f"Expected 'method' in function names, got {func_names}"
        assert (
            "standalone_func" in func_names
        ), f"Expected 'standalone_func' in function names, got {func_names}"

        # The "method" function should have parameters
        for func_node in collector.get(nodes.Function):
            assert isinstance(func_node, nodes.Function)
            if (
                isinstance(func_node.name, nodes.Name)
                and func_node.name.name == "method"
            ):
                assert isinstance(func_node.parameters, ListNode)
                params = list(func_node.parameters.children)
                assert len(params) > 0, "method should have parameters"
                break
        else:
            raise AssertionError("Function 'method' not found")

    def test_python_with_docstring_transform(self, tmp_path: Path) -> None:
        py_content = '''"""
Module with **markdown** docstring.
"""

def func():
    """Function with *emphasis*."""
    pass
'''
        (tmp_path / "markdown_example.py").write_text(py_content)

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]},
                        }
                    }
                }
            },
        )

        discover_settings = settings.for_plugin("docc.python.discover")
        discover = PythonDiscover(discover_settings)
        sources: Set[Source] = set(discover.discover(frozenset()))

        builder = PythonBuilder(discover_settings)
        documents: Dict[Source, Document] = {}
        builder.build(sources, documents)

        document = list(documents.values())[0]
        source = list(documents.keys())[0]
        index = Index()

        context = Context({Document: document, Source: source, Index: index})

        transform = PythonTransform(
            settings.for_plugin("docc.python.transform")
        )
        transform.transform(context)

        docstring_transform = DocstringTransform(
            settings.for_plugin("docc.mistletoe.transform")
        )
        docstring_transform.transform(context)

        # Verify docstrings were converted to MarkdownNode
        class MarkdownNodeChecker(Visitor):
            found_markdown = False

            def enter(self, node: Node) -> Visit:
                if isinstance(node, MarkdownNode):
                    self.found_markdown = True
                return Visit.TraverseChildren

            def exit(self, node: Node) -> None:
                pass

        checker = MarkdownNodeChecker()
        document.root.visit(checker)
        assert (
            checker.found_markdown
        ), "Docstrings should be converted to MarkdownNode"

        # Strengthened assertions: verify markdown content
        collector = NodeCollector()
        document.root.visit(collector)
        markdown_nodes = collector.get(MarkdownNode)
        assert (
            len(markdown_nodes) >= 1
        ), "Should have at least one MarkdownNode"

        # At least one markdown node should contain searchable text
        # from the module or function docstring
        all_search_text = []
        for md_node in markdown_nodes:
            assert isinstance(md_node, MarkdownNode)
            search_content = md_node.to_search()
            if isinstance(search_content, str):
                all_search_text.append(search_content)
        combined_text = " ".join(all_search_text)
        assert (
            "markdown" in combined_text.lower()
        ), f"Expected 'markdown' in search text, got: {combined_text}"
        assert (
            "emphasis" in combined_text.lower()
        ), f"Expected 'emphasis' in search text, got: {combined_text}"


def test_pipeline_python_to_html_references(tmp_path: Path) -> None:
    py_content = '''"""
Module with references.

See [other](ref:module.other) for more.
"""

def other():
    """Another function."""
    pass
'''
    (tmp_path / "references.py").write_text(py_content)

    settings = Settings(
        tmp_path,
        {
            "tool": {
                "docc": {
                    "plugins": {
                        "docc.python.discover": {"paths": [str(tmp_path)]},
                    }
                }
            }
        },
    )

    discover = PythonDiscover(settings.for_plugin("docc.python.discover"))
    sources: Set[Source] = set(discover.discover(frozenset()))

    builder = PythonBuilder(settings.for_plugin("docc.python.discover"))
    documents: Dict[Source, Document] = {}
    builder.build(sources, documents)

    document = list(documents.values())[0]
    source = list(documents.keys())[0]
    index = Index()

    context = Context({Document: document, Source: source, Index: index})

    PythonTransform(settings.for_plugin("docc.python.transform")).transform(
        context
    )

    DocstringTransform(
        settings.for_plugin("docc.mistletoe.transform")
    ).transform(context)

    ReferenceTransform(
        settings.for_plugin("docc.mistletoe.reference")
    ).transform(context)

    IndexTransform(settings.for_plugin("docc.references.index")).transform(
        context
    )

    # Verify the Reference was found and Index has the definition
    class ReferenceChecker(Visitor):
        found_reference = False

        def enter(self, node: Node) -> Visit:
            if isinstance(node, Reference):
                self.found_reference = True
                assert node.identifier == "module.other"
            return Visit.TraverseChildren

        def exit(self, node: Node) -> None:
            pass

    checker = ReferenceChecker()
    document.root.visit(checker)
    assert checker.found_reference, "Reference to module.other should be found"

    # Verify function was indexed
    locations = list(index.lookup("references.other"))
    assert (
        len(locations) == 1
    ), "Function 'other' should be indexed exactly once"


class TestEdgeCases:
    def test_empty_python_file(self, tmp_path: Path) -> None:
        (tmp_path / "empty.py").write_text("")

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]},
                        }
                    }
                }
            },
        )

        discover = PythonDiscover(settings.for_plugin("docc.python.discover"))
        sources: Set[Source] = set(discover.discover(frozenset()))

        builder = PythonBuilder(settings.for_plugin("docc.python.discover"))
        documents: Dict[Source, Document] = {}
        builder.build(sources, documents)

        assert len(documents) == 1
        # Verify the document was created and can be traversed
        document = list(documents.values())[0]
        source = list(documents.keys())[0]
        assert (
            len(list(document.root.children)) >= 0
        ), "Root should be traversable"

        # Transform the empty file to get the output node types
        index = Index()
        context = Context({Document: document, Source: source, Index: index})
        transform = PythonTransform(
            settings.for_plugin("docc.python.transform")
        )
        transform.transform(context)

        # After transform, should have a Module node
        collector = NodeCollector()
        document.root.visit(collector)
        modules = collector.get(nodes.Module)
        assert (
            len(modules) == 1
        ), "Empty file should produce exactly one Module"

        module = modules[0]
        assert isinstance(module, nodes.Module)

        # Empty module should have no class/function/attribute members
        assert (
            len(collector.get(nodes.Class)) == 0
        ), "Empty module should have no classes"
        assert (
            len(collector.get(nodes.Function)) == 0
        ), "Empty module should have no functions"
        assert (
            len(collector.get(nodes.Attribute)) == 0
        ), "Empty module should have no attributes"
        assert (
            len(collector.get(nodes.Docstring)) == 0
        ), "Empty module should have no docstrings"

    def test_python_with_syntax_error_handled(self, tmp_path: Path) -> None:
        (tmp_path / "syntax_error.py").write_text("def broken(\n")

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]},
                        }
                    }
                }
            },
        )

        discover = PythonDiscover(settings.for_plugin("docc.python.discover"))
        sources: Set[Source] = set(discover.discover(frozenset()))

        builder = PythonBuilder(settings.for_plugin("docc.python.discover"))
        documents: Dict[Source, Document] = {}

        with pytest.raises(libcst.ParserSyntaxError):
            builder.build(sources, documents)

    def test_nested_classes(self, tmp_path: Path) -> None:
        py_content = '''"""Module."""

class Outer:
    """Outer class."""

    class Inner:
        """Inner class."""

        def method(self) -> None:
            """Inner method."""
            pass
'''
        (tmp_path / "nested.py").write_text(py_content)

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]},
                        }
                    }
                }
            },
        )

        discover = PythonDiscover(settings.for_plugin("docc.python.discover"))
        sources: Set[Source] = set(discover.discover(frozenset()))

        builder = PythonBuilder(settings.for_plugin("docc.python.discover"))
        documents: Dict[Source, Document] = {}
        builder.build(sources, documents)

        assert len(documents) == 1

        # Verify nested class structure was captured
        document = list(documents.values())[0]
        import libcst

        from docc.plugins.python.cst import CstNode

        class NestedClassChecker(Visitor):
            found_outer = False
            found_inner = False
            found_method = False

            def enter(self, node: Node) -> Visit:
                if isinstance(node, CstNode):
                    cst = node.cst_node
                    if isinstance(cst, libcst.ClassDef):
                        if cst.name.value == "Outer":
                            self.found_outer = True
                        elif cst.name.value == "Inner":
                            self.found_inner = True
                    elif isinstance(cst, libcst.FunctionDef):
                        if cst.name.value == "method":
                            self.found_method = True
                return Visit.TraverseChildren

            def exit(self, node: Node) -> None:
                pass

        checker = NestedClassChecker()
        document.root.visit(checker)

        assert checker.found_outer, "Outer class should be found"
        assert checker.found_inner, "Inner class should be found"
        assert checker.found_method, "Inner method should be found"


class TestPythonTransformContract:
    """
    Behavioral-level tests for the Python source -> document tree pipeline.

    These tests assert on the OUTPUT contract (document tree structure using
    node types from docc.plugins.python.nodes), NOT on CST internals. They
    should survive a CST -> AST migration unchanged.
    """

    def test_module_with_class_and_function(self, tmp_path: Path) -> None:
        """Verify structural output for a module with class and function."""
        py_content = '''"""Module docstring."""

class MyClass:
    """A test class."""

    def method(self, x: int) -> str:
        """A method."""
        return str(x)

def standalone_func(arg: str) -> None:
    """Standalone function."""
    pass
'''
        document, source, context = _run_pipeline(tmp_path, py_content)

        collector = NodeCollector()
        document.root.visit(collector)

        # The document tree root should contain a Definition wrapping a Module
        definitions = collector.get(Definition)
        module_definitions = [
            d
            for d in definitions
            if isinstance(d, Definition) and isinstance(d.child, nodes.Module)
        ]
        assert (
            len(module_definitions) >= 1
        ), "Should have a Definition wrapping a nodes.Module"

        # There should be exactly one Module
        modules = collector.get(nodes.Module)
        assert len(modules) == 1, f"Expected 1 Module, got {len(modules)}"
        module = modules[0]
        assert isinstance(module, nodes.Module)

        # The Module should have the module docstring
        docstrings = collector.get(nodes.Docstring)
        docstring_texts = [
            d.text for d in docstrings if isinstance(d, nodes.Docstring)
        ]
        assert any(
            "Module docstring" in t for t in docstring_texts
        ), f"Expected module docstring, got: {docstring_texts}"

        # The Module members should include a Definition wrapping a Class
        # named "MyClass"
        classes = collector.get(nodes.Class)
        assert len(classes) >= 1, "Should have at least one Class node"
        class_names = [
            c.name.name
            for c in classes
            if isinstance(c, nodes.Class) and isinstance(c.name, nodes.Name)
        ]
        assert (
            "MyClass" in class_names
        ), f"Expected class 'MyClass', got: {class_names}"

        # MyClass should be wrapped in a Definition
        class_definitions = [
            d
            for d in definitions
            if isinstance(d, Definition) and isinstance(d.child, nodes.Class)
        ]
        assert (
            len(class_definitions) >= 1
        ), "Class should be wrapped in a Definition"

        # The Class should have members including a Function named "method"
        functions = collector.get(nodes.Function)
        func_names = [
            f.name.name
            for f in functions
            if isinstance(f, nodes.Function) and isinstance(f.name, nodes.Name)
        ]
        assert (
            "method" in func_names
        ), f"Expected function 'method', got: {func_names}"
        assert (
            "standalone_func" in func_names
        ), f"Expected function 'standalone_func', got: {func_names}"

        # The method Function should be wrapped in a Definition
        func_definitions = [
            d
            for d in definitions
            if isinstance(d, Definition)
            and isinstance(d.child, nodes.Function)
        ]
        assert (
            len(func_definitions) >= 1
        ), "Functions should be wrapped in Definitions"

        # Verify docstrings exist for class and functions
        assert any(
            "A test class" in t for t in docstring_texts
        ), f"Expected class docstring, got: {docstring_texts}"
        assert any(
            "A method" in t for t in docstring_texts
        ), f"Expected method docstring, got: {docstring_texts}"
        assert any(
            "Standalone function" in t for t in docstring_texts
        ), f"Expected standalone_func docstring, got: {docstring_texts}"

    def test_class_with_attributes(self, tmp_path: Path) -> None:
        """Verify that annotated class attributes produce Attribute nodes."""
        py_content = '''"""Module."""

class Config:
    """Configuration class."""

    timeout: int
    """Timeout in seconds."""

    name: str
'''
        document, source, context = _run_pipeline(
            tmp_path, py_content, filename="config.py"
        )

        collector = NodeCollector()
        document.root.visit(collector)

        # Class "Config" should exist
        classes = collector.get(nodes.Class)
        config_classes = [
            c
            for c in classes
            if isinstance(c, nodes.Class)
            and isinstance(c.name, nodes.Name)
            and c.name.name == "Config"
        ]
        assert (
            len(config_classes) == 1
        ), "Should have exactly one class named 'Config'"
        config_class = config_classes[0]
        assert isinstance(config_class, nodes.Class)

        # Config should have members
        assert isinstance(config_class.members, ListNode)
        member_list = list(config_class.members.children)
        assert len(member_list) >= 1, "Config should have members"

        # At least one member should be (or wrap) a nodes.Attribute
        attributes = collector.get(nodes.Attribute)
        assert len(attributes) >= 1, "Should have at least one Attribute node"

        # The attribute with docstring ("timeout") should have a Docstring
        attr_with_docstring = None
        for attr in attributes:
            assert isinstance(attr, nodes.Attribute)
            if isinstance(attr.docstring, nodes.Docstring):
                attr_with_docstring = attr
                break

        assert (
            attr_with_docstring is not None
        ), "At least one attribute should have a Docstring child"
        assert isinstance(attr_with_docstring.docstring, nodes.Docstring)
        assert "Timeout in seconds" in attr_with_docstring.docstring.text, (
            f"Expected 'Timeout in seconds' in docstring, "
            f"got: {attr_with_docstring.docstring.text}"
        )
