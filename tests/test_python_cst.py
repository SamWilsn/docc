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
from typing import Dict, Mapping, Set

import pytest

from docc.document import BlankNode, Document, ListNode
from docc.plugins.python import nodes
from docc.plugins.python.cst import (
    PythonBuilder,
    PythonDiscover,
    PythonSource,
)
from docc.settings import Settings
from docc.source import Source


@pytest.fixture
def settings_with_paths(tmp_path: Path) -> Settings:
    settings_dict: Dict[str, object] = {
        "tool": {
            "docc": {
                "plugins": {
                    "docc.python.discover": {"paths": [str(tmp_path)]},
                }
            }
        }
    }
    return Settings(tmp_path, settings_dict)


class TestPythonDiscover:
    def test_init_raises_on_non_sequence_paths(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {"docc.python.discover": {"paths": 123}}
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.python.discover")

        with pytest.raises(TypeError, match="paths must be a list"):
            PythonDiscover(plugin_settings)

    def test_init_raises_on_non_string_path(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {"docc.python.discover": {"paths": [123]}}
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.python.discover")

        with pytest.raises(
            TypeError, match="every python path must be a string"
        ):
            PythonDiscover(plugin_settings)

    def test_init_raises_on_empty_paths(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {"docc.python.discover": {"paths": []}}
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.python.discover")

        with pytest.raises(ValueError, match="python needs at least one path"):
            PythonDiscover(plugin_settings)

    def test_discover_finds_python_files(self, tmp_path: Path) -> None:
        (tmp_path / "test.py").write_text("# test")

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]}
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.python.discover")
        discover = PythonDiscover(plugin_settings)

        sources = list(discover.discover(frozenset()))
        assert len(sources) == 1
        assert isinstance(sources[0], PythonSource)

    def test_discover_finds_nested_python_files(self, tmp_path: Path) -> None:
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "nested.py").write_text("# nested")

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]}
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.python.discover")
        discover = PythonDiscover(plugin_settings)

        sources = list(discover.discover(frozenset()))
        assert len(sources) == 1
        assert "nested.py" in str(sources[0].relative_path)

    def test_excluded_paths(self, tmp_path: Path) -> None:
        subdir = tmp_path / "exclude_me"
        subdir.mkdir()
        (subdir / "test.py").write_text("# excluded")
        (tmp_path / "keep.py").write_text("# keep")

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {
                                "paths": [str(tmp_path)],
                                "excluded_paths": ["exclude_me"],
                            }
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.python.discover")
        discover = PythonDiscover(plugin_settings)

        sources = list(discover.discover(frozenset()))
        assert len(sources) == 1
        assert "keep.py" in str(sources[0].relative_path)

    def test_excluded_paths_non_sequence_raises(self, tmp_path: Path) -> None:
        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {
                                "paths": [str(tmp_path)],
                                "excluded_paths": 123,
                            }
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.python.discover")

        with pytest.raises(TypeError, match="excluded paths must be a list"):
            PythonDiscover(plugin_settings)


class TestPythonSource:
    def test_relative_path_property(self, tmp_path: Path) -> None:
        relative = PurePath("test.py")
        absolute = tmp_path / "test.py"
        absolute.write_text("# test")

        source = PythonSource(tmp_path, relative, absolute)
        assert source.relative_path == relative

    def test_output_path_property(self, tmp_path: Path) -> None:
        relative = PurePath("subdir") / "test.py"
        absolute = tmp_path / "subdir" / "test.py"
        absolute.parent.mkdir(exist_ok=True)
        absolute.write_text("# test")

        source = PythonSource(tmp_path, relative, absolute)
        assert source.output_path == relative

    def test_open_returns_file_handle(self, tmp_path: Path) -> None:
        content = "# test content\nx = 1"
        relative = PurePath("test.py")
        absolute = tmp_path / "test.py"
        absolute.write_text(content)

        source = PythonSource(tmp_path, relative, absolute)
        with source.open() as f:
            assert f.read() == content


class TestPythonBuilder:
    def test_build_simple_module(self, tmp_path: Path) -> None:
        content = '''"""Module docstring."""
x = 1
'''
        (tmp_path / "test.py").write_text(content)

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]}
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.python.discover")
        discover = PythonDiscover(plugin_settings)
        sources = set(discover.discover(frozenset()))

        builder = PythonBuilder(plugin_settings)
        documents: Dict[Source, Document] = {}
        builder.build(sources, documents)

        assert len(documents) == 1

    def test_build_removes_sources_from_unprocessed(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "test.py").write_text("x = 1")

        settings = Settings(
            tmp_path,
            {
                "tool": {
                    "docc": {
                        "plugins": {
                            "docc.python.discover": {"paths": [str(tmp_path)]}
                        }
                    }
                }
            },
        )
        plugin_settings = settings.for_plugin("docc.python.discover")
        discover = PythonDiscover(plugin_settings)
        sources: Set[Source] = set(discover.discover(frozenset()))
        original_count = len(sources)

        builder = PythonBuilder(plugin_settings)
        documents: Dict[Source, Document] = {}
        builder.build(sources, documents)

        assert len(sources) == 0
        assert len(documents) == original_count


class TestPythonNodes:
    def test_module_default_fields(self) -> None:
        module = nodes.Module()
        assert isinstance(module.name, BlankNode)
        assert isinstance(module.docstring, BlankNode)
        assert isinstance(module.members, ListNode)

    def test_module_to_search(self) -> None:
        module = nodes.Module()
        module.name = nodes.Name("test_module")
        result = module.to_search()
        assert isinstance(result, Mapping)
        assert result["type"] == "module"
        assert "test_module" in result["name"]

    def test_class_default_fields(self) -> None:
        cls = nodes.Class()
        assert isinstance(cls.decorators, ListNode)
        assert isinstance(cls.name, BlankNode)
        assert isinstance(cls.bases, ListNode)
        assert isinstance(cls.metaclass, BlankNode)
        assert isinstance(cls.docstring, BlankNode)
        assert isinstance(cls.members, ListNode)

    def test_class_to_search(self) -> None:
        cls = nodes.Class()
        cls.name = nodes.Name("TestClass")
        result = cls.to_search()
        assert isinstance(result, Mapping)
        assert result["type"] == "class"
        assert "TestClass" in result["name"]

    def test_function_default_fields(self) -> None:
        func = nodes.Function(asynchronous=False)
        assert func.asynchronous is False
        assert isinstance(func.decorators, ListNode)
        assert isinstance(func.name, BlankNode)
        assert isinstance(func.parameters, ListNode)
        assert isinstance(func.return_type, BlankNode)
        assert isinstance(func.docstring, BlankNode)
        assert isinstance(func.body, BlankNode)

    def test_function_async(self) -> None:
        func = nodes.Function(asynchronous=True)
        assert func.asynchronous is True

    def test_function_to_search(self) -> None:
        func = nodes.Function(asynchronous=False)
        func.name = nodes.Name("test_func")
        result = func.to_search()
        assert isinstance(result, Mapping)
        assert result["type"] == "function"
        assert "test_func" in result["name"]

    def test_parameter(self) -> None:
        param = nodes.Parameter()
        assert param.star is None
        assert isinstance(param.name, BlankNode)
        assert isinstance(param.type_annotation, BlankNode)

    def test_parameter_with_star(self) -> None:
        param = nodes.Parameter(star="*")
        assert param.star == "*"

        double_star_param = nodes.Parameter(star="**")
        assert double_star_param.star == "**"

    def test_attribute_to_search(self) -> None:
        attr = nodes.Attribute()
        attr.names = ListNode([nodes.Name("test_attr")])
        result = attr.to_search()
        assert isinstance(result, Mapping)
        assert result["type"] == "attribute"
        assert "test_attr" in result["name"]

    def test_name_children_empty(self) -> None:
        name = nodes.Name("test")
        assert tuple(name.children) == ()

    def test_name_replace_child_raises(self) -> None:
        name = nodes.Name("test")
        with pytest.raises(TypeError):
            name.replace_child(BlankNode(), BlankNode())

    def test_name_with_full_name(self) -> None:
        name = nodes.Name("test", "module.test")
        assert name.name == "test"
        assert name.full_name == "module.test"

    def test_docstring_children_empty(self) -> None:
        doc = nodes.Docstring("test docstring")
        assert tuple(doc.children) == ()

    def test_docstring_replace_child_raises(self) -> None:
        doc = nodes.Docstring("test")
        with pytest.raises(TypeError):
            doc.replace_child(BlankNode(), BlankNode())

    def test_docstring_to_search(self) -> None:
        doc = nodes.Docstring("This is documentation")
        assert doc.to_search() == "This is documentation"

    def test_type_node(self) -> None:
        type_node = nodes.Type()
        assert isinstance(type_node.child, BlankNode)

    def test_subscript_node(self) -> None:
        sub = nodes.Subscript()
        assert isinstance(sub.name, BlankNode)
        assert isinstance(sub.generics, BlankNode)

    def test_binary_operation(self) -> None:
        binop = nodes.BinaryOperation()
        assert isinstance(binop.left, BlankNode)
        assert isinstance(binop.operator, BlankNode)
        assert isinstance(binop.right, BlankNode)

    def test_bit_or(self) -> None:
        bit_or = nodes.BitOr()
        children = list(bit_or.children)
        assert len(children) == 0

    def test_list_node(self) -> None:
        list_node = nodes.List()
        assert isinstance(list_node.elements, ListNode)

    def test_tuple_node(self) -> None:
        tuple_node = nodes.Tuple()
        assert isinstance(tuple_node.elements, ListNode)

    def test_access_node(self) -> None:
        access = nodes.Access()
        assert isinstance(access.value, BlankNode)
        assert isinstance(access.attribute, BlankNode)


class TestPythonNodeRepr:
    def test_module_repr(self) -> None:
        module = nodes.Module()
        assert repr(module) == "Module(...)"

    def test_class_repr(self) -> None:
        cls = nodes.Class()
        assert repr(cls) == "Class(...)"

    def test_function_repr(self) -> None:
        func = nodes.Function(asynchronous=False)
        assert repr(func) == "Function(...)"


class TestPythonNodeReplaceChild:
    def test_replace_child_in_module(self) -> None:
        old_name = nodes.Name("old")
        new_name = nodes.Name("new")
        module = nodes.Module()
        module.name = old_name

        module.replace_child(old_name, new_name)
        assert module.name == new_name

    def test_replace_child_not_found(self) -> None:
        module = nodes.Module()
        original_name = module.name
        original_docstring = module.docstring
        original_members = module.members

        old = nodes.Name("old")
        new = nodes.Name("new")

        module.replace_child(old, new)

        # Verify original field values are unchanged after no-op replace
        assert module.name is original_name
        assert module.docstring is original_docstring
        assert module.members is original_members


class TestNameVisitor:
    def test_collect_single_name(self) -> None:
        name = nodes.Name("test")
        result = nodes._NameVisitor.collect(name)
        assert result == ["test"]

    def test_collect_multiple_names(self) -> None:
        names = [nodes.Name("a"), nodes.Name("b"), nodes.Name("c")]
        result = nodes._NameVisitor.collect(names)
        assert result == ["a", "b", "c"]

    def test_collect_from_list_node(self) -> None:
        list_node = ListNode([nodes.Name("x"), nodes.Name("y")])
        result = nodes._NameVisitor.collect(list_node)
        assert result == ["x", "y"]

    def test_collect_empty(self) -> None:
        blank = BlankNode()
        result = nodes._NameVisitor.collect(blank)
        assert result == []


def test_python_node_children_type_error() -> None:
    """
    PythonNode.children raises TypeError when a field annotated
    as Node contains a non-Node value.
    """
    module = nodes.Module()
    # Forcefully set a Node-typed field to a non-Node value
    object.__setattr__(module, "name", "not a node")

    with pytest.raises(TypeError, match="child not Node"):
        list(module.children)
