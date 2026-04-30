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

from io import StringIO
from typing import ClassVar, List, Literal, Tuple

import pytest
from typing_extensions import override

from docc.document import (
    BlankNode,
    Document,
    ListNode,
    Node,
    OutputNode,
    Visit,
    Visitor,
    _ExtensionVisitor,
    _StrVisitor,
)

Event = Tuple[Literal["enter", "exit"], int]


@pytest.mark.parametrize(
    "node, expected_repr",
    [
        (BlankNode(), "<blank>"),
        (ListNode(), "<list>"),
    ],
    ids=["BlankNode", "ListNode"],
)
def test_node_children_iterable(node: Node, expected_repr: str) -> None:
    assert hasattr(node.children, "__iter__")


@pytest.mark.parametrize(
    "node, expected_repr",
    [
        (BlankNode(), "<blank>"),
        (ListNode(), "<list>"),
    ],
    ids=["BlankNode", "ListNode"],
)
def test_node_repr(node: Node, expected_repr: str) -> None:
    assert repr(node) == expected_repr


class TestBlankNode:
    def test_children_returns_empty_tuple(self) -> None:
        node = BlankNode()
        assert tuple(node.children) == ()

    def test_replace_child_raises_type_error(self) -> None:
        node = BlankNode()
        with pytest.raises(TypeError):
            node.replace_child(BlankNode(), BlankNode())

    def test_bool_is_false(self) -> None:
        node = BlankNode()
        assert bool(node) is False


class TestListNode:
    def test_children_property_returns_list(self) -> None:
        first_child, second_child = BlankNode(), BlankNode()
        node = ListNode([first_child, second_child])
        assert list(node.children) == [first_child, second_child]

    def test_default_children_is_empty(self) -> None:
        node = ListNode()
        assert list(node.children) == []

    def test_replace_child(self) -> None:
        old_child = BlankNode()
        new_child = BlankNode()
        other_child = BlankNode()
        node = ListNode([old_child, other_child])

        node.replace_child(old_child, new_child)

        assert list(node.children) == [new_child, other_child]

    def test_replace_child_when_not_found(self) -> None:
        old_child = BlankNode()
        new_child = BlankNode()
        other_child = BlankNode()
        node = ListNode([other_child])

        node.replace_child(old_child, new_child)
        assert list(node.children) == [other_child]

    def test_bool_true_when_has_children(self) -> None:
        node = ListNode([BlankNode()])
        assert bool(node) is True

    def test_bool_false_when_empty(self) -> None:
        node = ListNode()
        assert bool(node) is False

    def test_len(self) -> None:
        node = ListNode([BlankNode(), BlankNode(), BlankNode()])
        assert len(node) == 3


class RecordingVisitor(Visitor):
    returns: ClassVar[Visit] = Visit.TraverseChildren
    events: List[Event]

    def __init__(self) -> None:
        self.events = []

    @override
    def enter(self, node: Node) -> Visit:
        self.events.append(("enter", id(node)))
        return self.returns

    @override
    def exit(self, node: Node) -> None:
        self.events.append(("exit", id(node)))


class SkippingVisitor(RecordingVisitor):
    returns: ClassVar[Visit] = Visit.SkipChildren


class TestNodeVisit:
    def test_visit_single_node(self) -> None:
        node = BlankNode()
        visitor = RecordingVisitor()
        node.visit(visitor)

        assert visitor.events == [
            ("enter", id(node)),
            ("exit", id(node)),
        ]

    def test_visit_with_children(self) -> None:
        first_child = BlankNode()
        second_child = BlankNode()
        parent = ListNode([first_child, second_child])

        visitor = RecordingVisitor()
        parent.visit(visitor)

        assert visitor.events == [
            ("enter", id(parent)),
            ("enter", id(first_child)),
            ("exit", id(first_child)),
            ("enter", id(second_child)),
            ("exit", id(second_child)),
            ("exit", id(parent)),
        ]

    def test_visit_skip_children(self) -> None:
        child = BlankNode()
        parent = ListNode([child])

        visitor = SkippingVisitor()
        parent.visit(visitor)

        assert visitor.events == [
            ("enter", id(parent)),
            ("exit", id(parent)),
        ]

    def test_visit_nested_structure(self) -> None:
        leaf = BlankNode()
        inner = ListNode([leaf])
        outer = ListNode([inner])

        visitor = RecordingVisitor()
        outer.visit(visitor)

        assert visitor.events == [
            ("enter", id(outer)),
            ("enter", id(inner)),
            ("enter", id(leaf)),
            ("exit", id(leaf)),
            ("exit", id(inner)),
            ("exit", id(outer)),
        ]

    def test_visit_depth_first(self) -> None:
        first_leaf = BlankNode()
        second_leaf = BlankNode()
        third_leaf = BlankNode()
        first_branch = ListNode([first_leaf, second_leaf])
        second_branch = ListNode([third_leaf])
        root = ListNode([first_branch, second_branch])

        visitor = RecordingVisitor()
        root.visit(visitor)

        enter_events = [e for e in visitor.events if e[0] == "enter"]
        assert enter_events[0] == ("enter", id(root))
        assert enter_events[1] == ("enter", id(first_branch))
        assert enter_events[2] == ("enter", id(first_leaf))


class TestNodeDump:
    def test_dump_to_stringio(self) -> None:
        node = BlankNode()
        output = StringIO()
        node.dump(file=output)
        result = output.getvalue()
        assert "<blank>" in result

    def test_dumps_returns_string(self) -> None:
        node = BlankNode()
        result = node.dumps()
        assert isinstance(result, str)
        assert "<blank>" in result

    def test_dump_nested_structure(self) -> None:
        child = BlankNode()
        parent = ListNode([child])

        result = parent.dumps()
        assert "<list>" in result
        assert "<blank>" in result


class TestStrVisitor:
    def test_builds_rich_tree(self) -> None:
        node = BlankNode()
        visitor = _StrVisitor()
        node.visit(visitor)

        assert visitor.root is not None
        assert "<blank>" in str(visitor.root.label)

    def test_nested_tree(self) -> None:
        child = BlankNode()
        parent = ListNode([child])

        visitor = _StrVisitor()
        parent.visit(visitor)

        assert visitor.root is not None


class TestDocument:
    def test_init_with_root(self) -> None:
        root = BlankNode()
        doc = Document(root)
        assert doc.root is root

    def test_extension_returns_none_when_no_output_nodes(self) -> None:
        root = BlankNode()
        doc = Document(root)
        assert doc.extension() is None

    def test_extension_returns_extension_from_output_node(self) -> None:
        from io import TextIOBase

        from docc.context import Context

        class TestOutputNode(OutputNode):
            @property
            def children(self):
                return ()

            def replace_child(self, old: Node, new: Node) -> None:
                pass

            @property
            def extension(self) -> str:
                return ".test"

            def output(
                self, context: Context, destination: TextIOBase
            ) -> None:
                pass

        root = TestOutputNode()
        doc = Document(root)
        assert doc.extension() == ".test"


class TestExtensionVisitor:
    def test_finds_extension(self) -> None:
        from io import TextIOBase

        from docc.context import Context

        class TestOutputNode(OutputNode):
            @property
            def children(self):
                return ()

            def replace_child(self, old: Node, new: Node) -> None:
                pass

            @property
            def extension(self) -> str:
                return ".html"

            def output(
                self, context: Context, destination: TextIOBase
            ) -> None:
                pass

        root = TestOutputNode()
        visitor = _ExtensionVisitor()
        root.visit(visitor)

        assert visitor.extension == ".html"

    def test_returns_none_when_no_output_nodes(self) -> None:
        root = BlankNode()
        visitor = _ExtensionVisitor()
        root.visit(visitor)

        assert visitor.extension is None

    def test_conflicting_extensions_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """
        When two OutputNodes have different extensions, the visitor
        logs a warning and keeps the first extension.
        """
        import logging
        from io import TextIOBase

        from docc.context import Context

        class HtmlOutputNode(OutputNode):
            @property
            def children(self):
                return ()

            def replace_child(self, old: Node, new: Node) -> None:
                pass

            @property
            def extension(self) -> str:
                return ".html"

            def output(
                self, context: Context, destination: TextIOBase
            ) -> None:
                pass

        class TxtOutputNode(OutputNode):
            @property
            def children(self):
                return ()

            def replace_child(self, old: Node, new: Node) -> None:
                pass

            @property
            def extension(self) -> str:
                return ".txt"

            def output(
                self, context: Context, destination: TextIOBase
            ) -> None:
                pass

        root = ListNode([HtmlOutputNode(), TxtOutputNode()])
        visitor = _ExtensionVisitor()

        with caplog.at_level(logging.WARNING):
            root.visit(visitor)

        # The first extension is kept
        assert visitor.extension == ".html"
        # A warning was logged about the conflict
        assert any("extension" in r.message for r in caplog.records)


class ConditionalSkipVisitor(RecordingVisitor):
    skip_after_first: bool

    def __init__(self, skip_after_first: bool = False) -> None:
        super().__init__()
        self.skip_after_first = skip_after_first
        self._first_seen = False

    @override
    def enter(self, node: Node) -> Visit:
        self.events.append(("enter", id(node)))
        if self.skip_after_first and not self._first_seen:
            self._first_seen = True
            return Visit.TraverseChildren
        elif self.skip_after_first:
            return Visit.SkipChildren
        return Visit.TraverseChildren


class TestVisitorEdgeCases:
    def test_visit_empty_list_node(self) -> None:
        node = ListNode([])
        visitor = RecordingVisitor()
        node.visit(visitor)

        assert visitor.events == [
            ("enter", id(node)),
            ("exit", id(node)),
        ]

    def test_visit_deeply_nested(self) -> None:
        node: Node = BlankNode()
        for _ in range(10):
            node = ListNode([node])

        visitor = RecordingVisitor()
        node.visit(visitor)

        enter_count = sum(1 for e in visitor.events if e[0] == "enter")
        assert enter_count == 11

    def test_visit_wide_tree(self) -> None:
        children: List[Node] = [BlankNode() for _ in range(100)]
        node = ListNode(children)

        visitor = RecordingVisitor()
        node.visit(visitor)

        enter_count = sum(
            1
            for e in visitor.events
            if e[0] == "enter" and e != ("enter", id(node))
        )
        assert enter_count == 100

    def test_conditional_skip(self) -> None:
        grandchild = BlankNode()
        child = ListNode([grandchild])
        parent = ListNode([child])

        visitor = ConditionalSkipVisitor(skip_after_first=True)
        parent.visit(visitor)

        assert ("enter", id(grandchild)) not in visitor.events


class ModifyingVisitor(Visitor):
    def __init__(self, old: Node, new: Node) -> None:
        self.old = old
        self.new = new
        self.stack: List[Node] = []

    @override
    def enter(self, node: Node) -> Visit:
        self.stack.append(node)
        return Visit.TraverseChildren

    @override
    def exit(self, node: Node) -> None:
        self.stack.pop()
        if node == self.old and self.stack:
            self.stack[-1].replace_child(self.old, self.new)


class SkipSpecificChildVisitor(Visitor):
    """Visitor that returns SkipChildren for a specific child node."""

    enter_calls: List[Node]
    exit_calls: List[Node]
    skip_target: Node

    def __init__(self, skip_target: Node) -> None:
        self.enter_calls = []
        self.exit_calls = []
        self.skip_target = skip_target

    @override
    def enter(self, node: Node) -> Visit:
        self.enter_calls.append(node)
        if node is self.skip_target:
            return Visit.SkipChildren
        return Visit.TraverseChildren

    @override
    def exit(self, node: Node) -> None:
        self.exit_calls.append(node)


def test_visit_skip_children_calls_exit() -> None:
    grandchild = BlankNode()
    skipped_child = ListNode([grandchild])
    other_child = BlankNode()
    parent = ListNode([skipped_child, other_child])

    visitor = SkipSpecificChildVisitor(skip_target=skipped_child)
    parent.visit(visitor)

    # exit must be called for the child that returned SkipChildren
    assert skipped_child in visitor.exit_calls

    # The grandchild should NOT have been entered (children were skipped)
    assert grandchild not in visitor.enter_calls

    # Both the skipped child and the other child should have exit called
    assert other_child in visitor.exit_calls
    assert parent in visitor.exit_calls


def test_visit_replace_during_exit() -> None:
    old_child = BlankNode()
    new_child = BlankNode()
    parent = ListNode([old_child])

    visitor = ModifyingVisitor(old_child, new_child)
    parent.visit(visitor)

    assert old_child not in parent.children
    assert new_child in parent.children
