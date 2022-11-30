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
Plugin that renders to HTML.
"""


import html.parser
import sys
import xml.etree.ElementTree as ET
from io import StringIO, TextIOBase
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import markupsafe
from jinja2 import (
    Environment,
    PackageLoader,
    pass_eval_context,
    select_autoescape,
)
from jinja2.nodes import EvalContext

from docc.document import BlankNode, Document, Node, OutputNode, Visit, Visitor
from docc.languages import python
from docc.plugins.loader import PluginError
from docc.settings import PluginSettings
from docc.transform import Transform

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points
else:
    from importlib.metadata import EntryPoint, entry_points


class TextNode(Node):
    """
    Node containing text.
    """

    _value: str

    def __init__(self, value: str) -> None:
        self._value = value

    @property
    def children(self) -> Sequence[Node]:
        """
        Child nodes belonging to this node.
        """
        return tuple()

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        raise TypeError("text nodes have no children")

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        return repr(self._value)


class HTMLTag(Node):
    """
    A node holding HTML.
    """

    tag_name: str
    attributes: Dict[str, Optional[str]]
    _children: List[Node]

    def __init__(
        self, tag_name: str, attributes: Dict[str, Optional[str]]
    ) -> None:
        self.tag_name = tag_name
        self.attributes = attributes
        self._children = []

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
        self._children = [new if x == old else x for x in self._children]

    def append(self, node: Union["HTMLTag", TextNode]) -> None:
        """
        Add the given node to the end of this instance's children.
        """
        self._children.append(node)

    def __repr__(self) -> str:
        """
        Textual representation of this instance.
        """
        output = f"<{self.tag_name}"
        for name, value in self.attributes.items():
            output += f" {name}"
            if value is not None:
                output += '="'
                output += html.escape(value, True)
                output += '"'
        output += ">"
        return output

    def _to_element(self) -> ET.Element:
        visitor = _ElementTreeVisitor()
        self.visit(visitor)
        return visitor.builder.close()


class HTMLRoot(OutputNode):
    """
    Node representing the top-level of an HTML document or fragment.
    """

    _children: List[Union[HTMLTag, TextNode]]

    def __init__(self) -> None:
        self._children = []

    @property
    def children(self) -> Iterable[Node]:
        """
        Child nodes belonging to this node.
        """
        return self._children

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        assert isinstance(new, (HTMLTag, TextNode))
        self._children = [new if x == old else x for x in self._children]

    def append(self, node: Union["HTMLTag", TextNode]) -> None:
        """
        Add a new HTML or text node to the end of this node's children.
        """
        self._children.append(node)

    def output(self, destination: TextIOBase) -> None:
        """
        Attempt to write this node to destination as HTML.
        """
        rendered = StringIO()
        for child in self.children:
            if isinstance(child, TextNode):
                rendered.write(child._value)
                continue

            assert isinstance(child, HTMLTag)
            element = child._to_element()
            markup = ET.tostring(element, encoding="unicode", method="html")
            rendered.write(markup)

        env = Environment(
            loader=PackageLoader("docc.plugins.html"),
            autoescape=select_autoescape(),
        )
        template = env.get_template("base.html")
        body = rendered.getvalue()
        destination.write(template.render(body=markupsafe.Markup(body)))


class _ElementTreeVisitor(Visitor):
    builder: ET.TreeBuilder

    def __init__(self) -> None:
        self.builder = ET.TreeBuilder()

    def enter_tag(self, node: HTMLTag) -> Visit:
        attributes: Dict[Union[str, bytes], Union[str, bytes]] = {
            name: value if value else ""
            for name, value in node.attributes.items()
        }
        self.builder.start(node.tag_name, attributes)
        return Visit.TraverseChildren

    def enter_text(self, node: TextNode) -> Visit:
        self.builder.data(node._value)
        return Visit.TraverseChildren

    def enter(self, node: Node) -> Visit:
        if isinstance(node, HTMLTag):
            return self.enter_tag(node)
        elif isinstance(node, TextNode):
            return self.enter_text(node)
        else:
            raise TypeError(f"unsupported node {node.__class__.__name__}")

    def exit_tag(self, node: HTMLTag) -> None:
        self.builder.end(node.tag_name)

    def exit_text(self, node: TextNode) -> None:
        pass  # Do nothing

    def exit(self, node: Node) -> None:
        if isinstance(node, HTMLTag):
            return self.exit_tag(node)
        elif isinstance(node, TextNode):
            return self.exit_text(node)
        else:
            raise TypeError(f"unsupported node {node.__class__.__name__}")


class HTMLVisitor(Visitor):
    """
    Visits a Document's tree and converts Nodes to HTML.
    """

    entry_points: Dict[str, EntryPoint]
    renderers: Dict[Type, Callable]
    root: HTMLRoot
    stack: List[Union[HTMLRoot, HTMLTag, TextNode, BlankNode]]

    def __init__(self) -> None:
        # Discover render functions.
        found = entry_points(group="docc.plugins.html")
        self.entry_points = {entry.name: entry for entry in found}
        self.root = HTMLRoot()
        self.stack = [self.root]
        self.renderers = {}

    def _renderer(self, type_: Type) -> Callable:
        try:
            return self.renderers[type_]
        except KeyError:
            pass

        key = f"{type_.__module__}:{type_.__qualname__}"
        try:
            renderer = self.entry_points[key].load()
        except KeyError as e:
            raise PluginError(f"no renderer found for `{key}`") from e

        if not callable(renderer):
            raise PluginError(f"renderer for `{key}` is not callable")

        self.renderers[type_] = renderer
        return renderer

    def enter(self, node: Node) -> Visit:
        """
        Called when visiting the given node, before any children (if any) are
        visited.
        """
        renderer = self._renderer(node.__class__)
        result = renderer(node)

        if not isinstance(result, tuple) or len(result) != 2:
            raise PluginError(
                f"`{renderer.__module__}:{renderer.__qualname__}` "
                "did not return a tuple with exactly two elements"
            )

        visit, tags = result

        if not isinstance(visit, Visit):
            raise PluginError(
                f"`{renderer.__module__}:{renderer.__qualname__}` "
                "did not return a Visit"
            )

        tags = [TextNode(x) if isinstance(x, str) else x for x in tags]

        if not all(isinstance(x, (TextNode, HTMLTag)) for x in tags):
            raise PluginError(
                f"`{renderer.__module__}:{renderer.__qualname__}` "
                "did not return str | HTMLTag"
            )

        for tag in tags:
            assert isinstance(self.stack[-1], (HTMLRoot, HTMLTag))
            self.stack[-1].append(tag)

        try:
            last = tags[-1]
        except IndexError:
            # Always append something so the exit implementation is simpler.
            last = BlankNode()

        self.stack.append(last)
        return visit

    def exit(self, node: Node) -> None:
        """
        Called after visiting the last child of the given node (or immediately
        if the node has no children.)
        """
        self.stack.pop()


class HTML(Transform):
    """
    A plugin that renders to HTML.
    """

    def __init__(self, settings: PluginSettings) -> None:
        pass

    def transform(self, document: Document) -> None:
        """
        Apply the transformation to the given document.
        """
        visitor = HTMLVisitor()
        document.root.visit(visitor)
        assert visitor.root is not None
        document.root = visitor.root


class _HTMLParser(html.parser.HTMLParser):
    root: HTMLRoot
    stack: List[Union[HTMLRoot, HTMLTag]]

    def __init__(self) -> None:
        super().__init__()
        self.root = HTMLRoot()
        self.stack = [self.root]

    def handle_starttag(
        self, tag_name: str, attributes: Sequence[Tuple[str, Optional[str]]]
    ) -> None:
        tag = HTMLTag(tag_name, dict(attributes))
        self.stack[-1].append(tag)
        self.stack.append(tag)

    def handle_endtag(self, tag_name: str) -> None:
        ended = self.stack.pop()
        assert isinstance(ended, HTMLTag)
        assert ended.tag_name == tag_name

    def handle_data(self, data: str) -> None:
        self.stack[-1].append(TextNode(data))

    def handle_entityref(self, name: str) -> None:
        raise TypeError()  # Not called when convert_charrefs is True.

    def handle_charref(self, name: str) -> None:
        raise TypeError()  # Not called when convert_charrefs is True.

    def handle_comment(self, data: str) -> None:
        raise NotImplementedError("HTML comments not yet supported")

    def handle_decl(self, declaration: str) -> None:
        raise NotImplementedError("HTML doctypes not yet supported")

    def handle_pi(self, data: str) -> None:
        raise NotImplementedError(
            "HTML processing instructions not yet supported"
        )

    def unknown_decl(self, data: str) -> None:
        raise NotImplementedError("unknown HTML declaration")


@pass_eval_context
def _html_filter(
    context: EvalContext, value: Any
) -> Union[markupsafe.Markup, str]:
    assert isinstance(value, Node)
    visitor = HTMLVisitor()
    value.visit(visitor)

    children = []
    for child in visitor.root.children:
        if isinstance(child, TextNode):
            children.append(child._value)
            continue

        assert isinstance(child, HTMLTag)
        element = child._to_element()
        markup = ET.tostring(element, encoding="unicode", method="html")
        children.append(markup)

    rendered = "".join(children)
    return markupsafe.Markup(rendered) if context.autoescape else rendered


def _render_template(
    template_name: str, node: Node
) -> Tuple[Visit, Sequence[Union[str, HTMLTag, TextNode]]]:
    env = Environment(
        loader=PackageLoader("docc.plugins.html"),
        autoescape=select_autoescape(),
    )
    env.filters["html"] = _html_filter
    template = env.get_template(template_name)
    parser = _HTMLParser()
    parser.feed(template.render(node=node))
    return (Visit.SkipChildren, parser.root._children)


def python_module(
    module: python.Module,
) -> Tuple[Visit, Sequence[Union[str, HTMLTag, TextNode]]]:
    """
    Render a python Module as HTML.
    """
    return _render_template("python/module.html", module)


def python_class(
    class_: python.Class,
) -> Tuple[Visit, Sequence[Union[str, HTMLTag, TextNode]]]:
    """
    Render a python Class as HTML.
    """
    return _render_template("python/class.html", class_)


def python_attribute(
    attribute: python.Attribute,
) -> Tuple[Visit, Sequence[Union[str, HTMLTag, TextNode]]]:
    """
    Render a python assignment as HTML.
    """
    return _render_template("python/attribute.html", attribute)


def python_function(
    function: python.Function,
) -> Tuple[Visit, Sequence[Union[str, HTMLTag, TextNode]]]:
    """
    Render a python Function as HTML.
    """
    return _render_template("python/function.html", function)


def python_name(
    name: python.Name,
) -> Tuple[Visit, Sequence[Union[str, HTMLTag, TextNode]]]:
    """
    Render a python Name as HTML.
    """
    return _render_template("python/name.html", name)
