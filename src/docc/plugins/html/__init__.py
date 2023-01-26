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
from os.path import commonpath
from pathlib import PurePath
from typing import (
    Callable,
    Dict,
    Final,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)
from urllib.parse import urlunsplit
from urllib.request import pathname2url

import markupsafe
from jinja2 import Environment, PackageLoader, pass_context, select_autoescape
from jinja2.runtime import Context
from typing_extensions import TypeAlias

from docc.document import BlankNode, Document, Node, OutputNode, Visit, Visitor
from docc.languages import python, verbatim
from docc.plugins import references
from docc.plugins.loader import PluginError
from docc.settings import PluginSettings
from docc.source import TextSource
from docc.transform import Transform

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points
else:
    from importlib.metadata import EntryPoint, entry_points


RenderResult: TypeAlias = Tuple[
    Visit, Sequence[Union[str, "HTMLTag", "TextNode"]]
]


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
        self,
        tag_name: str,
        attributes: Optional[Dict[str, Optional[str]]] = None,
    ) -> None:
        self.tag_name = tag_name
        self.attributes = {} if attributes is None else attributes
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
    def children(self) -> Iterable[Union[HTMLTag, TextNode]]:
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
    renderers: Dict[
        Type[Node],
        Callable[..., object],
    ]
    root: HTMLRoot
    stack: List[Union[HTMLRoot, HTMLTag, TextNode, BlankNode]]
    document: Document

    def __init__(self, document: Document) -> None:
        # Discover render functions.
        found = entry_points(group="docc.plugins.html")
        self.entry_points = {entry.name: entry for entry in found}
        self.root = HTMLRoot()
        self.stack = [self.root]
        self.renderers = {}
        self.document = document

    def _renderer(self, type_: Type[Node]) -> Callable[..., object]:
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
        result = renderer(self.document, node)

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
            top = self.stack[-1]
            assert isinstance(top, (HTMLRoot, HTMLTag))
            top.append(tag)

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
        visitor = HTMLVisitor(document)
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
        self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]
    ) -> None:
        element = HTMLTag(tag, dict(attrs))
        self.stack[-1].append(element)
        self.stack.append(element)

    def handle_endtag(self, tag: str) -> None:
        ended = self.stack.pop()
        assert isinstance(ended, HTMLTag)
        assert ended.tag_name == tag

    def handle_data(self, data: str) -> None:
        self.stack[-1].append(TextNode(data))

    def handle_entityref(self, name: str) -> None:
        raise TypeError()  # Not called when convert_charrefs is True.

    def handle_charref(self, name: str) -> None:
        raise TypeError()  # Not called when convert_charrefs is True.

    def handle_comment(self, data: str) -> None:
        raise NotImplementedError("HTML comments not yet supported")

    def handle_decl(self, decl: str) -> None:
        raise NotImplementedError("HTML doctypes not yet supported")

    def handle_pi(self, data: str) -> None:
        raise NotImplementedError(
            "HTML processing instructions not yet supported"
        )

    def unknown_decl(self, data: str) -> None:
        raise NotImplementedError("unknown HTML declaration")


@pass_context
def _html_filter(
    context: Context, value: object
) -> Union[markupsafe.Markup, str]:
    document = context["document"]
    assert isinstance(document, Document)
    assert isinstance(value, Node)
    visitor = HTMLVisitor(document)
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
    eval_context = context.eval_ctx
    return markupsafe.Markup(rendered) if eval_context.autoescape else rendered


def _render_template(
    document: object, template_name: str, node: Node
) -> RenderResult:
    assert isinstance(document, Document)
    env = Environment(
        loader=PackageLoader("docc.plugins.html"),
        autoescape=select_autoescape(),
    )
    env.filters["html"] = _html_filter
    template = env.get_template(template_name)
    parser = _HTMLParser()
    parser.feed(template.render(document=document, node=node))
    return (Visit.SkipChildren, parser.root._children)


def python_module(
    document: object,
    module: object,
) -> RenderResult:
    """
    Render a python Module as HTML.
    """
    assert isinstance(module, python.Module)
    return _render_template(document, "python/module.html", module)


def python_class(
    document: object,
    class_: object,
) -> RenderResult:
    """
    Render a python Class as HTML.
    """
    assert isinstance(class_, python.Class)
    return _render_template(document, "python/class.html", class_)


def python_attribute(
    document: object,
    attribute: object,
) -> RenderResult:
    """
    Render a python assignment as HTML.
    """
    assert isinstance(attribute, python.Attribute)
    return _render_template(document, "python/attribute.html", attribute)


def python_function(
    document: object,
    function: object,
) -> RenderResult:
    """
    Render a python Function as HTML.
    """
    assert isinstance(function, python.Function)
    return _render_template(document, "python/function.html", function)


def python_name(
    document: object,
    name: object,
) -> RenderResult:
    """
    Render a python Name as HTML.
    """
    assert isinstance(name, python.Name)
    return _render_template(document, "python/name.html", name)


def python_type(
    document: object,
    type_: object,
) -> RenderResult:
    """
    Render a python Type as HTML.
    """
    assert isinstance(type_, python.Type)
    return _render_template(document, "python/type.html", type_)


def python_generics(
    document: object,
    generics: object,
) -> RenderResult:
    """
    Render a python Generics as HTML.
    """
    assert isinstance(generics, python.Generics)
    return _render_template(document, "python/generics.html", generics)


def python_docstring(
    document: object,
    docstring: object,
) -> RenderResult:
    """
    Render a python Docstring as HTML.
    """
    assert isinstance(docstring, python.Docstring)
    return (Visit.TraverseChildren, docstring.text)


def blank_node(
    document: object,
    blank: object,
) -> RenderResult:
    """
    Render a blank node.
    """
    assert isinstance(blank, BlankNode)
    return (Visit.TraverseChildren, [])


class _VerbatimVisitor(verbatim.VerbatimVisitor):
    document: Final[Document]
    root: HTMLTag
    body: HTMLTag
    node_stack: List[Node]
    highlights_stack: List[Sequence[str]]

    def __init__(self, document: Document) -> None:
        super().__init__()
        self.document = document

        self.body = HTMLTag("tbody")

        self.root = HTMLTag("table")
        self.root.append(self.body)

        self.node_stack = []
        self.highlights_stack = []

    def line(self, source: TextSource, line: int) -> None:
        line_text = TextNode(str(line))

        line_cell = HTMLTag("th")
        line_cell.append(line_text)

        code_pre = HTMLTag("pre")

        code_cell = HTMLTag("td")
        code_cell.append(code_pre)

        row = HTMLTag("tr")
        row.append(line_cell)
        row.append(code_cell)

        self.body.append(row)

        self.node_stack = [code_pre]
        self._highlight(self.highlights_stack)

    def _highlight(self, highlight_groups: Sequence[Sequence[str]]) -> None:
        for highlights in highlight_groups:
            top = self.node_stack[-1]
            assert isinstance(top, HTMLTag)

            classes = [f"hi-{h}" for h in highlights] + ["hi"]
            span = HTMLTag(
                "span",
                attributes={
                    "class": " ".join(classes),
                },
            )
            top.append(span)
            self.node_stack.append(span)

    def text(self, text: str) -> None:
        top = self.node_stack[-1]
        assert isinstance(top, HTMLTag)
        top.append(TextNode(text))

    def begin_highlight(self, highlights: Sequence[str]) -> None:
        self.highlights_stack.append(highlights)
        self._highlight([highlights])

    def end_highlight(self) -> None:
        self.highlights_stack.pop()
        self.node_stack.pop()

    def enter_node(self, node: Node) -> Visit:
        """
        Visit a non-verbatim Node.
        """
        if isinstance(node, references.Reference):
            if "<" in node.identifier:
                # TODO: Create definitions for local variables.
                return Visit.TraverseChildren
            new_node = _render_reference(self.document, node)
            top = self.node_stack[-1]
            assert isinstance(top, HTMLTag)
            top.append(new_node)
            self.node_stack.append(new_node)
            return Visit.TraverseChildren
        else:
            return super().enter_node(node)

    def exit_node(self, node: Node) -> None:
        """
        Leave a non-verbatim Node.
        """
        if isinstance(node, references.Reference):
            if "<" in node.identifier:
                # TODO: Create definitions for local variables.
                return

            popped = self.node_stack.pop()

            # XXX: I'm concerned that a reference might end up on the stack
            #      during a line change.
            assert isinstance(popped, HTMLTag)
            assert popped.tag_name == "a"
        else:
            return super().exit_node(node)


def verbatim_verbatim(
    document: Document,
    node: verbatim.Verbatim,
) -> RenderResult:
    """
    Render a verbatim block as HTML.
    """
    assert isinstance(document, Document)
    assert isinstance(node, verbatim.Verbatim)

    visitor = _VerbatimVisitor(document)
    node.visit(visitor)
    return (Visit.SkipChildren, [visitor.root])


def references_definition(
    document: object,
    definition: object,
) -> RenderResult:
    """
    Render a Definition as HTML.
    """
    assert isinstance(document, Document)
    assert isinstance(definition, references.Definition)

    new_id = f"{definition.identifier}:{definition.specifier}"

    visitor = HTMLVisitor(document)
    definition.child.visit(visitor)

    children = list(visitor.root.children)

    if not children:
        children.append(HTMLTag("span"))

    first_child = children[0]

    if isinstance(first_child, TextNode):
        span = HTMLTag("span")
        span.append(first_child)
        children[0] = span
        first_child = span

    if "id" in first_child.attributes:
        raise NotImplementedError(
            f"multiple ids (adding {new_id} to {first_child.attributes['id']})"
        )

    first_child.attributes["id"] = new_id

    return (Visit.SkipChildren, children)


def references_reference(
    document: object,
    reference: object,
) -> Tuple[Visit, Sequence[Union[HTMLTag, TextNode]]]:
    """
    Render a Reference as HTML.
    """
    assert isinstance(document, Document)
    assert isinstance(reference, references.Reference)

    visitor = HTMLVisitor(document)
    reference.child.visit(visitor)

    children = list(visitor.root.children)

    if not children:
        children.append(TextNode(reference.identifier))

    anchor = _render_reference(document, reference)

    # TODO: handle tr, td, and other elements that can't be wrapped in an <a>.

    for child in children:
        anchor.append(child)

    return (Visit.SkipChildren, [anchor])


def _render_reference(
    document: Document, reference: references.Reference
) -> HTMLTag:
    anchor = HTMLTag("a")

    if reference.identifier == "src.docc.discover":
        print(document.source)
        document.root.dump()
    definitions = list(document.index.lookup(reference.identifier))

    if len(definitions) != 1:
        raise NotImplementedError()

    # XXX: This path stuff is most certainly broken.

    output_path = document.source.output_path
    definition_path = definitions[0].source.output_path

    if output_path == definition_path:
        relative_path = ""
    else:
        common_path = commonpath((output_path, definition_path))

        parents = len(output_path.relative_to(common_path).parents) - 1

        relative_path = (
            str(
                PurePath(*[".."] * parents)
                / definition_path.relative_to(common_path)
            )
            + ".html"
        )  # TODO: Don't hardcode extension.

    fragment = f"{definitions[0].identifier}:{definitions[0].specifier}"
    anchor.attributes["href"] = urlunsplit(
        (
            "",  # scheme
            "",  # host
            pathname2url(relative_path),  # path
            "",  # query
            fragment,  # fragment
        )
    )

    return anchor
