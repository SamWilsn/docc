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
    FrozenSet,
    Iterable,
    Iterator,
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
from jinja2 import Environment, PackageLoader
from jinja2 import nodes as j2
from jinja2 import pass_context, select_autoescape
from jinja2.ext import Extension
from jinja2.parser import Parser
from jinja2.runtime import Context as JinjaContext

from docc.context import Context
from docc.discover import Discover, T
from docc.document import BlankNode, Document, Node, OutputNode, Visit, Visitor
from docc.languages import python, verbatim
from docc.plugins import references
from docc.plugins.loader import PluginError
from docc.plugins.references import Index, ReferenceError
from docc.plugins.resources import ResourceSource
from docc.plugins.search import Search
from docc.settings import PluginSettings
from docc.source import Source, TextSource
from docc.transform import Transform

if sys.version_info < (3, 10):
    from importlib_metadata import EntryPoint, entry_points
else:
    from importlib.metadata import EntryPoint, entry_points


RenderResult = Optional["HTMLTag"]


class HTMLDiscover(Discover):
    """
    Create sources for static files necessary for HTML output.
    """

    def __init__(self, config: PluginSettings) -> None:
        """
        Construct a new instance with the given configuration.
        """
        pass

    def discover(self, known: FrozenSet[T]) -> Iterator[Source]:
        """
        Find sources.
        """
        yield ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "chota" / "dist" / "chota.min.css",
            PurePath("static") / "chota",
        )
        yield ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "docc.css",
            PurePath("static") / "docc",
        )
        yield ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "fuse" / "dist" / "fuse.min.js",
            PurePath("static") / "fuse",
        )
        yield ResourceSource.with_path(
            "docc.plugins.html",
            PurePath("static") / "search.js",
            PurePath("static") / "search",
        )


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

    def output(self, context: Context, destination: TextIOBase) -> None:
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
            extensions=[_ReferenceExtension],
            loader=PackageLoader("docc.plugins.html"),
            autoescape=select_autoescape(),
        )
        template = env.get_template("base.html")
        body = rendered.getvalue()
        static_path = _static_path_from(context)

        search_path = None
        search_base = None
        if Search in context:
            search_path = _search_path_from(context)
            search_base = _project_path_from(context)

        destination.write(
            template.render(
                body=markupsafe.Markup(body),
                static_path=static_path,
                search_path=search_path,
                search_base=search_base,
            )
        )

    @property
    def extension(self) -> str:
        """
        The preferred file extension for this node.
        """
        return ".html"


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
    context: Context

    def __init__(self, context: Context) -> None:
        # Discover render functions.
        found = entry_points(group="docc.plugins.html")
        self.entry_points = {entry.name: entry for entry in found}
        self.root = HTMLRoot()
        self.stack = [self.root]
        self.renderers = {}
        self.context = context

    def _renderer(self, node: Node) -> Callable[..., object]:
        type_ = node.__class__
        try:
            return self.renderers[type_]
        except KeyError:
            pass

        key = f"{type_.__module__}:{type_.__qualname__}"
        try:
            renderer = self.entry_points[key].load()
        except KeyError as e:
            raise PluginError(
                f"no renderer found for `{key}` (for node `{node}`)"
            ) from e

        if not callable(renderer):
            raise PluginError(f"renderer for `{key}` is not callable")

        self.renderers[type_] = renderer
        return renderer

    def enter(self, node: Node) -> Visit:
        """
        Called when visiting the given node, before any children (if any) are
        visited.
        """
        top = self.stack[-1]
        assert isinstance(top, (HTMLRoot, HTMLTag))

        renderer = self._renderer(node)
        result = renderer(self.context, top, node)

        if result is None:
            # Always append something so the exit implementation is simpler.
            self.stack.append(BlankNode())
            return Visit.SkipChildren
        elif isinstance(result, HTMLTag):
            self.stack.append(result)
            return Visit.TraverseChildren
        else:
            raise PluginError(
                f"`{renderer.__module__}:{renderer.__qualname__}` "
                "did not return `None` or `HTMLTag` instance"
            )

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

    def transform(self, context: Context) -> None:
        """
        Apply the transformation to the given document.
        """
        document = context[Document]
        if isinstance(document.root, OutputNode):
            return None

        visitor = HTMLVisitor(context)
        document.root.visit(visitor)
        assert visitor.root is not None
        document.root = visitor.root


class HTMLParser(html.parser.HTMLParser):
    """
    Subclass of Python's HTMLParser that converts into docc's syntax tree.
    """

    root: HTMLRoot
    stack: List[Union[HTMLRoot, HTMLTag]]

    def __init__(self) -> None:
        super().__init__()
        self.root = HTMLRoot()
        self.stack = [self.root]

    def handle_starttag(
        self, tag: str, attrs: Sequence[Tuple[str, Optional[str]]]
    ) -> None:
        """
        Handle opening tags.
        """
        element = HTMLTag(tag, dict(attrs))
        self.stack[-1].append(element)
        self.stack.append(element)

    def handle_endtag(self, tag: str) -> None:
        """
        Handle closing tags.
        """
        ended = self.stack.pop()
        assert isinstance(ended, HTMLTag)
        assert (
            ended.tag_name == tag
        ), f"mismatched tag `{ended.tag_name}` and `{tag}`"

    def handle_data(self, data: str) -> None:
        """
        Handle data.
        """
        self.stack[-1].append(TextNode(data))

    def handle_entityref(self, name: str) -> None:
        """
        Handle an entity reference.
        """
        raise TypeError()  # Not called when convert_charrefs is True.

    def handle_charref(self, name: str) -> None:
        """
        Handle a character reference.
        """
        raise TypeError()  # Not called when convert_charrefs is True.

    def handle_comment(self, data: str) -> None:
        """
        Handle an HTML comment.
        """
        raise NotImplementedError("HTML comments not yet supported")

    def handle_decl(self, decl: str) -> None:
        """
        Handle a doctype.
        """
        raise NotImplementedError("HTML doctypes not yet supported")

    def handle_pi(self, data: str) -> None:
        """
        Handle a processing instruction.
        """
        raise NotImplementedError(
            "HTML processing instructions not yet supported"
        )

    def unknown_decl(self, data: str) -> None:
        """
        Handle an unknown HTML declaration.
        """
        raise NotImplementedError("unknown HTML declaration")


class _FindVisitor(Visitor):
    class_: str
    found: List[Tuple[references.Definition, Node]]
    max_depth: int
    _definitions: List[references.Definition]

    def __init__(self, class_: str, max_depth: int = 1) -> None:
        self.class_ = class_
        self.found = []
        self.max_depth = max_depth
        self._definitions = []

    def enter(self, node: Node) -> Visit:
        if isinstance(node, references.Definition):
            self._definitions.append(node)

        if len(self._definitions) > self.max_depth:
            return Visit.SkipChildren

        try:
            definition = self._definitions[-1]
        except IndexError:
            return Visit.TraverseChildren

        type_ = node.__class__
        full_name = f"{type_.__module__}:{type_.__qualname__}"

        if full_name == self.class_:
            self.found.append((definition, node))
            return Visit.SkipChildren

        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        if isinstance(node, references.Definition):
            popped = self._definitions.pop()
            assert node == popped


class _ReferenceExtension(Extension):
    tags = {"reference"}

    def parse(self, parser: Parser) -> j2.Node:
        lineno = next(parser.stream).lineno

        # two arguments: the identifier of the reference, and the context
        args = [parser.parse_expression(), j2.ContextReference()]

        body = parser.parse_statements(
            ("name:endreference",), drop_needle=True
        )

        return j2.CallBlock(
            self.call_method("_reference_support", args), [], [], body
        ).set_lineno(lineno)

    def _reference_support(
        self, identifier: str, context: JinjaContext, caller: Callable[[], str]
    ) -> markupsafe.Markup:
        parser = HTMLParser()
        parser.feed(caller())

        children = parser.root._children

        output = ""
        for child in children:
            if isinstance(child, markupsafe.Markup):
                output += child
            elif isinstance(child, str):
                output += markupsafe.escape(child)
            else:
                reference = references.Reference(
                    identifier=identifier, child=child
                )
                output += _html_filter(context, reference)

        return markupsafe.Markup(output)


def _find_filter(
    value: object,
    class_: object,
) -> Sequence[Tuple[references.Definition, Node]]:
    assert isinstance(value, Node)
    assert isinstance(class_, str)

    visitor = _FindVisitor(class_)
    value.visit(visitor)
    return visitor.found


@pass_context
def _html_filter(
    context: JinjaContext, value: object
) -> Union[markupsafe.Markup, str]:
    ctx = context["context"]
    assert isinstance(ctx, Context)
    assert isinstance(value, Node)
    visitor = HTMLVisitor(ctx)
    value.visit(visitor)

    children = []
    for child in visitor.root.children:
        if isinstance(child, TextNode):
            children.append(html.escape(child._value))
            continue

        assert isinstance(child, HTMLTag)
        element = child._to_element()
        markup = ET.tostring(element, encoding="unicode", method="html")
        children.append(markup)

    rendered = "".join(children)
    eval_context = context.eval_ctx
    return markupsafe.Markup(rendered) if eval_context.autoescape else rendered


def _project_path_from(context: Context) -> str:
    return pathname2url(
        str(
            _make_relative(context[Source].output_path, PurePath("."))
            or PurePath()
        )
    )


def _search_path_from(context: Context) -> str:
    return pathname2url(
        str(
            _make_relative(context[Source].output_path, PurePath("search.js"))
            or PurePath()
        )
    )


def _static_path_from(context: Context) -> str:
    return pathname2url(
        str(
            _make_relative(context[Source].output_path, PurePath("static"))
            or PurePath()
        )
    )


def _render_template(
    context: object, parent: object, template_name: str, node: Node
) -> RenderResult:
    assert isinstance(context, Context)
    assert isinstance(parent, (HTMLTag, HTMLRoot))
    static_path = _static_path_from(context)
    env = Environment(
        extensions=[_ReferenceExtension],
        loader=PackageLoader("docc.plugins.html"),
        autoescape=select_autoescape(),
    )
    env.filters["html"] = _html_filter
    env.filters["find"] = _find_filter
    template = env.get_template(template_name)
    parser = HTMLParser()
    parser.feed(
        template.render(context=context, node=node, static_path=static_path)
    )
    for child in parser.root._children:
        parent.append(child)
    return None


def python_module(
    context: object,
    parent: object,
    module: object,
) -> RenderResult:
    """
    Render a python Module as HTML.
    """
    assert isinstance(module, python.Module)
    return _render_template(context, parent, "python/module.html", module)


def python_class(
    context: object,
    parent: object,
    class_: object,
) -> RenderResult:
    """
    Render a python Class as HTML.
    """
    assert isinstance(class_, python.Class)
    return _render_template(context, parent, "python/class.html", class_)


def python_attribute(
    context: object,
    parent: object,
    attribute: object,
) -> RenderResult:
    """
    Render a python assignment as HTML.
    """
    assert isinstance(attribute, python.Attribute)
    return _render_template(
        context, parent, "python/attribute.html", attribute
    )


def python_function(
    context: object,
    parent: object,
    function: object,
) -> RenderResult:
    """
    Render a python Function as HTML.
    """
    assert isinstance(function, python.Function)
    return _render_template(context, parent, "python/function.html", function)


def python_access(
    context: object,
    parent: object,
    access: object,
) -> RenderResult:
    """
    Render a python Access as HTML.
    """
    assert isinstance(access, python.Access)
    return _render_template(context, parent, "python/access.html", access)


def python_name(
    context: object,
    parent: object,
    name: object,
) -> RenderResult:
    """
    Render a python Name as HTML.
    """
    assert isinstance(name, python.Name)
    return _render_template(context, parent, "python/name.html", name)


def python_type(
    context: object,
    parent: object,
    type_: object,
) -> RenderResult:
    """
    Render a python Type as HTML.
    """
    assert isinstance(type_, python.Type)
    return _render_template(context, parent, "python/type.html", type_)


def python_list(
    context: object,
    parent: object,
    list_: object,
) -> RenderResult:
    """
    Render a python List as HTML.
    """
    assert isinstance(list_, python.List)
    return _render_template(context, parent, "python/list.html", list_)


def python_tuple(
    context: object,
    parent: object,
    tuple_: object,
) -> RenderResult:
    """
    Render a python List as HTML.
    """
    assert isinstance(tuple_, python.Tuple)
    return _render_template(context, parent, "python/tuple.html", tuple_)


def python_docstring(
    context: object,
    parent: object,
    docstring: object,
) -> RenderResult:
    """
    Render a python Docstring as HTML.
    """
    assert isinstance(docstring, python.Docstring)
    assert isinstance(parent, (HTMLRoot, HTMLTag))
    parent.append(TextNode(docstring.text))
    return None


def python_parameter(
    context: object,
    parent: object,
    parameter: object,
) -> RenderResult:
    """
    Render a python Parameter as HTML.
    """
    assert isinstance(parameter, python.Parameter)
    return _render_template(
        context, parent, "python/parameter.html", parameter
    )


def blank_node(
    context: object,
    parent: object,
    blank: object,
) -> RenderResult:
    """
    Render a blank node.
    """
    assert isinstance(blank, BlankNode)
    return None


class _VerbatimVisitor(verbatim.VerbatimVisitor):
    context: Final[Context]
    document: Final[Document]
    root: HTMLTag
    body: HTMLTag
    output_stack: List[Node]
    input_stack: List[Union[Sequence[str], references.Reference]]

    def __init__(self, context: Context) -> None:
        super().__init__()
        self.context = context
        self.document = context[Document]

        self.body = HTMLTag("tbody")

        self.root = HTMLTag("table", attributes={"class": "verbatim"})
        self.root.append(self.body)

        self.output_stack = []
        self.input_stack = []

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

        self.output_stack = [code_pre]
        self._highlight(self.input_stack)

    def _highlight(
        self,
        highlight_groups: Sequence[Union[Sequence[str], references.Reference]],
    ) -> None:
        for item in highlight_groups:
            top = self.output_stack[-1]
            assert isinstance(top, HTMLTag)

            if isinstance(item, references.Reference):
                new_node = _render_reference(self.context, item)
            else:
                classes = [f"hi-{h}" for h in item] + ["hi"]
                new_node = HTMLTag(
                    "span",
                    attributes={
                        "class": " ".join(classes),
                    },
                )

            top.append(new_node)
            self.output_stack.append(new_node)

    def text(self, text: str) -> None:
        top = self.output_stack[-1]
        assert isinstance(top, HTMLTag)
        top.append(TextNode(text))

    def begin_highlight(self, highlights: Sequence[str]) -> None:
        self.input_stack.append(highlights)
        self._highlight([highlights])

    def end_highlight(self) -> None:
        self.input_stack.pop()
        popped_node = self.output_stack.pop()
        assert isinstance(popped_node, HTMLTag)
        assert (
            popped_node.tag_name == "span"
        ), f"expected span, got `{popped_node.tag_name}`"

    def enter_node(self, node: Node) -> Visit:
        """
        Visit a non-verbatim Node.
        """
        if isinstance(node, references.Reference):
            if "<" in node.identifier:
                # TODO: Create definitions for local variables.
                return Visit.TraverseChildren
            self.input_stack.append(node)
            if self.output_stack:
                self._highlight([node])
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

            popped = self.input_stack.pop()
            assert popped == node

            popped_output = self.output_stack.pop()
            assert isinstance(popped_output, HTMLTag)
        else:
            return super().exit_node(node)


def verbatim_verbatim(
    context: Context,
    parent: object,
    node: verbatim.Verbatim,
) -> RenderResult:
    """
    Render a verbatim block as HTML.
    """
    assert isinstance(context, Context)
    assert isinstance(parent, (HTMLRoot, HTMLTag))
    assert isinstance(node, verbatim.Verbatim)

    visitor = _VerbatimVisitor(context)
    node.visit(visitor)
    parent.append(visitor.root)
    return None


def references_definition(
    context: object,
    parent: object,
    definition: object,
) -> RenderResult:
    """
    Render a Definition as HTML.
    """
    assert isinstance(context, Context)
    assert isinstance(parent, (HTMLRoot, HTMLTag))
    assert isinstance(definition, references.Definition)

    new_id = f"{definition.identifier}:{definition.specifier}"

    visitor = HTMLVisitor(context)
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

    for child in children:
        parent.append(child)

    return None


def references_reference(
    context: object,
    parent: object,
    reference: object,
) -> RenderResult:
    """
    Render a Reference as HTML.
    """
    assert isinstance(context, Context)
    assert isinstance(parent, (HTMLRoot, HTMLTag))
    assert isinstance(reference, references.Reference)

    anchor = _render_reference(context, reference)
    parent.append(anchor)

    if not reference.child:
        anchor.append(TextNode(reference.identifier))
        return None

    # TODO: handle tr, td, and other elements that can't be wrapped in an <a>.

    return anchor


def html_tag(
    context: object,
    parent: object,
    html_tag: object,
) -> RenderResult:
    """
    Render an HTMLTag as HTML.
    """
    assert isinstance(parent, (HTMLRoot, HTMLTag))
    assert isinstance(html_tag, HTMLTag)
    parent.append(html_tag)
    return None


def text_node(
    context: object,
    parent: object,
    text_node: object,
) -> RenderResult:
    """
    Render TextNode as HTML.
    """
    assert isinstance(parent, (HTMLRoot, HTMLTag))
    assert isinstance(text_node, TextNode)
    parent.append(text_node)
    return None


def _make_relative(from_: PurePath, to: PurePath) -> Optional[PurePath]:
    # XXX: This path stuff is most certainly broken.

    if from_ == to:
        # Can't represent an empty path with PurePath (becomes "." instead)
        return None

    common_path = commonpath((from_, to))

    parents = len(from_.relative_to(common_path).parents) - 1

    return PurePath(*[".."] * parents) / to.relative_to(common_path)


def _render_reference(
    context: Context, reference: references.Reference
) -> HTMLTag:
    try:
        definitions = list(context[Index].lookup(reference.identifier))
    except ReferenceError as error:
        raise ReferenceError(reference.identifier, context=context) from error

    if not definitions:
        raise NotImplementedError(
            f"no definitions for `{reference.identifier}`"
        )

    multi = len(definitions) > 1
    anchors = HTMLTag(
        "div", attributes={"class": "tooltip-content", "role": "tooltip"}
    )

    for definition in definitions:
        anchor = HTMLTag("a")
        anchors.append(anchor)

        # XXX: This path stuff is most certainly broken.

        output_path = context[Source].output_path
        definition_path = definition.source.output_path

        relative_path = _make_relative(output_path, definition_path)
        if relative_path is None:
            relative_path_str = ""
        else:
            # TODO: Don't hardcode extension.
            relative_path_str = str(relative_path) + ".html"

        fragment = f"{definition.identifier}:{definition.specifier}"
        anchor.attributes["href"] = urlunsplit(
            (
                "",  # scheme
                "",  # host
                pathname2url(relative_path_str),  # path
                "",  # query
                fragment,  # fragment
            )
        )

        if multi:
            anchor.append(TextNode(fragment))

    if not multi:
        anchor = anchors.children[0]
        assert isinstance(anchor, HTMLTag)
        return anchor

    container = HTMLTag(
        "div", attributes={"class": "tooltip", "tabindex": "0"}
    )
    container.append(anchors)

    return container
