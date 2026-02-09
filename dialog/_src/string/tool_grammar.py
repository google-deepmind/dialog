# Copyright 2026 The dialog Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tool grammar parser."""

from __future__ import annotations

import dataclasses
import functools
import html

from dialog._src import tags
from dialog._src.string import text_grammar
from etils import epy
import lark

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top
  from dialog._src import tool_helper
  # pylint: enable=g-import-not-at-top


def parse(text: str) -> text_grammar.Node:
  """Parses a tool call."""
  tree = _parser().parse(text)
  tree = Transformer().transform(tree)
  return tree


class ToolContent(text_grammar.Tree):
  """Content of a tool (`<|tool_xxx>...<tool_xxx|>`)."""

  def as_html(self) -> str:
    return text_grammar.span('json-container')(self.children)

  def as_conversation(self, *, kind: str):
    match self.children:
      case [ToolName() as tool_name, Dict() as content]:
        if kind != tool_name.kind:
          raise ValueError(
              f'Invalid tool format. Expected `{kind}:` but got'
              f' {tool_name.as_text()}'
          )

        # Normalize the dict to match MCP json format.
        data = content.as_conversation()
        if kind == tags.Tags.TOOL.name:
          data = tool_helper.as_mcp_def_dict(data)
        elif kind == tags.Tags.TOOL_CALL.name:
          data = {'arguments': data}
        elif kind == tags.Tags.TOOL_RESPONSE.name:
          data = {'structuredContent': data}
        else:
          raise ValueError(f'Unknown tool kind: {kind}')

        data['name'] = tool_name.as_conversation()
        return data
    raise ValueError(f'Unexpected tool format: {self.as_text()!r}')


class Dict(text_grammar.Tree):

  def as_html(self) -> str:
    text = ''.join(c.as_html() for c in self.children)
    return text

  def as_conversation(self):
    return dict(
        item.as_conversation()
        for item in self.children
        if isinstance(item, Item)
    )


class List(text_grammar.Tree):

  def as_html(self) -> str:
    text = ''.join(c.as_html() for c in self.children)
    return text

  def as_conversation(self):
    return [
        x.as_conversation()
        for x in self.children
        if not isinstance(x, Separator)
    ]


class Item(text_grammar.Tree):

  def as_html(self) -> str:
    text = ''.join(c.as_html() for c in self.children)
    return text

  def as_conversation(self):
    key, _, value = self.children
    return key.as_conversation(), value.as_conversation()


class ToolName(text_grammar.Leaf):
  """Name of a tool."""

  def as_conversation(self):
    kind, function_name = self.text.split(':', 1)
    if kind not in tags.Tags.TOOL_KIND_TO_TAG:
      raise ValueError(f'Unknown tool kind: {self.text}')
    return function_name

  @property
  def kind(self) -> str:
    kind, _ = self.text.split(':', 1)
    return tags.Tags.TOOL_KIND_TO_TAG[kind].name


@dataclasses.dataclass(frozen=True, kw_only=True)
class Primitive(text_grammar.Leaf):
  type: str

  def as_html(self) -> str:
    text = html.escape(self.text)
    return text_grammar.span(f'json-{self.type}')(text)

  def as_conversation(self):
    match self.type:
      case 'key':
        return self.text
      case 'string':
        quote = tags.Tags.QUOTE.tag
        return self.text.removeprefix(quote).removesuffix(quote)
      case 'number':
        # TODO(epot): int vs float
        if '.' in self.text:
          return float(self.text)
        else:
          return int(self.text)
      case 'bool':
        return _as_bool(self.text)
      case 'null':
        return None
      case _:
        raise ValueError(f'Unknown primitive type: {self.type}')


class Separator(text_grammar.Leaf):
  """`{`, `[`, `,`, `:`,...."""

  def as_conversation(self):
    raise ValueError(f'Separator should not be parsed: {self.text!r}')


def _as_bool(text: str) -> bool:
  if text == 'true':
    return True
  elif text == 'false':
    return False
  else:
    raise ValueError(f'Invalid boolean value: {text}')


class Transformer(lark.Transformer):
  """Transformer for the conversation grammar."""

  def start(self, items: list[text_grammar.Node]):
    return ToolContent(children=items)

  def dict(self, items: list[text_grammar.Node]):
    return Dict(children=items)

  def list(self, items: list[text_grammar.Node]):
    return List(children=items)

  def item(self, items: list[text_grammar.Node]):  # pytype: disable=unsupported-operands
    return Item(children=items)

  def TOOL_NAME(self, item: lark.Token):
    return ToolName(text=item.value)

  def __default_token__(self, token: lark.Token):
    if token.type in ('STRING', 'NUMBER', 'BOOL', 'NULL', 'KEY'):
      return Primitive(text=token.value, type=token.type.lower())

    # Convert leaf nodes to text (`{`, `[`, `,`, `:`,...)
    return Separator(text=token.value)


@functools.cache
def _parser() -> lark.Lark:
  """Returns a lark parser for the tool grammar."""
  grammar = r"""
      # Header
      start:  TOOL_NAME dict

      # Call name (e.g. `declaration:foo`)
      TOOL_NAME: CNAME COLON /[ a-zA-Z0-9_:\/\\.-]+/
      # TOOL_NAME: /[^{]+/  # Match all characters until `{`.

      # Body
      ?value: dict
           | list
           | STRING
           | NUMBER
           | BOOL
           | NULL

      dict: BRACKET_CURLY_LEFT (item (COMMA item)*)? BRACKET_CURLY_RIGHT
      list: BRACKET_SQUARE_LEFT (value (COMMA value)*)? BRACKET_SQUARE_RIGHT

      item: KEY COLON value

      KEY: CNAME

      STRING: QUOTE STRING_CONTENT QUOTE
      STRING_CONTENT: /.*?/s

      NUMBER: /-?\d+(\.\d+)?/
      BOOL: "true" | "false"
      NULL: "null"

      QUOTE: "<|\"|>"

      # Using named tokens makes error messages more readable (avoid
      # annonymous tokens like `__ANON_1`,...).
      COLON: ":"
      COMMA: ","
      BRACKET_SQUARE_LEFT: "["
      BRACKET_SQUARE_RIGHT: "]"
      BRACKET_CURLY_LEFT: "{"
      BRACKET_CURLY_RIGHT: "}"
      BRACKET_ROUND_LEFT: "("
      BRACKET_ROUND_RIGHT: ")"

      %import common.CNAME
  """
  return lark.Lark(grammar, start='start', parser='earley')
