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

from collections.abc import Callable, Container
import dataclasses
import functools
import html
import re
from typing import NoReturn

from etils import epy
import lark

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top
  from dialog._src import conversation
  from dialog._src import tags
  from dialog._src.string import tool_grammar
  # pylint: enable=g-import-not-at-top


def parse(text: str) -> Root:
  """Parses a text."""
  tree = _parser().parse(text)
  tree = Transformer().transform(tree)
  return tree


@dataclasses.dataclass(frozen=True, kw_only=True)
class Node:
  r"""Base node.

  Class hierarchy:
  - Leaf: Only have text content (no structure)
    - Text: Text content
    - Role: The `xxx\n` part of `<|turn>xxx\n`
    - Token: A token (e.g. <|xxx|> or <xxx|>)
  - Tree: Structure node, with children.
    - Root: The root of the tree (conversation)
    - Tag: A tag (<|xxx>...<xxx|>)
      - RoleContent: The content of a tag with role (`<|turn>role\n`)
    - Invalid: Any invalid node (e.g. unclosed tag, mismatch tags...)
  """

  def as_text(self) -> str:
    """Returns the raw text of the node."""
    raise NotImplementedError()

  def as_html(self) -> str:
    """Returns the HTML of the node."""
    raise NotImplementedError()

  def as_conversation(self):
    """Parse back to the `dialog.Conversation`."""
    raise NotImplementedError(f'Not yet implemented: {type(self).__name__}')

  def raise_if_invalid(self) -> None:
    """Raises an error if the node is invalid."""
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True, kw_only=True)
class Tree(Node):
  """Tree."""

  children: list[Node]

  def as_text(self) -> str:
    return ''.join(c.as_text() for c in self.children)

  def as_html(self) -> str:
    return ''.join(c.as_html() for c in self.children)

  def raise_if_invalid(self) -> None:
    """Raises an error if the tree contains invalid nodes."""
    for c in self.children:
      c.raise_if_invalid()


@dataclasses.dataclass(frozen=True, kw_only=True)
class Root(Tree):
  """Root conversation node."""

  def as_html(self) -> str:
    text = super().as_html()
    # TODO(epot): Add an extra `\n` when text ends with `\n`. Currently Chrome
    # does not render the trailing `\n`.
    return f'<div class="conversation-str">{text}</div>'

  def as_conversation(self) -> conversation.Conversation:
    """Parse back to the `dialog.Conversation`."""
    # First validate the structure to ensure there's no Invalid nodes.
    self.raise_if_invalid()

    turns = []
    for c in self.children:
      # Skip empty text nodes.
      if isinstance(c, Text) and not c.text.strip():
        continue
      if not isinstance(c, Tag) or c.name != 'turn':
        raise ValueError(
            'Conversation root must only contain `turn` tags. Got'
            f' {c.as_text()!r}'
        )
      turns.append(c.as_conversation())

    # Filter out the last model turn if empty.
    if _is_last_model_turn_empty(turns):
      turns = turns[:-1]
    return conversation.Conversation(turns)

  def as_chunk(self) -> conversation.Chunk:
    """Parse back to the `dialog.Chunk`."""
    err_msg = lambda: (
        'Invalid text. Expected a single `<|xxx>...<xxx|>` tag. Got:'
        f' `{self.as_text()!r}`'
    )
    if len(self.children) != 1:
      raise ValueError(err_msg())
    chunk = self.children[0]
    if not isinstance(chunk, Tag):
      raise ValueError(err_msg())
    start, _, end = chunk.children
    if (
        not isinstance(start, Token)
        or not isinstance(end, Token)
        or start.name != end.name
    ):
      raise ValueError(err_msg())

    chunk = chunk.as_conversation()
    return chunk


def _is_last_model_turn_empty(turns: list[conversation.Turn]) -> bool:
  """Returns whether the last model turn is empty."""
  if not turns:
    return False
  last_turn = turns[-1]
  if not isinstance(last_turn, conversation.Model):
    return False
  if any(not isinstance(c, conversation.Text) for c in last_turn.chunks):
    return False
  return all(not c.text for c in last_turn.chunks)  # pylint: disable=attribute-error


@dataclasses.dataclass(frozen=True, kw_only=True)
class Tag(Tree):
  """Node for a tag (`<|xxx>...<xxx|>`).

  * Is always `[start, content, end]`
  * Content is always a `Tree`
  """

  name: str

  def as_html(self) -> str:
    start = self.children[0]
    content = span(['content'])(
        span(['full-text'])(self.children[1:-1]) + span(['ellipsis'])('...')
    )
    end = self.children[-1]

    extra_class = [self.role] if self.role else []

    return span(['tag'] + extra_class)([start, content, end])

  @property
  def role(self) -> str | None:
    _, content, _ = self.children
    match content:
      case Tree(children=[Role(text=text), *_]):
        return text.removesuffix('\n')
      case _:
        return None

  def as_conversation(self) -> conversation.Conversation:
    """Parse back to the `dialog.Conversation`."""
    return self._conversation_cls(self._conversation_data)

  @property
  def _conversation_cls(self):
    match self.name:
      case tags.Tags.TURN.name:
        cls = conversation.Turn.ROLE_TO_CLS.get(self.role)
        if not cls:
          raise ValueError(f'Unknown turn role: {self.role!r}')
        return cls
      case tags.Tags.CHANNEL.name:
        if self.role != 'thought':
          raise ValueError(f'Unknown channel role: {self.role!r}')
        return conversation.Thought
      case tags.Tags.TOOL.name:
        return conversation.Tool
      case tags.Tags.TOOL_CALL.name:
        return conversation.ToolCall
      case tags.Tags.TOOL_RESPONSE.name:
        return conversation.ToolResponse
      case _:
        raise ValueError(f'Unknown tag name: {self.name}')

  @property
  def _conversation_data(self):
    _, content, _ = self.children
    if isinstance(content, tool_grammar.ToolContent):
      return content.as_conversation(kind=self.name)
    elif isinstance(content, Tree):
      return [
          c.as_conversation() for c in content.children if not isinstance(c, Role)  # pytype: disable=attribute-error
      ]
    else:
      raise ValueError(f'Unknown content type: {type(content)}')


@dataclasses.dataclass(frozen=True, kw_only=True)
class Invalid(Tree):
  """Node for a tag."""

  reason: str

  def as_html(self) -> str:
    return span('invalid', title=self.reason)(self.children)

  def raise_if_invalid(self) -> None:
    raise ValueError(
        f'Conversation parsing failed: {self.reason}.\nFor: {self.as_text()!r}'
    )


@dataclasses.dataclass(frozen=True, kw_only=True)
class Leaf(Node):
  """Node."""

  text: str

  def as_html(self) -> str:
    return html.escape(self.text)

  def as_text(self) -> str:
    return self.text

  def raise_if_invalid(self) -> None:
    pass


class Text(Leaf):
  """Node for text."""

  def as_conversation(self) -> conversation.Text:
    return conversation.Text(self.text)


class Role(Leaf):
  r"""Node for a role (`<|turn>role\n` or `<|channel>role\n`)."""

  def as_html(self) -> str:
    return span('role')(super().as_html())

  def as_conversation(self) -> conversation.Text:
    return conversation.Text(self.text)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Token(Leaf):
  """Node for text."""

  def as_html(self) -> str:
    return span('token')(super().as_html())

  @functools.cached_property
  def name(self) -> str:
    """Returns the name of the token."""
    return self.text.strip('<|>')

  def as_conversation(self) -> conversation.Text:
    if self.name == tags.Tags.IMAGE.name:
      return conversation.Image(None)
    if self.name == tags.Tags.AUDIO.name:
      return conversation.Audio(None)

    cls = conversation.ControlToken.NAME_TO_CLS.get(self.name)
    if not cls:
      raise ValueError(f'Unknown control token: {self.as_text()!r}')
    return cls()  # pylint: disable=no-value-for-parameter  # pytype: disable=missing-parameter


class Transformer(lark.Transformer):
  """Transformer for the conversation grammar."""

  def start(self, items: list[Node]) -> Node:
    return Root(children=items)

  def closed_tag(self, items: list[Node]) -> Node:
    start, *content, end = items
    assert isinstance(start, Token)
    assert isinstance(end, Token)
    content = _make_tag_content(content, start.name)
    tag = Tag(
        name=start.name,
        children=[
            _validate_tag_name(start, tags.Tags.CLOSED_TAGS),
            content,
            _validate_tag_name(end, tags.Tags.CLOSED_TAGS),
        ],
    )

    if start.name != end.name:
      return Invalid(
          children=[tag],
          reason='Start and end tag do not match.',
      )
    return tag

  def unclosed_tag(self, items: list[Node]) -> Node:
    start, *content = items
    end = Text(text='')  # Invisible node to match the structure.
    assert isinstance(start, Token)
    content = _make_tag_content(content, start.name)
    return Tag(
        name=start.name,
        children=[
            _validate_tag_name(start, tags.Tags.CLOSED_TAGS),
            content,
            end,
        ],
    )

  def orphan_close_token(self, items: list[Node]):
    return Invalid(
        children=items, reason='Orphan close token not mathing anything'
    )

  def __default__(self, data, children, meta) -> NoReturn:
    # Guard to ensure we process all tokens.
    raise ValueError(f'Invalid token: {data}')

  def __default_token__(self, token: lark.Token) -> Node:
    if token.type in (
        'OPEN_TOKEN',
        'CLOSE_TOKEN',
        'STANDALONE_TOKEN',
    ):
      leaf = Token(text=token.value)  # Names validated inside `Tag`
      if token.type == 'STANDALONE_TOKEN':
        return _validate_tag_name(leaf, tags.Tags.STANDALONE_TAGS)
      return leaf
    elif token.type == 'TEXT':
      return Text(text=token.value)
    else:
      raise ValueError(f'Unknown token type: {token.type}')


def _validate_tag_name(
    token: Token,
    names: Container[str],
) -> Node:
  if token.name not in names:
    return Invalid(children=[token], reason=f'Unknown tag name: {token.name}')
  return token


def _make_tag_content(items: list[Node], tag_name: str) -> Node:
  """Makes a role node."""
  if tag_name in tags.Tags.ROLE_TAGS:
    return _make_role_tag_content(items)
  elif tag_name in tags.Tags.TOOL_TAGS:
    return _make_tool_tag_content(items)
  else:
    return Tree(children=items)


def _make_role_tag_content(items: list[Node]) -> Node:
  match items:
    # Split the text prefix into role and content.
    case [Text(text=text), *rest]:
      if '\n' not in text:
        reason = 'Invalid format. Expected `<|tag>role\\n`'
        return Invalid(children=items, reason=reason)

      role, content = text.split('\n', maxsplit=1)
      if not re.fullmatch(r'\w+', role):
        reason = 'Invalid role format.'
        return Invalid(children=items, reason=reason)

      children = [Role(text=f'{role}\n')]
      if content:
        children.append(Text(text=content))
      children.extend(rest)

      return Tree(children=children)
    case _:
      reason = 'Invalid format. Expected `<|tag>role\\n`'
      return Invalid(children=items, reason=reason)


def _make_tool_tag_content(items: list[Node]) -> Node:
  """Makes a tool tag content."""
  match items:
    case [Text(text=text)]:
      try:
        return tool_grammar.parse(text)
      except Exception as e:  # pylint: disable=broad-except
        reason = f'Invalid tool tag. Parsing failed: {e}'
        return Invalid(children=items, reason=reason)
    case _:
      reason = 'Invalid tool tag. Contains non-text nodes.'
      return Invalid(children=items, reason=reason)


@functools.cache
def _parser() -> lark.Lark:
  """Returns a lark parser for the tool grammar."""
  grammar = r"""
      # Orphan only appear at the top-level
      # We repeat the same rules for the element, so `unclosed_tag`
      # has higher priority than `TEXT`
      start: (closed_tag
              | unclosed_tag
              | orphan_close_token
              | TEXT
              | STANDALONE_TOKEN)*

      ?element: closed_tag
              | unclosed_tag
              | STANDALONE_TOKEN
              | TEXT

      # <|xxx> <xxx|>
      # Use .2 for higher priority
      closed_tag.2: OPEN_TOKEN element* CLOSE_TOKEN
      unclosed_tag: OPEN_TOKEN element*
      orphan_close_token: CLOSE_TOKEN

      OPEN_TOKEN: "<|" NAME ">"
      CLOSE_TOKEN: "<" NAME "|>"
      STANDALONE_TOKEN: "<|" NAME "|>"

      # Priority -2 to match everything else before raw text.
      TEXT.-2: /
          .+?  # Match anyhing not greedily (stop at first tag)
          (?=  # Lookahead
              # Note: This pupposely will capture `<|"|>` tokens, as those are
              # handled with the separate `tool_grammar.py` parser.
              \Z  # End of document
              | (<\|\w+>)  # Open tag
              | (<\w+\|>)  # Close tag
              | (<\|\w+\|>)  # Standalone tag
          )
      /xs

      %import common.CNAME -> NAME
  """
  return lark.Lark(grammar, start='start', parser='lalr')


def span(
    class_: str | list[str], **kwargs
) -> Callable[[str | Node | list[str | Node]], str]:
  """Returns a span."""
  if isinstance(class_, str):
    class_ = [class_]
  class_ = ' '.join(class_)

  attrs = []
  for k, v in kwargs.items():
    attrs.append(f'{k}="{v}"')
  attrs = ' '.join(attrs)

  def _apply(text: str | Node | list[str | Node]) -> str:
    if not isinstance(text, list):
      text = [text]
    text = [t.as_html() if isinstance(t, Node) else t for t in text]
    text = ''.join(text)
    return f'<span class="{class_}" {attrs}>{text}</span>'

  return _apply
