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

"""Conversation class."""

from __future__ import annotations

import abc
from collections.abc import Iterable, Iterator, Sequence
import dataclasses
import functools
import html
import itertools
from typing import ClassVar, Self, TypeVar

from dialog._src import audio_helper
from dialog._src import auto_register
from dialog._src import html_helper
from dialog._src import img_helper
from dialog._src import mixin_utils
from dialog._src import resources
from dialog._src import tool_helper
from dialog._src.string import text_utils
from etils import enp
from etils import epy

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top
  import IPython.display  # pytype: disable=import-error
  from dialog._src import widget as widget_lib
  from dialog._src import streaming

  # pylint: enable=g-import-not-at-top


_T = TypeVar('_T')

# TODO(epot): More clear annotations (i.e. `Turns` vs `Turn`)
type TurnLike = (
    # Guard
    Conversation  # pytype: disable=name-error
    | str  # Reverse string parsing (`<|turn>user\n...<turn|>`)
    | Turn  # pytype: disable=name-error
    | Sequence[TurnLike]  # pytype: disable=name-error
)
type ChunkLike = (
    str
    | Chunk  # pytype: disable=name-error
    # A subset of the data is directly supported. i.e. One can do:
    # dialog.User(np.array((h, w, 3), dtype=np.uint8))
    # As alias for:
    # dialog.User(dialog.Image(np.array((h, w, 3), dtype=np.uint8)))
    # However, not all types are supported. i.e.
    # dialog.User('/path/to/image.png')  # creates a `dialog.Text`
    # vs
    # dialog.User(dialog.Image('/path/to/image.png'))
    | tool_helper.ToolLike
    | img_helper.ImageLike
)  # pytype: disable=name-error


# MARK: Conversation
class Conversation(
    mixin_utils.AddRepr,
    mixin_utils.Sequence['Turn', TurnLike],
):
  """A conversation."""

  REPR_CONTENT = 'turns'
  SEQUENCE_ATTRIBUTE = 'turns'

  def __init__(self, *turns: TurnLike):
    """Initializes the conversation."""

    turns = itertools.chain.from_iterable(
        Turn.turns_from_data(t) for t in turns
    )
    turns = list(turns)
    # Merge turns from the same type together.
    turns = _merge_similar_turns(turns)
    self.turns: list[Turn] = turns

  def as_text(self, training: bool = False) -> str:
    r"""Returns the text of the conversation.

    Text can be formatted for both training or inference. This affect
    how the last turn is formatted.

    * For training: Always ends by the end of turn token `<turn|>`
    * For inference:
        * Add an empty model turn at the end if missing: `<|turn>model\n`.
        * If conversation starts/ends with a model turn, do not add the
          beginning/end of turn tokens (`<|turn>`/`<turn|>`) on the edges. This
          helps concatenation with previous turns.

      The resulting text is the prompt to pre-fill the model.

    ```
    conv = Conversation(User('Hello'))

    conv.as_text(training=True)  == '''<|turn>user
    Hello<turn|>'''

    conv.as_text()  == '''<|turn>user
    Hello<turn|>
    <|turn>model
    '''
    ```

    Args:
      training: If True, the text is formatted for training. If False, the text
        is formatted for inference.

    Returns:
      The text `str` of the conversation.
    """

    if not self.turns:
      return ''

    # TODO(epot): Currently, internal formatter will not add `<|turn>model`
    # if the first turn is model. Should we do the same here ?

    # Inference mode, do not open or close model turns on the edges (to
    # help concatenation with previous turns).
    if not training:
      turns = self.turns
      if not isinstance(turns[-1], Model):
        turns = turns + [Model()]

      first_open = not isinstance(turns[0], Model)
      last_closed = not isinstance(turns[-1], Model)
      num_other_turns = len(turns) - 1
      opens = [first_open] + [True] * num_other_turns
      closes = [True] * num_other_turns + [last_closed]

      text = [
          turn.as_text(open=o, closed=c)
          for turn, o, c in zip(turns, opens, closes)
      ]
    else:
      text = [turn.as_text() for turn in self.turns]

    text = '\n'.join(text)
    return text_utils.ConversationStr(text)

  def as_html(
      self,
      *,
      collapsed: bool = False,
  ) -> str:
    """Returns the HTML of the conversation."""
    return html_helper.collapsible(
        content=[turn.as_html() for turn in self.turns],
        summary=html_helper.summary(
            title='Conversation',
            subtitle=f'({len(self.turns)} turns)',
            icons=self.title_icon,
            collapsible=bool(self.turns),
            is_collapsed=False,
        ),
        open=not collapsed,
    )

  def as_widget(self, collapsed: bool = False) -> widget_lib.Conversation:
    """Returns the widget of the conversation."""
    body = self.as_html(collapsed=collapsed)
    html_str = resources.read('elements.html') + body
    widget = widget_lib.Conversation(conversation=html_str)
    streaming.connect_stream(widget, self)
    return widget

  def merge_text(self) -> Self:
    """Merges all text chunks into a single text chunk."""
    return type(self)([t.merge_text() for t in self.turns])

  @property
  def title_icon(self) -> html_helper.IconSet:
    # Remove thinking icons
    icons = html_helper.IconSet([c.title_icon for c in self.turns])
    return icons

  def show(self, *, collapsed: bool = False) -> None:
    """Shows the conversation."""
    IPython.display.display(self.as_widget(collapsed=collapsed))

  def _ipython_display_(self) -> None:
    IPython.display.display(self.as_widget())

  def strip_thoughts(self, *, keep_last_turn: bool = True) -> Self:
    """Strips the thoughts from the conversation.

    Args:
      keep_last_turn: If True, the thoughts from the last model turn are kept.

    Returns:
      The conversation with the thoughts stripped.
    """
    if not self.turns:
      return self

    # Do not strip thoughts from the last turn if requested.
    if keep_last_turn:
      *prev_turns, last_turn = self.turns
      new_turns = [t.strip_thoughts() for t in prev_turns] + [last_turn]
    else:
      new_turns = [t.strip_thoughts() for t in self.turns]
    return type(self)(new_turns)

  # TODO(epot): `as_numpy` example


# MARK: Turns
class Turn(
    mixin_utils.AddRepr,
    mixin_utils.Sequence['Chunk', ChunkLike],
):
  """A turn in a conversation."""

  chunks: list[Chunk]
  ROLE: ClassVar[str]

  REPR_CONTENT = 'chunks'
  SEQUENCE_ATTRIBUTE = 'chunks'
  ADD_NOT_IMPLEMENTED_CLS = (Conversation,)

  ROLE_TO_CLS: ClassVar[dict[str, type[Turn]]] = {}

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    cls.ROLE_TO_CLS[cls.ROLE] = cls

  @classmethod
  def turns_from_data(cls, data: TurnLike) -> list[Self]:
    """Creates a turn from data."""
    if isinstance(data, Turn):
      return [data]
    elif isinstance(data, list | tuple):
      return list(
          itertools.chain.from_iterable(cls.turns_from_data(t) for t in data)
      )
    elif isinstance(data, Conversation):
      return data.turns
    elif isinstance(data, str):
      # Warning: This could potentially hide issues if users mix ctrl with
      # Gemma tokens.
      return (
          text_utils.ConversationStr(data)
          .as_conversation().turns
      )
    else:
      raise TypeError(f'Unsupported turn type: `{type(data).__name__}`.')

  def __init__(self, *chunks: ChunkLike):
    chunks = list(_flatten(chunks))
    self.chunks = [Chunk.from_data(c) for c in chunks]

  def as_text(self, *, closed: bool = True, open: bool = True) -> str:  # pylint: disable=redefined-builtin
    """Returns the text of the turn.

    Args:
      closed: If False, do not add the end-of-turn `<turn|>` token. Used to add
        the start of the incomplete next model turn.
      open: If True, do not add the beginning-of-turn `<|turn>` token.

    Returns:
      The text of the turn.
    """
    return _chunks_to_text(
        self.chunks,
        tag='turn',
        role=self.ROLE,
        closed=closed,
        open=open,
    )

  def as_html(self, collapsed: bool = False) -> str:
    """Returns the HTML of the turn."""
    return html_helper.collapsible(
        content=_merge_inline_chunks(self.chunks),
        summary=html_helper.summary(
            title=self.ROLE,
            icons=self.title_icon,
            collapsible=any(c.COLLAPSIBLE for c in self.chunks),
            is_collapsed=True,
        ),
        open=not collapsed,
        class_=self.ROLE,
    )

  def as_widget(self, collapsed: bool = False) -> widget_lib.Conversation:
    """Returns the widget of the conversation."""
    body = self.as_html(collapsed=collapsed)
    html_str = resources.read('elements.html') + body
    return widget_lib.Conversation(conversation=html_str)

  def merge_text(self) -> Self:
    """Merges all text chunks into a single text chunk."""
    return type(self)(_merge_text_chunks(self.chunks))

  def _ipython_display_(self) -> None:
    IPython.display.display(self.as_widget())

  @functools.cached_property
  def title_icon(self) -> html_helper.IconSet:
    """Returns whether the turn has tools."""
    return html_helper.IconSet([c.title_icon for c in self.chunks])

  def strip_thoughts(self) -> Self:
    """Strips the thoughts from the turn."""
    chunks = [c for c in self.chunks if not isinstance(c, Thought)]
    return type(self)(chunks)


class User(Turn):
  """A user turn."""

  ROLE = 'user'


class Model(Turn):
  """An assistant turn."""

  ROLE = 'model'


class System(Turn):
  """A system turn."""

  ROLE = 'system'


# MARK: Chunks
class Chunk(auto_register.RegisterSubclasses[ChunkLike], abc.ABC):
  """A chunk of text in a turn."""

  # Inline chunks are merged together into a single html tag.
  INLINE: ClassVar[bool] = False
  COLLAPSIBLE: ClassVar[bool] = False
  ICON: ClassVar[html_helper.Icon] | None = None

  @abc.abstractmethod
  def as_text(self) -> str:
    """Returns the text of the chunk."""
    raise NotImplementedError()

  @abc.abstractmethod
  def as_html(self) -> str:
    """Returns the text of the chunk."""
    raise NotImplementedError()

  @property
  def title_icon(self) -> html_helper.Icon | None:
    """Returns the title icon of the chunk."""
    if self.ICON:
      return self.ICON
    return None


class Thought(
    Chunk,
    mixin_utils.AddRepr,
    mixin_utils.Sequence['Chunk', ChunkLike],
):
  """A thought chunk."""

  COLLAPSIBLE = True
  ICON = html_helper.Icon(emoji='💭', tooltip='Thought')

  REPR_CONTENT = 'chunks'
  SEQUENCE_ATTRIBUTE = 'chunks'

  def __init__(self, *chunks: ChunkLike):
    self.chunks = [Chunk.from_data(c) for c in _flatten(chunks)]

  def as_text(self) -> str:
    """Returns the text of the chunk."""
    return _chunks_to_text(
        self.chunks,
        tag='channel',
        role='thought',
    )

  def as_html(self) -> str:
    """Returns the text of the chunk."""
    num_chars = 0
    for c in self.chunks:
      if isinstance(c, Text):
        num_chars += len(c.text)

    return html_helper.collapsible(
        content=_merge_inline_chunks(self.chunks),
        summary=html_helper.summary(
            title='Thought',
            subtitle=f'({num_chars:,} characters)',
            icons=self.title_icon.remove(self.ICON),
            collapsible=any(c.COLLAPSIBLE for c in self.chunks),
            is_collapsed=True,
        ),
        open=False,
    )

  def merge_text(self) -> Self:
    """Merges all text chunks into a single text chunk."""
    return type(self)(_merge_text_chunks(self.chunks))

  @property
  def title_icon(self) -> html_helper.IconSet:
    return html_helper.IconSet(
        [self.ICON] + [c.title_icon for c in self.chunks]
    )


@dataclasses.dataclass(frozen=True, repr=False)
class Text(Chunk, mixin_utils.AddRepr):
  """A text chunk."""

  text: str

  INLINE = True

  REPR_CONTENT = 'text'

  @classmethod
  def _from_data(cls, data: ChunkLike) -> Self | None:
    if not isinstance(data, str):
      return None
    return cls(text=data)

  def as_text(self) -> str:
    return self.text

  def as_html(self) -> str:
    if not self.text:
      return ''
    return f'<span>{html.escape(self.text)}</span>'


class Audio(Chunk, mixin_utils.AddRepr):
  """An audio chunk."""

  ICON = html_helper.Icon(emoji='🔊', tooltip='Audio')

  REPR_CONTENT = 'data'

  def __init__(self, data: audio_helper.AudioLike):
    self.data = audio_helper.Audio.from_data(data)

  @classmethod
  def _from_data(cls, data: ChunkLike) -> Self | None:
    if audio_helper.Audio.is_data_supported_without_doubt(data):
      return cls(data)
    return None

  def as_text(self) -> str:
    """Returns the text of the chunk."""
    return '<|audio|>'  # Audio placeholder

  def as_html(self) -> str:
    """Returns the HTML of the chunk."""
    return f'<audio controls src="{self.data.html_src}"></audio>'


class Image(Chunk, mixin_utils.AddRepr):
  """An image chunk."""

  ICON = html_helper.Icon(emoji='🖼️', tooltip='Image')

  REPR_CONTENT = 'data'

  def __init__(self, data: img_helper.ImageLike):
    self.data = img_helper.Image.from_data(data)

  @classmethod
  def _from_data(cls, data: ChunkLike) -> Self | None:
    if img_helper.Image.is_data_supported_without_doubt(data):
      return cls(data)
    return None

  @classmethod
  def _debug_msg_from_data(cls, data: ChunkLike) -> str | None:
    if enp.lazy.is_array(data):
      return (
          'Images are expected to be 3D uint8 arrays, got'
          f' shape={data.shape}, dtype={data.dtype}'  # pylint: disable=attribute-error  # pytype: disable=attribute-error
      )

  def as_text(self) -> str:
    """Returns the text of the chunk."""
    return '<|image|>'  # Image placeholder

  def as_html(self) -> str:
    """Returns the HTML of the chunk."""
    if self.data.missing:
      return '<img class="missing" src="" />'
    return f'<img src="{self.data.html_src}" />'


class Tool(Chunk, mixin_utils.AddRepr):
  """A tool definition chunk."""

  COLLAPSIBLE = True
  ICON = html_helper.Icon(emoji='🔨', tooltip='Tool use')

  REPR_CONTENT = 'data'

  def __init__(self, data: tool_helper.ToolLike):
    self.data = tool_helper.Tool.from_data(data)

  @classmethod
  def _from_data(cls, data: ChunkLike) -> Self | None:
    if tool_helper.Tool.is_data_supported_without_doubt(data):
      return cls(data)
    return None

  @classmethod
  def _debug_msg_from_data(cls, data: ChunkLike) -> str | None:
    # Only add a debug message for `Tool` as the message will be
    # the same for `ToolCall` and `ToolResponse`.
    if isinstance(data, dict):
      return (
          'Json dict cannot be provided directly, but should be wrapped in'
          ' `dialog.Tool(json)`, `dialog.ToolCall(json)`,...'
      )

  def as_text(self) -> str:
    """Returns the text of the chunk."""
    return self.data.as_text()  # pytype: disable=attribute-error

  def as_html(self) -> str:
    """Returns the text of the chunk."""
    parameters = ', '.join(self.data.parameters.keys())

    return html_helper.collapsible(
        summary=html_helper.summary(
            # TODO(epot): Add emoji ?
            title=self.data.name,
            subtitle=f'({parameters})',
        ),
        content=html_helper.json_to_html(self.data.full_json),
        open=False,
    )


class ToolCall(Chunk, mixin_utils.AddRepr):
  """A tool call chunk."""

  COLLAPSIBLE = True
  ICON = Tool.ICON

  REPR_CONTENT = 'data'

  def __init__(self, data: tool_helper.ToolCallLike):
    self.data = tool_helper.ToolCall.from_data(data)

  @classmethod
  def _from_data(cls, data: ChunkLike) -> Self | None:
    if tool_helper.ToolCall.is_data_supported_without_doubt(data):
      return cls(data)
    return None

  def as_text(self) -> str:
    """Returns the text of the chunk."""
    return self.data.as_text()  # pytype: disable=attribute-error

  def as_html(self) -> str:
    """Returns the text of the chunk."""
    args = tool_helper.args_to_compact_repr(self.data.arguments)

    return html_helper.collapsible(
        summary=html_helper.summary(
            title=self.data.name,
            subtitle=f'({args})',
        ),
        content=html_helper.json_to_html(self.data.full_json),
        open=False,
    )


class ToolResponse(Chunk, mixin_utils.AddRepr):
  """A tool response chunk."""

  COLLAPSIBLE = True
  ICON = Tool.ICON
  FAILED_ICON = html_helper.Icon(emoji='❌', tooltip='Tool call failed')

  REPR_CONTENT = 'data'

  def __init__(self, data: tool_helper.ToolResponseLike):
    self.data = tool_helper.ToolResponse.from_data(data)

  @classmethod
  def _from_data(cls, data: ChunkLike) -> Self | None:
    if tool_helper.ToolResponse.is_data_supported_without_doubt(data):
      return cls(data)
    return None

  def as_text(self) -> str:
    """Returns the text of the chunk."""
    return self.data.as_text()  # pytype: disable=attribute-error

  def as_html(self) -> str:
    """Returns the text of the chunk."""
    response = tool_helper.response_to_compact_repr(self.data.response)

    return html_helper.collapsible(
        summary=html_helper.summary(
            title=self.data.name,
            icons=self.title_icon.remove(self.ICON),
            subtitle=f'-> {response}',
        ),
        content=html_helper.json_to_html(self.data.full_json),
        open=False,
    )

  @property
  def title_icon(self) -> html_helper.IconSet:
    icons = [self.ICON]
    if self.data.is_error:  # pytype: disable=attribute-error
      icons.append(self.FAILED_ICON)
    # TODO(epot): Also include image/audio answers.
    return html_helper.IconSet(icons)


@dataclasses.dataclass(frozen=True, repr=False)
class ControlToken(mixin_utils.AddRepr, Chunk):
  """Base class for control token chunk."""

  name: str

  INLINE = True

  REPR_CONTENT = 'name'
  NAME_TO_CLS: ClassVar[dict[str, type[ControlToken]]] = {}

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)
    name = cls.name  # pytype: disable=attribute-error
    if isinstance(name, dataclasses.Field):
      name = name.default
    cls.NAME_TO_CLS[name] = cls

  def as_text(self) -> str:
    """Returns the text of the chunk."""
    return f'<|{self.name}|>'

  def as_html(self) -> str:
    """Returns the text of the chunk."""
    parts = []
    if self.ICON:
      parts.append(self.ICON.as_html())
    parts.append(self.name)

    text = ' '.join(parts)
    return f'<span class="control-token">{text}</span>'


@dataclasses.dataclass(frozen=True, repr=False)
class Think(ControlToken):
  """Thinking-on chunk."""

  name: str = dataclasses.field(default='think', init=False)
  ICON = html_helper.Icon(emoji='🤔', tooltip='Thinking activated')
  REPR_CONTENT = None


# @dataclasses.dataclass(frozen=True, repr=False)
# class FastThinkOn(ControlToken):
#   """Activate fast think chunk (default if no thinking is activated)."""

#   name: str = dataclasses.field(default='fast_think', init=False)
#   ICON = html_helper.Icon(emoji='⚡', tooltip='Fast thinking')
#   REPR_CONTENT = None


class Stream(Chunk, mixin_utils.AddRepr):
  """A special chunk for streaming model responses as they get generated.

  Usage:

  ```python
  stream = dialog.Stream()
  response = dialog.Model(stream)
  response.show()  # Display the empty model response.

  # Update the response as chunks are streamed in.
  stream.add('Hello')
  stream.add(' world!')
  ```
  """

  ICON = html_helper.Icon(emoji='📡', tooltip='Stream')

  def __init__(self):
    self.handler = streaming.StreamHandler()

  def as_text(self) -> str:
    """Returns the text of the chunk."""
    raise ValueError('Stream cannot be converted to text.')

  def as_html(self) -> str:
    """Returns the HTML of the chunk."""
    return '<div class="stream"><div class="stream-tmp"></div></div>'

  def set_widget(self, widget: widget_lib.Conversation) -> None:
    """Sets the widget of the stream."""
    self.handler.widget = widget

  def add(self, text: str) -> None:
    """Adds text to the stream."""
    self.handler.add(text)


# TODO(epot): Move internal utils to a separate file.
# MARK: Utils


class _Inline(list[str]):

  def as_html(self) -> str:
    """Returns the HTML of the inline chunks."""
    content = ''.join(self)
    if not content:
      return ''
    return f'<p>{content}</p>'


def _merge_text_chunks(chunks: list[Chunk]) -> list[Chunk]:
  """Merges text chunks into a single chunk."""
  new_chunks = []
  for chunk in chunks:
    if isinstance(chunk, Text):
      if new_chunks and isinstance(new_chunks[-1], Text):
        new_chunks[-1] = Text(new_chunks[-1].text + chunk.text)
      else:
        new_chunks.append(chunk)
    elif isinstance(chunk, Thought):
      new_chunks.append(chunk.merge_text())
    else:
      new_chunks.append(chunk)
  return new_chunks


def _merge_inline_chunks(chunks: list[Chunk]) -> str:
  """Merges inline chunks into a single HTML tag."""
  parts = []
  for chunk in chunks:
    value = chunk.as_html()

    if chunk.INLINE:
      if parts and isinstance(parts[-1], _Inline):
        parts[-1].append(value)
      else:
        parts.append(_Inline([value]))
    else:
      parts.append(value)

  # Merge all inline chunks together.
  parts = [p.as_html() if isinstance(p, _Inline) else p for p in parts]
  return ''.join(parts)


STREAM_ERROR_ICON = html_helper.Icon(emoji='❌', tooltip='Invalid format!')


html_helper.register_icon_order([
    # Thoughts
    Think.ICON,
    # FastThinkOn.ICON,
    Thought.ICON,
    # Tool use
    Tool.ICON,
    ToolResponse.FAILED_ICON,
    # Multimodal
    Image.ICON,
    Audio.ICON,
    # Streaming
    Stream.ICON,
    STREAM_ERROR_ICON,
])


def _chunks_to_text(
    chunks: list[Chunk],
    *,
    tag: str,
    role: str,
    closed: bool = True,
    open: bool = True,  # pylint: disable=redefined-builtin
) -> str:
  """Converts a list of chunks to text."""
  content = ''
  if open:
    content += f'<|{tag}>{role}\n'
  content += ''.join(chunk.as_text() for chunk in chunks)
  if closed:
    content += f'<{tag}|>'
  return content


def _flatten(values: Iterable[_T]) -> Iterator[_T]:
  """Flattens a list of turns."""
  for v in values:
    if isinstance(v, list | tuple):
      yield from _flatten(v)
    else:
      yield v


def _merge_similar_turns(turns: list[Turn]) -> list[Turn]:
  """Merges similar turns."""
  if not turns:
    return []
  if len(turns) == 1:
    return turns

  all_turns = []
  curr_turn, *other_turns = turns
  for turn in other_turns:
    if curr_turn.ROLE == turn.ROLE:
      curr_turn = type(curr_turn)(
          curr_turn.chunks,
          turn.chunks,
      )
    else:
      all_turns.append(curr_turn)
      curr_turn = turn

  all_turns.append(curr_turn)
  return all_turns
