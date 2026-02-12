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

"""Streaming utils."""

from __future__ import annotations

import dataclasses
import html

from dialog._src import conversation
from dialog._src import html_helper
from dialog._src import tags
from dialog._src import widget as widget_lib
from dialog._src.string import str_compat
from dialog._src.string import text_utils


_TAGS = (
    tags.Tags.CHANNEL,
    tags.Tags.TOOL_CALL,
    tags.Tags.TOOL_RESPONSE,
)


@dataclasses.dataclass(kw_only=True)
class StreamHandler:
  """A stream."""

  widget: widget_lib.Conversation | None = None

  data: str = ''
  tag: tags.ClosedTag | None = None

  def add(self, text: str) -> None:
    """Adds text to the stream."""
    if not self.widget:
      raise ValueError(
          'To use `dialog.Stream`, you first need to display the associated'
          ' conversation.'
      )

    if (
        not self.tag
        and text.startswith(tuple(t.open for t in _TAGS))
        and text.endswith(tuple(t.close for t in _TAGS))
    ):
      assert not self.data
      self.widget.stream_add_final_html = _str_to_final_html(text)
    # Standard case: Add text at the top level.
    elif not self.tag and text not in (t.open for t in _TAGS):
      assert not self.data
      self.widget.stream_add_final_html = html.escape(text)
    # Closing tag: Finalize the current chunk.
    elif self.tag and text.endswith(self.tag.close):
      self.data += text
      self.widget.stream_add_final_html = _str_to_final_html(self.data)
      self.widget.stream_replace_tmp_html = ''
      self.data = ''
      self.tag = None
    # Currently open tag.
    elif self.tag:
      self.data += text
      self.widget.stream_replace_tmp_html = _str_to_tmp_html(
          self.tag, self.data
      )
    # Opening tag.
    elif not self.tag and text in (t.open for t in _TAGS):
      self.tag = {t.open: t for t in _TAGS}[text]
      self.data += text
      self.widget.stream_replace_tmp_html = _str_to_tmp_html(
          self.tag, self.data
      )
    else:  # Unexpected case.
      raise RuntimeError('Should never happen.')


def _str_to_final_html(data: str) -> str:
  """Converts a string to a final HTML."""

  # Hack: to parse, we first wrap in a full turn, and then un-wrap the chunks.
  conv = text_utils.ConversationStr(f'<|turn>model\n{data}<turn|>')

  try:
    (turn,) = conv.as_conversation()
  except Exception:  # pylint: disable=broad-exception-caught
    return html_helper.collapsible(
        summary=html_helper.summary(
            title='Invalid format!',
            icons=html_helper.IconSet([conversation.STREAM_ERROR_ICON]),
            subtitle=f'({len(data):,} characters)',
        ),
        content='<p>' + html.escape(data) + '</p>',
        open=True,
    )
  else:
    return conversation.merge_inline_chunks(turn.chunks)


def _str_to_tmp_html(tag: tags.ClosedTag, data: str) -> str:
  """Converts a string to a temporary HTML."""
  return html_helper.collapsible(
      summary=html_helper.summary(
          title=tag.name,
          subtitle=f'({len(data):,} characters)',
      ),
      content='<p>' + html.escape(data) + '</p>',
      open=True,
  )


def connect_stream(
    widget: widget_lib.Conversation,
    conv: conversation.Conversation | conversation.Turn,
) -> None:
  """Connects the stream to the widget."""
  if isinstance(conv, conversation.Turn):
    conv = [conv]
  for turn in conv:
    for chunk in turn:
      if isinstance(chunk, conversation.Stream):
        chunk.set_widget(widget)
