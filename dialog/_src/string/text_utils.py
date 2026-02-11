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

"""Text utils."""

from __future__ import annotations

from dialog._src.string import text_grammar
from etils import epy

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top
  # pytype: disable=import-error
  import IPython.display
  # pytype: enable=import-error
  from dialog._src import widget as widget_lib
  from dialog._src import conversation
  # pylint: enable=g-import-not-at-top


# TODO(epot):
# - Text
#   - Json
#   - Convert back to Conversation.
# - Documentation:
#   - Conversation
#     - Text
#   - Tool
#   - Images, modailties
#   - Gemini Example compatibility (from/to, including tool use)
# - Conversation
#   - Add incomplete model turn when text export
#   - Add token escape for ctrl tokens (for user-text)
#   - Strip thoughts !!


class ConversationStr(str):
  """A conversation string."""

  def as_ast(self) -> text_grammar.Root:
    return text_grammar.parse(self)

  def as_html(self) -> str:
    return self.as_ast().as_html()

  def as_widget(self):
    return widget_lib.ConversationStr(conversation=self.as_html())

  def as_conversation(self) -> conversation.Conversation:
    """Returns the conversation."""
    return self.as_ast().as_conversation()

  def as_chunk(self) -> conversation.Chunk:
    """Returns the chunk."""
    return self.as_ast().as_chunk()

  def _ipython_display_(self) -> None:
    IPython.display.display(self.as_widget())
