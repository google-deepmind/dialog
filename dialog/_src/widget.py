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

"""Widget for dialog."""

import anywidget  # pytype: disable=import-error
from dialog._src import resources
import traitlets  # pytype: disable=import-error


class Conversation(anywidget.AnyWidget):
  """Widget for dialog.

  Attributes:
    conversation: The conversation HTML.
    stream_replace_tmp_html: Push the stream HTML of the model response.
    stream_add_final_html: Push the stream HTML of the model response.
  """

  _esm = resources.read('conversation.js')
  _css = resources.read('conversation.css')

  conversation = traitlets.Unicode().tag(sync=True)
  stream_replace_tmp_html = traitlets.Unicode().tag(sync=True)
  stream_add_final_html = traitlets.Unicode().tag(sync=True)


class ConversationStr(anywidget.AnyWidget):
  """Widget for text."""

  _esm = resources.read('conversation_str.js')
  _css = resources.read('conversation_str.css')

  conversation = traitlets.Unicode().tag(sync=True)
