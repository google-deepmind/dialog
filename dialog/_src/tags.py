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

"""Tags and constants."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Tag:
  """Tag (`<|tag><tag|>` or `<|tag|>`)."""

  name: str


@dataclasses.dataclass(frozen=True)
class ClosedTag(Tag):
  """Tag (`<|tag><tag|>`)."""

  @property
  def open(self) -> str:
    return f'<|{self.name}>'

  @property
  def close(self) -> str:
    return f'<{self.name}|>'


@dataclasses.dataclass(frozen=True)
class StandaloneTag(Tag):
  """Standalone tag (`<|tag|>`)."""

  @property
  def tag(self) -> str:
    return f'<|{self.name}|>'


def _as_dict(tags: list[Tag]) -> dict[str, Tag]:
  """Converts a list of tags to a dict."""
  return {tag.name: tag for tag in tags}


class Tags:
  """Tags."""

  # Structure
  TURN = ClosedTag('turn')
  CHANNEL = ClosedTag('channel')

  # Tools
  TOOL = ClosedTag('tool')
  TOOL_CALL = ClosedTag('tool_call')
  TOOL_RESPONSE = ClosedTag('tool_response')

  # Thinking
  THINK = StandaloneTag('think')
  # FAST_THINK_ON = StandaloneTag('fast_think')

  # Multimodal
  IMAGE = StandaloneTag('image')
  AUDIO = StandaloneTag('audio')
  VIDEO = StandaloneTag('video')

  QUOTE = StandaloneTag('"')

  # Closing tags <|tag><tag|>
  ROLE_TAGS = _as_dict([
      TURN,
      CHANNEL,
  ])
  TOOL_TAGS = _as_dict([
      TOOL,
      TOOL_CALL,
      TOOL_RESPONSE,
  ])
  CLOSED_TAGS = {
      **ROLE_TAGS,
      **TOOL_TAGS,
  }

  # Standalone tags <|tag|>
  THINKING_TAGS = _as_dict([
      THINK,
      # FAST_THINK_ON,
  ])
  MULTIMODAL_TAGS = _as_dict([
      IMAGE,
      AUDIO,
      VIDEO,
  ])
  STANDALONE_TAGS = {
      **THINKING_TAGS,
      # Multi-modal tags are used as standalone placeholder.
      **MULTIMODAL_TAGS,
  }

  # Mapping from tool kind to tag.
  # Tool kind is: `<|tool>declaration:name{}<tool|>`
  TOOL_KIND_TO_TAG = {
      'declaration': TOOL,
      'call': TOOL_CALL,
      'response': TOOL_RESPONSE,
  }
