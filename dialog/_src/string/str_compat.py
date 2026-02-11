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

"""Format string utilities."""

import enum
import functools
import re

from dialog._src import tags


class Format(enum.StrEnum):
  # fmt: off
  """The format of the string.

  Attributes:
    GEMMA4: `<|turn>`,... tokens.
    GEMMA3: `<start_of_turn>`,... tokens.
  """
  # fmt: on

  GEMMA4 = enum.auto()  # <|turn>
  GEMMA3 = enum.auto()  # <start_of_turn>

  def from_gemma4(self, text: str) -> str:
    """Convert the text from GEMMA format to the self format."""
    match self:
      case self.GEMMA4:
        return text
      case _:
        return _sub_xx_to_yy(self.re_gemma4_to_self, text, self.GEMMA4_TO_SELF)

  def to_gemma4(self, text: str) -> str:
    """Convert the text from the self format to GEMMA format."""
    match self:
      case self.GEMMA4:
        return text
      case _:
        return _sub_xx_to_yy(self.re_self_to_gemma4, text, self.SELF_TO_GEMMA4)

  @functools.cached_property
  def SELF_TO_GEMMA4(self) -> dict[str, str]:  # pylint: disable=invalid-name
    """Returns the mapping from the self format to GEMMA format."""
    return {
        self.GEMMA4: {},
        self.GEMMA3: _GEMMA3_TO_GEMMA4,
    }[self]

  @functools.cached_property
  def GEMMA4_TO_SELF(self) -> dict[str, str]:  # pylint: disable=invalid-name
    """Returns the mapping from GEMMA format to the self format."""
    return {v: k for k, v in self.SELF_TO_GEMMA4.items()}

  @functools.cached_property
  def re_self_to_gemma4(self) -> re.Pattern[str]:
    """Returns a regex to match the self format tokens."""
    return _compile_re(self.SELF_TO_GEMMA4)

  @functools.cached_property
  def re_gemma4_to_self(self) -> re.Pattern[str]:
    """Returns a regex to match the self format tokens."""
    return _compile_re(self.GEMMA4_TO_SELF)

_GEMMA3_TO_GEMMA4 = {
    # Tool use
    '<start_function_declaration>': tags.Tags.TOOL.open,  # <|tool>
    '<end_function_declaration>': tags.Tags.TOOL.close,  # <tool|>
    '<start_function_call>': tags.Tags.TOOL_CALL.open,  # <|tool_call>
    '<end_function_call>': tags.Tags.TOOL_CALL.close,  # <tool_call|>'
    '<start_function_response>': tags.Tags.TOOL_RESPONSE.open,
    '<end_function_response>': tags.Tags.TOOL_RESPONSE.close,
    '<escape>': tags.Tags.QUOTE.tag,  # <|"|>
    # Thinking
    '<unusedXX>': tags.Tags.THINK.tag,  # <|think|>  # << TODO!!!
    # '<ctrl93>': tags.Tags.FAST_THINK_ON.tag,  # <|fast_think|>
    '<start_receiver>': tags.Tags.CHANNEL.open,  # <|channel>
    '<end_receiver>': tags.Tags.CHANNEL.close,  # <channel|>
    # Turns
    '<start_of_turn>': tags.Tags.TURN.open,  # <|turn>
    '<end_of_turn>': tags.Tags.TURN.close,  # <turn|>
    # Multi-modal tokens
}


def _compile_re(mapping: dict[str, str]) -> re.Pattern[str]:
  """Compile the regex from the mapping keys."""
  keys = sorted(mapping.keys())
  keys = map(re.escape, keys)
  return re.compile('|'.join(keys))


def _sub_xx_to_yy(
    pattern: re.Pattern[str],
    text: str,
    mapping: dict[str, str],
) -> str:
  """Replace the `<ctrlXX>` tokens by `<|...|>` tokens."""
  return pattern.sub(lambda match: mapping[match.group(0)], text)
