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

r"""Tool helper."""

from __future__ import annotations

from collections.abc import Callable
import dataclasses
import sys
import textwrap
from typing import Any, ClassVar, Self, TypeGuard

from dialog._src import auto_register
from dialog._src import mixin_utils
from dialog._src import schema_utils
from dialog._src import tags
from dialog._src.string import text_utils
from etils import epy

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  import mcp
  import pydantic
  from dialog._src import conversation
  # pytype: enable=import-error  # pylint: enable=g-import-not-at-top


type ToolLike = (
    epy.typing.JsonDict
    | mcp.Tool
)

type ToolCallLike = (
    epy.typing.JsonDict
    | mcp.types.CallToolRequestParams
)

type ToolResponseLike = (
    epy.typing.JsonDict
    | mcp.types.CallToolResult
)

_MCP_TO_GEMMA = {
    'inputSchema': 'parameters',
    'outputSchema': 'response',
    'annotations': 'toolAnnotations',
}


class Tool(
    auto_register.RegisterSubclasses[ToolLike],
    mixin_utils.HasReprContent,
):
  """A normalized tool definition."""

  name: str

  # The unmodified original Json data (displayed).
  full_json: epy.typing.JsonDict

  def_json: epy.typing.JsonDict

  @property
  def parameters(self) -> epy.typing.JsonDict:
    return self.def_json.get('parameters', {}).get('properties', {})  # pylint: disable=attribute-error

  def as_text(self) -> str:
    """Returns the text of the tool definition."""

    return _json_to_text_root(
        name=self.name,
        content=self.def_json,
        tag=tags.Tags.TOOL.name,
        kind='declaration',
    )

  def __repr_content__(self) -> epy.typing.JsonDict:
    return self.full_json


class ToolCall(
    auto_register.RegisterSubclasses[ToolCallLike],
    mixin_utils.HasReprContent,
):
  """A normalized tool call."""

  name: str

  # The unmodified original Json data (displayed).
  full_json: epy.typing.JsonDict

  arguments: epy.typing.JsonDict

  def as_text(self) -> str:
    """Returns the text of the tool definition."""
    return _json_to_text_root(
        name=self.name,
        content=self.arguments,
        tag=tags.Tags.TOOL_CALL.name,
        kind='call',
    )

  def __repr_content__(self) -> epy.typing.JsonDict:
    return self.full_json


class ToolResponse(
    auto_register.RegisterSubclasses[ToolCallLike],
    mixin_utils.HasReprContent,
):
  """A normalized tool call."""

  name: str

  # The unmodified original Json data (displayed).
  full_json: epy.typing.JsonDict

  response: epy.typing.JsonDict

  # Part of the MCP specification, but never seen by the model.
  is_error: bool

  def as_text(self) -> str:
    """Returns the text of the tool definition."""
    return _json_to_text_root(
        name=self.name,
        content=self.response,
        tag=tags.Tags.TOOL_RESPONSE.name,
        kind='response',
    )

  def __repr_content__(self) -> epy.typing.JsonDict:
    return self.full_json


@dataclasses.dataclass(frozen=True)
class _MCPToolBase[_McpT: pydantic.BaseModel]:
  """Common base class for MCP tools."""

  data: _McpT  # pytype: disable=name-error

  # Use callable for lazy-imports.
  MCP_CLS: ClassVar[Callable[[], type[_McpT]]]  # pytype: disable=name-error
  DIALOG_CLS: ClassVar[Callable[[], type[conversation.Chunk]]]

  @classmethod
  def _from_data(cls, data: ToolLike) -> Self | None:
    """Converts a tool definition to MCP tool."""
    if isinstance(data, str):
      return _tool_from_text(data, cls.DIALOG_CLS())

    if isinstance(data, dict):  # Json to MCP
      mcp_cls = cls.MCP_CLS()
      try:
        data = mcp_cls.model_validate(data, strict=True)
      except Exception as e:  # pylint: disable=broad-except
        e.add_note(f'Error for: {epy.pretty_repr(data)}')
        raise
    if not cls._is_mcp(data):
      return None
    if data.model_extra and (extra := set(data.model_extra) - {'name'}):  # pytype: disable=attribute-error
      raise ValueError(
          f'Extra MCP fields not allowed: {sorted(extra)}, for {data}.'
      )
    return cls(data)

  @classmethod
  def is_data_supported_without_doubt(cls, data: Any) -> bool:
    return cls._is_mcp(data)

  @classmethod
  def _is_mcp(cls, data: Any) -> TypeGuard[_McpT]:  # pytype: disable=name-error
    return 'mcp' in sys.modules and isinstance(data, cls.MCP_CLS())

  @property
  def name(self) -> str:
    return self.data.name  # pylint: disable=attribute-error

  @property
  def full_json(self) -> epy.typing.JsonDict:
    return self.data.model_dump(exclude_none=True)


@dataclasses.dataclass(frozen=True)
class _MCPTool(_MCPToolBase['mcp.Tool'], Tool):
  """Tool definition from a MCP Json."""

  MCP_CLS = staticmethod(lambda: mcp.Tool)
  DIALOG_CLS = staticmethod(lambda: conversation.Tool)

  @classmethod
  def _from_data(cls, data: ToolLike) -> Self | None:
    # `inputSchema` is optional.
    if isinstance(data, dict) and 'inputSchema' not in data:
      data = dict(data)
      data['inputSchema'] = {}
      self = super()._from_data(data)
      assert self is not None
      self.data.inputSchema = None
      return self  # pytype: disable=bad-return-type
    return super()._from_data(data)

  @property
  def def_json(self) -> epy.typing.JsonDict:
    json = self.data.model_dump(exclude_none=True)

    # TODO(epot): Normalization / conversion of Json.

    # Unused fields.
    for field in (
        'name',  # Displayed above.
        # Should be `title` moved to `annotations.title` if present ?
        # Likely not as the model has likely seen very few title during
        # training.
        'title',
        'icons',
        'meta',
        'execution',
    ):
      _ = json.pop(field, None)

    # Rename fields to match Gemma expectations.
    for before, after in _MCP_TO_GEMMA.items():
      if before in json:
        json[after] = json.pop(before)

    if 'parameters' in json:
      json['parameters'] = schema_utils.normalize_schema(json['parameters'])

    if 'response' in json:
      json['response'] = schema_utils.normalize_schema(json['response'])

    # Finally, re-order to ensure consistent order.
    return schema_utils.order_alphabetically(json)


def as_mcp_def_dict(data: epy.typing.JsonDict) -> epy.typing.JsonDict:
  """Converts a tool definition to MCP Json."""
  data = data.copy()
  for after, before in _MCP_TO_GEMMA.items():
    if before in data:
      data[after] = data.pop(before)
  return data


@dataclasses.dataclass(frozen=True)
class _MCPToolCall(_MCPToolBase['mcp.types.CallToolRequestParams'], ToolCall):
  """Tool call from a MCP Json."""

  MCP_CLS = staticmethod(lambda: mcp.types.CallToolRequestParams)
  DIALOG_CLS = staticmethod(lambda: conversation.ToolCall)

  @property
  def arguments(self) -> epy.typing.JsonDict:
    return self.data.arguments or {}


@dataclasses.dataclass(frozen=True)
class _MCPToolResponse(_MCPToolBase['mcp.types.CallToolResult'], ToolResponse):
  """A normalized tool response."""

  REQUIRED_FIELDS = (
      'name',  # Non-standard MCP field, but required by Gemma.
      'structuredContent',
  )
  OPTIONAL_FIELDS = (
      'isError',
      'content',  # Unused.
  )
  MCP_CLS = staticmethod(lambda: mcp.types.CallToolResult)
  DIALOG_CLS = staticmethod(lambda: conversation.ToolResponse)

  @classmethod
  def _from_data(cls, data: ToolResponseLike) -> Self | None:
    if isinstance(data, dict) and 'content' not in data:
      data = dict(data)
      data['content'] = []
    return super()._from_data(data)  # pytype: disable=bad-return-type

  def __post_init__(self):
    if 'name' not in self.data.model_extra:
      raise ValueError(
          'ToolResponse requires a `name` field matching the ToolCall.\nGot:'
          f' {self.data!r}'
      )
    # MCP standard also supports `content` but only `structuredContent` is
    # supported by Gemma.
    # Could eventually auto-convert `content` to `structuredContent` ?
    if self.data.structuredContent is None:
      raise ValueError(
          'ToolResponse requires a `structuredContent` field.\nGot:'
          f' {self.data!r}'
      )

  @property
  def response(self) -> epy.typing.JsonDict:
    return self.data.structuredContent

  @property
  def is_error(self) -> bool:
    return self.data.isError


def response_to_compact_repr(json: epy.typing.JsonDict) -> str:
  """Returns a compact repr of the response."""
  if len(json) == 1:
    return _arg_to_compact_repr(list(json.values())[0])
  else:
    return '{' + args_to_compact_repr(json) + '}'


def args_to_compact_repr(items: epy.typing.JsonDict) -> str:
  """Returns a compact repr of the arguments."""
  args = []
  len_count = 0
  for k, v in items.items():
    v = _arg_to_compact_repr(v)
    arg = f'{k}={v}'
    len_count += len(arg)
    if len_count > 80:
      args.append('...')
      break
    else:
      args.append(arg)
  return ', '.join(args)


def _arg_to_compact_repr(v: epy.typing.Json) -> str:
  if isinstance(v, str):
    v = textwrap.shorten(v, width=20, placeholder='...')
    v = repr(v)
  else:
    v = textwrap.shorten(repr(v), width=20, placeholder='...')
  return v


def _json_to_text_root(
    *,
    name: str,
    content: epy.typing.JsonDict,
    tag: str,
    kind: str,
):
  content = _json_to_text(content)
  return f'<|{tag}>{kind}:{name}{content}<{tag}|>'


def _json_to_text(json: epy.typing.Json) -> str:
  """Returns the repr of the JSON."""
  # TODO(epot): Are the types normalized to UPPER or lower case ?
  if isinstance(json, str):
    return f'<|"|>{json}<|"|>'
  elif isinstance(json, dict):
    items = [f'{k}:{_json_to_text(v)}' for k, v in json.items()]  # pytype: disable=attribute-error
    content = ','.join(items)
    return '{' + content + '}'
  elif isinstance(json, list):
    values = [_json_to_text(v) for v in json]
    content = ','.join(values)
    return f'[{content}]'
  # TODO(epot): Add tests !!!!!
  elif isinstance(json, bool):
    return 'true' if json else 'false'
  elif isinstance(json, int):
    return str(json)
  elif isinstance(json, float):
    if json.is_integer():  # `1.0` is formatted as `1`  # pytype: disable=attribute-error
      return str(int(json))
    return str(json)
  elif isinstance(json, type(None)):
    return 'null'
  else:
    raise ValueError(f'Unsupported JSON type: {type(json)}')


def _tool_from_text[_McpT](text: str, dialog_cls: type[_McpT]) -> _McpT:
  """Converts a tool definition to MCP tool."""
  chunk = text_utils.ConversationStr.from_gemini_str(text).as_chunk()
  if not isinstance(chunk, dialog_cls):
    raise ValueError(
        f'Invalid tool text. Expected `{dialog_cls.__name__}` content. Got:'
        f' `{type(chunk).__name__}`. For: {text!r}'
    )
  return chunk.data
