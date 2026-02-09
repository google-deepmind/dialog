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

"""Utility functions for working with JSON schemas."""

from __future__ import annotations

from etils import epy


def normalize_schema(schema: epy.typing.JsonDict) -> epy.typing.JsonDict:
  """Normalizes the schema.

  The [OpenAPI
  schema](https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.2.0.md)
  standard does not fully match the internal
  proto implementation.
  This function normalizes the schema to match the internal proto.

  OpenAPI schema:
  https://github.com/OAI/OpenAPI-Specification/blob/main/versions/3.0.2.md#schema-object

  Note: This function is quite forgiving, i.e. accepts non-standard schemas,
  instead, rely on the model being able to generalize to unseen specs.

  Args:
    schema: The schema to normalize.

  Returns:
    The normalized schema.
  """

  try:
    return _normalize_schema(schema)
  except Exception as e:  # pylint: disable=broad-except
    epy.reraise(
        e,
        prefix=f'Failed to normalize schema:\n{epy.pretty_repr(schema)}\n',
    )


def order_alphabetically[_JsonT](json: _JsonT) -> _JsonT:
  """Orders the JSON."""
  match json:
    case dict():
      return {k: order_alphabetically(v) for k, v in sorted(json.items())}
    case list():
      return [order_alphabetically(v) for v in json]
    case _:
      return json


def _normalize_schema(schema: epy.typing.JsonDict) -> epy.typing.JsonDict:
  """Normalizes the schema."""
  if not isinstance(schema, dict):
    raise ValueError(f'Schema must be a dict, got: `{epy.pretty_repr(schema)}`')
  schema = schema.copy()

  schema.pop('$schema', None)

  for key, normalize_fn in (
      ('type', _type_to_upper),
      ('properties', _normalize_schema_dict),
      ('items', _normalize_schema),
      ('allOf', _normalize_schema_list),
      ('anyOf', _normalize_schema_list),
      ('oneOf', _normalize_schema_list),
      ('not', _normalize_schema),
      # 'additionalProperties'
      # 'defs'
  ):
    if key in schema:
      schema[key] = normalize_fn(schema[key])

  return schema


def _normalize_schema_dict(
    schema_dict: epy.typing.JsonDict,
) -> epy.typing.JsonDict:
  """Normalizes a schema dict."""
  if not isinstance(schema_dict, dict):
    raise ValueError(
        f'Schema dict must be a dict, got: `{epy.pretty_repr(schema_dict)}`'
    )
  return {k: _normalize_schema(v) for k, v in schema_dict.items()}


def _normalize_schema_list(
    schema_list: list[epy.typing.JsonDict],
) -> list[epy.typing.JsonDict]:
  """Normalizes a schema list."""
  if not isinstance(schema_list, list):
    raise ValueError(
        f'Schema list must be a list, got: `{epy.pretty_repr(schema_list)}`'
    )
  return [_normalize_schema(v) for v in schema_list]


def _type_to_upper(type_: str | list[str]) -> str | list[str]:
  match type_:
    case str():
      return type_.upper()
    case list():  # Not officially supported, but commonly used.
      return [_type_to_upper(t) for t in type_]  # pytype: disable=bad-return-type
    case _:
      raise ValueError(f'Unsupported type: {type_}')
