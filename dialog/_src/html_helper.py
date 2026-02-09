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

"""Helper functions to build HTML."""

from __future__ import annotations

import dataclasses

from etils import epy


_COLLAPSIBLE_TEMPLATE = """\
<details {open} {class_}>
  <summary>{summary}</summary>
  <div class="content">{content}</div>
</details>
"""

ICON_ORDER = []


def register_icon_order(icons: list[IconBase]) -> None:
  """Registers the icon order."""
  ICON_ORDER.extend(icons)


class IconBase:

  def as_html(self) -> str:
    raise NotImplementedError()


@dataclasses.dataclass(frozen=True, eq=True)
class Icon(IconBase):
  """An icon."""

  emoji: str
  tooltip: str | None = None

  def as_html(self, *, root: bool = True) -> str:
    """Returns the HTML of the icon."""
    if root:
      return f'<span class="title-icon">{self.as_html(root=False)}</span>'

    if self.tooltip:
      tooltip = f'title="{self.tooltip}"'
    else:
      tooltip = ''
    return f'<span {tooltip}>{self.emoji}</span>'


class IconSet(IconBase):
  """A set of icons."""

  icons: set[Icon]

  def __init__(self, icons: list[IconBase | None]):
    filtered_icons = set()
    for icon in icons:
      if icon is None:
        continue
      elif isinstance(icon, Icon):
        filtered_icons.add(icon)
      elif isinstance(icon, IconSet):
        filtered_icons.update(icon.icons)
      else:
        raise ValueError(f'Unsupported icon type: {type(icon)}')
    self.icons = filtered_icons

  def remove(self, icon: Icon) -> IconSet:
    """Returns a new IconSet without the given icon."""
    return IconSet([i for i in self.icons if i != icon])

  def as_html(self) -> str:
    """Returns the HTML of the icon set."""
    sorted_icons = []
    for icon in ICON_ORDER:
      if icon in self.icons:
        sorted_icons.append(icon)

    if extra_icons := (set(self.icons) - set(ICON_ORDER)):
      raise ValueError(f'Unsorted icons: {extra_icons}')

    html_str = ''.join(icon.as_html(root=False) for icon in sorted_icons)
    return f'<span class="title-icon">{html_str}</span>'


def collapsible(
    *,
    summary: str,  # pylint: disable=redefined-outer-name
    content: str | list[str],
    open: bool = False,  # pylint: disable=redefined-builtin
    class_: str | None = None,
) -> str:
  """Returns a collapsible HTML element."""
  open_str = 'open' if open else ''
  class_str = f'class="{class_}"' if class_ else ''
  if not isinstance(content, str):
    content = ''.join(content)
  return _COLLAPSIBLE_TEMPLATE.format(
      summary=summary,
      content=content,
      open=open_str,
      class_=class_str,
  )


def summary(
    *,
    title: str,
    subtitle: str | None = None,
    icons: IconSet | None = None,
    collapsible: bool = False,  # pylint: disable=redefined-outer-name
    is_collapsed: bool | None = None,
) -> str:
  """Returns the HTML of the summary."""
  parts = [title]

  if subtitle:
    parts.append(f'<span class="subtitle">{subtitle}</span>')

  if icons and icons.icons:
    parts.append(icons.as_html())

  if collapsible:
    assert is_collapsed is not None
    parts.append(_collapse_icon(is_collapsed=is_collapsed))

  return ''.join(parts)


_ICON_TEMPLATE = """\
<div class="collapse-icon">
  <svg viewBox="0 0 24 24">
    <use href="#icon-{}"></use>
  </svg>
</div>
"""


def _collapse_icon(is_collapsed: bool) -> str:
  """Returns the HTML of the icon."""
  name = 'expand' if is_collapsed else 'collapse'
  return _ICON_TEMPLATE.format(name)


def json_to_html(json: epy.typing.JsonDict) -> str:
  """Returns the HTML of the JSON."""
  content = _json_to_html(json)
  return f'<div class="json-container">{content}</div>'


_TYPE_TO_CLASS = {
    str: 'json-string',
    bool: 'json-boolean',
    int: 'json-number',
    float: 'json-number',
    type(None): 'json-null',
}


def _json_to_html(json: epy.typing.Json) -> str:
  """Returns the HTML of the JSON."""
  if isinstance(json, str):
    if '"' in json:
      return f'"<span class="json-string">{json}</span>"'
    else:
      return f'<span class="json-string">"{json}"</span>'
  elif isinstance(json, tuple(_TYPE_TO_CLASS)):
    class_ = _TYPE_TO_CLASS[type(json)]
    return f'<span class="{class_}">{json}</span>'
  elif isinstance(json, dict):
    items = []
    for k, v in json.items():  # pytype: disable=attribute-error
      items.append(
          '<div class="json-item">'
          f'<span class="json-key">{k}</span>: {_json_to_html(v)}'
          '</div>'
      )
    items = '\n'.join(items)
    return f'<div class="json-dict">{items}</div>'
  elif isinstance(json, list):
    if not json:
      return '[]'
    items = []
    for v in json:
      items.append(
          '<div class="json-item">'
          f'<span class="json-key">-</span> {_json_to_html(v)}'
          '</div>'
      )
      # items.append(f'{_json_to_html(v)}')
    items = '\n'.join(items)
    return f'<div class="json-list">{items}</div>'
  else:
    raise ValueError(f'Unsupported JSON type: {type(json)}')
