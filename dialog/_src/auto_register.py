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

"""Auto register classes."""

from __future__ import annotations

from typing import ClassVar, Self


class RegisterSubclasses[_DataLikeT]:
  """Base class for auto register classes."""

  SUBCLASSES: ClassVar[list[type[Self]]]
  ROOT_CLS: ClassVar[type[Self]]

  def __init_subclass__(cls, **kwargs):
    super().__init_subclass__(**kwargs)

    # Direct child
    if RegisterSubclasses in cls.__bases__:
      cls.ROOT_CLS = cls
      cls.SUBCLASSES = []
    else:
      cls.SUBCLASSES.append(cls)

  @classmethod
  def from_data(cls, data: _DataLikeT) -> Self:  # pytype: disable=name-error
    """Creates the instance from the data."""
    if isinstance(data, cls.ROOT_CLS):
      return data

    for subclass in cls.SUBCLASSES:
      if obj := subclass._from_data(data):  # pylint: disable=protected-access
        return obj

    extra_info = []
    for subclass in cls.SUBCLASSES:
      if curr_info := subclass._debug_msg_from_data(data):  # pylint: disable=protected-access
        extra_info.append(curr_info)
    extra_info = '\n'.join(extra_info)
    if extra_info:
      extra_info = f'\n{extra_info}'
    raise ValueError(
        f'Unsupported {cls.__name__} input: {type(data)}{extra_info}'
    )

  @classmethod
  def _from_data(cls, data: _DataLikeT) -> Self | None:  # pytype: disable=name-error
    """If the subclass can consume the data, returns an instance."""
    del data
    return None

  @classmethod
  def _debug_msg_from_data(cls, data: _DataLikeT) -> str | None:  # pytype: disable=name-error
    """Returns a debug message from the data."""
    del data
    return None

  @classmethod
  def is_data_supported_without_doubt(cls, data: _DataLikeT) -> bool:  # pytype: disable=name-error
    """Returns whether the subclass can consume the data.

    The `without_doubt` is here as it should be absolutely certain that the
    data cannot be consumed by another chunk. For example, `str` can be
    consumed by both `dialog.Image` and `dialog.Text`.
    This functions returns `True` for a subset of what `from_data` succeeds on.

    This allow to bypass boilerplate when creating the turns. i.e. one can
    directly do:

    ```python
    dialog.User(
        'What can you say about those images?'
        np.zeros((h, w, 3), dtype=np.uint8),
        pathlib.Path('/tmp/image.png'),
    )
    ```

    Rather than:

    ```python
    dialog.User(
        dialog.Text('What can you say about those images?'),
        dialog.Image(np.zeros((h, w, 3), dtype=np.uint8)),
        dialog.Image(pathlib.Path('/tmp/image.png')),
    )
    ```

    Args:
      data: The data to check.

    Returns:
      Whether the subclass can consume the data.
    """
    if cls == cls.ROOT_CLS:
      for subclass in cls.SUBCLASSES:
        if (
            subclass.is_data_supported_without_doubt  # pylint: disable=comparison-with-callable
            == RegisterSubclasses.is_data_supported_without_doubt
        ):
          continue
        if subclass.is_data_supported_without_doubt(data):
          return True

    return False
