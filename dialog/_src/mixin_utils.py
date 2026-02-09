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

"""Utils for `__repr__`."""

from __future__ import annotations

import abc
import collections.abc
from collections.abc import Iterator
import typing
from typing import Any, ClassVar, Self

from etils import epy


class HasReprContent(abc.ABC):
  """Protocol to define `__repr_content__` of a class."""

  @abc.abstractmethod
  def __repr_content__(self) -> str | dict[str, Any] | list[Any]:
    raise NotImplementedError()


class AddRepr:
  """Mixin to add a `__repr__` method to classes.

  Usage:

  ```python
  class MyClass(repr_utils.AddRepr):
    REPR_CONTENT = 'some_content'
  ```

  Will display something like:

  ```python
  repr(obj) == f'MyClass({obj.some_content})'
  ```
  """

  REPR_CONTENT: ClassVar[str | None]

  def __repr__(self) -> str:
    if self.REPR_CONTENT is None:
      value = ()
    else:
      value = getattr(self, self.REPR_CONTENT)

    # Unwrap the content if needed.
    if isinstance(value, HasReprContent):
      value = value.__repr_content__()  # pylint: disable=protected-access

    if not isinstance(value, str | dict | list | tuple):
      value = epy.Lines.Repr(repr(value))

    return epy.Lines.make_block(
        header=type(self).__name__,
        content=value,
    )


class Sequence[_ElemT, _ElemLikeT]:
  """Iterable mixin."""

  SEQUENCE_ATTRIBUTE: ClassVar[str]

  # Class for which `__add__` is not implemented.
  # This allow to execute:
  # Turn() + Conversation()
  # as:
  # Conversation().__radd__(Turn())
  ADD_NOT_IMPLEMENTED_CLS: tuple[ClassVar[type[Any]], ...] = ()

  def __init__(self, *args: _ElemLikeT):  # pytype: disable=name-error
    raise NotImplementedError()

  def __len__(self) -> int:
    return len(self._sequence)

  def __iter__(self) -> Iterator[_ElemT]:  # pytype: disable=name-error
    return iter(self._sequence)

  @typing.overload
  def __getitem__(self, index: int) -> _ElemT:  # pytype: disable=name-error
    ...

  @typing.overload
  def __getitem__(self, index: slice) -> Self:
    ...

  def __getitem__(self, index):
    if isinstance(index, slice):
      return type(self)(self._sequence[index])  # pylint: disable=too-many-function-args
    else:
      return self._sequence[index]

  def __add__(self, other: collections.abc.Sequence[_ElemLikeT]) -> Self:  # pytype: disable=name-error
    if isinstance(other, self.ADD_NOT_IMPLEMENTED_CLS):
      return NotImplemented  # pytype: disable=bad-return-type
    return type(self)(self._sequence, other)

  def __radd__(self, other: collections.abc.Sequence[_ElemLikeT]) -> Self:  # pytype: disable=name-error
    if isinstance(other, self.ADD_NOT_IMPLEMENTED_CLS):
      return NotImplemented  # pytype: disable=bad-return-type
    return type(self)(other, self._sequence)

  def __eq__(self, other: Any) -> bool:
    if not isinstance(other, self.__class__):
      return NotImplemented
    return self._sequence == other._sequence  # pylint: disable=protected-access

  @property
  def _sequence(self) -> list[_ElemT]:  # pytype: disable=name-error
    return getattr(self, self.SEQUENCE_ATTRIBUTE)
