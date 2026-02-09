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

"""Image helper."""

from __future__ import annotations

import base64 as base64lib
import dataclasses
import functools
import io
import os
from typing import ClassVar, Self

from dialog._src import auto_register
from etils import enp
from etils import epath
from etils import epy
import numpy as np
import PIL.Image
import requests

type ImageLike = (
    # A path, url
    str
    | epath.PathLike
    | np.ndarray
    | PIL.Image.Image
    | bytes
)

_QUALITY = 75
_MAX_HEIGHT_PX = 700


class Image(auto_register.RegisterSubclasses[ImageLike]):
  """Image data."""

  # One of `numpy`, or `full_pil` must be implemented. The other one is
  # inferred.

  @property
  def numpy(self) -> np.ndarray:
    """Returns the image as a numpy array."""
    return np.asarray(self.full_pil)

  @functools.cached_property
  def full_pil(self) -> PIL.Image.Image:
    """Returns the image as a PIL image."""
    return PIL.Image.fromarray(self.numpy)

  @functools.cached_property
  def small_pil(self) -> PIL.Image.Image:
    """Compresses and resize the raw image bytes for faster rendering."""
    img = self.full_pil

    # Resize the image if it's too large.
    original_width, original_height = img.size
    if original_height > _MAX_HEIGHT_PX:
      # Calculate aspect ratio and new width.
      aspect_ratio = original_height / original_width
      new_width = int(_MAX_HEIGHT_PX / aspect_ratio)
      img = img.resize((new_width, _MAX_HEIGHT_PX))
    return img

  @functools.cached_property
  def html_src(self) -> str:
    """Returns the image as a base64 string."""
    if self.missing:
      return ''

    img_bytes, mime_type = _pil_to_bytes_and_mime(self.small_pil)

    img_b64 = base64lib.b64encode(img_bytes).decode('utf-8')
    return f'data:{mime_type};base64,{img_b64}'

  @functools.cached_property
  def missing(self) -> bool:
    return False


@dataclasses.dataclass(frozen=True)
class _ArrayImage(Image):
  """Image data from a numpy array."""

  data: enp.typing.Array

  def __post_init__(self):
    if not enp.lazy.is_array(self.data):
      raise ValueError(f'Expected an array, got {type(self.data)}')

    if (
        self.data.ndim != 3
        or self.data.shape[-1] != 3
        or self.data.dtype != np.uint8
    ):
      raise ValueError(
          f'Expected a 3D uint8 array, got shape={self.data.shape},'
          f' dtype={self.data.dtype}'
      )

  @classmethod
  def _from_data(cls, data: ImageLike) -> Self | None:
    if enp.lazy.is_array(data):
      return cls(data)
    return None

  @classmethod
  def is_data_supported_without_doubt(cls, data: ImageLike) -> bool:
    return (
        enp.lazy.is_array(data)
        and data.ndim == 3  # pytype: disable=attribute-error
        and data.shape[-1] == 3  # pytype: disable=attribute-error
        and data.dtype == np.uint8  # pytype: disable=attribute-error
    )

  @functools.cached_property
  def numpy(self) -> np.ndarray:
    return np.asarray(self.data)


@dataclasses.dataclass(frozen=True)
class _PILImage(Image):
  """Image data from a PIL image."""

  data: PIL.Image.Image

  @classmethod
  def _from_data(cls, data: ImageLike) -> Self | None:
    if isinstance(data, PIL.Image.Image):
      return cls(data)
    return None

  @classmethod
  def is_data_supported_without_doubt(cls, data: ImageLike) -> bool:
    return isinstance(data, PIL.Image.Image)

  @functools.cached_property
  def full_pil(self) -> PIL.Image.Image:
    return self.data


# Url needs to be defined before `Path`.
@dataclasses.dataclass(frozen=True)
class _UrlImage(Image):
  """Image data from a URL."""

  url: str

  @classmethod
  def _from_data(cls, data: ImageLike) -> Self | None:
    # TODO(epot): Should filter http(s):// URLs as well.
    if isinstance(data, str) and data.startswith(('http://', 'https://')):
      return cls(data)
    return None

  @functools.cached_property
  def full_pil(self) -> PIL.Image.Image:
    response = requests.get(self.url)
    response.raise_for_status()
    return PIL.Image.open(io.BytesIO(response.content))


@dataclasses.dataclass(frozen=True)
class _PathImage(Image):
  """Image data from a path."""

  path: epath.Path

  def __post_init__(self):
    if self.path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
      raise ValueError(
          f'Unsupported image extension: {self.path.suffix.lower()}'
      )

  SUPPORTED_EXTENSIONS: ClassVar[tuple[str, ...]] = (
      '.png',
      '.jpg',
      '.jpeg',
      '.gif',
      '.webp',
  )

  @classmethod
  def _from_data(cls, data: ImageLike) -> Self | None:
    # TODO(epot): Should filter http(s):// URLs as well.
    if isinstance(data, epath.PathLike):
      return cls(epath.Path(data))
    return None

  @classmethod
  def is_data_supported_without_doubt(cls, data: ImageLike) -> bool:
    return isinstance(data, epath.PathLike) and os.fspath(
        data
    ).lower().endswith(_PathImage.SUPPORTED_EXTENSIONS)

  @functools.cached_property
  def full_pil(self) -> PIL.Image.Image:
    with self.path.open('rb') as f:
      img = PIL.Image.open(f)
      img.load()
    return img


@dataclasses.dataclass(frozen=True)
class _BytesImage(Image):
  """Image data from bytes."""

  data: bytes

  @classmethod
  def _from_data(cls, data: ImageLike) -> Self | None:
    if isinstance(data, bytes):
      return cls(data)
    return None

  @functools.cached_property
  def full_pil(self) -> PIL.Image.Image:
    return PIL.Image.open(io.BytesIO(self.data))


@dataclasses.dataclass(frozen=True)
class _MissingImage(Image):
  """Missing image data."""

  @classmethod
  def _from_data(cls, data: ImageLike) -> Self | None:
    if data is None or (isinstance(data, str) and not data):
      return cls()
    return None

  @functools.cached_property
  def full_pil(self) -> PIL.Image.Image:
    raise ValueError('Missing image')

  @functools.cached_property
  def missing(self) -> bool:
    return True


# From
# https://modelcontextprotocol.io/specification/2025-06-18/server/tools#image-content
# class _MCPImage(Image):
#   """Image data from MCP."""


def _pil_to_bytes_and_mime(img: PIL.Image.Image) -> tuple[bytes, str]:
  """Converts a PIL image to bytes and mime type."""
  img_bytes = io.BytesIO()

  img_format = img.format
  if img_format is None:
    img_format = 'JPEG'

  if img_format.upper() == 'JPEG':
    img.save(img_bytes, format='JPEG', quality=_QUALITY)
    mime_type = 'image/jpeg'
  elif img_format.upper() == 'PNG':
    img.save(img_bytes, format='PNG', optimize=True)
    mime_type = 'image/png'
  elif img_format.upper() == 'WEBP':
    img.save(img_bytes, format='WEBP', quality=_QUALITY)
    mime_type = 'image/webp'
  else:
    img = img.convert('RGB')
    img.save(img_bytes, format='JPEG', quality=_QUALITY)
    mime_type = 'image/jpeg'
  return img_bytes.getvalue(), mime_type
