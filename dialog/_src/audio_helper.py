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

"""Audio helper functions."""

from __future__ import annotations

import base64
import dataclasses
import functools
import io
import sys
from typing import ClassVar, Self
import wave

from dialog._src import auto_register
from etils import epath
from etils import epy
import numpy as np
import requests

with epy.lazy_imports():
  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
  import pydub
  # pytype: enable=import-error  # pylint: enable=g-import-not-at-top


type AudioLike = (
    # A path, url
    str
    | epath.PathLike
    # Could eventually supports arrays based on demand.
    # | np.ndarray
    | pydub.AudioSegment
    | bytes
    | None  # Missing audio.
)


class Audio(auto_register.RegisterSubclasses[AudioLike]):
  """Audio data."""

  @functools.cached_property
  def html_src(self) -> str:
    """Returns the audio as a base64 string."""
    if self.missing:
      return ''

    # b64encode returns bytes, so we decode to utf-8 to get a string
    audio_b64 = base64.b64encode(self.bytes).decode('utf-8')
    return f'data:{self.mime_type};base64,{audio_b64}'

  @functools.cached_property
  def bytes(self) -> bytes:
    """Returns the audio as bytes."""
    raise NotImplementedError()

  @functools.cached_property
  def mime_type(self) -> str:
    """Returns the audio as a mime type."""
    raise NotImplementedError()

  @functools.cached_property
  def missing(self) -> bool:
    return False


class _PydubBaseAudio(Audio):
  """Base class for audio data from a pydub segment."""

  @functools.cached_property
  def segment(self) -> pydub.AudioSegment:
    """Returns the audio as a pydub segment."""
    raise NotImplementedError()

  @functools.cached_property
  def bytes(self) -> bytes:
    """Returns the audio as bytes."""
    buffer = io.BytesIO()

    # For more compression, we could do:
    # sound = sound.set_channels(1)  # Switch to mono
    # sound = sound.set_frame_rate(22050)

    self.segment.export(
        buffer,
        format='webm',
        codec='libopus',
        bitrate='64k',
    )

    raw_bytes = buffer.getvalue()
    return raw_bytes

  @functools.cached_property
  def mime_type(self) -> str:
    return 'audio/webm'


@dataclasses.dataclass(frozen=True)
class _PydubAudio(_PydubBaseAudio):
  """Audio data from a path."""

  data: pydub.AudioSegment

  @classmethod
  def _from_data(cls, data: AudioLike) -> Self | None:
    if isinstance(data, pydub.AudioSegment):
      return cls(data)
    return None

  @classmethod
  def is_data_supported_without_doubt(cls, data: AudioLike) -> bool:
    return 'pydub' in sys.modules and isinstance(data, pydub.AudioSegment)

  @functools.cached_property
  def segment(self) -> pydub.AudioSegment:
    return self.data


# Url needs to be defined before `Path`.
@dataclasses.dataclass(frozen=True)
class _UrlAudio(_PydubBaseAudio):
  """Audio data from a URL."""

  url: str

  @classmethod
  def _from_data(cls, data: AudioLike) -> Self | None:
    if isinstance(data, str) and data.startswith(('http://', 'https://')):
      return cls(data)
    return None

  @functools.cached_property
  def segment(self) -> pydub.AudioSegment:
    """Returns the audio as a pydub segment."""
    response = requests.get(self.url)
    response.raise_for_status()
    return pydub.AudioSegment.from_file(io.BytesIO(response.content))


@dataclasses.dataclass(frozen=True)
class _PathAudio(_PydubBaseAudio):
  """Audio data from a path."""

  path: epath.Path

  @classmethod
  def _from_data(cls, data: AudioLike) -> Self | None:
    if isinstance(data, epath.PathLike):
      return cls(epath.Path(data))
    return None

  @functools.cached_property
  def segment(self) -> pydub.AudioSegment:
    """Returns the audio as a pydub segment."""
    with self.path.open('rb') as f:
      return pydub.AudioSegment.from_file(f)


@dataclasses.dataclass(frozen=True)
class _BytesAudio(_PydubBaseAudio):
  """Audio data from bytes."""

  data: bytes

  @classmethod
  def _from_data(cls, data: AudioLike) -> Self | None:
    if isinstance(data, bytes):
      return cls(data)
    return None

  @functools.cached_property
  def segment(self) -> pydub.AudioSegment:
    return pydub.AudioSegment.from_file(io.BytesIO(self.data))


@dataclasses.dataclass(frozen=True)
class _MissingAudio(Audio):
  """Missing image data."""

  @classmethod
  def _from_data(cls, data: AudioLike) -> Self | None:
    if data is None or (isinstance(data, str) and not data):
      return cls()
    return None

  @functools.cached_property
  def missing(self) -> bool:
    return True


# MARK: Decoders
# Converts from a mime type to wave bytes.
# Currently unsupported mime types:
# - 'audio/x-raw-tokens'
# - 'audio/x-youtube'
# - 'audio/hardtoken'


class _Decoder:
  """Decoder for audio data."""

  MIME_TYPES: ClassVar[tuple[str, ...]]

  @classmethod
  def from_mime_type(cls, mime_type: str) -> Self | None:
    """Returns whether the mime type matches."""
    mime_type = mime_type.lower()
    for subcls in cls.__subclasses__():
      if mime_type in subcls.MIME_TYPES:
        return subcls()

    return None

  def decode(self, data: gemini_example_pb2.Audio) -> bytes:
    """Decodes audio data from bytes."""
    raise NotImplementedError()


class _WavDecoder(_Decoder):
  """Decoder for WAV audio data."""

  MIME_TYPES = (
      # Common alias from https://mimetype.io/audio/wav.
      'audio/wav',
      'audio/wave',
      'audio/x-wav',
      'audio/vnd.wav',
  )

  def decode(self, data: gemini_example_pb2.Audio) -> bytes:
    """Decodes audio data from bytes."""
    return data.value.content


@dataclasses.dataclass(frozen=True, kw_only=True)
class _PcmDecoder(_Decoder):
  """Decodes audio data from bytes."""

  FLOAT_MIME_TYPES = ('audio/pcm;encoding=float;bits=32',)
  MIME_TYPES = (
      'audio/l16',
      'audio/pcm',
      *FLOAT_MIME_TYPES,
  )

  def decode(self, data: gemini_example_pb2.Audio) -> bytes:
    """Decodes audio data from bytes."""
    if data.value.mime_type in self.FLOAT_MIME_TYPES:
      dtype = np.float32
      validate_fn = _validate_pcm_float_32
    else:
      dtype = np.int16
      validate_fn = _validate_pcm_int_16

    array = np.frombuffer(data.value.content, dtype=dtype)
    validate_fn(array)

    with io.BytesIO() as f:
      with wave.open(f, 'wb') as wav_file:
        wav_file.setnchannels(array.shape[1] if array.ndim > 1 else 1)
        wav_file.setsampwidth(array.dtype.itemsize)
        wav_file.setframerate(data.sample_rate)
        wav_file.writeframes(array.tobytes())

    return f.getvalue()


def _validate_pcm_int_16(data: np.ndarray) -> None:
  msg = 'Invalid pcm_int_16 audio. Reason: '
  if not isinstance(data, np.ndarray):
    raise TypeError(msg + f'{type(data)} is not a numpy array')
  if data.ndim != 1:
    raise ValueError(msg + f'Expected 1D array, got {data.ndim}')
  if data.dtype != np.int16:
    raise ValueError(msg + f'Expected int16, got {data.dtype}')


def _validate_pcm_float_32(data: np.ndarray) -> None:
  msg = 'Invalid pcm_float_32 audio. Reason: '
  if not isinstance(data, np.ndarray):
    raise TypeError(msg + f'{type(data)} is not a numpy array')
  if data.ndim != 1:
    raise ValueError(msg + f'Expected 1D array, got {data.ndim}')
  if data.dtype != np.float32:
    raise ValueError(msg + f'Expected float32, got {data.dtype}')
  if np.max(data) > 1.0 or np.min(data) < -1.0:
    raise ValueError(
        msg + f'Expected range [-1, 1], got [{np.max(data)}, {np.min(data)}]'
    )
