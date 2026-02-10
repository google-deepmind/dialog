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

import pathlib

import dialog
import pydub
import pydub.generators


def test_audio(tmp_path: pathlib.Path):
  tone = pydub.generators.Sine(freq=440).to_audio_segment(duration=1_000)
  tone -= 10  # Adjust the volume (-10 dB)

  filepath = tmp_path / 'audio.wav'
  tone.export(filepath, format='wav')

  bytes_data = filepath.read_bytes()

  audio_pydub = dialog.Audio(tone)
  audio_path = dialog.Audio(filepath)
  audio_bytes = dialog.Audio(bytes_data)

  _assert_equal(audio_pydub, audio_path)
  _assert_equal(audio_pydub, audio_bytes)


def _assert_equal(audio_1: dialog.Audio, audio_2: dialog.Audio):
  assert audio_1.as_text() == audio_2.as_text()
