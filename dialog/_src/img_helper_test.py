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

import io
import pathlib

import dialog
import numpy as np
import PIL.Image


def test_adapter_image(tmp_path: pathlib.Path):
  img = PIL.Image.new('RGB', (100, 100), color='black')

  as_bytes = io.BytesIO()
  img.save(as_bytes, format='JPEG')
  raw_bytes = as_bytes.getvalue()

  img_path = tmp_path / 'img.jpg'
  img_path.write_bytes(raw_bytes)

  img_pil = dialog.Image(img)
  img_np = dialog.Image(np.asarray(img))
  img_bytes = dialog.Image(raw_bytes)
  img_path = dialog.Image(img_path)

  _assert_equal(img_pil, img_np)
  _assert_equal(img_pil, img_bytes)
  _assert_equal(img_pil, img_path)


def _assert_equal(
    img_1: dialog.Image,
    img_2: dialog.Image,
) -> None:
  assert img_1.as_text() == img_2.as_text()
