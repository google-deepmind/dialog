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

import dialog


_GEMMA4 = """<|turn>system
<|think|>

<|tool>declaration:get_weather{parameters:{properties:{location:{type:<|"|>STRING<|"|>},timedelta:{description:<|"|>Number of days in the future. Default to 0.<|"|>,type:<|"|>NUMBER<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|>

Ask follow up questions.<turn|>
<|turn>user
Describe this image: <|image|><turn|>
<|turn>model
<|channel>thought
Let me inspect those.<channel|>Hello, this image is ..."""


_GEMMA3 = """<start_of_turn>system
<unusedXX>

<start_function_declaration>declaration:get_weather{parameters:{properties:{location:{type:<escape>STRING<escape>},timedelta:{description:<escape>Number of days in the future. Default to 0.<escape>,type:<escape>NUMBER<escape>}},type:<escape>OBJECT<escape>}}<end_function_declaration>

Ask follow up questions.<end_of_turn>
<start_of_turn>user
Describe this image: <|image|><end_of_turn>
<start_of_turn>model
<start_receiver>thought
Let me inspect those.<end_receiver>Hello, this image is ..."""


def test_format_gemma4():
  assert dialog.Format.GEMMA4.to_gemma4(_GEMMA4) == _GEMMA4
  assert dialog.Format.GEMMA4.from_gemma4(_GEMMA4) == _GEMMA4

  assert dialog.Format.GEMMA3.to_gemma4(_GEMMA3) == _GEMMA4
  assert dialog.Format.GEMMA3.from_gemma4(_GEMMA4) == _GEMMA3
