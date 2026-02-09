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

# Most tests are internal in `_src/gemini_example/` to ensure consistency
# between dialog and gemini_example standard formatter.


def test_add():
  conv = dialog.Conversation()

  assert len(conv) == 0  # pylint: disable=g-explicit-length-test

  conv += dialog.User('Hello')
  conv += dialog.Model('Hi')

  assert len(conv) == 2

  assert conv == dialog.Conversation(
      dialog.User('Hello'),
      dialog.Model('Hi'),
  )

  conv = dialog.System('Be nice') + conv
  assert conv == dialog.Conversation(
      dialog.System('Be nice'),
      dialog.User('Hello'),
      dialog.Model('Hi'),
  )


_STR = """<|turn>system
You are a helpful assistant.<|tool>declaration:file_explorer{parameters:{properties:{method:{type:<|"|>STRING<|"|>},path:{type:<|"|>STRING<|"|>}},type:<|"|>OBJECT<|"|>}}<tool|><|tool>declaration:get_current_time{description:<|"|>Get the current time<|"|>}<tool|><turn|>
<|turn>user
Describe this image: <|image|><turn|>
<|turn>model
<|channel>thought
Let me think...<channel|>What a pretty turtle!"""


def test_parse():
  conv = dialog.Conversation(_STR)
  assert conv.as_text() == _STR
  assert len(conv) == 3

  conv = dialog.Conversation(
      dialog.System(
          'You are a helpful assistant.',
          dialog.Tool({
              'name': 'file_explorer',
              'inputSchema': {
                  'type': 'object',
                  'properties': {
                      'method': {'type': 'string'},
                      'path': {'type': 'string'},
                  },
              },
          }),
          dialog.Tool({
              'name': 'get_current_time',
              'description': 'Get the current time',
          }),
      ),
      dialog.User('Describe this image: ', dialog.Image(None)),
      dialog.Model(
          dialog.Thought('Let me think...'),
          'What a pretty turtle!',
      ),
  )
  assert conv.as_text() == _STR
