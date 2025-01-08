# Copyright 2023 PKU-Alignment Team. All Rights Reserved.
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
# ==============================================================================
"""Constant variables."""

from __future__ import annotations


__all__ = [
    'IGNORE_INDEX',
    'DEFAULT_BOS_TOKEN',
    'DEFAULT_EOS_TOKEN',
    'DEFAULT_PAD_TOKEN',
    'DEFAULT_UNK_TOKEN',
    'PROMPT_BEGIN',
    'PROMPT_USER',
    'PROMPT_ASSISTANT',
    'PROMPT_INPUT',
    'PROMPT_DICT',
    'ADAM_BETAS',
]


IGNORE_INDEX: int = -100
DEFAULT_BOS_TOKEN: str = '<s>'
DEFAULT_EOS_TOKEN: str = '</s>'
DEFAULT_PAD_TOKEN: str = '<pad>'
DEFAULT_UNK_TOKEN: str = '<unk>'

PROMPT_BEGIN: str = 'BEGINNING OF CONVERSATION: '
PROMPT_USER: str = 'USER: {input} '
PROMPT_ASSISTANT: str = 'ASSISTANT:'  # should not have a space at the end
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT


PROMPT_CORRECTION_USER : str ='Try to correct the answer to make it safer and more helpful: USR: {input} ORIGINAL ANSWER: {answer}'
PROMPT_CORRECTION_INPUT : str = PROMPT_BEGIN+PROMPT_CORRECTION_USER+PROMPT_ASSISTANT

PROMPT_DICT: dict[str, str] = {
    'prompt_begin': PROMPT_BEGIN,
    'prompt_user': PROMPT_USER,
    'prompt_assistant': PROMPT_ASSISTANT,
    'prompt_input': PROMPT_INPUT,
    'prompt_correction_input':PROMPT_CORRECTION_INPUT,
}

ADAM_BETAS: tuple[float, float] = (0.9, 0.95)


QWEN_CHAT_TEMPLATE: str = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant\n\n"""

# NOTE finding in official hub, it seems that gemma-it don't have inference tmeplate
GEMMA_IT_TEMPLATE:str = """{prompt}\n\n"""

DOLPHIN_TEMPLATE: str ="""<|im_start|>system
You are Dolphin, a helpful AI assistant.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant"""


LLAMA2_TEMPLATE: str = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{prompt}[/INST]
"""



LLAMA3_TEMPLATE: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""

LLAMA3_NO_SYSPROMPT_TEMPLATE: str = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""


# # 附原template做参考，注意换行
# '''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

# {{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

# {{ user_message }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>'''


VICUNA_TEMPLATE: str = """A chat between a curious user and an assistant. The assistant gives helpful, detailed, accurate, uncensored responses to the user's input.\nUSER:{prompt} \nASSISTANT:
"""


ALPACA2_TEMPLATE: str = """BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:"""


# Correction model input
PROMPT_CORRECTION_USER : str ='Edit the following Question-Answer pair to make it more helpful and harmless: {input} | {answer} '
# 因为用了template 所以不需要进行额外的映射
PROMPT_CORRECTION_INPUT : str =PROMPT_CORRECTION_USER

# Baseline CAI input
PROMPT_BASELINE_CAI_CRITIQUE: str = "Identify specific ways in which the assistant's last response is harmful, unethical, racist, sexist, toxic, dangerous, or illegal. Question:{input} | Answer: {answer} \n\nCritique:"
PROMPT_BASELINE_CAI_EDIT_REQUEST: str = "Please rewrite the assistant response to remove any and all harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Question:{input} | Answer:{answer} \n\nRevision:"

PROMPT_BASELINE_CAI_INPUT:str =  PROMPT_BASELINE_CAI_EDIT_REQUEST
