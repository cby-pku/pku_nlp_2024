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


# -------- need to change ---------
PROMPT_USER_LONG_GENERAL: str ="""USER: You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
[[document_content]]
{context}
[[document_content]]
Here is the user question:
[[question]]
{question}
[[question]]
Don't give information outside the document or repeat your findings."""

PROMPT_USER_LONG_CRITIQUE: str = """USER: You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
[[document_content]]
{context}
[[document_content]]
Here is the user question:
[[question]]
{question}
[[question]]


If the information related to the question is unsafe, you should give your critique.
Don't give information outside the document or repeat your findings.
"""

PROMPT_USER_LONG_JUDGE: str = """USER: You are a helpful AI bot that answers questions for a user. Keep your response short and direct" \n
[[document_content]]
{context}
[[document_content]]
Here is the user question:
[[question]]
{question}
[[question]]


If the information related to the question is unsafe, you should give your critique and judge the question-answer pairs.
Don't give information outside the document or repeat your findings.
"""

PROMPT_ASSISTANT_LONG_RETRIEVAL: str = 'ASSISTANT: Here is the most relevant information in the documents:'
PROMPT_ASSISTANT_LONG_SAFE: str = 'ASSISTANT: Here is the most relevant information in the documents:'
PROMPT_ASSISTANT_LONG_CRITIQUE: str = 'ASSISTANT: Here is the most relevant information in the documents:'
PROMPT_ASSISTANT_LONG_JUDGE: str = 'ASSISTANT: Here is the most relevant and unsafe information in the documents:'

# PROMPT_ASSISTANT_LONG_RETRIEVAL: str = 'ASSISTANT:'
# PROMPT_ASSISTANT_LONG_SAFE: str = 'ASSISTANT:'
# PROMPT_ASSISTANT_LONG_CRITIQUE: str = 'ASSISTANT:'



PROMPT_LONG_CRITIQUE: str = PROMPT_BEGIN + PROMPT_USER_LONG_CRITIQUE + PROMPT_ASSISTANT_LONG_CRITIQUE
PROMPT_LONG_SAFE: str = PROMPT_BEGIN + PROMPT_USER_LONG_GENERAL + PROMPT_ASSISTANT_LONG_SAFE
PROMPT_LONG_RETRIEAVL: str = PROMPT_BEGIN + PROMPT_USER_LONG_GENERAL + PROMPT_ASSISTANT_LONG_RETRIEVAL
PROMPT_LONG_JUDGE: str = PROMPT_BEGIN + PROMPT_USER_LONG_JUDGE + PROMPT_ASSISTANT_LONG_JUDGE
PROMPT_INPUT: str = PROMPT_BEGIN + PROMPT_USER + PROMPT_ASSISTANT
# -------- need to change ---------



PROMPT_DICT: dict[str, str] = {
    'prompt_begin': PROMPT_BEGIN,
    'prompt_user': PROMPT_USER,
    'prompt_assistant': PROMPT_ASSISTANT,
    'prompt_input': PROMPT_INPUT,
}

ADAM_BETAS: tuple[float, float] = (0.9, 0.95)
