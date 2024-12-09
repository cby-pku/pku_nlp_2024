
import argparse
import hashlib
import itertools
import json
import logging
import os
import random
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Union

import openai
import ray
import tqdm
# from tqdm import tqdm

import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from web_utils import *

# from prompts import *
# from translate_en_ch.system_prompt import SYSTEM_PROMPT,USER_PROMPT

from config.system_prompt import SYSTEM_PROMPT,USER_PROMPT

HERE = Path(__file__).absolute().parent

# DEFAULT_OPENAI_CHAT_COMPLETION_MODELS = [
#     'gpt-4-1106-preview'
# ]
# DEFAULT_OPENAI_CHAT_COMPLETION_MODELS = [
#     'gpt-3.5-turbo-1106',
# ]
DEFAULT_OPENAI_CHAT_COMPLETION_MODELS = [
    # 'gpt-4-1106-preview',
    # 'gpt-4-0125-preview',
    # 'gpt-4-turbo-preview',
    # 'gpt-3.5-turbo'
    # 'claude-3-opus-20240229'
    'gpt-4o'
]


def generate_hash_uid(to_hash: dict | tuple | list | str):
    """Generates a unique hash for a given model and arguments."""
    # Convert the dictionary to a JSON string
    json_string = json.dumps(to_hash, sort_keys=True)

    # Generate a hash of the JSON string
    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid

def hash_checker(result: dict):
    return (result['safer_id'] is not None) or result['better_id'] is not None

def string_to_seed(s):
    """将字符串转换为整数种子"""
    # 使用hashlib生成字符串的哈希值
    hash_value = hashlib.sha256(s.encode()).hexdigest()
    # 将哈希值（16进制字符串）转换为整数
    seed = int(hash_value, 16)
    return seed

def fixed_random(s):
    """根据给定的字符串s生成固定范围[a, b]内的随机整数"""
    seed = string_to_seed(s)
    random.seed(seed)
    return random.choice([0,1])

def post_process(response, type: str):
    
    #(.*?) 表示非贪婪模式的捕获组，匹配任意字符（除了换行符 \n）零次或多次
    if(type == "safety"):
        order = response['order']
        pattern = r'More harmful: \[\[(.*?)\]\]'
        matches = re.findall(pattern, response['output'], re.IGNORECASE)
        result = {'qac': response['transformed_input'],'safer_id': None, 'reasoning': response['output']}
        if len(matches) == 1:
            match = matches[0].lower()
            if match == 'responsea':
                result['safer_id'] = 1 - order
            elif match == 'responseb':
                result['safer_id'] = 0 + order
            elif match == 'equal':
                result['safer_id'] = -1
        else:
            pattern = r'\*\*More harmful\*\*: \[\[(.*?)\]\]'
            matches_1 = re.findall(pattern, response['output'], re.IGNORECASE)
            if len(matches_1) == 1:
                match = matches_1[0].lower()
                if match == 'responsea':
                    result['safer_id'] = 1 - order
                elif match == 'responseb':
                    result['safer_id'] = 0 + order
                elif match == 'equal':
                    result['safer_id'] = -1
        
        return result
    

    
    result = {
            'input':response['transformed_input'],
            'output':response['output']
        }
    return result


@ray.remote(num_cpus=1)
def request_openai(
    id: int,
    input: dict[str, str],
    openai_api_keys: list[(str, str)],
    openai_model: str,
    base_url: str | None = None,
    cache_dir: Path | str | None = None,
) -> list[dict[str, object]]:
    openai_api_keys = itertools.cycle(openai_api_keys)
    openai_api_keys = next(openai_api_keys)
    
    platform = openai_api_keys[1]
    openai_api_key = openai_api_keys[0]
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

    system_prompt = input['system_prompt']
    user_prompt = input['user_prompt']
    sha256 = hashlib.sha256((system_prompt + user_prompt).encode('utf-8')).hexdigest()

    if cache_dir is not None:
        output_file = cache_dir / f'{sha256}.json'
        # FIXME 在这里把cache关掉
        if output_file.exists():
            with output_file.open(mode='rt', encoding='utf-8') as f:
                try:
                    return id, json.load(f)
                except json.JSONDecodeError:
                    output_file = None
    else:
        output_file = None

    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt},
    ]

    result = input.copy()
    if(platform == 'baichuan'):
        result.update(
            baichuan_gpt_api(
                input = messages,
                openai_api_keys=openai_api_key,
                openai_model=openai_model,
            )
        )
    elif(platform == 'bean'):
        result.update(
            bean_gpt_api(
                input = messages,
                openai_api_keys=openai_api_key,
                openai_model=openai_model,
            )
        )
    else:
        result.update(
            request_openai_noexcept(
                messages=messages,
                openai_api_keys=openai_api_key,
                openai_model=openai_model,
                base_url=base_url,
            ),
        )
    
    result['sha256'] = sha256

    if output_file is not None:
        with output_file.open(mode='wt', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    return id, result


def batch_request_openai(
    inputs: list[dict[str, Any]],
    openai_api_keys: list[str],
    openai_models: list[str],
    base_url: str | None = None,
    num_workers: int = 8,
    cache_dir: Path | str | None = None,
) -> list[dict[str, object]]:
    openai_api_keys = sorted(set(openai_api_keys))
    openai_models = sorted(set(openai_models))
    if cache_dir is not None:
        cache_dir = Path(cache_dir).expanduser().absolute()
        cache_dir.mkdir(parents=True, exist_ok=True)

    pending = deque(enumerate(inputs))
    not_ready = []
    results = [None for _ in range(len(pending))]
    openai_api_keys_cycle = itertools.cycle(
        [openai_api_keys[i:] + openai_api_keys[:i] for i in range(len(openai_api_keys))],
    )
    with tqdm.tqdm(total=len(pending)) as pbar:
        while len(not_ready) > 0 or len(pending) > 0:
            while len(not_ready) < num_workers and len(pending) > 0:
                idx, input = pending.popleft()
                current_key=next(openai_api_keys_cycle)
                not_ready.append(
                    request_openai.remote(
                        idx,
                        input,
                        openai_api_keys=current_key,
                        openai_model=random.choice(openai_models),  # noqa: S311
                        base_url=base_url,
                        cache_dir=cache_dir,
                    ),
                )
                

            ready, not_ready = ray.wait(not_ready, timeout=1)
            for idx, result in ray.get(ready):
                results[idx] = result
            pbar.update(len(ready))

    return results


def get_openai_api_keys(
    openai_api_keys: list[str],
    openai_api_key_file: Path | str | None,
) -> list[str]:
    openai_api_keys = list(openai_api_keys or [])

    if openai_api_key_file is not None:
        openai_api_key_file = Path(openai_api_key_file).expanduser().absolute()
        with openai_api_key_file.open(mode='rt', encoding='utf-8') as f:
            for line in f:
                line = re.sub(r'#.*', '', line).strip()
                parts = tuple(line.split(','))
                if not line:
                    continue
                if not line.startswith('sk-'):
                    raise ValueError(f'Invalid OpenAI API key: {line}')
                openai_api_keys.append(parts)

    openai_api_keys = list(dict.fromkeys(openai_api_keys))

    if len(openai_api_keys) == 0:
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if openai_api_key is not None:
            openai_api_keys.append(openai_api_key)
        else:
            raise ValueError('No OpenAI API key provided.')

    for i, [openai_api_key, platform] in enumerate(openai_api_keys, start=1):
        if not openai_api_key.startswith('sk-'):
            raise ValueError(f'Invalid OpenAI API key: {openai_api_key}')
        print(f'{platform} API key #{i}: {openai_api_key}')

    return openai_api_keys

def get_annotator_correction_prompt(data, type):
    # restriction_list = RestrictionList #### 第一步所有restrictions都是violation
    
    # TODO 在这里更改输入，如果只翻译question就只填question，如果翻译1Q1A就只填一个response
    if type == "math-evaluation":
        question = data['question'] 
        answer = data['answer'] # 标准答案
        response = data['response'] # 模型答案
        return SYSTEM_PROMPT,USER_PROMPT.format(question=question, answer=answer, response=response)

    
    else:
        raise RuntimeError("not implemented type")

def transform_data(data, type):
    

    if (type == "math-evaluation") :
        new_data = {}
        new_data["solution"] = data['solution']
        new_data["question"] = data.get('prompt', data.get('question',data.get('problem')))
        new_data["answer"] = data.get('original_answer', data.get('answer'))
        new_data["correction"] = data.get('correction_answer', data.get('correction'))


        return new_data
    
    # order = fixed_random(data.get('prompt', data.get('question')))
    order = 1
    # 0 原序 1 反序
    # 新的字典
    new_data = {}
    if (type == "reasoning") or (type=='safety') or (type=='utility') or (type=='empathy') or (type=='dialoguesum') or (type=='mt-bench'):
        # new_data["target"] = data['target']
        new_data["question"] = data.get('prompt', data.get('question'))
        if order == 0:
            new_data["answer"] = data.get('original_answer', data.get('answer',data.get('answer1')))
            new_data["correction"] = data.get('correction_answer', data.get('correction'))
        else:
            new_data["answer"] = data.get('correction_answer', data.get('correction'))
            new_data["correction"] = data.get('original_answer', data.get('answer',data.get('answer1')))

        new_data['order'] = order
        return new_data, order


    # 获取category中为True的键作为violation
    if ('category' not in data) or (not data['category']):
        violation = []
    else:
        violation = data['category']
    # elif isinstance(data["category"][0], dict):
    #     # 第一种格式（字典）
    #     violation = [key for key_value_dict in data["category"] for key, value in key_value_dict.items() if value]
    # else:
    #     # 第二种格式（列表）
    #     violation = [key for key in data["category"]]

    # 新字典中的字段赋值
    new_data["violation"] = violation
    new_data["question"] = data.get('prompt', data.get('question'))
    if order == 0:
        new_data["answer"] = data.get('original_answer', data.get('answer'))
        new_data["correction"] = data.get('correction_answer', data.get('correction'))
    else:
        new_data["answer"] = data.get('correction_answer', data.get('correction'))
        new_data["correction"] = data.get('original_answer', data.get('answer'))
    new_data['order'] = order
    return new_data, order


def prepare_inputs(input_file: Path | str, shuffle: bool = False, type: str = "safety", platform : str = "openai") -> list[Any]:
    input_file = Path(input_file).expanduser().absolute()

    if input_file.suffix.lower() == '.json':
        with input_file.open(mode='rt', encoding='utf-8') as f:
            raw_inputs = json.load(f)
    elif input_file.suffix.lower() == '.jsonl':
        with input_file.open(mode='rt', encoding='utf-8') as f:
            raw_inputs = [json.loads(line) for line in f]

    inputs = []
    i=0
    

    if (type == 'translate_en_ch') or (type == 'translate_ch_en') or (type == 'math-evaluation'):
        for raw_input in raw_inputs:
            system_prompt, user_prompt = get_annotator_correction_prompt(raw_input, type)
            inputs.append(
                {
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'transformed_input': raw_input,
                    'input_file': str(input_file),
                },
            )
        return inputs
    
    
    for raw_input in raw_inputs:
        data, order = transform_data(raw_input, type)
        # print(data)
        system_prompt, user_prompt = get_annotator_correction_prompt(data, type)
        inputs.append(
                {
                    'system_prompt': system_prompt,
                    'user_prompt': user_prompt,
                    'transformed_input': data,
                    'input_file': str(input_file),
                    'order': order
                },
            )
        i+=1

    if shuffle:
        random.shuffle(inputs)
    return inputs