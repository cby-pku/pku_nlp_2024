
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


def baichuan_gpt_api(
    input: dict[str, str],
    openai_api_keys: str,
    openai_model: str,
) -> Any:
    """Baichuan GPT API"""

    messages = input
    output = {}
    output['message'] = input
    output['model'] = openai_model

    openai_api = 'http://47.236.144.103'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer '+ openai_api_keys,
        'Connection':'close',
        }
    
    retry_strategy = Retry(
        #NOTE
        total=50,  # 最大重试次数（包括首次请求）
        backoff_factor=1,  # 重试之间的等待时间因子
        status_forcelist=[429, 500, 502, 503, 504],  # 需要重试的状态码列表
        allowed_methods=["POST"],  # 只对POST请求进行重试
        raise_on_redirect=False,  # Don't raise exception
        raise_on_status=False,  # Don't raise exception
    )

    params_gpt = {
        'model': openai_model,
        'messages': messages,
        'temperature': 0.05,
        'max_tokens': 4096,
        "stop": None,
    }

    url = openai_api + '/v1/chat/completions'

    # retry_strategy = Retry(
    #     total=5,  # Maximum retry count
    #     backoff_factor=0.1,  # Wait factor between retries
    #     status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to force a retry on
    #     allowed_methods=['POST'],  # Retry only for POST request
    #     raise_on_redirect=False,  # Don't raise exception
    #     raise_on_status=False,  # Don't raise exception
    # )

    # http = urllib3.PoolManager(
    #     retries=retry_strategy,
    # )
    encoded_data = json.dumps(params_gpt).encode('utf-8')

    print('Baichuan Proxy API Called...')

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    
    # if response.status_code != 200:
    #     err_msg = f'Access openai error, Key platform Baichuan, Key id: {openai_api_keys}, status code: {response.status_code}, status info : {response.text}\n request detail: {encoded_data}'
    #     logging.error(err_msg)
    #     response = 'Baichuan Proxy API Failed...'
    # output['output'] = response
    # return output
    
    # NOTE
    max_try = 30
    while max_try > 0:
        try:
            response = session.post(url, headers=headers, data=json.dumps(params_gpt), timeout=(5, 150)) # NOTE
            if response.status_code == 200:
                response = json.loads(response.text)['choices'][0]['message']['content']
                # print(response)
                logging.info(response)
                break
            err_msg = f'Access openai error, Key platform Baichuan, Key id: {openai_api_keys}, status code: {response.status_code}, status info : {response.text}\n request detail: {encoded_data}'
            logging.error(err_msg)
            time.sleep(random.randint(5, 30) * 0.1)
            max_try -= 1
        except requests.exceptions.Timeout:
            logging.error("Request timed out")
            max_try -= 1
        except Exception as e:
            err_msg = f'Access openai error, Key platform Baichuan, Key id: {openai_api_keys}, status code: {"N/A"}, status info : {"N/A"}\n request detail: {encoded_data}, Error: {str(e)}'
            logging.error(err_msg)
            max_try -= 1
    else:
        print('Baichuan Proxy API Failed...')
        # print('Using OpenAI API...')
        # response = ray.get(gpt_api.remote(system_content, user_content))
        response = 'Baichuan Proxy API Failed...'

    output['output'] = response

    return output


def bean_gpt_api(
    input: dict[str, str],
    openai_api_keys: str,
    openai_model: str,
) -> Any:
    """Bean GPT API"""


    messages = input
    output = {}
    output['message'] = input
    output['model'] = openai_model

    # openai_api = 'https://apejhvxcd.cloud.sealos.io'
    openai_api = 'https://api.61798.cn'
    # openai_api = 'https://api.close2openai.com'

    headers = {
        'Content-Type': 'application/json',
        'Authorization': openai_api_keys,
        'Connection':'close',
        }

    retry_strategy = Retry(
        total=50,  # Maximum retry count
        backoff_factor=0.1,  # Wait factor between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to force a retry on
        allowed_methods=['POST'],  # Retry only for POST request
        raise_on_redirect=False,  # Don't raise exception
        raise_on_status=False,  # Don't raise exception
    )
    params_gpt = {
        'model': openai_model,
        'messages': messages,
        'temperature': 0.05,
        'max_tokens': 4096,
        "stop": None,
    }

    url = openai_api + '/v1/chat/completions'

    # retry_strategy = Retry(
    #     total=5,  # Maximum retry count
    #     backoff_factor=0.1,  # Wait factor between retries
    #     status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to force a retry on
    #     allowed_methods=['POST'],  # Retry only for POST request
    #     raise_on_redirect=False,  # Don't raise exception
    #     raise_on_status=False,  # Don't raise exception
    # )

    # http = urllib3.PoolManager(
    #     retries=retry_strategy,
    # )
    encoded_data = json.dumps(params_gpt).encode('utf-8')

    print('Bean Proxy API Called...')

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    max_try = 30
    while max_try > 0:
        try:
            response = session.post(url, headers=headers, data=json.dumps(params_gpt), timeout=(5, 250))
            if response.status_code == 200:
                response = json.loads(response.text)['choices'][0]['message']['content']
                # print(response)
                logging.info(response)
                break
            err_msg = f'Access openai error, Key platform Bean, Key id: {openai_api_keys}, status code: {response.status_code}, status info : {response.text}\n request detail: {encoded_data}'
            logging.error(err_msg)
            time.sleep(random.randint(5, 30) * 0.1)
            max_try -= 1
        except requests.exceptions.Timeout:
            logging.error("Request timed out")
            max_try -= 1
        except Exception as e:
            err_msg = f'Access openai error, Key platform Bean, Key id: {openai_api_keys}, status code: {"N/A"}, status info : {"N/A"}\n request detail: {encoded_data}, Error: {str(e)}'
            logging.error(err_msg)
            max_try -= 1
    else:
        print('Bean Proxy API Failed...')
        # print('Using OpenAI API...')
        # response = ray.get(gpt_api.remote(system_content, user_content))
        response = 'Bean Proxy API Failed...'

    output['output'] = response
    return output

def request_openai_noexcept(
    messages: list[dict[str, str]],
    openai_api_keys: str,
    openai_model: str,
    base_url: str | None = None,
) -> list[dict[str, object]]:
    output = None
    hit_rate_limit = 0
    while True:
        client = openai.OpenAI(api_key=openai_api_keys, base_url=base_url)
        try:
            output = client.chat.completions.create(
                messages=messages,
                model=openai_model,
                max_tokens=8192,
                temperature=0.05,
            )
            break
        except openai.OpenAIError as e:
            logging.error(e)
            if 'maximum context length' in str(e).lower():
                return {
                    'messages': messages,
                    'output': 'ERROR: reached maximum context length',
                    'model': openai_model,
                }
            if 'repetitive patterns' in str(e).lower():
                return {
                    'messages': messages,
                    'output': 'ERROR: Sorry! We have encountered an issue with repetitive patterns in your prompt. Please try again with a different prompt.',
                    'model': openai_model,
                }
            # if 'rate limit' in str(e).lower():
            #     hit_rate_limit += 1
            #     if hit_rate_limit >= 20:
            #         next_openai_api_key = next(openai_api_keys)
            #         logging.info(
            #             f'Hit rate limit for key {openai_api_keys}. '
            #             f'Switching to {next_openai_api_key}.',
            #         )
            #         openai_api_key = next_openai_api_key
            #         hit_rate_limit = 0
            # elif 'quota' in str(e).lower():
            #     next_openai_api_key = next(openai_api_keys)
            #     logging.info(
            #         f'Hit quota for key {openai_api_keys}. Switching to {next_openai_api_key}.',
            #     )
            #     openai_api_key = next_openai_api_key
            time.sleep(random.randint(5, 30) * 0.1)  # noqa: S311
    return {
        'messages': messages,
        'output': output.choices[0].message.content,
        'model': openai_model,
    }