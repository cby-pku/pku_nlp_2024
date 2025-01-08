'''This python file is used to prepare context given different document length \ needle type \ context length'''
import asyncio
import glob
import json
import os
import time
from tqdm import tqdm
import numpy as np
from typing import Optional

from asyncio import Semaphore
from datetime import datetime, timezone
import transformers
from transformers import AutoTokenizer
def log(string):
    print()
    print("=="*60)
    print()
    print(string)
    print()
    print("=="*60)
    print()
    
os.environ['TOKENIZERS_PARALLELISM'] = '(true | false)'
class PrepareTester:
    """
    This class is used to generate the context and prompt
    """
    
    def __init__(self,

                #直接用加载好的tokenizer
                model_to_test: transformers.PreTrainedTokenizerBase,
                               
                # NOTE 这是个数组,这里面应该有不同的context length
                context_lengths: None,
                
                # NOTE 这里面应该读三个键值, needle_answer. needle_type, retrieval_question
                needles = None,
                document_depth_percents = None,
                
                
                haystack_dir = "PaulGrahamEssays",
                final_context_length_buffer = 200,
                num_concurrent_requests = 1,
                **kwargs
                ):
        """
        :param contex_length: the total context length for context in prompt
        :depth_percent: needle position percent in the context
        :needle_type: to read different needle types (for NIAH & NIAHS)
        """
        self.model_to_test = model_to_test
        self.context_lengths = context_lengths
        
        self.needles = needles
        self.document_depth_percents = document_depth_percents

        self.haystack_dir = haystack_dir
        self.final_context_length_buffer = 200
        self.num_concurrent_requests = num_concurrent_requests
        self.testing_results = []
    
    async def bound_evaluate_and_log(self,sem,*args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def run_test(self):
        # NOTE 使用信号量控制并发请求的数目
        sem = Semaphore(self.num_concurrent_requests)
        
        # Run through each iteration of context_lengths and depths and needle_types
        tasks = []
        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                for needle in self.needles:
                    task = self.bound_evaluate_and_log(sem,context_length,depth_percent,needle)
                    
                    tasks.append(task)
        
        # NOTE 与下面冲突
        # await asyncio.gather(*tasks)
        
        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Preparing context"):
            await f
            
        
    async def evaluate_and_log(self,context_length,depth_percent,needle):
        
        needle_text = needle['needle_text']
        needle_type = needle['needle_type']
        retrieval_question = needle['retrieval_question']
        # needle_answer. needle_type, retrieval_question
        
        context = await self.generate_context(context_length,depth_percent,needle_text)
        
        
        # 然后这里就可以直接返回了
        
        results = {
            'context':context,
            'context_length':context_length,
            'needle_depth_percent':depth_percent,
            'needle_text':needle_text,
            'needle_type':needle_type,
            'retrieval_question': retrieval_question
        }
        
        self.testing_results.append(results)
    
    
    def encode_text_to_tokens(self,text:str) -> list[str]:
        # NOTE 因为只用tokenizer 所以这里其实model_to_test 就是tokenizer
        return self.model_to_test.encode(text)
        # return self.model_to_test.encode(text).ids
    
    def decode_tokens(self, tokens: list[int], context_length: Optional[int] = None) -> str:
        return self.model_to_test.decode(tokens[:context_length])
    
    
    def get_context_length_in_tokens(self,context):
        return len(self.encode_text_to_tokens(context))

        
    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        
        print(f'Max Context Length: {max_context_length}')
        # Package Directory
        base_dir = os.path.abspath(os.path.dirname(__file__))
        
        print(f'Current Base Directory :{base_dir}')
        while self.get_context_length_in_tokens(context) < max_context_length:
            
            for file in glob.glob(os.path.join(base_dir,self.haystack_dir,"*.txt")):
                with open(file, 'r', encoding = 'utf-8') as f:
                    context += f.read()
        return context
            
    def encode_and_trim(self,context,context_length):
        tokens = self.encode_text_to_tokens(context)
        if (len(tokens)) > context_length:
            # 只decode context_length前的tokens
            context = self.decode_tokens(tokens,context_length)
        return context

    # NOTE 这是主方法，可以用于generation context
    async def generate_context(self,context_length, depth_percent,needle_text):
        
        # NOTE 应该这里会比较慢，原来的工程里也加了异步处理，第二个很耗时间的是generate
        
        # Get haystack files loaded into a sttring 
        context = self.read_context_files()
        
        # NOTE 这里和原来的有很大的不一样，不用encode，直接truncate就好
        # Truncate the haystack dir essays to the context length you desire
        context = self.encode_and_trim(context,context_length)
        
        # Insert your random statement according to your depth percent
        context = self.insert_needle(context, depth_percent,context_length,needle_text)
        
        return context
        
    
    # NOTE 这个可以插入needle, 在generation的时候可以在主方法里循环遍历 needle type
    def insert_needle(self,context,depth_percent,context_length,needle_text):
        
        tokens_needle = self.encode_text_to_tokens(needle_text)
        tokens_context = self.encode_text_to_tokens(context)
        
        # NOTE
        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.encode_text_to_tokens('.')
            
            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context
    
    
    
    # FIXME 调用的地方
    def get_results(self):
        return self.testing_results
    
    # FIXME 调用的地方
    def prepare_context_all(self):
        # 用这行代码可以保证异步方法完全运行完成并写入数据后再返回
        log('Preparation of Context Begin...')
        asyncio.run(self.run_test())
        log('Preparation of Context End..')
        

if __name__=='__main__':
    
    prepare = PrepareTester()
    prepare.prepare_context_all()  # 确保异步操作完成
    results = prepare.get_results()
    print(results)  # 打印结果
        

