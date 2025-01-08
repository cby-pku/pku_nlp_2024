import os
import sys
import math
import torch
import argparse
import textwrap
import transformers
from peft import PeftModel
from transformers import GenerationConfig, TextStreamer
from .llama_attn_replace import replace_llama_attn
import json
from tqdm import tqdm

PROMPT_DICT = {
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_no_input_llama2": (
        "[INST] <<SYS>>\n"
        "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\n"
        "If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\n"
        "<</SYS>> \n\n {instruction} [/INST]"
    ),
    "prompt_llama2": "[INST]{instruction}[/INST]"
}

# FIXME 实际会传入一个context 然后返回一个answers数组
class LongLoraProvider:
    def __init__(self,
                model_name_or_path,
                context,
                context_size: int = 32768,
                max_gen_len: int = 512,
                flash_attn: bool = True,
                temperature: float = 0.8,
                top_p: float = 0.95,
                **kwargs
    ):
        """
            **param: model_name_or_path: the model for inference
            **param: context: same with answer_generation.py
            
            Your output should follow the format:
            {
            'output':answer,
            'context_length':context_entry['context_length'],
            'needle_depth_percent':context_entry['needle_depth_percent'],
            'needle_type':context_entry['needle_type'],
            'model_name_or_path':model_name_or_path,
            }
            
            
            Example for original inference:
            
            python3 inference.py  
                --base_model ${MODEL} 
                --question "What are the main contributions and novelties of this work?" 
                --context_size 32768 
                --max_gen_len 512  
                --flash_attn True 
                --material ${CONTEXT}
            
        """
        self.base_model = model_name_or_path
        self.cache_dir = './cache'
        self.context = context
        self.context_size = context_size
        self.temperature = temperature
        self.top_p = top_p
        self.max_gen_len = max_gen_len
        self.flash_attn = flash_attn
        self.final_results = []
        self.testing_results = []
        
    

    def log(self,string):
        print()
        print("=="*60)
        print()
        print(string)
        print()
        print("=="*60)
        print()
    
    def build_generator(self,
        model, tokenizer, temperature=0.6, top_p=0.9, max_gen_len=4096, use_cache=True
    ):
        def response(prompt):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # streamer = TextStreamer(tokenizer)
            
            output = model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                use_cache=use_cache,
            )
            
            out = tokenizer.decode(output[0], skip_special_tokens=True)
            self.log(out)

            # 移除开头的 "<s>"
            if out.startswith("<s>"):
                out = out[len("<s>"):].strip()

            # 检查是否包含 prompt 的内容
            split_out = out.split(prompt)
            if len(split_out) > 1:
                out = split_out[1].strip()
            else:
                out = split_out[0].strip()  # 如果不包含，直接使用整个输出
            
            # 移除开头的 "[INST] \n"
            if out.startswith("[INST] \n"):
                out = out[len("[INST] \n"):].strip()
                
            return out

        return response

    def inference(self):
        if self.flash_attn:
            replace_llama_attn(inference=True)

        # Set RoPE scaling factor
        config = transformers.AutoConfig.from_pretrained(
            self.base_model,
            cache_dir=self.cache_dir,
        )

        orig_ctx_len = getattr(config, "max_position_embeddings", None)
        if orig_ctx_len and self.context_size > orig_ctx_len:
            scaling_factor = float(math.ceil(self.context_size / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}

        # Load model and tokenizer
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.base_model,
            config=config,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.resize_token_embeddings(32001)

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.base_model,
            cache_dir=self.cache_dir,
            model_max_length=self.context_size if self.context_size > orig_ctx_len else orig_ctx_len,
            padding_side="right",
            use_fast=False,
        )

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        model.eval()

        respond = self.build_generator(model, tokenizer, temperature=self.temperature, top_p=self.top_p,

                                max_gen_len=self.max_gen_len, use_cache=True)

        
        
        # context 里面是 context \ retrieval_question
        prompt_no_input = PROMPT_DICT["prompt_llama2"]
        
        self.log('Long Lora begin inference...')
        
        for context_entry in tqdm(self.context):
            material = context_entry['context']
            question = context_entry['retrieval_question']
            prompt = prompt_no_input.format_map({"instruction": material + "\n%s"%question})
            output = respond(prompt=prompt)
            self.testing_results.append(output)
        
        self.log('Long Lora finished inference...')
        
    def get_results(self):
        for context_entry,answer in zip(self.context,self.testing_results):
            self.final_results.append(
                {
                    'output':answer,
                    'context_length':context_entry['context_length'],
                    'needle_depth_percent':context_entry['needle_depth_percent'],
                    'needle_type':context_entry['needle_type'],
                    'model_name_or_path':self.base_model,
                }
            )
            
        return self.final_results
    


