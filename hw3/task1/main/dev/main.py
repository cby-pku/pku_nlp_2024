import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
from customized_gpt2 import CustomizedGPT2LMHeadModel


@torch.no_grad()
def customized_greedy_decoding(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    input_ids = tokenized_batch['input_ids']
    attention_mask = tokenized_batch['attention_mask']
    res = input_ids
    start_time = time.time()
    past_key_values = None

    for timestep in range(MAX_NEW_LENGTH):
        outputs = custom_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True
        )
        logits = outputs['logits']
        past_key_values = outputs['past_key_values']

        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        res = torch.cat([res, next_token], dim=-1)

        input_ids = next_token
        attention_mask = torch.ones_like(input_ids)  # 更新 attention_mask
        # attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)

    return res, time.time() - start_time



@torch.no_grad()
def golden_greedy_decoding_wo_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        # NOTE debug 要不要加这行代码
        # print(f'tokenized_batch: {tokenized_batch}')
        tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
        # print(f'tokenized_batch: {tokenized_batch}')
        outputs = original_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    return res, time.time() - start_time


if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 2
    times = [0, 0]

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", use_cache=False) # NOTE maybe need to set use_cache=True
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map='cuda', use_cache=True)
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map="cuda", use_cache=True)

    with open("data.txt") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]

    for i in tqdm(range(0, (len(prompt_dataset) + bsz - 1) // bsz)):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        golden_wo_cache_res, golden_wo_cache_time = golden_greedy_decoding_wo_cache(batch)
        custom_res, custom_time = customized_greedy_decoding(batch)

        times[0] += golden_wo_cache_time
        times[1] += custom_time
        
        print(f'golden_wo_cache_res: {golden_wo_cache_res}')
        print(f'custom_res: {custom_res}')

        assert torch.equal(golden_wo_cache_res, custom_res), "Decoding results are not equal"

    print("Time taken for golden greedy decoding without KV cache: ", times[0])
    print("Time taken for customized greedy decoding: ", times[1])
