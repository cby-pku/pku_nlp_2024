import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from typing import List
import numpy as np
import pandas as pd

from tqdm import tqdm
class InferenceProfiler:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
    def load_model(self, quantization=None):
        if quantization == "int8":
            return AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                device_map="auto",
                load_in_8bit=True
            )
        elif quantization == "int4":
            return AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                load_in_4bit=True
            )
        else:
            return AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

    def measure_inference(self, 
                         prompts: List[str], 
                         max_length: int = 50,
                         use_cache: bool = True,
                         quantization: str = None) -> dict:
        model = self.load_model(quantization)
        
        # 对于量化模型，我们需要将输入放在与模型第一层相同的设备上
        if quantization:
            device = model.device
        else:
            device = self.device
        
        # NOTE recode the GPU memory usage
        if torch.cuda.is_available():
            # NOTE reset the peak memory stats
            torch.cuda.reset_peak_memory_stats()
            # NOTE get the current memory allocated
            start_mem = torch.cuda.memory_allocated()
        
        total_tokens = 0
        start_time = time.time()
        first_token_time = None
        
        for prompt in tqdm(prompts):
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_length=max_length,
                    use_cache=use_cache,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            if first_token_time is None:
                first_token_time = time.time() - start_time
            
            total_tokens += output.shape[1]
        
        end_time = time.time()
        
        results = {
            "total_time": end_time - start_time,
            "tokens_per_second": total_tokens / (end_time - start_time),
            "first_token_latency": first_token_time
        }
        
        if torch.cuda.is_available():
            peak_mem = torch.cuda.max_memory_allocated()
            results["peak_memory_mb"] = peak_mem / 1024 / 1024
            results["memory_increase_mb"] = (peak_mem - start_mem) / 1024 / 1024
            
        return results

def load_prompts_from_file(file_path: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]

def main():
    model_variants = [
        "/data/models/Llama-2-7b-hf",
        # "meta-llama/Llama-2-13b-hf",
        # "meta-llama/Llama-2-70b-hf",
    ]
    
    for model_name in model_variants:
        print(f"\nTesting model: {model_name}")
        profiler = InferenceProfiler(model_name)
        
        test_prompts = load_prompts_from_file("test_prompts.txt")
        
        configs = [
            {"name": "Baseline (no cache)", "cache": False, "quant": None},
            {"name": "With KV-cache", "cache": True, "quant": None},
            {"name": "INT8 + KV-cache", "cache": True, "quant": "int8"},
            {"name": "INT4 + KV-cache", "cache": True, "quant": "int4"},
            {"name": "INT8 + No KV-cache", "cache": False, "quant": "int8"},
            {"name": "INT4 + No KV-cache", "cache": False, "quant": "int4"},
        ]
        
        print("Start testing...\n")
        
        all_results = []
        
        for config in configs:
            print(f"Testing config: {config['name']}")
            results = profiler.measure_inference(
                test_prompts,
                use_cache=config["cache"],
                quantization=config["quant"]
            )
            
            print(f"Total time: {results['total_time']:.2f} seconds")
            print(f"Inference speed: {results['tokens_per_second']:.2f} tokens/second")
            if "peak_memory_mb" in results:
                print(f"Peak memory usage: {results['peak_memory_mb']:.2f} MB")
                print(f"Memory increase: {results['memory_increase_mb']:.2f} MB")
            print("\n")
            
            # NOTE add the results to the list·
            result_dict = {
                "config_name": config["name"],
                "total_time": results["total_time"],
                "tokens_per_second": results["tokens_per_second"],
                "first_token_latency": results["first_token_latency"]
            }
            if "peak_memory_mb" in results:
                result_dict["peak_memory_mb"] = results["peak_memory_mb"]
                result_dict["memory_increase_mb"] = results["memory_increase_mb"]
            
            all_results.append(result_dict)
        
        df = pd.DataFrame(all_results)
        model_short_name = model_name.split('/')[-1].lower()
        df.to_csv(f"inference_results_llama2.csv", index=False)
        print(f"Results have been saved to inference_results_llama2.csv")

if __name__ == "__main__":
    main()