import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
from typing import List
import numpy as np
import pandas as pd
from tqdm import tqdm

class GPUProfiler:
    """Handles GPU memory profiling for inference."""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reset_memory_stats(self):
        """Reset GPU memory statistics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def get_memory_usage(self) -> float:
        """Returns the current GPU memory allocated in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_peak_memory(self) -> float:
        """Returns the peak GPU memory used in MB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0
    
    def get_memory_increase(self, start_mem: float, end_mem: float) -> float:
        """Returns the difference in memory usage in MB."""
        return (end_mem - start_mem)

class ModelLoader:
    """Loads the GPT2 model with the required configuration (e.g., quantization)."""
    
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load(self, quantization: str = None) -> GPT2LMHeadModel:
        """Load model with specific quantization or baseline configuration."""
        if quantization == "int8":
            return GPT2LMHeadModel.from_pretrained(self.model_name, device_map="auto", load_in_8bit=True)
        elif quantization == "int4":
            return GPT2LMHeadModel.from_pretrained(self.model_name, device_map="auto", load_in_4bit=True)
        else:
            model = GPT2LMHeadModel.from_pretrained(self.model_name)
            return model.to("cuda" if torch.cuda.is_available() else "cpu")

class InferenceRunner:
    """Handles the actual inference execution and profiling."""
    
    def __init__(self, model_loader: ModelLoader, gpu_profiler: GPUProfiler):
        self.model_loader = model_loader
        self.gpu_profiler = gpu_profiler

    def measure_inference(self, prompts: List[str], batch_size: int, max_length: int = 50, use_cache: bool = True, quantization: str = None) -> dict:
        """运行推理并测量整体性能"""
        model = self.model_loader.load(quantization)
        device = model.device if quantization else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.gpu_profiler.reset_memory_stats()
        start_mem = self.gpu_profiler.get_memory_usage()
        
        # 记录总体开始时间
        start_time = time.time()
        total_tokens = 0
        
        # 按批次处理所有数据
        for i in tqdm(range(0, (len(prompts) + batch_size - 1) // batch_size)):
            batch = prompts[i * batch_size: (i + 1) * batch_size]
            input_ids = self.model_loader.tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)['input_ids']
            
            with torch.no_grad():
                output = model.generate(
                    input_ids, 
                    max_length=max_length, 
                    use_cache=use_cache, 
                    pad_token_id=self.model_loader.tokenizer.eos_token_id
                )
            total_tokens += output.shape[0] * output.shape[1]

        # 记录总体结束时间
        end_time = time.time()
        peak_mem = self.gpu_profiler.get_peak_memory()
        memory_increase = self.gpu_profiler.get_memory_increase(start_mem, peak_mem)
        
        results = {
            "total_time": end_time - start_time,
            "tokens_per_second": total_tokens / (end_time - start_time),
            "peak_memory_mb": peak_mem,
            "memory_increase_mb": memory_increase
        }
        
        return results

class ResultSaver:
    """Saves inference results to a CSV file."""
    
    @staticmethod
    def save(results: List[dict], output_path: str = "inference_results_gpt2.csv"):
        """Save the results in CSV format."""
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"Results have been saved to {output_path}")

def load_prompts(file_path: str) -> List[str]:
    """Loads prompts from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        prompts = file.readlines()
    return [prompt.strip() for prompt in prompts]

def run_inference(configs: List[dict], prompts: List[str], model_loader: ModelLoader, gpu_profiler: GPUProfiler, batch_sizes: List[int]) -> List[dict]:
    """为每个配置和批次大小运行推理"""
    inference_runner = InferenceRunner(model_loader, gpu_profiler)
    results = []

    for config in configs:
        print(f"\nTesting config: {config['name']}")
        for batch_size in batch_sizes:
            print(f"Batch size: {batch_size}")
            result = inference_runner.measure_inference(
                prompts,
                batch_size=batch_size,
                use_cache=config["cache"],
                quantization=config["quant"]
            )
            result.update({
                "config_name": config["name"],
                "batch_size": batch_size
            })
            results.append(result)

    return results

def main():
    model_loader = ModelLoader()
    gpu_profiler = GPUProfiler()

    # Load prompts from file
    with open("data.txt") as f:
        test_prompts = [line.strip() for line in f.readlines()]

    # Define configurations
    configs = [
        {"name": "Baseline (no cache)", "cache": False, "quant": None},
        {"name": "With KV-cache", "cache": True, "quant": None},
    ]

    batch_sizes = [1, 2, 4, 8, 16]
    
    # 运行推理并收集结果
    results = run_inference(configs, test_prompts, model_loader, gpu_profiler, batch_sizes)
    
    # 保存结果
    ResultSaver.save(results)

if __name__ == "__main__":
    main()
