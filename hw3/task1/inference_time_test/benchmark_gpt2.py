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

    def measure_inference(self, prompts: List[str], max_length: int = 50, use_cache: bool = True, quantization: str = None) -> dict:
        """Runs inference on provided prompts and measures performance."""
        model = self.model_loader.load(quantization)
        device = model.device if quantization else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Reset GPU memory stats
        self.gpu_profiler.reset_memory_stats()
        
        # Measure inference time
        start_time = time.time()
        total_tokens = 0
        first_token_time = None
        start_mem = self.gpu_profiler.get_memory_usage()

        for prompt in tqdm(prompts):
            input_ids = self.model_loader.tokenizer.encode(prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                output = model.generate(input_ids, max_length=max_length, use_cache=use_cache, pad_token_id=self.model_loader.tokenizer.eos_token_id)

            if first_token_time is None:
                first_token_time = time.time() - start_time

            total_tokens += output.shape[1]

        end_time = time.time()
        peak_mem = self.gpu_profiler.get_peak_memory()
        memory_increase = self.gpu_profiler.get_memory_increase(start_mem, peak_mem)
        
        results = {
            "total_time": end_time - start_time,
            "tokens_per_second": total_tokens / (end_time - start_time),
            "first_token_latency": first_token_time,
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

def run_inference(configs: List[dict], prompts: List[str], model_loader: ModelLoader, gpu_profiler: GPUProfiler) -> List[dict]:
    """Runs inference for all configurations and collects results."""
    inference_runner = InferenceRunner(model_loader, gpu_profiler)
    results = []

    for config in configs:
        print(f"Testing config: {config['name']}")
        result = inference_runner.measure_inference(prompts, use_cache=config["cache"], quantization=config["quant"])
        result["config_name"] = config["name"]
        results.append(result)

    return results

def main():
    model_loader = ModelLoader()
    gpu_profiler = GPUProfiler()

    # Load prompts from file
    test_prompts = load_prompts("test_prompts.txt")

    # Define configurations
    configs = [
        {"name": "Baseline (no cache)", "cache": False, "quant": None},
        {"name": "With KV-cache", "cache": True, "quant": None},
        {"name": "INT8 + KV-cache", "cache": True, "quant": "int8"},
        {"name": "INT4 + KV-cache", "cache": True, "quant": "int4"},
        {"name": "INT8 + No KV-cache", "cache": False, "quant": "int8"},
        {"name": "INT4 + No KV-cache", "cache": False, "quant": "int4"},
    ]

    print("Start testing...\n")
    
    # Run inference for all configurations
    results = run_inference(configs, test_prompts, model_loader, gpu_profiler)

    # Save the results to CSV
    ResultSaver.save(results)

if __name__ == "__main__":
    main()
