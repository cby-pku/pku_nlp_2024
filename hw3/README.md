## PartI: Inference Efficiency Test
```
cd task1/inference_efficiency_test
```
Test and compare inference efficiency (basic) metrics and compare the inference throughput (tokens/s) between different models

- A naïve (baseline) LLM inference implementation.
- An implementation that utilizes the KV-cache directly from HuggingFace Transformers.

Experiment with quantization techniques: Measure and compare GPU memory usage and inference speed under different quantization levels. Record your findings and discuss how quantization impacts the trade-off between memory and speed. 



## PartII: Customized GPT-2 Model
```
cd task1/main
```
Implement a customized GPT-2 model with KV-cache by inheriting from the `GPT2LMHeadModel` class provided in the `transformers` library.

You can use the following command to run the code:
```
python main.py
```

## PartIII: LLM Reasoning Techniques
```
cd task2/reasoning
```
Implement the reasoning techniques discussed in the lecture.

