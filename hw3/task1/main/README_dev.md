## PartI: Inference Efficiency Test
```
cd inference_efficiency_test
```
Test and compare inference efficiency (basic) metrics and compare the inference throughput (tokens/s) between different models

- A naïve (baseline) LLM inference implementation.
- An implementation that utilizes the KV-cache directly from HuggingFace Transformers.

Experiment with quantization techniques: Measure and compare GPU memory usage and inference speed under different quantization levels. Record your findings and discuss how quantization impacts the trade-off between memory and speed. 

