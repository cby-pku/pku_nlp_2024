## Part I: Inference Efficiency Test
```
cd task1/inference_efficiency_test
```
Test and compare inference efficiency (basic) metrics and compare the inference throughput (tokens/s) between different models

- A naïve (baseline) LLM inference implementation.
- An implementation that utilizes the KV-cache directly from HuggingFace Transformers.

Experiment with quantization techniques: Measure and compare GPU memory usage and inference speed under different quantization levels. Record your findings and discuss how quantization impacts the trade-off between memory and speed. 



## Part II: Customized GPT-2 Model
```
cd task1/main
```
Implement a customized GPT-2 model with KV-cache by inheriting from the `GPT2LMHeadModel` class provided in the `transformers` library.

You can use the following command to run the code:
```
python main.py
```

## Part III: LLM Reasoning Techniques
```
cd task2/reasoning
```
Implement the reasoning techniques discussed in the lecture.

Use the following command to run the code:

```
python main.py

``` 


Use the following command to convert the results.csv to results.json
```
python tools/csv_to_json.py
```

And consider the evaluation of the reasoning techniques, we use GPT4o as the evaluator.

Use the following command to run the code:
```
python tools/gpt4o_eval.py
```

