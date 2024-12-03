# GPT-2 Inference Evaluation Tool

The file measures:
- Inference speed (tokens/second)
- Total processing time
- GPU memory usage (if CUDA is available)
- Impact of different optimization techniques:
  - KV-cache
  - INT8 quantization
  - INT4 quantization


## Usage

```bash
python benchmark.py
```

Results will be saved to `inference_results.csv`.

## Customization

### Modifying Test Prompts
Edit the `test_prompts` list in the `main()` function:

```python
test_prompts = [
    "Your custom prompt 1",
    "Your custom prompt 2",
    # Add more prompts...
]
```

### Adding New Configurations
Add new configurations to the `configs` list in `main()`:

```python
configs = [
    {
        "name": "Your Config Name",
        "cache": True/False,  # Enable/disable KV-cache
        "quant": None/"int8"/"int4"  # Quantization type
    },
    # Add more configs...
]
```

### Adjusting Generation Parameters
Modify the `measure_inference()` method parameters:
- `max_length`: Maximum length of generated sequences
- `use_cache`: Enable/disable KV-cache
- `quantization`: Quantization type (None, "int8", or "int4")

