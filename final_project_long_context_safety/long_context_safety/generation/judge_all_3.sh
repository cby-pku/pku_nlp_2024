
OUTPUT_DIR='/data/projects/beavertails2/boyuan/emnlp2024/niah/niah-safe/test/final_judge_results_supple'
INPUTFILE='/data/projects/beavertails2/boyuan/emnlp2024/niah/niah-safe/test/needles/judge/qa.json'

#!/bin/bash

# 但是因为是在answer_generation里面写的判断逻辑，所以不用显式特判在sh里
# dpo
# Define the directory paths

MODEL='/data/projects/beavertails2/boyuan/models/safe-dpo/safe-dpo-alpaca2-7b-reproduced'

bash answer_generation.sh \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --input_path "$INPUTFILE"




MODEL='/data/projects/beavertails2/boyuan/models/safe-dpo/safe-dpo-alpaca3-8b-reproduced'

bash answer_generation.sh \
  --model_name_or_path "$MODEL" \
  --output_dir "$OUTPUT_DIR" \
  --input_path "$INPUTFILE"
