
OUTPUT_DIR='/data/projects/beavertails2/boyuan/emnlp2024/niah/niah-safe/test/results'
INPUTFILE='/data/projects/beavertails2/boyuan/emnlp2024/niah/niah-safe/test/needles/critique/qa.json'

#!/bin/bash

# 但是因为是在answer_generation里面写的判断逻辑，所以不用显式特判在sh里
# dpo
# Define the directory paths


MODEL_DIR="/data/projects/beavertails2/boyuan/models/safe-dpo"
for MODEL in "$MODEL_DIR"/*/; do
  echo "Processing model: $MODEL"

  # if [[ "$MODEL" == *"dpo-alpaca-7b-reproduced"* ]]; then
  #   continue
  # fi
  # # if [[ "$MODEL" == *"dpo-alpaca-llama-2-7b-80k"* ]]; then
  # #   continue
  # # fi
  # # if [[ "$MODEL" == *"dpo-alpaca-llama2-7b-32k"* ]]; then
  # #   continue
  # # fi
  # if [[ "$MODEL" == *"dpo-alpaca2-7b-reproduced"* ]]; then
  #   continue
  # fi
  # if [[ "$MODEL" == *"dpo-alpaca3-8b-reproduced"* ]]; then
  #   continue
  # fi

  # Run the answer_generation.sh script with the current model
  bash answer_generation.sh \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --input_path "$INPUTFILE"
done


# sft
MODEL_DIR="/data/projects/beavertails2/boyuan/models/reproduced"
for MODEL in "$MODEL_DIR"/*/; do
  echo "Processing model: $MODEL"

  # Run the answer_generation.sh script with the current model
  bash answer_generation.sh \
    --model_name_or_path "$MODEL" \
    --output_dir "$OUTPUT_DIR" \
    --input_path "$INPUTFILE"
done


# pretrained
# to be continued