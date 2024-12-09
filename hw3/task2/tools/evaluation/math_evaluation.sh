#!/bin/bash
TYPE='math-evaluation'
INPUT_FOLDER='/data/align-anything/boyuan/nlp-workspace/pku_nlp_2024/hw3/task2/reasoning/evaluation_results'
OUTPUT_FOLDER='/data/align-anything/boyuan/nlp-workspace/pku_nlp_2024/hw3/task2/reasoning/evaluation_results'
# 遍历 input_folder 下的所有 json 文件
find "${INPUT_FOLDER}" -type f -name '*.json' | while read INPUT_PATH; do

  
  # 调用处理脚本
  bash new_script.sh \
      --input-file "${INPUT_PATH}" \
      --output-folder "${OUTPUT_FOLDER}" \
      --type "${TYPE}"
      
done

