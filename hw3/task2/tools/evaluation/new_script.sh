#!/bin/bash

if [ -z "${BASH_VERSION}" ]; then
    echo "Please use bash to run this script." >&2
    exit 1
fi

set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" &>/dev/null && pwd)"
ROOT_DIR="$(dirname "${SCRIPT_DIR}")"
SCRIPT_NAME=$(basename "$0") 
SCRIPT_NAME_WITHOUT_EXTENSION="${SCRIPT_NAME%.sh}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# 默认参数配置
folder_names=("gpt4/" "llama2-7B-chat/" "alpaca-7B/"  "gpt3.5-turbo-1106/" "llama2-13B-chat/" "llama2-70B-chat/" "vicuna-7B/" "vicuna-13B-v1.5/" "vicuna-33B-v1.3/" "claude2/" "beaver/" "gpt3.5-turbo-1106/")

export LOGLEVEL="${LOGLEVEL:-WARNING}"
INPUT_PATH=""
OUTPUT_PATH=""
OUTPUT_FOLDER=""
OUTPUT_NAME=""
MODEL=""
PLATFORM="openai"
TEMPLATE=''

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    arg="$1"
    shift
    case "${arg}" in
        --input-file)
            INPUT_PATH="$1"
            shift
            ;;
        --input-file=*)
            INPUT_PATH="${arg#*=}"
            ;;
        --output-folder)
            OUTPUT_FOLDER="$1"
            shift
            ;;
        --output-folder=*)
            OUTPUT_FOLDER="${arg#*=}"
            ;;
        --type)
            TEMPLATE="$1"
            shift
            ;;
        --type=*)
            TEMPLATE="${arg#*=}"
            ;;
        --folder-name)
            folder_names=("")
            shift
            ;;
        *)
            echo "Unknown parameter passed: '${arg}'" >&2
            exit 1
            ;;
    esac
done

# 检查必需参数是否存在
if [[ -z "$INPUT_PATH" ]]; then
    echo "Error: --input-file is required"
    exit 1
fi

if [[ -z "$TEMPLATE" ]]; then
    echo "Error: --type is required"
    exit 1
fi

# 分配可用端口
MASTER_PORT_START=10000
MASTER_PORT_END=65535
MASTER_PORT="$(
    comm -23 \
        <(seq "${MASTER_PORT_START}" "${MASTER_PORT_END}" | sort) \
        <(ss -Htan | awk '{ print $4 }' | awk -F ':' '{ print $NF }' | sort -u) |
        shuf | head -n 1
)"

# 确保输出目录存在
CACHE_DIR="${SCRIPT_DIR}/.cache/${SCRIPT_NAME_WITHOUT_EXTENSION}"
mkdir -p "${CACHE_DIR}"

# 执行 Python 脚本
python3 main.py --debug \
    --openai-api-key-file "${SCRIPT_DIR}/config/openai_api_keys.txt" \
    --input-file "${INPUT_PATH}" \
    --output-folder ${OUTPUT_FOLDER} \
    --num-workers 50 \
    --type "${TEMPLATE}" \
    --shuffle
