from __future__ import annotations
import os
import sys
import json
import argparse
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from vllm import SamplingParams, LLM
from models.constants_template import (
    PROMPT_INPUT,
    PROMPT_LONG_CRITIQUE,
    PROMPT_LONG_SAFE,
    PROMPT_LONG_RETRIEAVL,
    PROMPT_LONG_JUDGE
)
from generate_context_prompt import PrepareTester
from special_provider.lora_inference import LongLoraProvider

class InferenceEngine:
    def __init__(self, model_name_or_path: str, context: list[dict], output_dir: str):
        self.model_name_or_path = model_name_or_path
        self.context = context
        self.output_dir = output_dir
        self.gpu_num = 8
    
    def generate_vllm(self) -> list[dict]:
        sampling_params = SamplingParams(
            top_k=30,
            top_p=0.95,
            temperature=0.4,
            
            # NOTE 这里的 max_tokens 也要改啊, 但是这里是inference output 的max tokens
            max_tokens=1024,
            frequency_penalty=1.2,
        )
        llm = LLM(
            model=self.model_name_or_path,
            tokenizer=self.model_name_or_path,
            tokenizer_mode='auto',
            trust_remote_code=False,
            download_dir=None,
            tensor_parallel_size=self.gpu_num,
            block_size=16,
            gpu_memory_utilization=0.9,
            max_num_seqs=128,
        )
        prompts = self._formulate_prompts(self.context)
        outputs = llm.generate(prompts, sampling_params)

        results = [
            {
                'output': output.outputs[0].text,
                'context_length': context_entry['context_length'],
                'needle_depth_percent': context_entry['needle_depth_percent'],
                'needle_type': context_entry['needle_type'],
                'model_name_or_path': self.model_name_or_path,
            }
            for output, context_entry in zip(outputs, self.context)
        ]
        return results

    def generate_special_provider(self) -> list[dict]:
        lora_generator = LongLoraProvider(
            model_name_or_path=self.model_name_or_path,
            context=self.context
        )
        lora_generator.inference()
        return lora_generator.get_results()

    def _formulate_prompts(self, context: list[dict]) -> list[str]:
        prompts = []
        for entry in context:
            needle_type = entry['needle_type']
            if 'safe' in needle_type:
                template = PROMPT_LONG_SAFE
            elif 'critique' in needle_type:
                template = PROMPT_LONG_CRITIQUE
            elif 'retrieval' in needle_type:
                template = PROMPT_LONG_RETRIEAVL
            elif 'judge' in needle_type:
                template = PROMPT_LONG_JUDGE
            else:
                raise ValueError(f"Unknown needle_type: {needle_type}")

            input_seq = template.format(
                context=entry['context'],
                question=entry['retrieval_question']
            )
            prompt = PROMPT_INPUT.format(input=input_seq)
            prompts.append(prompt)
        return prompts

    def save_answers_by_context_length(self, answers: list[dict]) -> None:
        categorized_answers = {}
        for answer in answers:
            context_length = answer['context_length']
            if context_length not in categorized_answers:
                categorized_answers[context_length] = []
            categorized_answers[context_length].append(answer)

        os.makedirs(self.output_dir, exist_ok=True)
        for context_length, answers_list in categorized_answers.items():
            final_output_path = os.path.join(self.output_dir, f'{context_length}.json')
            with open(final_output_path, 'w', encoding='utf-8') as f:
                json.dump(answers_list, f, indent=5, ensure_ascii=False)
            self._log(f'Data for context length {context_length} has been successfully saved at {final_output_path}')

    @staticmethod
    def _log(message: str) -> None:
        print(f"\n{'=='*30}\n\n{message}\n\n{'=='*30}\n")


class NeedlePreparer:
    def __init__(self, input_path: str):
        self.input_path = input_path

    def prepare_needles(self) -> tuple[list[dict], str]:
        with open(self.input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        path_type = os.path.basename(os.path.dirname(self.input_path))
        needles = []
        for entry in data:
            needle_type = self._determine_needle_type(entry, path_type)
            needles.append({
                'needle_type': needle_type,
                'needle_text': entry['answer'],
                'retrieval_question': entry['prompt'],
            })
        return needles, needles[0]['needle_type']

    @staticmethod
    def _determine_needle_type(entry: dict, path_type: str) -> str:
        if 'severity' in entry or path_type in ['safe', 'critique', 'judge']:
            return f"{path_type}-{entry['severity']}-{entry['idx']}"
        return path_type


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Inference via multiple models')
    parser.add_argument('--model_name_or_path', type=str, required=True, help='The name or path of the model to load from')
    parser.add_argument('--output_name', type=str, help='The name of the output json file', required=False)
    parser.add_argument('--output_dir', type=str, default=None, help='Where to store the eval output')
    parser.add_argument('--input_path', type=str, default='problem.json', help='Path to input problem JSON file')
    parser.add_argument('--template', type=str, default='template for inference qa', help='[OPTIONAL] Template name')
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    needle_preparer = NeedlePreparer(args.input_path)
    needles, needle_type = needle_preparer.prepare_needles()

    context_json_path = _prepare_context_json(args.model_name_or_path, needles, needle_type)
    with open(context_json_path, 'r', encoding='utf-8') as f:
        context = json.load(f)

    engine = InferenceEngine(args.model_name_or_path, context, args.output_dir)
    if 'Yarn' in args.model_name_or_path or 'longlora' in args.model_name_or_path or 'LongAlpaca' in args.model_name_or_path:
        engine._log('You are using LongLora series models to inference')
        answers = engine.generate_special_provider()
    else:
        answers = engine.generate_vllm()

    model_name = args.model_name_or_path.rstrip('/').split('/')[-1]
    final_output_dir = os.path.join(args.output_dir, model_name, needle_type)
    engine.save_answers_by_context_length(answers, final_output_dir)


def _prepare_context_json(model_name_or_path: str, needles: list[dict], needle_type: str) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    folder_name = os.path.join(parent_directory, f"context/{needle_type.split('-')[0]}")
    os.makedirs(folder_name, exist_ok=True)
    context_json_path = os.path.join(folder_name, "context.json")

    if not os.path.exists(context_json_path):
        context_preparer = PrepareTester(
            model_to_test=tokenizer,
            context_lengths=[i * 1000 for i in range(1, 33)],
            needles=needles,
            document_depth_percents=[i * 10 for i in range(1, 11)]
        )
        context_preparer.prepare_context_all()
        context = context_preparer.get_results()

        with open(context_json_path, 'w', encoding='utf-8') as f:
            json.dump(context, f, indent=5, ensure_ascii=False)
    
    return context_json_path


if __name__ == '__main__':
    sys.exit(main())
