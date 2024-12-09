import random
import csv
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from deepseek import get_deepseek_response, load_api_keys
from datasets import load_dataset
from tqdm import tqdm

def load_gsm8k_test_data():
    dataset = load_dataset("openai/gsm8k", "main")
    dataset = dataset['test']
    test_data = list(zip(dataset['question'], dataset['answer']))
    test_data_subset = random.sample(test_data, 100)
    return test_data_subset

def naive_prompting(system_prompt, question, api_key):
    return get_deepseek_response(system_prompt, question, api_key)

def chain_of_thought_prompting(system_prompt, question, api_key):
    cot_prompt = f"{question} Let's think step by step."
    return get_deepseek_response(system_prompt, cot_prompt, api_key)

def in_context_learning(system_prompt, question, examples, api_key):
    icl_prompt = "\n".join([f"Q: {ex[0]} A: {ex[1]}" for ex in examples]) + f"\nQ: {question} A:"
    return get_deepseek_response(system_prompt, icl_prompt, api_key)

def reflexion_prompting(system_prompt, question, api_key):
    reflexion_prompt = f"{question} Reflect on the possible solutions."
    return get_deepseek_response(system_prompt, reflexion_prompt, api_key)

def process_question(question, answer, system_prompt, api_key, examples):
    naive_response = naive_prompting(system_prompt, question, api_key)
    cot_response = chain_of_thought_prompting(system_prompt, question, api_key)
    icl_response = in_context_learning(system_prompt, question, examples, api_key)
    reflexion_response = reflexion_prompting(system_prompt, question, api_key)
    
    return {
        "Question": question,
        "Correct Answer": answer,
        "Naive Response": naive_response,
        "CoT Response": cot_response,
        "ICL Response": icl_response,
        "Reflexion Response": reflexion_response
    }

if __name__ == "__main__":
    api_keys = load_api_keys('./config/key.txt')
    api_key = api_keys[0]
    system_prompt = "You are a helpful assistant."
    
    test_data = load_gsm8k_test_data()
    examples = test_data[:5]  

    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(process_question, question, answer, system_prompt, api_key, examples) 
                  for question, answer in test_data]
        for future in tqdm(futures, total=len(futures), desc="Processing questions"):
            results.append(future.result())

    # naive_accuracy = sum(1 for result in results if result["Correct Answer"] in result["Naive Response"]) / len(results)
    # cot_accuracy = sum(1 for result in results if result["Correct Answer"] in result["CoT Response"]) / len(results)
    # icl_accuracy = sum(1 for result in results if result["Correct Answer"] in result["ICL Response"]) / len(results)
    # reflexion_accuracy = sum(1 for result in results if result["Correct Answer"] in result["Reflexion Response"]) / len(results)

    # print(f"Naive Accuracy: {naive_accuracy:.2%}")
    # print(f"CoT Accuracy: {cot_accuracy:.2%}")
    # print(f"ICL Accuracy: {icl_accuracy:.2%}")
    # print(f"Reflexion Accuracy: {reflexion_accuracy:.2%}")

    df = pd.DataFrame(results)
    df.to_csv('results.csv', index=False)