import json
import os

# 读取原始JSON文件
with open('/data/align-anything/boyuan/nlp-workspace/pku_nlp_2024/hw3/task2/reasoning/results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 创建一个目录来存储拆分后的JSON文件
output_dir = '/data/align-anything/boyuan/nlp-workspace/pku_nlp_2024/hw3/task2/reasoning/evaluation_results'
os.makedirs(output_dir, exist_ok=True)

# 创建一个字典来存储每个方法的所有响应
all_responses = {
    'Naive': [],
    'CoT': [],
    'ICL': [],
    'Reflexion': []
}

# 遍历每个问题
for item in data:
    question = item['Question']
    correct_answer = item['Correct Answer']
    
    # 将每个方法的响应添加到对应的列表中
    for method in all_responses.keys():
        response = item[f'{method} Response']
        output_data = {
            'question': question,
            'answer': correct_answer,
            'response': response
        }
        all_responses[method].append(output_data)

# 为每个方法创建一个合并的JSON文件
for method, responses in all_responses.items():
    filename = f'{output_dir}/{method}_responses.json'
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(responses, outfile, indent=4, ensure_ascii=False)

print("JSON文件已成功合并并保存。")