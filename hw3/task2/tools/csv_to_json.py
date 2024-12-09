import csv
import json

def csv_to_json(csv_file_path, json_file_path):
    # 打开CSV文件
    with open(csv_file_path, mode='r', encoding='utf-8') as csv_file:
        # 创建CSV阅读器
        csv_reader = csv.DictReader(csv_file)
        
        # 将CSV数据转换为字典列表
        data = [row for row in csv_reader]
    
    # 将字典列表写入JSON文件
    with open(json_file_path, mode='w', encoding='utf-8') as json_file:
        json.dump(data, json_file, indent=4, ensure_ascii=False)

# 使用示例
csv_file_path = '/data/align-anything/boyuan/nlp-workspace/pku_nlp_2024/hw3/task2/reasoning/results.csv'
json_file_path = '/data/align-anything/boyuan/nlp-workspace/pku_nlp_2024/hw3/task2/reasoning/results.json'
csv_to_json(csv_file_path, json_file_path)