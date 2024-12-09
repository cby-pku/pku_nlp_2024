import os
import json
import csv

def calculate_accuracy(directory):
    results = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('_re_output.json'):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                data = json.load(file)
                total = len(data)
                true_count = sum(1 for item in data if item.get('output') == '[[Output]]: true')
                accuracy = true_count / total if total > 0 else 0
                results.append((os.path.basename(filename), accuracy))

    # 将结果写入CSV文件
    csv_filename = os.path.join(directory, 'accuracy_results.csv')
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Filename', 'Accuracy'])
        csv_writer.writerows(results)

    print(f"Accuracy results saved to {csv_filename}")

# 使用示例
calculate_accuracy('/data/align-anything/boyuan/nlp-workspace/pku_nlp_2024/hw3/task2/reasoning/evaluation_results')