# NOTE 首先进行数据集的拆分

import os
import random

# Paths
input_dataset = '/data/align-anything/boyuan/nlp-workspace/hw1/task3/datasets/eng_jpn.txt'
train_dataset = '/data/align-anything/boyuan/nlp-workspace/hw1/task3/datasets/train.txt'
val_dataset = '/data/align-anything/boyuan/nlp-workspace/hw1/task3/datasets/val.txt'
test_dataset = '/data/align-anything/boyuan/nlp-workspace/hw1/task3/datasets/test.txt'

# Load dataset
with open(input_dataset, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Shuffle data
random.seed(42)
random.shuffle(lines)

# Split dataset: 8:1:1
num_samples = len(lines)
train_split = int(0.8 * num_samples)
val_split = int(0.9 * num_samples)

train_lines = lines[:train_split]
val_lines = lines[train_split:val_split]
test_lines = lines[val_split:]

# Save splits
with open(train_dataset, 'w', encoding='utf-8') as train_file:
    train_file.writelines(train_lines)

with open(val_dataset, 'w', encoding='utf-8') as val_file:
    val_file.writelines(val_lines)

with open(test_dataset, 'w', encoding='utf-8') as test_file:
    test_file.writelines(test_lines)

print("Dataset split completed: train, validation, and test sets saved.")
