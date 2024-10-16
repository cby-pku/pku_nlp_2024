import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import jieba
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = [self.vocab.get(token, self.vocab['<UNK>']) for token in jieba.lcut(text)]
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]  
        else:
            tokens += [self.vocab['<PAD>']] * (self.max_len - len(tokens)) 
            
        return torch.tensor(tokens), torch.tensor(label)

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        tokens = jieba.lcut(text)
        counter.update(tokens)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    current_index = 2

    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = current_index
            current_index += 1

    return vocab

def load_dataset(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().rsplit('\t', 1) 
            texts.append(text)
            labels.append(int(label))
            
    
    return texts, labels