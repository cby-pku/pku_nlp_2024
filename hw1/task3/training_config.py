import random
import MeCab
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import math
from nltk.translate.bleu_score import corpus_bleu

class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences
    
    def __len__(self):
        return len(self.src_sequences)
    
    def __getitem__(self, idx):
        return self.src_sequences[idx], self.trg_sequences[idx]

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lengths = [len(s) for s in src_batch]
    trg_lengths = [len(t) for t in trg_batch]
    max_src_len = max(src_lengths)
    max_trg_len = max(trg_lengths)
    src_padded = [s + [jpn_word2idx['<PAD>']] * (max_src_len - len(s)) for s in src_batch]
    trg_padded = [t + [eng_word2idx['<PAD>']] * (max_trg_len - len(t)) for t in trg_batch]
    return (torch.tensor(src_padded, dtype=torch.long),
            torch.tensor(trg_padded, dtype=torch.long),
            src_lengths, trg_lengths)