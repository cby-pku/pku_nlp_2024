import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import jieba
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

class CNN_Text(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, kernel_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super(CNN_Text, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_size, out_channels=num_filters, kernel_size=k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        if (x >= len(self.embedding.weight)).any():
            raise ValueError(f"Out-of-range token index found: {x.max()} (vocab size is {len(self.embedding.weight)})")
        x = self.embedding(x).permute(0, 2, 1)
        convs = [torch.relu(conv(x)) for conv in self.convs]
        pools = [torch.max(conv, dim=2)[0] for conv in convs]
        out = torch.cat(pools, dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out