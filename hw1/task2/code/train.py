import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import jieba
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb

# Initialize wandb

wandb.init(project="cnn-text-classification", name="sentence_classification", config={
    "epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 32,
    "embed_size": 128
})

config = wandb.config

# Dataset loading and preprocessing
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
        
        # Tokenize the text and convert tokens to vocabulary indices, map unknown tokens to <UNK>
        tokens = [self.vocab.get(token, self.vocab['<UNK>']) for token in jieba.lcut(text)]
        
        # Ensure token list length matches the required max length
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]  # Truncate if too long
        else:
            tokens += [self.vocab['<PAD>']] * (self.max_len - len(tokens))  # Pad if too short

        return torch.tensor(tokens), torch.tensor(label)

# Modify build_vocab to ensure <PAD> and <UNK> tokens are always included at the beginning
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

# CNN model
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

# Training function
def train_model(train_loader, dev_loader, model, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_dev_acc = 0
    patience, patience_counter = 3, 0  # Early stopping parameters

    # Track model with wandb
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        train_loader_tqdm = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, (inputs, labels) in enumerate(train_loader_tqdm):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

            train_loader_tqdm.set_postfix({"Batch Loss": loss.item()})

        train_acc = correct / len(train_loader.dataset)
        dev_acc = evaluate(model, dev_loader)

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "train_acc": train_acc,
            "dev_acc": dev_acc
        })

        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Dev Acc: {dev_acc:.4f}')

        # Early stopping
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            print(f"  New best dev accuracy: {best_dev_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


# Loading dataset from files
def load_dataset(file_path):
    texts = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            text, label = line.strip().rsplit('\t', 1)  # Assuming text and label are separated by a tab
            texts.append(text)
            labels.append(int(label))
            
    
    return texts, labels

def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        loader_tqdm = tqdm(loader, desc="Evaluating")
        for inputs, labels in loader_tqdm:
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)

# Load your data from files
train_dataset_path = '/data/align-anything/boyuan/nlp-workspace/hw1/task2/datasets/train.txt'
dev_dataset_path = '/data/align-anything/boyuan/nlp-workspace/hw1/task2/datasets/dev.txt'
test_dataset_path = '/data/align-anything/boyuan/nlp-workspace/hw1/task2/datasets/test.txt'

train_texts, train_labels = load_dataset(train_dataset_path)
dev_texts, dev_labels = load_dataset(dev_dataset_path)
test_texts, test_labels = load_dataset(test_dataset_path)

# Tokenization and vocabulary building
vocab = build_vocab(train_texts, min_freq=2)
max_len = 100

# Creating datasets and dataloaders
train_dataset = TextDataset(train_texts, train_labels, vocab, max_len)
dev_dataset = TextDataset(dev_texts, dev_labels, vocab, max_len)
test_dataset = TextDataset(test_texts, test_labels, vocab, max_len)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

# Model and training
vocab_size = len(vocab)
embed_size = config.embed_size
num_classes = len(set(train_labels))

model = CNN_Text(vocab_size=vocab_size, embed_size=embed_size, num_classes=num_classes)
train_model(train_loader, dev_loader, model, epochs=config.epochs, lr=config.learning_rate)

# Final evaluation on test set
print("\nEvaluating on test set...")
test_acc = evaluate(model, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")

# Log the final test accuracy to wandb
wandb.log({"test_accuracy": test_acc})
wandb.finish()
