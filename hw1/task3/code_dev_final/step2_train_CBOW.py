import random
import MeCab
import nltk
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# Download nltk resources
nltk.download('punkt')

# Initialize MeCab tokenizer for Japanese
mecab = MeCab.Tagger("-Owakati")

# 1. Data Preprocessing

## 1.1 Read and split dataset
def read_and_split_data(file_path):
    # Read dataset
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Shuffle data
    random.shuffle(lines)

    # Split dataset
    total = len(lines)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[train_size + val_size:]

    # Save split data
    os.makedirs('datasets', exist_ok=True)
    with open('datasets/train.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open('datasets/val.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    with open('datasets/test.txt', 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    
    return train_lines, val_lines, test_lines

# Call the function to read and split data
train_lines, val_lines, test_lines = read_and_split_data('datasets/eng_jpn.txt')

## 1.2 Define tokenization functions

### Japanese tokenization
def tokenize_japanese(text):
    return mecab.parse(text).strip().split()

### English tokenization
def tokenize_english(text):
    return nltk.word_tokenize(text.lower())

## 1.3 Build vocabulary
def build_vocab(tokenized_texts, min_freq=1):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    tokens = [word for word, freq in counter.items() if freq >= min_freq]
    word2idx = {word: idx+2 for idx, word in enumerate(tokens)}  # Reserve 0:<PAD>, 1:<UNK>
    word2idx['<PAD>'] = 0
    word2idx['<UNK>'] = 1
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

# Process training set
jpn_sentences = []
eng_sentences = []

for line in train_lines:
    parts = line.strip().split('\t')
    if len(parts) != 2:
        continue  # Skip invalid lines
    jpn_sentence, eng_sentence = parts
    jpn_tokens = tokenize_japanese(jpn_sentence)
    eng_tokens = tokenize_english(eng_sentence)
    jpn_sentences.append(jpn_tokens)
    eng_sentences.append(eng_tokens)

# Build vocabularies
jpn_word2idx, jpn_idx2word = build_vocab(jpn_sentences)
eng_word2idx, eng_idx2word = build_vocab(eng_sentences)

# 2. Prepare CBOW training data
window_size = 2

def generate_cbow_data(sentences, word2idx, window_size):
    data = []
    for tokens in sentences:
        indices = [word2idx.get(token, word2idx['<UNK>']) for token in tokens]
        for i in range(window_size, len(indices) - window_size):
            context = indices[i - window_size:i] + indices[i + 1:i + window_size + 1]
            target = indices[i]
            data.append((context, target))
    return data

# Generate training data
jpn_cbow_data = generate_cbow_data(jpn_sentences, jpn_word2idx, window_size)
eng_cbow_data = generate_cbow_data(eng_sentences, eng_word2idx, window_size)

# 3. Implement CBOW Model
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(CBOWModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def forward(self, context_idxs):
        embeds = self.embeddings(context_idxs)
        context_mean = embeds.mean(dim=1)
        out = self.linear(context_mean)
        log_probs = nn.functional.log_softmax(out, dim=1)
        return log_probs

# 4. Train the Model

## 4.1 Set up models, loss functions, and optimizers for both languages
embedding_dim = 256
vocab_size_jpn = len(jpn_word2idx)
vocab_size_eng = len(eng_word2idx)

# Initialize models, loss functions, and optimizers
jpn_model = CBOWModel(vocab_size_jpn, embedding_dim)
eng_model = CBOWModel(vocab_size_eng, embedding_dim)
jpn_loss_function = nn.NLLLoss()
eng_loss_function = nn.NLLLoss()
jpn_optimizer = optim.Adam(jpn_model.parameters(), lr=0.001)
eng_optimizer = optim.Adam(eng_model.parameters(), lr=0.001)

## 4.2 DataLoader preparation
def prepare_dataloader(cbow_data, batch_size):
    contexts = [context for context, target in cbow_data]
    targets = [target for context, target in cbow_data]
    dataset = torch.utils.data.TensorDataset(torch.tensor(contexts, dtype=torch.long),
                                             torch.tensor(targets, dtype=torch.long))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

batch_size = 64
jpn_dataloader = prepare_dataloader(jpn_cbow_data, batch_size)
eng_dataloader = prepare_dataloader(eng_cbow_data, batch_size)

## 4.3 Define evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for context_batch, target_batch in dataloader:
            log_probs = model(context_batch)
            _, predicted = torch.max(log_probs, 1)
            total += target_batch.size(0)
            correct += (predicted == target_batch).sum().item()
    accuracy = correct / total
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

## 4.4 Training loop
epochs = 5

for epoch in range(epochs):
    # Japanese model training
    jpn_model.train()
    total_loss_jpn = 0
    for context_batch, target_batch in jpn_dataloader:
        jpn_model.zero_grad()
        log_probs = jpn_model(context_batch)
        loss = jpn_loss_function(log_probs, target_batch)
        loss.backward()
        jpn_optimizer.step()
        total_loss_jpn += loss.item()
    print(f"Epoch {epoch+1} - Japanese Model Loss: {total_loss_jpn:.2f}")
    
    # English model training
    eng_model.train()
    total_loss_eng = 0
    for context_batch, target_batch in eng_dataloader:
        eng_model.zero_grad()
        log_probs = eng_model(context_batch)
        loss = eng_loss_function(log_probs, target_batch)
        loss.backward()
        eng_optimizer.step()
        total_loss_eng += loss.item()
    print(f"Epoch {epoch+1} - English Model Loss: {total_loss_eng:.2f}")
    
    # Evaluate models on test set
    print("Evaluating Japanese Model on Test Set:")
    jpn_test_data = generate_cbow_data(jpn_sentences, jpn_word2idx, window_size)
    jpn_test_loader = prepare_dataloader(jpn_test_data, batch_size)
    evaluate_model(jpn_model, jpn_test_loader)
    
    print("Evaluating English Model on Test Set:")
    eng_test_data = generate_cbow_data(eng_sentences, eng_word2idx, window_size)
    eng_test_loader = prepare_dataloader(eng_test_data, batch_size)
    evaluate_model(eng_model, eng_test_loader)

# 5. Save the models
os.makedirs('models', exist_ok=True)
torch.save(jpn_model.state_dict(), 'models/jpn_cbow_model.pth')
torch.save(eng_model.state_dict(), 'models/eng_cbow_model.pth')
print("Models saved successfully!")
