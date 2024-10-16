import random
import MeCab
import nltk
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import os
from tqdm import tqdm

nltk.download('punkt')
mecab = MeCab.Tagger("-Owakati")

SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

def load_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        word2idx = eval(f.read())
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

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

def main():
    jpn_word2idx, jpn_idx2word = load_vocab('vocab/jpn_word2idx.txt')
    eng_word2idx, eng_idx2word = load_vocab('vocab/eng_word2idx.txt')

    train_lines = open('datasets/train.txt', 'r', encoding='utf-8').readlines()
    val_lines = open('datasets/val.txt', 'r', encoding='utf-8').readlines()
    test_lines = open('datasets/test.txt', 'r', encoding='utf-8').readlines()

    jpn_train_sentences = []
    eng_train_sentences = []
    jpn_val_sentences = []
    eng_val_sentences = []
    jpn_test_sentences = []
    eng_test_sentences = []

    for line in train_lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue 
        jpn_sentence, eng_sentence = parts
        jpn_tokens = mecab.parse(jpn_sentence).strip().split()
        eng_tokens = nltk.word_tokenize(eng_sentence.lower())
        jpn_train_sentences.append(jpn_tokens)
        eng_train_sentences.append(eng_tokens)
    for line in val_lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        jpn_sentence, eng_sentence = parts
        jpn_tokens = mecab.parse(jpn_sentence).strip().split()
        eng_tokens = nltk.word_tokenize(eng_sentence.lower())
        jpn_val_sentences.append(jpn_tokens)
        eng_val_sentences.append(eng_tokens)

    for line in test_lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        jpn_sentence, eng_sentence = parts
        jpn_tokens = mecab.parse(jpn_sentence).strip().split()
        eng_tokens = nltk.word_tokenize(eng_sentence.lower())
        jpn_test_sentences.append(jpn_tokens)
        eng_test_sentences.append(eng_tokens)

    jpn_cbow_data = generate_cbow_data(jpn_train_sentences, jpn_word2idx, window_size)
    eng_cbow_data = generate_cbow_data(eng_train_sentences, eng_word2idx, window_size)

    jpn_val_cbow_data = generate_cbow_data(jpn_val_sentences, jpn_word2idx, window_size)
    eng_val_cbow_data = generate_cbow_data(eng_val_sentences, eng_word2idx, window_size)


    jpn_test_cbow_data = generate_cbow_data(jpn_test_sentences, jpn_word2idx, window_size)
    eng_test_cbow_data = generate_cbow_data(eng_test_sentences, eng_word2idx, window_size)

    embedding_dim = 256
    vocab_size_jpn = len(jpn_word2idx)
    vocab_size_eng = len(eng_word2idx)

    jpn_model = CBOWModel(vocab_size_jpn, embedding_dim)
    eng_model = CBOWModel(vocab_size_eng, embedding_dim)
    jpn_loss_function = nn.NLLLoss()
    eng_loss_function = nn.NLLLoss()
    jpn_optimizer = optim.Adam(jpn_model.parameters(), lr=0.001)
    eng_optimizer = optim.Adam(eng_model.parameters(), lr=0.001)

    def prepare_dataloader(cbow_data, batch_size):
        contexts = [context for context, target in cbow_data]
        targets = [target for context, target in cbow_data]
        dataset = torch.utils.data.TensorDataset(torch.tensor(contexts, dtype=torch.long),
                                                 torch.tensor(targets, dtype=torch.long))
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return dataloader

    batch_size = 64
    jpn_train_loader = prepare_dataloader(jpn_cbow_data, batch_size)
    eng_train_loader = prepare_dataloader(eng_cbow_data, batch_size)
    jpn_val_loader = prepare_dataloader(jpn_val_cbow_data, batch_size)
    eng_val_loader = prepare_dataloader(eng_val_cbow_data, batch_size)
    jpn_test_loader = prepare_dataloader(jpn_test_cbow_data, batch_size)
    eng_test_loader = prepare_dataloader(eng_test_cbow_data, batch_size)
    
    def evaluate_model(model, dataloader, loss_function):
        model.eval()
        total_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for context_batch, target_batch in dataloader:
                log_probs = model(context_batch)
                loss = loss_function(log_probs, target_batch)
                total_loss += loss.item()
                _, predicted = torch.max(log_probs, 1)
                total += target_batch.size(0)
                correct += (predicted == target_batch).sum().item()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total
        return avg_loss, accuracy

    epochs = 10

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        jpn_model.train()
        total_loss_jpn = 0
        with tqdm(jpn_train_loader, desc=f"Training Japanese Model", leave=False) as pbar:
            for context_batch, target_batch in pbar:
                jpn_model.zero_grad()
                log_probs = jpn_model(context_batch)
                loss = jpn_loss_function(log_probs, target_batch)
                loss.backward()
                jpn_optimizer.step()
                total_loss_jpn += loss.item()
                pbar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})
        avg_loss_jpn = total_loss_jpn / len(jpn_train_loader)
        print(f"Japanese Model Training Loss: {avg_loss_jpn:.4f}")

        val_loss_jpn, val_accuracy_jpn = evaluate_model(jpn_model, jpn_val_loader, jpn_loss_function)
        print(f"Japanese Model Validation Loss: {val_loss_jpn:.4f}, Accuracy: {val_accuracy_jpn:.4f}")

        eng_model.train()
        total_loss_eng = 0
        with tqdm(eng_train_loader, desc=f"Training English Model", leave=False) as pbar:
            for context_batch, target_batch in pbar:
                eng_model.zero_grad()
                log_probs = eng_model(context_batch)
                loss = eng_loss_function(log_probs, target_batch)
                loss.backward()
                eng_optimizer.step()
                total_loss_eng += loss.item()
                pbar.set_postfix({'Batch Loss': f"{loss.item():.4f}"})
        avg_loss_eng = total_loss_eng / len(eng_train_loader)
        print(f"English Model Training Loss: {avg_loss_eng:.4f}")

        val_loss_eng, val_accuracy_eng = evaluate_model(eng_model, eng_val_loader, eng_loss_function)
        print(f"English Model Validation Loss: {val_loss_eng:.4f}, Accuracy: {val_accuracy_eng:.4f}")

    print("\nEvaluating Models on Test Set...")
    test_loss_jpn, test_accuracy_jpn = evaluate_model(jpn_model, jpn_test_loader, jpn_loss_function)
    print(f"Japanese Model Test Loss: {test_loss_jpn:.4f}, Accuracy: {test_accuracy_jpn:.4f}")

    test_loss_eng, test_accuracy_eng = evaluate_model(eng_model, eng_test_loader, eng_loss_function)
    print(f"English Model Test Loss: {test_loss_eng:.4f}, Accuracy: {test_accuracy_eng:.4f}")

    os.makedirs('models', exist_ok=True)
    torch.save(jpn_model.state_dict(), 'models/jpn_cbow_model.pth')
    torch.save(eng_model.state_dict(), 'models/eng_cbow_model.pth')
    print("\nModels saved successfully!")

if __name__ == '__main__':
    main()
