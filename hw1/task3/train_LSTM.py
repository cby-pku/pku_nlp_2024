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
from models import *
from training_config import *
from evaluate import *

nltk.download('punkt')
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
mecab = MeCab.Tagger("-Owakati")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


def load_vocab(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        word2idx = eval(f.read())
    idx2word = {idx: word for word, idx in word2idx.items()}
    return word2idx, idx2word

jpn_word2idx, jpn_idx2word = load_vocab('vocab/jpn_word2idx.txt')
eng_word2idx, eng_idx2word = load_vocab('vocab/eng_word2idx.txt')

vocab_size_jpn = len(jpn_word2idx)
vocab_size_eng = len(eng_word2idx)



embedding_dim = 256 

jpn_cbow_model = CBOWModel(vocab_size_jpn, embedding_dim)
jpn_cbow_model.load_state_dict(torch.load('models/jpn_cbow_model.pth', map_location=device))
jpn_embeddings = jpn_cbow_model.embeddings.weight.data.clone()

eng_cbow_model = CBOWModel(vocab_size_eng, embedding_dim)
eng_cbow_model.load_state_dict(torch.load('models/eng_cbow_model.pth', map_location=device))
eng_embeddings = eng_cbow_model.embeddings.weight.data.clone()

print("Loaded pre-trained CBOW embeddings.")


def tokenize_japanese(text):
    return mecab.parse(text).strip().split()

def tokenize_english(text):
    return nltk.word_tokenize(text.lower())

def process_data(file_path, src_word2idx, trg_word2idx):
    src_sequences = []
    trg_sequences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        src_sentence, trg_sentence = parts
        src_tokens = tokenize_japanese(src_sentence)
        trg_tokens = tokenize_english(trg_sentence)
        src_indices = [src_word2idx.get(token, src_word2idx['<UNK>']) for token in src_tokens]
        trg_indices = [trg_word2idx.get(token, trg_word2idx['<UNK>']) for token in trg_tokens]
        src_sequences.append(src_indices)
        trg_sequences.append(trg_indices)
    return src_sequences, trg_sequences

train_src, train_trg = process_data('datasets/train.txt', jpn_word2idx, eng_word2idx)
val_src, val_trg = process_data('datasets/val.txt', jpn_word2idx, eng_word2idx)
test_src, test_trg = process_data('datasets/test.txt', jpn_word2idx, eng_word2idx)


batch_size = 64
train_dataset = TranslationDataset(train_src, train_trg)
val_dataset = TranslationDataset(val_src, val_trg)
test_dataset = TranslationDataset(test_src, test_trg)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

print("Data loaders are prepared.")



INPUT_DIM = vocab_size_jpn
OUTPUT_DIM = vocab_size_eng
ENC_EMB_DIM = embedding_dim
DEC_EMB_DIM = embedding_dim
HID_DIM = 512

attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, jpn_embeddings)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, eng_embeddings, attn)
model = Seq2Seq(enc, dec).to(device)


optimizer = optim.Adam(model.parameters(), lr=0.001)
PAD_IDX = eng_word2idx['<PAD>']
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

early_stopping = EarlyStopping(patience=5, min_delta=0.001)

def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    progress_bar = tqdm(iterator, desc="Training", leave=False)
    for src, trg, src_lengths, _ in progress_bar:
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, src_lengths, trg)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg, src_lengths, _ in tqdm(iterator, desc="Evaluating", leave=False):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, src_lengths, trg, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

print("Starting training with early stopping...")
for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch+1}/{N_EPOCHS}")
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, val_loader, criterion)

    early_stopping(valid_loss)
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'models/seq2seq_model_early_stopping.pth')

    train_ppl = math.exp(train_loss)
    valid_ppl = math.exp(valid_loss)
    print(f"\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl:.2f}")
    print(f"\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl:.2f}")
    
    if early_stopping.early_stop:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

print("Training completed.")

model.load_state_dict(torch.load('models/seq2seq_model_early_stopping.pth'))
print("Loaded the best model for evaluation.")



train_loss = evaluate(model, train_loader, criterion)
train_ppl = math.exp(train_loss)
train_bleu = calculate_bleu(train_dataset, model, jpn_word2idx, eng_idx2word, device)
print(f"Training Set - Loss: {train_loss:.3f} | PPL: {train_ppl:.2f} | BLEU: {train_bleu*100:.2f}")

val_loss = evaluate(model, val_loader, criterion)
val_ppl = math.exp(val_loss)
val_bleu = calculate_bleu(val_dataset, model, jpn_word2idx, eng_idx2word, device)
print(f"Validation Set - Loss: {val_loss:.3f} | PPL: {val_ppl:.2f} | BLEU: {val_bleu*100:.2f}")

test_loss = evaluate(model, test_loader, criterion)
test_ppl = math.exp(test_loss)
test_bleu = calculate_bleu(test_dataset, model, jpn_word2idx, eng_idx2word, device)
print(f"Test Set - Loss: {test_loss:.3f} | PPL: {test_ppl:.2f} | BLEU: {test_bleu*100:.2f}")


test_cases = [
    "私の名前は愛です",
    "昨日はお肉を食べません",
    "いただきますよう",
    "秋は好きです",
    "おはようございます"
]

print("\nTranslations of test cases:")
for idx, case in enumerate(test_cases):
    translation = translate_sentence(case, model, jpn_word2idx, eng_idx2word, device)
    print(f"Case {idx+1}: {case}")
    print(f"Translation: {' '.join(translation)}\n")




print("\nEvaluation Metrics:")
print(f"Training Set - Loss: {train_loss:.3f} | PPL: {train_ppl:.2f} | BLEU: {train_bleu*100:.2f}")
print(f"Validation Set - Loss: {val_loss:.3f} | PPL: {val_ppl:.2f} | BLEU: {val_bleu*100:.2f}")
print(f"Test Set - Loss: {test_loss:.3f} | PPL: {test_ppl:.2f} | BLEU: {test_bleu*100:.2f}")

print("\nPredictions on Test Cases:")
for idx, case in enumerate(test_cases):
    translation = translate_sentence(case, model, jpn_word2idx, eng_idx2word, device)
    print(f"Case {idx+1}: {case}")
    print(f"Translation: {' '.join(translation)}\n")
