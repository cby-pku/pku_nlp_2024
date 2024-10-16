import random
import MeCab
import nltk
from collections import Counter
import os

nltk.download('punkt')
mecab = MeCab.Tagger("-Owakati")

def read_and_split_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    random.shuffle(lines)
    total = len(lines)
    train_size = int(0.8 * total)
    val_size = int(0.1 * total)

    train_lines = lines[:train_size]
    val_lines = lines[train_size:train_size + val_size]
    test_lines = lines[train_size + val_size:]
    os.makedirs('datasets', exist_ok=True)
    with open('datasets/train.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open('datasets/val.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    with open('datasets/test.txt', 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
    
    return train_lines, val_lines, test_lines

def tokenize_japanese(text):
    return mecab.parse(text).strip().split()

def tokenize_english(text):
    return nltk.word_tokenize(text.lower())

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

def main():
    input_file = './eng_jpn.txt'
    train_lines, val_lines, test_lines = read_and_split_data(input_file)

    jpn_sentences = []
    eng_sentences = []

    for line in train_lines:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        jpn_sentence, eng_sentence = parts
        jpn_tokens = tokenize_japanese(jpn_sentence)
        eng_tokens = tokenize_english(eng_sentence)
        jpn_sentences.append(jpn_tokens)
        eng_sentences.append(eng_tokens)
    jpn_word2idx, jpn_idx2word = build_vocab(jpn_sentences)
    eng_word2idx, eng_idx2word = build_vocab(eng_sentences)

    os.makedirs('vocab', exist_ok=True)
    with open('vocab/jpn_word2idx.txt', 'w', encoding='utf-8') as f:
        f.write(str(jpn_word2idx))
    with open('vocab/eng_word2idx.txt', 'w', encoding='utf-8') as f:
        f.write(str(eng_word2idx))

    print("Data preparation completed.")

if __name__ == '__main__':
    main()
