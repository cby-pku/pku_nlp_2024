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

def translate_sentence(sentence, model, src_word2idx, trg_idx2word, device, max_len=50):
    model.eval()
    tokens = tokenize_japanese(sentence)
    indices = [src_word2idx.get(token, src_word2idx['<UNK>']) for token in tokens]
    src_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    src_length = [len(indices)]
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor, src_length)
    mask = model.create_mask(src_tensor)
    input = torch.tensor([eng_word2idx['<PAD>']], dtype=torch.long).to(device)
    outputs = []
    for _ in range(max_len):
        output, hidden = model.decoder(input, hidden, encoder_outputs, mask)
        top1 = output.argmax(1).item()
        if top1 == eng_word2idx['<PAD>']:
            break
        outputs.append(top1)
        input = torch.tensor([top1], dtype=torch.long).to(device)
    trg_tokens = [trg_idx2word.get(idx, '<UNK>') for idx in outputs]
    return trg_tokens

def calculate_bleu(dataset, model, src_word2idx, trg_idx2word, device):
    references = []
    hypotheses = []
    for src_seq, trg_seq in tqdm(zip(dataset.src_sequences, dataset.trg_sequences), total=len(dataset), desc="Calculating BLEU"):
        src_sentence = ' '.join([jpn_idx2word.get(idx, '<UNK>') for idx in src_seq])
        trg_sentence = [eng_idx2word.get(idx, '<UNK>') for idx in trg_seq]
        pred_tokens = translate_sentence(src_sentence, model, src_word2idx, trg_idx2word, device)
        references.append([trg_sentence])
        hypotheses.append(pred_tokens)
    bleu_score = corpus_bleu(references, hypotheses)
    return bleu_score