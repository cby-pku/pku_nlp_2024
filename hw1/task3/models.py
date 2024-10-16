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
    

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        :param patience: How many epochs to wait after the last time validation loss improved.
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, embeddings):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, src, src_lengths):
        embedded = self.embedding(src)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded,
            torch.tensor(src_lengths, dtype=torch.int64),
            batch_first=True,
            enforce_sorted=False
        )
        packed_outputs, (hidden, cell) = self.lstm(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # No summing of bidirectional outputs
        # outputs shape: [batch_size, seq_len, hidden_dim * 2]
        # Concatenate final forward and backward hidden states
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        return outputs, hidden.unsqueeze(0)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        # Repeat decoder hidden state src_len times
        hidden = hidden.permute(1, 0, 2).repeat(1, src_len, 1)
        # Concatenate hidden and encoder_outputs
        # hidden: [batch_size, src_len, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim * 2]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy: [batch_size, src_len, hidden_dim]
        attention = self.v(energy).squeeze(2)
        # attention: [batch_size, src_len]
        attention = attention.masked_fill(mask == 0, -1e10)
        return torch.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, embeddings, attention):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding.from_pretrained(embeddings)
        self.attention = attention
        self.lstm = nn.LSTM(hidden_dim * 2 + embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2 + embedding_dim, output_dim)
        
    def forward(self, input, hidden, encoder_outputs, mask):
        input = input.unsqueeze(1)  # [batch_size, 1]
        embedded = self.embedding(input)  # [batch_size, 1, embedding_dim]
        a = self.attention(hidden, encoder_outputs, mask)  # [batch_size, src_len]
        a = a.unsqueeze(1)  # [batch_size, 1, src_len]
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, hidden_dim * 2]
        lstm_input = torch.cat((embedded, weighted), dim=2)  # [batch_size, 1, embedding_dim + hidden_dim * 2]
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, torch.zeros_like(hidden)))
        output = output.squeeze(1)  # [batch_size, hidden_dim]
        embedded = embedded.squeeze(1)  # [batch_size, embedding_dim]
        weighted = weighted.squeeze(1)  # [batch_size, hidden_dim * 2]
        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def create_mask(self, src):
        mask = (src != jpn_word2idx['<PAD>'])
        return mask
        
    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(0)
        max_len = trg.size(1)
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, max_len, trg_vocab_size).to(device)
        encoder_outputs, hidden = self.encoder(src, src_lengths)
        input = trg[:, 0]
        mask = self.create_mask(src)
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs
