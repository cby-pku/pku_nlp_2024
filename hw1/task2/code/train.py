import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import jieba
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import wandb
from model import *
from dataset import *


wandb.init(project="cnn-text-classification", name="sentence_classification", config={
    "epochs": 10,
    "learning_rate": 0.001,
    "batch_size": 32,
    "embed_size": 128
})

config = wandb.config


def train_model(train_loader, dev_loader, model, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_dev_acc = 0
    patience, patience_counter = 3, 0  
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
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": total_loss / len(train_loader),
            "train_acc": train_acc,
            "dev_acc": dev_acc
        })

        print(f'Epoch {epoch+1}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.4f}, Dev Acc: {dev_acc:.4f}')

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
            print(f"  New best dev accuracy: {best_dev_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def evaluate(model, loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        loader_tqdm = tqdm(loader, desc="Evaluating")
        for inputs, labels in loader_tqdm:
            outputs = model(inputs)
            correct += (outputs.argmax(1) == labels).sum().item()
    return correct / len(loader.dataset)


def main():

    train_dataset_path = '../datasets/train.txt'
    dev_dataset_path = '../datasets/dev.txt'
    test_dataset_path = '../datasets/test.txt'

    train_texts, train_labels = load_dataset(train_dataset_path)
    dev_texts, dev_labels = load_dataset(dev_dataset_path)
    test_texts, test_labels = load_dataset(test_dataset_path)

    vocab = build_vocab(train_texts, min_freq=2)
    max_len = 100

    train_dataset = TextDataset(train_texts, train_labels, vocab, max_len)
    dev_dataset = TextDataset(dev_texts, dev_labels, vocab, max_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    vocab_size = len(vocab)
    embed_size = config.embed_size
    num_classes = len(set(train_labels))

    model = CNN_Text(vocab_size=vocab_size, embed_size=embed_size, num_classes=num_classes)
    train_model(train_loader, dev_loader, model, epochs=config.epochs, lr=config.learning_rate)

    print("\nEvaluating on test set...")
    test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")

    wandb.log({"test_accuracy": test_acc})
    wandb.finish()
    
if __name__ == '__main__':
    main()
