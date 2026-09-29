"""
Shared components for PyTorch CNN → TFLite pipeline.
Single source of truth for model, data, training, eval, plotting.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

from archive.config import (
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
    AUG_FLIP_PROB,
    AUG_PADDING,
    BATCH_SIZE,
    CONV1_OUT,
    CONV2_OUT,
    DATA_ROOT,
    DEVICE,
    FC_HIDDEN,
    NORMALIZE_MEAN,
    NORMALIZE_STD,
    NUM_CLASSES,
)


# ---- Model (defined ONCE) ----
class CustomCNN(nn.Module):
    """CIFAR-10 CNN: 2 conv blocks + 2 FC layers."""

    def __init__(
        self,
        conv1_out=CONV1_OUT,
        conv2_out=CONV2_OUT,
        fc_hidden=FC_HIDDEN,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(3, conv1_out, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, 3, padding=1)
        self.fc1 = nn.Linear(conv2_out * 8 * 8, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        # NCHW: [B,3,32,32] -> conv1 -> [B,16,32,32] -> pool -> [B,16,16,16]
        x = self.pool(F.relu(self.conv1(x)))
        # -> conv2 -> [B,32,16,16] -> pool -> [B,32,8,8]
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, CONV2_OUT * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_device():
    return torch.device(DEVICE)


# ---- Transforms ----
def get_test_transform():
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)]
    )


def get_train_transform(augment=False):
    if augment:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=AUG_PADDING),
                transforms.RandomHorizontalFlip(AUG_FLIP_PROB),
                transforms.ColorJitter(
                    brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST
                ),
                transforms.ToTensor(),
                transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
            ]
        )
    return get_test_transform()


# ---- DataLoaders ----
def get_dataloaders(train_transform, test_transform, batch_size=BATCH_SIZE):
    trainset = datasets.CIFAR10(
        DATA_ROOT, train=True, download=True, transform=train_transform
    )
    testset = datasets.CIFAR10(
        DATA_ROOT, train=False, download=True, transform=test_transform
    )
    return (
        torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False),
    )


# ---- Training / Eval ----
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        pred = model(inputs).argmax(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def run_training(
    model, trainloader, testloader, optimizer, epochs, desc="", device=None
):
    if device is None:
        device = get_device()
    criterion = nn.CrossEntropyLoss()
    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for epoch in range(1, epochs + 1):
        loss, tr_acc = train_epoch(model, trainloader, optimizer, criterion, device)
        te_acc = evaluate(model, testloader, device)
        history["train_loss"].append(loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        print(
            f"{desc} Epoch [{epoch}/{epochs}] - Loss: {loss:.4f} | Train: {tr_acc:.2f}% | Test: {te_acc:.2f}%"
        )
    return history


# ---- Plotting ----
def plot_baseline(history, save_path):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(
        range(1, len(history["train_loss"]) + 1), history["train_loss"], marker="o"
    )
    axes[0].set_title("Train Loss per Epoch")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(
        range(1, len(history["test_acc"]) + 1),
        history["test_acc"],
        marker="o",
        color="orange",
    )
    axes[1].set_title("Test Accuracy per Epoch")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy %")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_augment(adam_hist, sgd_hist, save_path):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    _fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Train vs Test Accuracy
    axes[0, 0].plot(adam_hist["train_acc"], label="Adam Train", marker="o")
    axes[0, 0].plot(adam_hist["test_acc"], label="Adam Test", marker="s")
    axes[0, 0].plot(
        sgd_hist["train_acc"], label="SGD Train", marker="o", linestyle="--"
    )
    axes[0, 0].plot(sgd_hist["test_acc"], label="SGD Test", marker="s", linestyle="--")
    axes[0, 0].set_title("Training vs Test Accuracy")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Accuracy %")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Loss
    axes[0, 1].plot(adam_hist["train_loss"], label="Adam", marker="o")
    axes[0, 1].plot(sgd_hist["train_loss"], label="SGD", marker="s")
    axes[0, 1].set_title("Training Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Generalization Gap
    adam_gap = [a - b for a, b in zip(adam_hist["train_acc"], adam_hist["test_acc"])]
    sgd_gap = [a - b for a, b in zip(sgd_hist["train_acc"], sgd_hist["test_acc"])]
    axes[1, 0].plot(adam_gap, label="Adam", marker="o")
    axes[1, 0].plot(sgd_gap, label="SGD", marker="s")
    axes[1, 0].set_title("Generalization Gap (Train - Test)")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Gap %")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Final Accuracy Bar
    axes[1, 1].bar(
        ["Adam", "SGD"],
        [adam_hist["test_acc"][-1], sgd_hist["test_acc"][-1]],
        color=["blue", "orange"],
    )
    axes[1, 1].set_title("Final Test Accuracy")
    axes[1, 1].set_ylabel("Accuracy %")
    axes[1, 1].set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
