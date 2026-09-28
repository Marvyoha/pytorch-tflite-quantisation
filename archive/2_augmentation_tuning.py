"""
Day 11: Data Augmentation & Hyperparameter Tuning for CIFAR-10 Custom CNN
Steps:
1. Introduce image data augmentation for training set only
2. Re-train CustomCNN with augmentations and compare overfitting
3. Benchmark Adam vs SGD with Momentum
4. Save best model checkpoint for ONNX/TFLite export
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torchvision import transforms

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Introduce Image Data Augmentation
# Test set remains un-augmented for reliable evaluation
test_transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Augmented training transform applied only to training set
train_transform_aug = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),  # Random crop with 4px padding
        transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip with 50% probability
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Slight color variation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Load CIFAR-10 with augmentation for training
trainset_aug = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=train_transform_aug
)
trainloader_aug = torch.utils.data.DataLoader(trainset_aug, batch_size=32, shuffle=True)

# Load CIFAR-10 test set without augmentation
testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=test_transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


# Custom CNN Architecture - same as baseline
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv1: 3 input channels, 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions

        # Conv2: 16 input channels, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 8, 8]
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output logits
        return x


# Training function for reusability
def train_model(model, trainloader, testloader, optimizer, epochs=10, description=""):
    criterion = nn.CrossEntropyLoss()
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(trainloader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        train_accuracies.append(train_acc)

        # Evaluate on test set
        model.eval()
        correct_test, total_test = 0, 0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        test_acc = 100 * correct_test / total_test
        test_accuracies.append(test_acc)

        print(
            f"{description} Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%"
        )

    return train_losses, train_accuracies, test_accuracies


# Step 2: Re-train CNN with Augmentations using Adam
print("\n=== Experiment A: Adam with Data Augmentation ===")
model_aug_adam = CustomCNN().to(device)
optimizer_aug_adam = optim.Adam(model_aug_adam.parameters(), lr=0.001)

adam_train_loss, adam_train_acc, adam_test_acc = train_model(
    model_aug_adam,
    trainloader_aug,
    testloader,
    optimizer_aug_adam,
    epochs=10,
    description="Adam",
)

# Step 3: Optimizer & Learning Rate Experiments
print("\n=== Experiment B: SGD with Momentum and Data Augmentation ===")
model_aug_sgd = CustomCNN().to(device)
optimizer_sgd = optim.SGD(
    model_aug_sgd.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4
)

sgd_train_loss, sgd_train_acc, sgd_test_acc = train_model(
    model_aug_sgd,
    trainloader_aug,
    testloader,
    optimizer_sgd,
    epochs=10,
    description="SGD",
)

# Compare final metrics
final_adam_test = adam_test_acc[-1]
final_sgd_test = sgd_test_acc[-1]

print("\n=== Final Comparison ===")
print(f"Adam final test accuracy: {final_adam_test:.2f}%")
print(f"SGD final test accuracy: {final_sgd_test:.2f}%")

# Step 4: Save Best Model Checkpoint
if final_adam_test >= final_sgd_test:
    best_model = model_aug_adam
    best_optimizer_name = "Adam"
    best_test_acc = final_adam_test
else:
    best_model = model_aug_sgd
    best_optimizer_name = "SGD"
    best_test_acc = final_sgd_test

torch.save(best_model.state_dict(), "trained_models/cnn_cifar10_best.pth")
print("\nBest augmented CNN model saved to trained_models/cnn_cifar10_best.pth")
print(f"Best optimizer: {best_optimizer_name} with test accuracy: {best_test_acc:.2f}%")

# Save plots
os.makedirs("graphs", exist_ok=True)

plt.figure(figsize=(12, 8))

# Plot 1: Training vs Test Accuracy comparison
plt.subplot(2, 2, 1)
plt.plot(adam_train_acc, label="Adam Train Acc", marker="o")
plt.plot(adam_test_acc, label="Adam Test Acc", marker="s")
plt.plot(sgd_train_acc, label="SGD Train Acc", marker="o", linestyle="--")
plt.plot(sgd_test_acc, label="SGD Test Acc", marker="s", linestyle="--")
plt.title("Training vs Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")
plt.legend()
plt.grid(True)

# Plot 2: Loss comparison
plt.subplot(2, 2, 2)
plt.plot(adam_train_loss, label="Adam Loss", marker="o")
plt.plot(sgd_train_loss, label="SGD Loss", marker="s")
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Plot 3: Generalization gap
plt.subplot(2, 2, 3)
adam_gap = [a - b for a, b in zip(adam_train_acc, adam_test_acc)]
sgd_gap = [a - b for a, b in zip(sgd_train_acc, sgd_test_acc)]
plt.plot(adam_gap, label="Adam Gap", marker="o")
plt.plot(sgd_gap, label="SGD Gap", marker="s")
plt.title("Generalization Gap (Train - Test)")
plt.xlabel("Epoch")
plt.ylabel("Gap %")
plt.legend()
plt.grid(True)

# Plot 4: Final test accuracy comparison
plt.subplot(2, 2, 4)
optimizers = ["Adam", "SGD"]
accuracies = [final_adam_test, final_sgd_test]
plt.bar(optimizers, accuracies, color=["blue", "orange"])
plt.title("Final Test Accuracy Comparison")
plt.ylabel("Accuracy %")
plt.ylim(0, 100)

plt.tight_layout()
plt.savefig("graphs/augmentation_optimizer_comparison.png")
plt.close()

print("Plots saved to graphs/augmentation_optimizer_comparison.png")
