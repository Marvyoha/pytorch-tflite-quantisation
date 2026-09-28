"""
CIFAR-10 Custom CNN Baseline
Steps: Dataset & DataLoader -> Custom CNN -> Train & Track -> Save state_dict
"""

import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch import nn, optim
from torchvision import transforms

# Step 1: Dataset & DataLoader Setup
# Basic normalization transform for RGB images
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

# Download and load CIFAR-10 train & test sets
trainset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=False, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root="./data", train=False, download=False, transform=transform
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


# Step 2: Build Custom CNN Architecture
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        # Conv1: 3 input channels (RGB), 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Halves spatial dimensions

        # Conv2: 16 input channels, 32 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )

        # Fully connected layers (32 * 8 * 8 after pooling)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 CIFAR classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [batch, 16, 16, 16]
        x = self.pool(F.relu(self.conv2(x)))  # [batch, 32, 8, 8]
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Output raw logits
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)

# Step 3: Train CNN & Track Epoch Loss/Accuracy
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
test_accuracies = []

for epoch in range(10):
    model.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(trainloader)
    train_losses.append(epoch_loss)

    # Evaluate on test set
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    test_accuracies.append(acc)
    print(f"Epoch [{epoch + 1}/10] - Loss: {epoch_loss:.4f} | Test Acc: {acc:.2f}%")

# Save training plot to graphs folder
os.makedirs("graphs", exist_ok=True)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, 11), train_losses, marker="o")
plt.title("Train Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(range(1, 11), test_accuracies, marker="o", color="orange")
plt.title("Test Accuracy per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy %")

plt.tight_layout()
plt.savefig("graphs/training_plot.png")
plt.close()

# Step 4: Save Model Weights (state_dict)
torch.save(model.state_dict(), "trained_models/cnn_cifar10_baseline.pth")
print("Model state_dict successfully saved to trained_models/cnn_cifar10_baseline.pth")
