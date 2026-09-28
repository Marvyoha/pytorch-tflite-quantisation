"""
Day 12: Debug & Catch-Up + TFLite Preview
Hr 1-2: Verify checkpoint, shape sanity, consistent validation accuracy
Hr 4-6: Preview TFLite conversion steps
"""

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision import transforms

# Reuse same normalization as Days 10-11
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

# Architecture must match training code exactly to avoid shape mismatches
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Shape check: [B,3,32,32] -> conv1 -> [B,16,32,32] -> pool -> [B,16,16,16]
        x = self.pool(F.relu(self.conv1(x)))
        # -> conv2 -> [B,32,16,16] -> pool -> [B,32,8,8]
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)

# Load best checkpoint from Day 11
ckpt_path = "trained_models/cnn_cifar10_best.pth"
try:
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Loaded checkpoint {ckpt_path}")
except FileNotFoundError:
    # fallback to baseline
    ckpt_path = "trained_models/cnn_cifar10_baseline.pth"
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    print(f"Fallback loaded {ckpt_path}")

model.eval()

# Debug: run single batch shape sanity check
with torch.no_grad():
    sample, _ = next(iter(testloader))
    sample = sample.to(device)
    out = model(sample)
    assert out.shape == (sample.shape[0], 10), f"Output shape mismatch {out.shape}"
    print(f"Shape sanity passed: input {sample.shape} -> output {out.shape}")

# Consistent validation accuracy
def evaluate(loader):
    correct = total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = model(inputs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total

acc1 = evaluate(testloader)
acc2 = evaluate(testloader)
print(f"Validation Accuracy run 1: {acc1:.2f}%")
print(f"Validation Accuracy run 2: {acc2:.2f}%")
print(f"Consistent: {abs(acc1-acc2) < 1e-6}")

# ponytail: TFLite preview only, no conversion yet - add when torch.onnx export verified
# Preview steps for tomorrow:
# 1. Export to ONNX: torch.onnx.export(model, dummy_input, "model.onnx")
# 2. Convert ONNX -> TFLite using ai-edge-torch or tf.lite.TFLiteConverter.from_saved_model
# 3. Quantize post-training for size/latency
print("\nTFLite preview complete. Ready for export.")
