"""
Day 13: PyTorch CNN -> TensorFlow Lite with Dynamic Range Quantization
Hr 1: Environment setup
Hr 1-3: Build equivalent Keras model from PyTorch weights
Hr 3-4.5: Convert to TFLite (float + quantized)
Hr 4.5-5.5: Benchmark file size & accuracy
"""

import os
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import transforms, datasets
import numpy as np

# Ensure output directory exists
os.makedirs("trained_models", exist_ok=True)

# --------------------------
# 1. Model definition - must match training
# --------------------------
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [B,16,16,16]
        x = self.pool(F.relu(self.conv2(x)))  # [B,32,8,8]
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomCNN().to(device)

# Load best checkpoint
ckpt_path = "trained_models/cnn_cifar10_best.pth"
if not os.path.exists(ckpt_path):
    ckpt_path = "trained_models/cnn_cifar10_baseline.pth"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Loaded checkpoint {ckpt_path}")

# --------------------------
# 2. Build equivalent Keras model with weight transfer
# --------------------------
import tensorflow as tf
from tensorflow.keras import layers, models

# Extract PyTorch weights
weights = {}
for name, param in model.named_parameters():
    weights[name] = param.detach().numpy()

# Build Keras model with NHWC input, but we'll transpose before dense layers
keras_model = models.Sequential([
    layers.Input(shape=(32, 32, 3), name="input"),  # NHWC
    layers.Conv2D(16, 3, padding='same', activation='relu', name="conv1"),
    layers.MaxPooling2D(2, name="pool1"),
    layers.Conv2D(32, 3, padding='same', activation='relu', name="conv2"),
    layers.MaxPooling2D(2, name="pool2"),
    # Transpose NHWC -> NCHW before flatten to match PyTorch order
    layers.Lambda(lambda x: tf.transpose(x, [0, 3, 1, 2]), name="nchw_transpose"),
    layers.Flatten(name="flatten"),
    layers.Dense(128, activation='relu', name="fc1"),
    layers.Dense(10, name="fc2")
])

# Set weights (convert NCHW -> NHWC for conv, transpose for dense)
keras_model.get_layer("conv1").set_weights([
    np.transpose(weights['conv1.weight'], (2, 3, 1, 0)),  # (3,3,3,16)
    weights['conv1.bias']
])
keras_model.get_layer("conv2").set_weights([
    np.transpose(weights['conv2.weight'], (2, 3, 1, 0)),  # (3,3,16,32)
    weights['conv2.bias']
])
keras_model.get_layer("fc1").set_weights([
    weights['fc1.weight'].T,  # (2048, 128)
    weights['fc1.bias']
])
keras_model.get_layer("fc2").set_weights([
    weights['fc2.weight'].T,  # (128, 10)
    weights['fc2.bias']
])

# Verify numerical equivalence
print("Verifying numerical equivalence...")
x_test = torch.randn(1, 3, 32, 32)
with torch.no_grad():
    pt_out = model(x_test).numpy()

inp_np = np.transpose(x_test.numpy(), (0, 2, 3, 1))  # NCHW -> NHWC
keras_out = keras_model.predict(inp_np, verbose=0)
print(f"PyTorch output: {pt_out[0, :5]}")
print(f"Keras output:   {keras_out[0, :5]}")
print(f"Match: {np.allclose(pt_out, keras_out, atol=1e-4)}")

# Export SavedModel
keras_model.export("trained_models/saved_model")
print("Keras SavedModel exported")

# --------------------------
# 3. Convert to TFLite (float + dynamic range quantized)
# --------------------------
tflite_float_path = "trained_models/cnn_model_float.tflite"
tflite_int8_path = "trained_models/cnn_model_int8.tflite"

# Float TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("trained_models/saved_model")
tflite_float = converter.convert()
with open(tflite_float_path, "wb") as f:
    f.write(tflite_float)
print(f"Float TFLite saved: {len(tflite_float)/1024:.1f} KB")

# Dynamic range quantized TFLite
converter_q = tf.lite.TFLiteConverter.from_saved_model("trained_models/saved_model")
converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_int8 = converter_q.convert()
with open(tflite_int8_path, "wb") as f:
    f.write(tflite_int8)
print(f"INT8 TFLite saved: {len(tflite_int8)/1024:.1f} KB")

# --------------------------
# 4. Benchmark file size & accuracy
# --------------------------
def get_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0.0

pth_size = get_size_mb(ckpt_path)
float_size = get_size_mb(tflite_float_path)
int8_size = get_size_mb(tflite_int8_path)

print("\n=== File Size Benchmark ===")
print(f"PyTorch .pth:      {pth_size:.3f} MB")
print(f"TFLite Float:      {float_size:.3f} MB")
print(f"TFLite INT8:       {int8_size:.3f} MB")
print(f"Compression ratio: {pth_size/int8_size:.1f}x")

# Accuracy verification
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
testset = datasets.CIFAR10(root="./data", train=False, download=False, transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# PyTorch baseline
correct_pt = total = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        if i >= 5: break
        inputs, labels = inputs.to(device), labels.to(device)
        preds = model(inputs).argmax(dim=1)
        correct_pt += (preds == labels).sum().item()
        total += labels.size(0)
acc_pt = 100 * correct_pt / total

# TFLite INT8 inference
interpreter = tf.lite.Interpreter(model_path=tflite_int8_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

correct_tflite = total_tflite = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(testloader):
        if i >= 5: break
        inp_np = inputs.numpy().astype(np.float32)
        inp_np = np.transpose(inp_np, (0, 2, 3, 1))  # NCHW -> NHWC
        labels_np = labels.numpy()
        for j in range(inp_np.shape[0]):
            interpreter.set_tensor(input_details[0]['index'], inp_np[j:j+1])
            interpreter.invoke()
            out = interpreter.get_tensor(output_details[0]['index'])
            pred = np.argmax(out[0])
            if pred == labels_np[j]:
                correct_tflite += 1
            total_tflite += 1
acc_tflite = 100 * correct_tflite / total_tflite if total_tflite else 0

print(f"\nPyTorch accuracy:  {acc_pt:.2f}%")
print(f"TFLite INT8 accuracy: {acc_tflite:.2f}%")
print(f"Accuracy delta:    {abs(acc_pt-acc_tflite):.2f}%")

# Write markdown benchmark report
report_path = "trained_models/benchmark_report.md"
with open(report_path, "w") as f:
    f.write("# TFLite Conversion Benchmark\n\n")
    f.write("| Model | File Size MB |\n")
    f.write("|---|---|\n")
    f.write(f"| PyTorch .pth | {pth_size:.3f} |\n")
    f.write(f"| TFLite Float | {float_size:.3f} |\n")
    f.write(f"| TFLite INT8 | {int8_size:.3f} |\n\n")
    f.write(f"PyTorch accuracy: {acc_pt:.2f}%\n")
    f.write(f"TFLite INT8 accuracy: {acc_tflite:.2f}%\n")
    f.write(f"Compression ratio: {pth_size/int8_size:.1f}x\n")
print(f"Benchmark report saved to {report_path}")

print("\nDay 13 complete.")