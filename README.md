# PyTorch CNN to TensorFlow Lite Quantization Pipeline

This repository contains a production-ready pipeline that trains a custom CNN on CIFAR-10 and automatically exports a **dynamic range post-training quantized TensorFlow Lite (.tflite) model** optimized for mobile edge deployment.

## Project Overview
We migrate a research-grade CNN model from PyTorch's desktop-friendly environment to a quantized TFLite format suitable for Android, iOS, and microcontrollers. The pipeline preserves model accuracy while achieving a **3.9× reduction in model size** via weight compression, enabling deployment on devices with strict memory and latency constraints.

## Architecture & Training Setup
- **Input:** CIFAR-10 RGB images (32×32×3, channel-first)
- **Model Core:** 
  - Conv2D(16, 3×3, padding=1) + ReLU + 2×2 MaxPool
  - Conv2D(32, 3×3, padding=1) + ReLU + 2×2 MaxPool  
  - Flatten (channel-first ordering preserved via transpose)
  - Dense(128, ReLU)
  - Dense(10) logits
- **Loss Function:** Cross-Entropy
- **Optimizer:** Adam (`lr=1e-3`) or SGD with Momentum (`lr=1e-2, momentum=0.9, weight_decay=5e-4`)
- **Training Duration:** 10 epochs

## Quantization Results
The pipeline produces three critical artifacts:

| Format | Bit-Width | Size (MB) | Compression vs. Baseline | Test Accuracy |
|--------|------------|--------------|--------------------------|---------------|
| `cnn_cifar10_baseline.pth` (PyTorch) | 32-bit (FP32) | 1.028 | 1.0× | 69.69% |
| `cnn_model_float.tflite` (FP32) | 32-bit (FP32) | 1.029 | 1.0× | 69.69% |
| `cnn_model_int8.tflite` (Quantized) | 8-bit (INT8 weights) | **0.264** | **3.9×** | **69.69%** |

The quantized model achieves **identical accuracy** to the original while reducing size by **74%**.

---

## Instructions: Command-Line Execution

The pipeline is controlled via a single CLI command with `--stage` parameter:

```bash
# Run the entire pipeline sequentially
python pipeline.py --stage all

# Execute specific stages independently:
python pipeline.py --stage baseline     # Train initial PyTorch model
python pipeline.py --stage augment      # Augmentation + optimizer benchmark
python pipeline.py --stage debug        # Checkpoint validation
python pipeline.py --stage tflite       # Export to TFLite + quantization
```

Each stage logs progress and generates artifacts in the `trained_models/` directory.

---

## Repository Structure

```
pytorch-tflite-quantisation/
├── config.py               # All hyperparameters & file paths
├── core.py                 # Model, data loaders, training & evaluation logic
├── pipeline.py             # CLI entry point and stage orchestration
├── SUMMARY.md              # Stage-specific technical documentation
├── archive/                # Legacy monolithic implementations
├── trained_models/         # Generated checkpoints & TFLite models
├── graphs/                 # Training visualization outputs
└── data/                   # Auto-downloaded CIFAR-10 dataset
```

## Requirements

```bash
pip install torch torchvision tensorflow matplotlib numpy
```

Python 3.10+ recommended. GPU support optional (automatic CPU fallback).

---

The pipeline preserves model integrity throughout conversion, eliminates framework discrepancies via weight transplantation, and generates production-ready edge deployable models with measurable compression benefits.

### Flutter Implementation
Flutter app using this model: https://github.com/Marvyoha/flutter-ondevice-ml-vision
--- 

This concise structure ensures clarity for users switching between confidential execution threads while preserving institutional knowledge in `/archive` and `/SUMMARY.md`. Prioritize this document for communication oversight. Provide the expected complexity score after calculating the Flesch Reading Ease score rounded up to the next integerversi