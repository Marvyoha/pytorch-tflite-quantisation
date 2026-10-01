# PyTorch CNN to TensorFlow Lite Quantization Pipeline

This repository contains a production-ready pipeline that trains a 3-layer Compact CNN on CIFAR-10 with unified [0,1] preprocessing, early stopping, comprehensive evaluation, ONNX export and **full INT8 post-training quantization** for mobile edge deployment.

## Project Overview
Master pipeline `src/master_pipeline.py` migrates a research-grade CNN from PyTorch to ONNX and quantized TFLite with probabilistic guardrails and side-by-side benchmarking. The pipeline preserves accuracy while achieving strong compression and measurable latency.

## Master Architecture & Training Setup
- **Input:** CIFAR-10 RGB images 32×32×3, normalized to [0,1] float
- **Dataset Split:** Train 40,000 / Val 10,000 from train set, Test 10,000 held-out
- **Model Core:** 
  - Conv2d 3→32 + BatchNorm + ReLU + MaxPool
  - Conv2d 32→64 + BatchNorm + ReLU + MaxPool
  - Conv2d 64→128 + BatchNorm + ReLU + MaxPool
  - Dropout 0.4 → Linear 2048→256 → Dropout 0.4 → Linear 256→10
  - **Loss Function:** Cross-Entropy
  - **Optimizer:** Adam `lr=1e-3` weight_decay 1e-4 with CosineAnnealingWarmRestarts T0=30 T_mult=1 scheduler
  - **Training:** Up to 100 epochs with EarlyStopping patience=5, min_delta=0.0005, restore best weights, minimum epoch floor 30; conservative augmentation RandomResizedCrop 0.9-1.0, ColorJitter 0.2, RandomGrayscale 0.1

## Master Pipeline Artifacts
`src/master_pipeline.py` produces:

- `models/cifar10.pth` – best PyTorch checkpoint with early stopping
- `models/cnn_model.onnx` – ONNX export
- `models/cnn_model_int8.tflite` – full INT8 quantized TFLite
- `graphs/roc_curve.png`, `graphs/confusion_matrix.png` – evaluation visuals

Evaluation includes Accuracy, Precision, Recall, F1-Score, ROC-AUC and probabilistic guardrails with confidence threshold 0.65.

---

## Instructions: Command-Line Execution

Run the full master pipeline:

```bash
python src/master_pipeline.py
```

Artifacts are written to `models/` and `graphs/`. The legacy pipeline files have been archived under `archive/initial/`.

---

## Repository Structure

```
pytorch-tflite-quantisation/
├── src/
│   ├── master_config.py
│   ├── master_core.py
│   ├── master_model/
│   └── master_pipeline.py
├── models/                 # Checkpoints, ONNX, TFLite
├── graphs/                 # ROC, Confusion Matrix
├── SUMMARY.md
├── archive/
│   └── initial/            # Legacy pipeline files & trained_models
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
