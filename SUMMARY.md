## Project Overview
End-to-end pipeline training a custom CNN on CIFAR-10, augmenting/optimizing, validating checkpoints, and exporting to quantized TensorFlow Lite for edge deployment.

## Architecture (Single Definition)

```
Input: [B, 3, 32, 32] (NCHW)
 ├── Conv2d(3→16, 3×3, pad=1) + ReLU + MaxPool2d(2×2) → [B, 16, 16, 16]
 ├── Conv2d(16→32, 3×3, pad=1) + ReLU + MaxPool2d(2×2) → [B, 32, 8, 8]
 ├── Flatten (channel-first) → [B, 2048]
 ├── Linear(2048→128) + ReLU → [B, 128]
 └── Linear(128→10) → [B, 10] logits
```

**Keras equivalent** (NHWC with NCHW transpose before Flatten):
```
Input: [B, 32, 32, 3] → Conv2D(16) → Pool → Conv2D(32) → Pool
 → Lambda(transpose NHWC→NCHW) → Flatten → Dense(128) → Dense(10)
```

---

## Stage 1: Baseline Training (`--stage baseline`)

**Config:** Adam lr=1e-3, 10 epochs, batch=32, no augmentation  
**Outputs:** `trained_models/cnn_cifar10_baseline.pth`, `graphs/training_plot.png`

- Normalization: mean/std = (0.5, 0.5, 0.5) per channel
- Tracks per-epoch train loss + test accuracy
- Saves `state_dict` only (lightweight, portable)

---

## Stage 2: Augmentation & Optimizer Benchmark (`--stage augment`)

**Augmentations (training only):**
- RandomCrop(32, padding=4)
- RandomHorizontalFlip(p=0.5)
- ColorJitter(brightness=0.2, contrast=0.2)

**Experiments:**
| Optimizer | LR | Momentum | Weight Decay |
|-----------|----|----------|--------------|
| Adam      | 1e-3 | — | — |
| SGD       | 1e-2 | 0.9 | 5e-4 |

**Outputs:** `trained_models/cnn_cifar10_best.pth`, `graphs/augmentation_optimizer_comparison.png`  
Best model selected by final test accuracy.

---

## Stage 3: Debug & Checkpoint Validation (`--stage debug`)

**Checks:**
1. Shape sanity: single forward pass, verify `[B, 10]` output
2. Deterministic inference: two full test-set passes, compare accuracy (tolerance 1e-6)

Loads `best.pth` if exists, falls back to `baseline.pth`.

---

## Stage 4: TFLite Export & Quantization (`--stage tflite`)

**Weight Transfer:**
- Conv weights: NCHW `(out, in, h, w)` → NHWC `(h, w, in, out)` via `np.transpose(..., (2,3,1,0))`
- Dense weights: `.T` (PyTorch `[out, in]` → Keras `[in, out]`)
- Bias: unchanged
- **Critical:** `Lambda(tf.transpose(x, [0,3,1,2]))` before Flatten matches PyTorch channel-first flatten order

**Conversion:**
1. Keras `model.export()` → SavedModel
2. `TFLiteConverter.from_saved_model()` → float `.tflite`
3. Same converter + `optimizations=[Optimize.DEFAULT]` → dynamic-range INT8 `.tflite`

**Benchmark Results (actual run):**

| Model | Size | Accuracy |
|-------|------|----------|
| PyTorch `.pth` | 1.028 MB | 69.69% |
| TFLite Float | 1.029 MB | — |
| TFLite INT8 | **0.264 MB** | **69.69%** |

**Compression: 3.9× | Accuracy delta: 0.00%**

Report saved to `trained_models/benchmark_report.md`.

---

## Configuration (`config.py`)

All hyperparameters, paths, and constants in one file. Modify here to change:
- Device, batch size, epochs, learning rates
- Augmentation strengths
- Model widths (conv channels, FC hidden)
- Output directories

---

## CLI Usage

```bash
# Full pipeline (sequential)
python pipeline.py --stage all

# Individual stages
python pipeline.py --stage baseline
python pipeline.py --stage augment
python pipeline.py --stage debug
python pipeline.py --stage tflite
```

---

## File Structure

```
pytorch-tflite-quantisation/
├── config.py              # Hyperparameters & paths
├── core.py                # Shared: model, data, train, eval, plots
├── pipeline.py            # CLI entry point
├── SUMMARY.md             # This file
├── archive/               # Old day scripts
│   ├── 1_cifar10_custom_cnn_baseline.py
│   ├── 2_augmentation_tuning.py
│   ├── 3_cnn_debug_preview.py
│   └── 4_pytorch_to_tflite.py
├── trained_models/        # Checkpoints, TFLite models, reports
├── graphs/                # Training plots
└── data/                  # CIFAR-10 (auto-download)
```

---

## Requirements

```bash
pip install torch torchvision tensorflow matplotlib numpy
```

Python 3.10+ recommended. GPU optional (CPU fallback automatic).

---

## Archive Contents

Original day-by-day scripts preserved in `archive/` for reference:
- `1_cifar10_custom_cnn_baseline.py` — baseline training
- `2_augmentation_tuning.py` — augmentation + Adam vs SGD
- `3_cnn_debug_preview.py` — checkpoint validation
- `4_pytorch_to_tflite.py` — TFLite export (working version)