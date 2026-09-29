## Project Overview
Master pipeline `src/master_pipeline.py` trains a 3-layer Compact CNN on CIFAR-10 with 80/20 train/validation split, early stopping, unified [0,1] preprocessing, comprehensive evaluation, ONNX export, full INT8 TFLite quantization with representative dataset, probabilistic guardrails, and side-by-side benchmarking.

## Architecture
```
Input: [B, 3, 32, 32] NCHW, [0,1] float
├── Conv2d(3→32,3×3,pad=1)+BN+ReLU+MaxPool2d → [B,32,16,16]
├── Conv2d(32→64,3×3,pad=1)+BN+ReLU+MaxPool2d → [B,64,8,8]
├── Conv2d(64→128,3×3,pad=1)+BN+ReLU+MaxPool2d → [B,128,4,4]
├── Flatten → [B,2048]
├── Dropout(0.4)
├── Linear(2048→256)+ReLU
├── Dropout(0.4)
└── Linear(256→10) logits
```

Preprocessing unified across PyTorch, ONNX, TFLite with channel ordering sanity check atol 1e-4.

## Data Split & Preprocessing
- CIFAR-10 train True → random_split 40,000 train / 10,000 val
- CIFAR-10 test False isolated 10,000 images
- Train augmentations: RandomCrop 32 pad4, RandomHorizontalFlip 0.5, ColorJitter brightness 0.2 contrast 0.2
- Val/Test: ToTensor only → [0,1] float, no normalization

## Training
- EarlyStopping patience=5, min_delta=0.001, restore_best_weights=True
- Optimizer Adam lr=1e-3 weight_decay=1e-4
- Epochs up to 50
- Checkpoint saved to `models/cifar10.pth`

## Evaluation Suite
- Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC via sklearn
- Plots: `graphs/roc_curve.png`, `graphs/confusion_matrix.png`

## Export & Quantization
- ONNX: `torch.onnx.export` → `models/cnn_model.onnx`
- TFLite full INT8 PTQ:
  - Representative dataset generator 150 samples from train subset
  - optimizations=[Optimize.DEFAULT]
  - supported_ops=[TFLITE_BUILTINS_INT8]
  - inference_input_type=int8, inference_output_type=int8
  - Output: `models/cnn_model_int8.tflite`

## Probabilistic Guardrails
- Softmax probabilities, CONFIDENCE_THRESHOLD=0.65
- max(prob) >= 0.65 → return label + confidence
- else → "Uncertain / High Entropy"

## Benchmarking
Side-by-side comparison PyTorch .pth, ONNX .onnx, TFLite INT8:
- Classification metrics
- Mean inference latency ms per image over 100 warm runs
- File size MB
- Logit divergence MAE vs PyTorch

Markdown table printed at runtime.

## File Structure
```
src/
  master_config.py
  master_core.py
  master_model/
  master_pipeline.py
models/
graphs/
archive/initial/  # legacy pipeline
```

## Requirements
pip install torch torchvision tensorflow onnxruntime scikit-learn matplotlib numpy

## Execution
python src/master_pipeline.py
