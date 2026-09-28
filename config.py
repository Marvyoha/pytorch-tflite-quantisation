# Hardware
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

# Data
DATA_ROOT = "./data"
BATCH_SIZE = 32
NUM_WORKERS = 0
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Augmentation
AUG_PADDING = 4
AUG_FLIP_PROB = 0.5
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2

# Training
EPOCHS = 10
ADAM_LR = 1e-3
SGD_LR = 1e-2
SGD_MOMENTUM = 0.9
SGD_WEIGHT_DECAY = 5e-4

# Model
CONV1_OUT = 16
CONV2_OUT = 32
FC_HIDDEN = 128
NUM_CLASSES = 10

# Paths
MODEL_DIR = "trained_models"
GRAPHS_DIR = "graphs"
BASELINE_CKPT = f"{MODEL_DIR}/cnn_cifar10_baseline.pth"
BEST_CKPT = f"{MODEL_DIR}/cnn_cifar10_best.pth"
ONNX_PATH = f"{MODEL_DIR}/cnn_model.onnx"
TFLITE_FLOAT = f"{MODEL_DIR}/cnn_model_float.tflite"
TFLITE_INT8 = f"{MODEL_DIR}/cnn_model_int8.tflite"
SAVED_MODEL_DIR = f"{MODEL_DIR}/saved_model"
BENCHMARK_REPORT = f"{MODEL_DIR}/benchmark_report.md"