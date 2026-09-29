import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "/content/data" if os.getenv("COLAB_GPU") else "./data"
BATCH_SIZE = 128
NUM_WORKERS = 0

AUG_PADDING = 4
AUG_FLIP_PROB = 0.5
AUG_BRIGHTNESS = 0.2
AUG_CONTRAST = 0.2

EPOCHS = int(os.getenv("COLAB_EPOCHS", 50))

CONV1_OUT = 32
CONV2_OUT = 64
CONV3_OUT = 128
FC_HIDDEN = 256
NUM_CLASSES = 10
DROPOUT = 0.4

CONFIDENCE_THRESHOLD = 0.65

MODEL_DIR = "models"
GRAPHS_DIR = "graphs"

CKPT_PATH = f"{MODEL_DIR}/cifar10.pth"
ONNX_PATH = f"{MODEL_DIR}/cnn_model.onnx"
TFLITE_INT8_PATH = f"{MODEL_DIR}/cnn_model_int8.tflite"

ROC_PATH = f"{GRAPHS_DIR}/roc_curve.png"
CM_PATH = f"{GRAPHS_DIR}/confusion_matrix.png"
