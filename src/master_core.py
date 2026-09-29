import copy
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torchvision import datasets, transforms

from src.master_config import (
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
    AUG_FLIP_PROB,
    AUG_PADDING,
    BATCH_SIZE,
    CKPT_PATH,
    CM_PATH,
    CONFIDENCE_THRESHOLD,
    CONV1_OUT,
    CONV2_OUT,
    CONV3_OUT,
    DATA_ROOT,
    DEVICE,
    DROPOUT,
    EPOCHS,
    FC_HIDDEN,
    GRAPHS_DIR,
    MODEL_DIR,
    NUM_CLASSES,
    NUM_WORKERS,
    ONNX_PATH,
    ROC_PATH,
    TFLITE_INT8_PATH,
)

CLASSES = (
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


class CompactCNN(nn.Module):
    def __init__(
        self,
        conv1_out=CONV1_OUT,
        conv2_out=CONV2_OUT,
        conv3_out=CONV3_OUT,
        fc_hidden=FC_HIDDEN,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, conv1_out, 3, padding=1),
            nn.BatchNorm2d(conv1_out),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(conv1_out, conv2_out, 3, padding=1),
            nn.BatchNorm2d(conv2_out),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(conv2_out, conv3_out, 3, padding=1),
            nn.BatchNorm2d(conv3_out),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv3_out * 4 * 4, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.best_weights = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class StandardPreprocessing:
    @staticmethod
    def torch_transform(train=False):
        if train:
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=AUG_PADDING),
                    transforms.RandomHorizontalFlip(AUG_FLIP_PROB),
                    transforms.ColorJitter(
                        brightness=AUG_BRIGHTNESS, contrast=AUG_CONTRAST
                    ),
                    transforms.ToTensor(),
                ]
            )
        return transforms.Compose([transforms.ToTensor()])

    @staticmethod
    def preprocess_image_array(img_np):
        if img_np.dtype != np.float32:
            img_np = img_np.astype(np.float32) / 255.0
        return img_np


def verify_preprocessing_alignment():
    sample = torch.rand(1, 3, 32, 32)
    pyt_arr = sample.numpy()
    tf_arr = np.transpose(pyt_arr, (0, 2, 3, 1))
    reconstructed_pyt = np.transpose(tf_arr, (0, 3, 1, 2))
    assert np.allclose(pyt_arr, reconstructed_pyt, atol=1e-4), (
        "Channel alignment mismatch"
    )


def get_cifar10_datasets():
    full_train = datasets.CIFAR10(
        DATA_ROOT,
        train=True,
        download=False,
        transform=StandardPreprocessing.torch_transform(train=True),
    )
    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset = torch.utils.data.random_split(
        full_train, [40000, 10000], generator=generator
    )

    val_dataset = datasets.CIFAR10(
        DATA_ROOT,
        train=True,
        download=False,
        transform=StandardPreprocessing.torch_transform(train=False),
    )
    val_subset.dataset = val_dataset

    test_dataset = datasets.CIFAR10(
        DATA_ROOT,
        train=False,
        download=False,
        transform=StandardPreprocessing.torch_transform(train=False),
    )

    return train_subset, val_subset, test_dataset


def get_dataloaders(train_subset, val_subset, test_dataset, batch_size=BATCH_SIZE):
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, epochs=EPOCHS):
    device = torch.device(DEVICE)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=5, min_delta=0.001, restore_best_weights=True
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = 100.0 * val_correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )

        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    early_stopping.restore(model)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    print(f"Saved best model checkpoint to {CKPT_PATH}")
    return model


@torch.no_grad()
def evaluate_pytorch(model, test_loader):
    device = torch.device(DEVICE)
    model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro")
    rec = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    roc = roc_auc_score(all_labels, all_probs, multi_class="ovr")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": roc,
        "labels": all_labels,
        "preds": all_preds,
        "probs": all_probs,
    }


def generate_evaluation_plots(metrics):
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    labels = metrics["labels"]
    probs = metrics["probs"]
    preds = metrics["preds"]

    plt.figure(figsize=(8, 6))
    for i in range(NUM_CLASSES):
        y_true_binary = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, probs[:, i])
        plt.plot(fpr, tpr, label=f"Class {CLASSES[i]}")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("CIFAR-10 One-vs-Rest ROC Curves")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(ROC_PATH)
    plt.close()

    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("CIFAR-10 Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=45)
    plt.yticks(tick_marks, CLASSES)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(CM_PATH)
    plt.close()


def export_onnx(model):
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32)
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.onnx.export(
        model.cpu(),
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"Exported ONNX model to {ONNX_PATH}")


def convert_full_int8_tflite(model, train_subset):
    import tensorflow as tf

    pyt_weights = {n: p.detach().cpu().numpy() for n, p in model.named_parameters()}
    pyt_buffers = {n: b.detach().cpu().numpy() for n, b in model.named_buffers()}

    keras_input = tf.keras.Input(shape=(32, 32, 3), name="input")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", name="conv1")(keras_input)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool1")(x)

    x = tf.keras.layers.Conv2D(64, 3, padding="same", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool2")(x)

    x = tf.keras.layers.Conv2D(128, 3, padding="same", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling2D(2, name="pool3")(x)

    x = tf.keras.layers.Lambda(
        lambda t: tf.transpose(t, [0, 3, 1, 2]), name="nchw_transpose"
    )(x)
    x = tf.keras.layers.Flatten(name="flatten")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fc1")(x)
    keras_output = tf.keras.layers.Dense(10, name="fc2")(x)

    keras_model = tf.keras.Model(inputs=keras_input, outputs=keras_output)

    keras_model.get_layer("conv1").set_weights(
        [
            np.transpose(pyt_weights["block1.0.weight"], (2, 3, 1, 0)),
            pyt_weights["block1.0.bias"],
        ]
    )
    keras_model.get_layer("bn1").set_weights(
        [
            pyt_weights["block1.1.weight"],
            pyt_weights["block1.1.bias"],
            pyt_buffers["block1.1.running_mean"],
            pyt_buffers["block1.1.running_var"],
        ]
    )

    keras_model.get_layer("conv2").set_weights(
        [
            np.transpose(pyt_weights["block2.0.weight"], (2, 3, 1, 0)),
            pyt_weights["block2.0.bias"],
        ]
    )
    keras_model.get_layer("bn2").set_weights(
        [
            pyt_weights["block2.1.weight"],
            pyt_weights["block2.1.bias"],
            pyt_buffers["block2.1.running_mean"],
            pyt_buffers["block2.1.running_var"],
        ]
    )

    keras_model.get_layer("conv3").set_weights(
        [
            np.transpose(pyt_weights["block3.0.weight"], (2, 3, 1, 0)),
            pyt_weights["block3.0.bias"],
        ]
    )
    keras_model.get_layer("bn3").set_weights(
        [
            pyt_weights["block3.1.weight"],
            pyt_weights["block3.1.bias"],
            pyt_buffers["block3.1.running_mean"],
            pyt_buffers["block3.1.running_var"],
        ]
    )

    keras_model.get_layer("fc1").set_weights(
        [pyt_weights["fc1.weight"].T, pyt_weights["fc1.bias"]]
    )
    keras_model.get_layer("fc2").set_weights(
        [pyt_weights["fc2.weight"].T, pyt_weights["fc2.bias"]]
    )

    def representative_dataset_gen():
        for i in range(150):
            img_tensor, _ = train_subset[i]
            img_np = np.transpose(img_tensor.numpy(), (1, 2, 0))
            img_np = np.expand_dims(img_np, axis=0).astype(np.float32)
            yield [img_np]

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()
    with open(TFLITE_INT8_PATH, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Saved full INT8 TFLite model to {TFLITE_INT8_PATH}")


def predict_with_guardrails(logits, threshold=CONFIDENCE_THRESHOLD):
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    results = []
    for p in probs:
        max_idx = int(np.argmax(p))
        max_prob = float(p[max_idx])
        if max_prob >= threshold:
            results.append(
                {
                    "status": "Accepted",
                    "class": CLASSES[max_idx],
                    "confidence": max_prob,
                }
            )
        else:
            results.append(
                {
                    "status": "Uncertain / High Entropy",
                    "class": None,
                    "confidence": max_prob,
                }
            )
    return results


def run_side_by_side_benchmark(test_loader):
    import onnxruntime as ort
    import tensorflow as tf

    pyt_model = CompactCNN()
    pyt_model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    pyt_model.eval()

    ort_session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])

    tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_INT8_PATH)
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    in_scale, in_zero_point = input_details[0]["quantization"]
    out_scale, out_zero_point = output_details[0]["quantization"]

    images, labels = [], []
    for imgs, lbls in test_loader:
        images.append(imgs)
        labels.append(lbls)
    all_imgs = torch.cat(images, dim=0)
    all_labels = torch.cat(labels, dim=0).numpy()

    # PyTorch inference
    with torch.no_grad():
        pyt_logits = pyt_model(all_imgs).numpy()
    pyt_probs = np.exp(pyt_logits - np.max(pyt_logits, axis=1, keepdims=True)) / np.sum(
        np.exp(pyt_logits - np.max(pyt_logits, axis=1, keepdims=True)),
        axis=1,
        keepdims=True,
    )
    pyt_preds = np.argmax(pyt_logits, axis=1)

    # ONNX inference
    onnx_inputs = {
        ort_session.get_inputs()[0].name: all_imgs.numpy().astype(np.float32)
    }
    onnx_logits = ort_session.run(None, onnx_inputs)[0]
    onnx_probs = np.exp(
        onnx_logits - np.max(onnx_logits, axis=1, keepdims=True)
    ) / np.sum(
        np.exp(onnx_logits - np.max(onnx_logits, axis=1, keepdims=True)),
        axis=1,
        keepdims=True,
    )
    onnx_preds = np.argmax(onnx_logits, axis=1)

    # TFLite inference
    tflite_logits = []
    tflite_preds = []
    for i in range(len(all_imgs)):
        img_hwc = np.transpose(all_imgs[i].numpy(), (1, 2, 0))
        img_hwc = np.expand_dims(img_hwc, axis=0)
        img_int8 = np.round(img_hwc / in_scale + in_zero_point).astype(np.int8)
        tflite_interpreter.set_tensor(input_details[0]["index"], img_int8)
        tflite_interpreter.invoke()
        q_out = tflite_interpreter.get_tensor(output_details[0]["index"])
        dequant_out = (q_out.astype(np.float32) - out_zero_point) * out_scale
        tflite_logits.append(dequant_out[0])
        tflite_preds.append(np.argmax(dequant_out[0]))

    tflite_logits = np.array(tflite_logits)
    tflite_preds = np.array(tflite_preds)
    tflite_probs = np.exp(
        tflite_logits - np.max(tflite_logits, axis=1, keepdims=True)
    ) / np.sum(
        np.exp(tflite_logits - np.max(tflite_logits, axis=1, keepdims=True)),
        axis=1,
        keepdims=True,
    )

    # Latency over 100 warm single-image runs
    sample_img = all_imgs[0:1]
    sample_np = sample_img.numpy().astype(np.float32)
    sample_hwc = np.transpose(sample_np, (0, 2, 3, 1))
    sample_int8 = np.round(sample_hwc / in_scale + in_zero_point).astype(np.int8)

    for _ in range(10):
        with torch.no_grad():
            _ = pyt_model(sample_img)
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: sample_np})
        tflite_interpreter.set_tensor(input_details[0]["index"], sample_int8)
        tflite_interpreter.invoke()

    t0 = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = pyt_model(sample_img)
    pyt_lat = (time.perf_counter() - t0) / 100.0 * 1000.0

    t0 = time.perf_counter()
    for _ in range(100):
        _ = ort_session.run(None, {ort_session.get_inputs()[0].name: sample_np})
    onnx_lat = (time.perf_counter() - t0) / 100.0 * 1000.0

    t0 = time.perf_counter()
    for _ in range(100):
        tflite_interpreter.set_tensor(input_details[0]["index"], sample_int8)
        tflite_interpreter.invoke()
        _ = tflite_interpreter.get_tensor(output_details[0]["index"])
    tflite_lat = (time.perf_counter() - t0) / 100.0 * 1000.0

    def calc_metrics(preds, probs):
        return {
            "acc": accuracy_score(all_labels, preds) * 100.0,
            "prec": precision_score(all_labels, preds, average="macro") * 100.0,
            "rec": recall_score(all_labels, preds, average="macro") * 100.0,
            "f1": f1_score(all_labels, preds, average="macro") * 100.0,
            "roc": roc_auc_score(all_labels, probs, multi_class="ovr"),
        }

    pyt_m = calc_metrics(pyt_preds, pyt_probs)
    onnx_m = calc_metrics(onnx_preds, onnx_probs)
    tflite_m = calc_metrics(tflite_preds, tflite_probs)

    mae_tflite = float(np.mean(np.abs(pyt_logits - tflite_logits)))
    mae_onnx = float(np.mean(np.abs(pyt_logits - onnx_logits)))

    def get_size_mb(path):
        return os.path.getsize(path) / (1024 * 1024)

    pyt_size = get_size_mb(CKPT_PATH)
    onnx_size = get_size_mb(ONNX_PATH)
    tflite_size = get_size_mb(TFLITE_INT8_PATH)

    table = f"""
| Format | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC | Latency (ms) | Size (MB) | MAE vs PyTorch Logits |
|---|---|---|---|---|---|---|---|---|
| **PyTorch (.pth)** | {pyt_m["acc"]:.2f} | {pyt_m["prec"]:.2f} | {pyt_m["rec"]:.2f} | {pyt_m["f1"]:.2f} | {pyt_m["roc"]:.4f} | {pyt_lat:.2f} | {pyt_size:.3f} | 0.0000 |
| **ONNX (.onnx)** | {onnx_m["acc"]:.2f} | {onnx_m["prec"]:.2f} | {onnx_m["rec"]:.2f} | {onnx_m["f1"]:.2f} | {onnx_m["roc"]:.4f} | {onnx_lat:.2f} | {onnx_size:.3f} | {mae_onnx:.4f} |
| **TFLite INT8 (.tflite)** | {tflite_m["acc"]:.2f} | {tflite_m["prec"]:.2f} | {tflite_m["rec"]:.2f} | {tflite_m["f1"]:.2f} | {tflite_m["roc"]:.4f} | {tflite_lat:.2f} | {tflite_size:.3f} | {mae_tflite:.4f} |
"""
    print(table)
    return table
