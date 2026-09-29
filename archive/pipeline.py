#!/usr/bin/env python3
"""
PyTorch CNN → TFLite Pipeline
Stages: baseline → augment → debug → tflite
Usage: python pipeline.py --stage baseline|augment|debug|tflite|all
"""

import argparse
import os

import numpy as np
import torch

from archive.config import (
    ADAM_LR,
    BASELINE_CKPT,
    BENCHMARK_REPORT,
    BEST_CKPT,
    DEVICE,
    EPOCHS,
    GRAPHS_DIR,
    MODEL_DIR,
    NUM_CLASSES,
    SAVED_MODEL_DIR,
    SGD_LR,
    SGD_MOMENTUM,
    SGD_WEIGHT_DECAY,
    TFLITE_FLOAT,
    TFLITE_INT8,
)
from archive.core import (
    CustomCNN,
    evaluate,
    get_dataloaders,
    get_test_transform,
    get_train_transform,
    plot_augment,
    plot_baseline,
    run_training,
)


def stage_baseline():
    print("=== Stage 1: Baseline Training ===")
    model = CustomCNN().to(DEVICE)
    trainloader, testloader = get_dataloaders(
        get_train_transform(False), get_test_transform()
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
    history = run_training(
        model, trainloader, testloader, optimizer, EPOCHS, "Baseline"
    )
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), BASELINE_CKPT)
    plot_baseline(history, f"{GRAPHS_DIR}/training_plot.png")
    print(f"Saved baseline checkpoint: {BASELINE_CKPT}")


def stage_augment():
    print("=== Stage 2: Augmentation & Optimizer Benchmark ===")
    trainloader, testloader = get_dataloaders(
        get_train_transform(True), get_test_transform()
    )

    # Adam
    model_adam = CustomCNN().to(DEVICE)
    opt_adam = torch.optim.Adam(model_adam.parameters(), lr=ADAM_LR)
    h_adam = run_training(model_adam, trainloader, testloader, opt_adam, EPOCHS, "Adam")

    # SGD
    model_sgd = CustomCNN().to(DEVICE)
    opt_sgd = torch.optim.SGD(
        model_sgd.parameters(),
        lr=SGD_LR,
        momentum=SGD_MOMENTUM,
        weight_decay=SGD_WEIGHT_DECAY,
    )
    h_sgd = run_training(model_sgd, trainloader, testloader, opt_sgd, EPOCHS, "SGD")

    # Save best
    best = model_adam if h_adam["test_acc"][-1] >= h_sgd["test_acc"][-1] else model_sgd
    torch.save(best.state_dict(), BEST_CKPT)
    plot_augment(h_adam, h_sgd, f"{GRAPHS_DIR}/augmentation_optimizer_comparison.png")
    print(f"Best checkpoint saved: {BEST_CKPT}")


def stage_debug():
    print("=== Stage 3: Debug & Checkpoint Validation ===")
    model = CustomCNN().to(DEVICE)
    ckpt = BEST_CKPT if os.path.exists(BEST_CKPT) else BASELINE_CKPT
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    _, testloader = get_dataloaders(get_train_transform(False), get_test_transform())

    # Shape check
    sample, _ = next(iter(testloader))
    out = model(sample.to(DEVICE))
    assert out.shape == (sample.shape[0], NUM_CLASSES), f"Shape mismatch: {out.shape}"
    print(f"Shape OK: {sample.shape} → {out.shape}")

    # Consistency
    acc1 = evaluate(model, testloader, DEVICE)
    acc2 = evaluate(model, testloader, DEVICE)
    print(
        f"Acc run1: {acc1:.2f}% | run2: {acc2:.2f}% | Consistent: {abs(acc1 - acc2) < 1e-6}"
    )


def stage_tflite():
    print("=== Stage 4: TFLite Export & Quantization ===")
    import tensorflow as tf

    layers = tf.keras.layers
    models = tf.keras.models

    # Load PyTorch weights
    model = CustomCNN().to(DEVICE)
    ckpt = BEST_CKPT if os.path.exists(BEST_CKPT) else BASELINE_CKPT
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    weights = {n: p.detach().numpy() for n, p in model.named_parameters()}

    # Build Keras (NHWC + NCHW transpose before flatten)
    keras_model = models.Sequential(
        [
            layers.Input(shape=(32, 32, 3), name="input"),
            layers.Conv2D(16, 3, padding="same", activation="relu", name="conv1"),
            layers.MaxPooling2D(2, name="pool1"),
            layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2"),
            layers.MaxPooling2D(2, name="pool2"),
            layers.Lambda(
                lambda x: tf.transpose(x, [0, 3, 1, 2]), name="nchw_transpose"
            ),
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="fc1"),
            layers.Dense(10, name="fc2"),
        ]
    )
    keras_model.get_layer("conv1").set_weights(
        [np.transpose(weights["conv1.weight"], (2, 3, 1, 0)), weights["conv1.bias"]]
    )
    keras_model.get_layer("conv2").set_weights(
        [np.transpose(weights["conv2.weight"], (2, 3, 1, 0)), weights["conv2.bias"]]
    )
    keras_model.get_layer("fc1").set_weights(
        [weights["fc1.weight"].T, weights["fc1.bias"]]
    )
    keras_model.get_layer("fc2").set_weights(
        [weights["fc2.weight"].T, weights["fc2.bias"]]
    )

    # Verify numerical equivalence
    x_test = torch.randn(1, 3, 32, 32)
    pt_out = model(x_test).detach().numpy()
    keras_out = keras_model.predict(
        np.transpose(x_test.numpy(), (0, 2, 3, 1)), verbose=0
    )
    assert np.allclose(pt_out, keras_out, atol=1e-4), "Weight transfer failed"
    print("Weight transfer verified: PyTorch ≈ Keras")

    # Export SavedModel
    os.makedirs(MODEL_DIR, exist_ok=True)
    keras_model.export(SAVED_MODEL_DIR)

    # Float TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    with open(TFLITE_FLOAT, "wb") as f:
        f.write(converter.convert())
    print(f"Float TFLite: {TFLITE_FLOAT}")

    # Dynamic range quantized TFLite
    converter_q = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter_q.optimizations = [tf.lite.Optimize.DEFAULT]
    with open(TFLITE_INT8, "wb") as f:
        f.write(converter_q.convert())
    print(f"INT8 TFLite:  {TFLITE_INT8}")

    # Benchmark
    _, testloader = get_dataloaders(get_train_transform(False), get_test_transform())
    acc_pt = evaluate(model, testloader, DEVICE)

    interpreter = tf.lite.Interpreter(TFLITE_INT8)
    interpreter.allocate_tensors()
    in_det, out_det = interpreter.get_input_details(), interpreter.get_output_details()
    correct = total = 0
    for i, (inputs, labels) in enumerate(testloader):
        if i >= 5:
            break
        inp = np.transpose(inputs.numpy(), (0, 2, 3, 1)).astype(np.float32)
        for j in range(inp.shape[0]):
            interpreter.set_tensor(in_det[0]["index"], inp[j : j + 1])
            interpreter.invoke()
            pred = np.argmax(interpreter.get_tensor(out_det[0]["index"])[0])
            correct += pred == labels.numpy()[j]
            total += 1
    acc_tflite = 100 * correct / total

    # Report
    def sz(p):
        return os.path.getsize(p) / 1024 / 1024

    with open(BENCHMARK_REPORT, "w") as f:
        f.write("# TFLite Conversion Benchmark\n\n")
        f.write("| Model | MB |\n|---|---|\n")
        f.write(f"| PyTorch .pth | {sz(ckpt):.3f} |\n")
        f.write(f"| TFLite Float | {sz(TFLITE_FLOAT):.3f} |\n")
        f.write(f"| TFLite INT8  | {sz(TFLITE_INT8):.3f} |\n\n")
        f.write(f"PyTorch accuracy:  {acc_pt:.2f}%\n")
        f.write(f"TFLite INT8 accuracy: {acc_tflite:.2f}%\n")
        f.write(f"Compression ratio: {sz(ckpt) / sz(TFLITE_INT8):.1f}x\n")
    print(f"Benchmark report: {BENCHMARK_REPORT}")
    print(
        f"PyTorch: {acc_pt:.2f}% | TFLite INT8: {acc_tflite:.2f}% | Delta: {abs(acc_pt - acc_tflite):.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(description="PyTorch CNN → TFLite Pipeline")
    parser.add_argument(
        "--stage",
        choices=["baseline", "augment", "debug", "tflite", "all"],
        default="all",
    )
    args = parser.parse_args()

    stages = {
        "baseline": stage_baseline,
        "augment": stage_augment,
        "debug": stage_debug,
        "tflite": stage_tflite,
    }

    if args.stage == "all":
        for s in ["baseline", "augment", "debug", "tflite"]:
            stages[s]()
    else:
        stages[args.stage]()


if __name__ == "__main__":
    main()
