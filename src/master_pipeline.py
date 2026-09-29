import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.master_config import (
    CONFIDENCE_THRESHOLD,
    EPOCHS,
)
from src.master_core import (
    CompactCNN,
    convert_full_int8_tflite,
    evaluate_pytorch,
    export_onnx,
    generate_evaluation_plots,
    get_cifar10_datasets,
    get_dataloaders,
    predict_with_guardrails,
    run_side_by_side_benchmark,
    train_model,
    verify_preprocessing_alignment,
)


def main():
    print("=== Step 1: Preprocessing & Data Pipeline Initialization ===")
    verify_preprocessing_alignment()
    train_subset, val_subset, test_dataset = get_cifar10_datasets()
    train_loader, val_loader, test_loader = get_dataloaders(
        train_subset, val_subset, test_dataset
    )
    print(
        f"Split complete: {len(train_subset)} Train | {len(val_subset)} Val | {len(test_dataset)} Test"
    )

    print("\n=== Step 2: Training 3-Layer Compact CNN with Regularization ===")
    model = CompactCNN()
    model = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    print("\n=== Step 3: Evaluation Suite (Precision, Recall, F1, ROC-AUC) ===")
    metrics = evaluate_pytorch(model, test_loader)
    print(f"Accuracy:  {metrics['accuracy'] * 100.0:.2f}%")
    print(f"Precision: {metrics['precision'] * 100.0:.2f}%")
    print(f"Recall:    {metrics['recall'] * 100.0:.2f}%")
    print(f"F1-Score:  {metrics['f1'] * 100.0:.2f}%")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    generate_evaluation_plots(metrics)
    print("Plots saved: graphs/roc_curve.png, graphs/confusion_matrix.png")

    print("\n=== Step 4: ONNX Export ===")
    export_onnx(model)

    print("\n=== Step 5: Full INT8 Quantization (TFLite PTQ) ===")
    convert_full_int8_tflite(model, train_subset)

    print(
        f"\n=== Step 6: Probabilistic Guardrails (Confidence Threshold = {CONFIDENCE_THRESHOLD}) ==="
    )
    sample_logits = metrics["probs"][:10]  # First 10 samples
    guardrail_results = predict_with_guardrails(sample_logits)
    for idx, res in enumerate(guardrail_results):
        print(
            f"Sample {idx:02d}: Status={res['status']} | Class={res['class']} | Conf={res['confidence']:.3f}"
        )

    print("\n=== Step 7: Side-by-Side Model Format Benchmarking ===")
    run_side_by_side_benchmark(test_loader)


if __name__ == "__main__":
    main()
