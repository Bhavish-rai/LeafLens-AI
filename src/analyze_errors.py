
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"

MODEL_FILES = {
    "MobileNetV2": REPORTS / "mobilenetv2_test_predictions.csv",
    "EfficientNetB0": REPORTS / "efficientnetb0_test_predictions.csv",
}

ERRORS_FILE = REPORTS / "misclassified_predictions.csv"
CLASS_REPORT_FILE = REPORTS / "error_analysis_by_class.csv"
COMPARISON_FILE = REPORTS / "error_analysis_model_comparison.csv"
PLOT_FILE = REPORTS / "misclassification_comparison.png"


def load_predictions(model_name, file_path):
    if not file_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {file_path}")

    df = pd.read_csv(file_path)

    # Support different column names across the two model CSV files.
    actual_col = next(
        (c for c in ["actual_class", "true_class", "actual_label",
                     "true_label", "ground_truth", "y_true", "label"]
         if c in df.columns),
        None,
    )

    predicted_col = next(
        (c for c in ["predicted_class", "predicted_label",
                     "predicted", "y_pred", "prediction"]
         if c in df.columns),
        None,
    )

    image_col = next(
        (c for c in ["image_path", "image", "filepath", "filename", "path"]
         if c in df.columns),
        None,
    )

    if actual_col is None or predicted_col is None:
        raise ValueError(
            f"Cannot identify actual/predicted columns in {file_path.name}.\n"
            f"Available columns: {list(df.columns)}"
        )

    if image_col is None:
        df["image_path"] = [f"row_{i}" for i in range(len(df))]
        image_col = "image_path"

    result = pd.DataFrame({
        "image_path": df[image_col].astype(str),
        "actual_class": df[actual_col].astype(str),
        "predicted_class": df[predicted_col].astype(str),
    })

    if "confidence" in df.columns:
        result["confidence"] = pd.to_numeric(
            df["confidence"], errors="coerce"
        )

    result["model"] = model_name
    result["correct"] = (
        result["actual_class"] == result["predicted_class"]
    )

    print(f"\n{model_name}")
    print(f"Test images: {len(result):,}")
    print(f"Correct predictions: {result['correct'].sum():,}")
    print(f"Incorrect predictions: {(~result['correct']).sum():,}")
    print(f"Accuracy: {result['correct'].mean() * 100:.2f}%")

    return result


def main():
    REPORTS.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("LEAFLENS AI - MODEL ERROR ANALYSIS")
    print("=" * 55)

    predictions = [
        load_predictions(name, path)
        for name, path in MODEL_FILES.items()
    ]

    combined = pd.concat(predictions, ignore_index=True)

    # 1. Save every incorrect prediction.
    errors = combined.loc[~combined["correct"]].copy()
    errors.to_csv(ERRORS_FILE, index=False)

    print(f"\nMisclassified predictions saved: {ERRORS_FILE.name}")
    print(f"Total errors across both models: {len(errors):,}")

    # 2. Generate per-class precision, recall and F1-score.
    class_rows = []

    for model_name, group in combined.groupby("model", sort=False):
        report = classification_report(
            group["actual_class"],
            group["predicted_class"],
            output_dict=True,
            zero_division=0,
        )

        for class_name, values in report.items():
            if class_name in {"accuracy", "macro avg", "weighted avg"}:
                continue

            class_rows.append({
                "model": model_name,
                "class": class_name,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1-score"],
                "support": int(values["support"]),
            })

    class_report = pd.DataFrame(class_rows)
    class_report.to_csv(CLASS_REPORT_FILE, index=False)

    print(f"Per-class report saved: {CLASS_REPORT_FILE.name}")

    print("\n--- Five weakest classes per model (by F1-score) ---")
    for model_name, group in class_report.groupby("model", sort=False):
        print(f"\n{model_name}")
        print(
            group.nsmallest(5, "f1_score")[
                ["class", "precision", "recall", "f1_score", "support"]
            ].to_string(index=False, float_format=lambda x: f"{x:.4f}")
        )

    # 3. Create model-level comparison.
    summary = (
        combined.groupby("model")
        .agg(
            test_images=("correct", "size"),
            correct_predictions=("correct", "sum"),
        )
        .reset_index()
    )

    summary["incorrect_predictions"] = (
        summary["test_images"] - summary["correct_predictions"]
    )
    summary["accuracy_percent"] = (
        summary["correct_predictions"] / summary["test_images"] * 100
    )
    summary["error_rate_percent"] = (
        summary["incorrect_predictions"] / summary["test_images"] * 100
    )

    summary.to_csv(COMPARISON_FILE, index=False)

    print("\n--- Model comparison ---")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    # 4. Compare correctness on images common to both CSV files.
    pivot = combined.pivot(
        index="image_path",
        columns="model",
        values="correct",
    )

    if {"MobileNetV2", "EfficientNetB0"}.issubset(pivot.columns):
        comparable = pivot.dropna()
        mobile = comparable["MobileNetV2"].astype(bool)
        efficient = comparable["EfficientNetB0"].astype(bool)

        print("\n--- Same-image comparison ---")
        print(f"Common images compared: {len(comparable):,}")
        print(f"Both correct: {(mobile & efficient).sum():,}")
        print(f"Both incorrect: {(~mobile & ~efficient).sum():,}")
        print(
            "EfficientNetB0 correct, MobileNetV2 wrong: "
            f"{(~mobile & efficient).sum():,}"
        )
        print(
            "MobileNetV2 correct, EfficientNetB0 wrong: "
            f"{(mobile & ~efficient).sum():,}"
        )

    # 5. Generate a high-resolution error comparison graph.
    plt.figure(figsize=(8, 5))
    plt.bar(summary["model"], summary["incorrect_predictions"])
    plt.title("Misclassified Test Images by Model")
    plt.xlabel("Model")
    plt.ylabel("Number of Incorrect Predictions")
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n--- Analysis completed successfully ---")
    print("Generated files:")

    for path in [
        ERRORS_FILE,
        CLASS_REPORT_FILE,
        COMPARISON_FILE,
        PLOT_FILE,
    ]:
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
