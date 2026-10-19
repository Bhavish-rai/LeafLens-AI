
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "PlantVillage"
TEST_FILE = DATA_DIR / "splits" / "color_test.txt"

IMAGE_SIZE = (224, 224)
LATENCY_BATCH_SIZE = 8
LATENCY_RUNS = 10

MODEL_CONFIGS = [
    {
        "name": "MobileNetV2",
        "model_path": MODELS_DIR / "leaflens_mobilenetv2_best.keras",
        "metrics_path": REPORTS_DIR / "mobilenetv2_test_metrics.json",
    },
    {
        "name": "EfficientNetB0",
        "model_path": MODELS_DIR / "leaflens_efficientnetb0_best.keras",
        "metrics_path": REPORTS_DIR / "efficientnetb0_test_metrics.json",
    },
]

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load official test images for latency testing
# --------------------------------------------------

def load_sample_batch():
    if not TEST_FILE.exists():
        raise FileNotFoundError(f"Test split not found: {TEST_FILE}")

    paths = [
        line.strip()
        for line in TEST_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:LATENCY_BATCH_SIZE]

    images = []

    for relative_path in paths:
        image_path = DATA_DIR / relative_path

        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image_bytes = tf.io.read_file(str(image_path))
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False,
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32)
        images.append(image)

    return tf.stack(images)


# --------------------------------------------------
# 3. Measure standardized model inference latency
# --------------------------------------------------

def measure_inference_latency(model, sample_batch):
    # Warm up the model first. This call is excluded from timing.
    _ = model(sample_batch, training=False)

    # Synchronize execution before starting measurements.
    start = time.perf_counter()
    _ = model(sample_batch, training=False)
    first_call_seconds = time.perf_counter() - start

    # Repeat inference using the same images and batch size.
    durations = []

    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        _ = model(sample_batch, training=False)
        durations.append(time.perf_counter() - start)

    average_batch_seconds = float(np.mean(durations))
    image_count = int(sample_batch.shape[0])

    return {
        "latency_batch_size": image_count,
        "average_batch_inference_ms": round(
            average_batch_seconds * 1000, 2
        ),
        "average_per_image_ms": round(
            average_batch_seconds * 1000 / image_count, 2
        ),
        "first_call_ms": round(first_call_seconds * 1000, 2),
    }


# --------------------------------------------------
# 4. Compare metrics, model sizes and parameters
# --------------------------------------------------

def main():
    sample_batch = load_sample_batch()
    results = []

    for config in MODEL_CONFIGS:
        model_name = config["name"]
        model_path = config["model_path"]
        metrics_path = config["metrics_path"]

        print(f"\nAnalyzing {model_name}...")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if not metrics_path.exists():
            raise FileNotFoundError(
                f"Evaluation metrics not found: {metrics_path}"
            )

        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)

        model = tf.keras.models.load_model(model_path)

        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        total_parameters = model.count_params()

        trainable_parameters = int(
            sum(
                np.prod(weight.shape)
                for weight in model.trainable_weights
            )
        )

        latency = measure_inference_latency(model, sample_batch)

        result = {
            "model": model_name,
            "test_samples": metrics["test_samples"],
            "accuracy": metrics["accuracy"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1_score": metrics["macro_f1_score"],
            "weighted_f1_score": metrics["weighted_f1_score"],
            "model_size_mb": round(file_size_mb, 2),
            "total_parameters": int(total_parameters),
            "trainable_parameters": trainable_parameters,
            "test_evaluation_time_seconds": metrics.get(
    "evaluation_time_seconds"
),
            **latency,
        }

        results.append(result)

        print(f"Test accuracy: {result['accuracy']:.2%}")
        print(f"Macro F1-score: {result['macro_f1_score']:.2%}")
        print(f"Model size: {result['model_size_mb']} MB")
        print(f"Total parameters: {result['total_parameters']:,}")
        print(
            "Average inference per image:",
            f"{result['average_per_image_ms']} ms",
        )

        del model

    # --------------------------------------------------
    # 5. Save comparison table
    # --------------------------------------------------

    comparison_df = pd.DataFrame(results)

    csv_path = REPORTS_DIR / "model_comparison.csv"
    comparison_df.to_csv(csv_path, index=False)

    # --------------------------------------------------
    # 6. Generate accuracy and macro F1 chart
    # --------------------------------------------------

    model_names = comparison_df["model"].tolist()
    x = np.arange(len(model_names))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 5))

    accuracy_bars = ax.bar(
        x - width / 2,
        comparison_df["accuracy"] * 100,
        width,
        label="Accuracy",
    )

    f1_bars = ax.bar(
        x + width / 2,
        comparison_df["macro_f1_score"] * 100,
        width,
        label="Macro F1-score",
    )

    ax.set_title("LeafLens AI: Model Performance Comparison")
    ax.set_ylabel("Score (%)")
    ax.set_xlabel("Model")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    ax.bar_label(accuracy_bars, fmt="%.2f%%", padding=3, fontsize=9)
    ax.bar_label(f1_bars, fmt="%.2f%%", padding=3, fontsize=9)

    fig.tight_layout()

    chart_path = REPORTS_DIR / "model_comparison.png"
    fig.savefig(chart_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --------------------------------------------------
    # 7. Print final summary
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("LEAFLENS AI — MODEL COMPARISON")
    print("=" * 65)

    display_columns = [
        "model",
        "accuracy",
        "macro_f1_score",
        "model_size_mb",
        "total_parameters",
        "average_per_image_ms",
    ]

    print(
        comparison_df[display_columns].to_string(index=False)
    )

    print("\nSaved comparison files:")
    print(csv_path)
    print(chart_path)


if __name__ == "__main__":
    main()
