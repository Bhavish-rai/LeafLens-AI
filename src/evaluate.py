import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

DATA_DIR = Path("data/PlantVillage")
SPLIT_DIR = DATA_DIR / "splits"
MODEL_PATH = Path("models/leaflens_mobilenetv2_best.keras")
REPORT_DIR = Path("reports")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("LEAFLENS AI - OFFICIAL TEST EVALUATION")
print("=" * 60)

# Load the saved model and class names
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

with open(REPORT_DIR / "class_names.json", encoding="utf-8") as file:
    class_names = json.load(file)

model = tf.keras.models.load_model(MODEL_PATH)

# Read official test paths without splitting on spaces
test_file = SPLIT_DIR / "color_test.txt"

test_paths = [
    line.strip().replace("\\", "/")
    for line in test_file.read_text(encoding="utf-8-sig").splitlines()
    if line.strip()
]

image_paths = []
true_labels = []

for relative_path in test_paths:
    parts = relative_path.split("/")

    if len(parts) < 4:
        raise ValueError(f"Unexpected path: {relative_path}")

    class_name = parts[2]

    if class_name not in class_names:
        raise ValueError(f"Unknown class: {class_name}")

    image_path = DATA_DIR.joinpath(*parts)

    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image: {image_path}")

    image_paths.append(str(image_path.resolve()))
    true_labels.append(class_names.index(class_name))

true_labels = np.asarray(true_labels, dtype=np.int32)

print(f"Test images: {len(image_paths)}")
print(f"Classes: {len(class_names)}")

# Load and preprocess test images
def load_test_image(path, label):
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    # Match MobileNetV2 training preprocessing
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label


test_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, true_labels)
)

test_dataset = test_dataset.map(
    load_test_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(
    tf.data.AUTOTUNE
)

# Predict the official test set for the first time
print("\nPredicting official test images...")

probabilities = model.predict(test_dataset, verbose=1)
predicted_labels = np.argmax(probabilities, axis=1)
confidence_scores = np.max(probabilities, axis=1)

# Calculate metrics
accuracy = accuracy_score(true_labels, predicted_labels)

macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
)

weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        average="weighted",
        zero_division=0,
    )
)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

# Save overall metrics
metrics = {
    "model": "MobileNetV2",
    "evaluation_split": "official_test",
    "test_samples": len(true_labels),
    "number_of_classes": len(class_names),
    "accuracy": float(accuracy),
    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1_score": float(macro_f1),
    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1_score": float(weighted_f1),
}

with open(
    REPORT_DIR / "mobilenetv2_test_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=2)

with open(
    REPORT_DIR / "mobilenetv2_classification_report.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(report, file, indent=2)

# Save predictions
predictions = pd.DataFrame({
    "image_path": test_paths,
    "actual_class": [class_names[i] for i in true_labels],
    "predicted_class": [class_names[i] for i in predicted_labels],
    "confidence": confidence_scores,
    "correct": true_labels == predicted_labels,
})

predictions.to_csv(
    REPORT_DIR / "mobilenetv2_test_predictions.csv",
    index=False,
)

# Save confusion matrix as CSV and image
matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=list(range(len(class_names))),
)

pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names,
).to_csv(REPORT_DIR / "mobilenetv2_confusion_matrix.csv")

fig, ax = plt.subplots(figsize=(22, 19))

image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

ax.set(
    title="LeafLens AI - MobileNetV2 Test Confusion Matrix",
    xlabel="Predicted class",
    ylabel="Actual class",
)

ax.set_xticks(range(len(class_names)))
ax.set_xticklabels(class_names, rotation=90, fontsize=7)
ax.set_yticks(range(len(class_names)))
ax.set_yticklabels(class_names, fontsize=7)

fig.tight_layout()
fig.savefig(
    REPORT_DIR / "mobilenetv2_confusion_matrix.png",
    dpi=180,
    bbox_inches="tight",
)
plt.close(fig)

print("\n" + "=" * 60)
print("OFFICIAL TEST RESULTS")
print("=" * 60)

for name, value in metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.4f}")
    else:
        print(f"{name}: {value}")

print("\nFiles saved in the reports/ directory.")
print("Official test evaluation completed.")