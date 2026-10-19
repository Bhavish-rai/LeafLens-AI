
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "PlantVillage"
SPLIT_DIR = DATA_DIR / "splits"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

TEST_FILE = SPLIT_DIR / "color_test.txt"
CLASS_NAMES_FILE = REPORTS_DIR / "class_names.json"
MODEL_FILE = MODELS_DIR / "leaflens_efficientnetb0_best.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading EfficientNetB0 evaluation setup...")


# --------------------------------------------------
# 2. Load class names and trained model
# --------------------------------------------------

if not MODEL_FILE.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_FILE}")

if not TEST_FILE.exists():
    raise FileNotFoundError(f"Official test split not found: {TEST_FILE}")

if not CLASS_NAMES_FILE.exists():
    raise FileNotFoundError(f"Class names not found: {CLASS_NAMES_FILE}")

class_names = json.loads(
    CLASS_NAMES_FILE.read_text(encoding="utf-8")
)

model = tf.keras.models.load_model(MODEL_FILE)

print("Model loaded successfully.")
print("Number of classes:", len(class_names))


# --------------------------------------------------
# 3. Load official test paths and labels
# --------------------------------------------------

relative_paths = [
    line.strip()
    for line in TEST_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

class_to_index = {
    name: index for index, name in enumerate(class_names)
}

image_paths = []
true_labels = []

for relative_path in relative_paths:
    parts = Path(relative_path).parts

    if len(parts) < 4:
        raise ValueError(f"Unexpected dataset path: {relative_path}")

    image_path = DATA_DIR / relative_path
    class_name = parts[2]

    if class_name not in class_to_index:
        raise ValueError(f"Unknown class: {class_name}")

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_paths.append(str(image_path))
    true_labels.append(class_to_index[class_name])

true_labels = np.array(true_labels, dtype=np.int32)

print("Official test images:", len(image_paths))


# --------------------------------------------------
# 4. Preprocess official test images
# --------------------------------------------------

def load_test_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # EfficientNetB0 contains its own input rescaling.
    # Therefore, keep pixel values in the 0–255 range.
    return image, label


test_dataset = tf.data.Dataset.from_tensor_slices(
    (image_paths, true_labels)
)

test_dataset = test_dataset.map(
    load_test_image,
    num_parallel_calls=tf.data.AUTOTUNE,
)

test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


# --------------------------------------------------
# 5. Generate predictions
# --------------------------------------------------

print("\nEvaluating on the official test set...")
start_time = time.time()

probabilities = model.predict(test_dataset, verbose=1)

evaluation_seconds = time.time() - start_time

predicted_labels = np.argmax(probabilities, axis=1)
confidence_scores = np.max(probabilities, axis=1)


# --------------------------------------------------
# 6. Calculate evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(true_labels, predicted_labels)

macro_precision = precision_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0,
)

weighted_precision = precision_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

weighted_recall = recall_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0,
)

report = classification_report(
    true_labels,
    predicted_labels,
    labels=np.arange(len(class_names)),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

metrics = {
    "model": "EfficientNetB0",
    "evaluation_split": "official_test",
    "test_samples": int(len(true_labels)),
    "number_of_classes": int(len(class_names)),
    "accuracy": round(float(accuracy), 4),
    "macro_precision": round(float(macro_precision), 4),
    "macro_recall": round(float(macro_recall), 4),
    "macro_f1_score": round(float(macro_f1), 4),
    "weighted_precision": round(float(weighted_precision), 4),
    "weighted_recall": round(float(weighted_recall), 4),
    "weighted_f1_score": round(float(weighted_f1), 4),
    "evaluation_time_seconds": round(evaluation_seconds, 2),
}


# --------------------------------------------------
# 7. Save metrics and prediction results
# --------------------------------------------------

metrics_path = REPORTS_DIR / "efficientnetb0_test_metrics.json"
report_path = REPORTS_DIR / "efficientnetb0_classification_report.json"
predictions_path = REPORTS_DIR / "efficientnetb0_test_predictions.csv"

metrics_path.write_text(
    json.dumps(metrics, indent=4),
    encoding="utf-8",
)

report_path.write_text(
    json.dumps(report, indent=4),
    encoding="utf-8",
)

prediction_df = pd.DataFrame(
    {
        "image_path": relative_paths,
        "true_class": [class_names[i] for i in true_labels],
        "predicted_class": [
            class_names[i] for i in predicted_labels
        ],
        "confidence": confidence_scores,
        "correct": true_labels == predicted_labels,
    }
)

prediction_df.to_csv(predictions_path, index=False)


# --------------------------------------------------
# 8. Generate and save confusion matrix
# --------------------------------------------------

matrix = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=np.arange(len(class_names)),
)

matrix_csv_path = REPORTS_DIR / "efficientnetb0_confusion_matrix.csv"
matrix_png_path = REPORTS_DIR / "efficientnetb0_confusion_matrix.png"

pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names,
).to_csv(matrix_csv_path)

fig, ax = plt.subplots(figsize=(20, 18))

image = ax.imshow(matrix, interpolation="nearest")
fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

ax.set(
    title="EfficientNetB0 — Official Test Confusion Matrix",
    xlabel="Predicted Class",
    ylabel="True Class",
)

ax.set_xticks(np.arange(len(class_names)))
ax.set_yticks(np.arange(len(class_names)))

ax.set_xticklabels(class_names, rotation=90, fontsize=7)
ax.set_yticklabels(class_names, fontsize=7)

fig.tight_layout()
fig.savefig(matrix_png_path, dpi=200, bbox_inches="tight")
plt.close(fig)


# --------------------------------------------------
# 9. Print final results
# --------------------------------------------------

print("\n" + "=" * 55)
print("EFFICIENTNETB0 OFFICIAL TEST RESULTS")
print("=" * 55)

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nSaved files:")
print(metrics_path)
print(report_path)
print(predictions_path)
print(matrix_csv_path)
print(matrix_png_path)
