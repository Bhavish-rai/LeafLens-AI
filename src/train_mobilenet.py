
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    CSVLogger,
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SIZE = 0.15

DATA_DIR = Path("data/PlantVillage")
SPLIT_DIR = DATA_DIR / "splits"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

print("=" * 60)
print("LEAFLENS AI - MOBILENETV2 TRAINING")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"Available GPUs: {tf.config.list_physical_devices('GPU')}")


# --------------------------------------------------
# 2. Load the official training paths
# --------------------------------------------------

def read_training_paths():
    split_file = SPLIT_DIR / "color_train.txt"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Training split not found: {split_file}"
        )

    lines = [
        line.strip().replace("\\", "/")
        for line in split_file.read_text(
            encoding="utf-8-sig"
        ).splitlines()
        if line.strip()
    ]

    paths = []
    labels = []

    for relative_path in lines:
        parts = relative_path.split("/")

        if len(parts) < 4:
            raise ValueError(
                f"Unexpected image path: {relative_path}"
            )

        image_path = DATA_DIR.joinpath(*parts)

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # The class name is the directory under raw/color.
        class_name = parts[2]

        paths.append(str(image_path.resolve()))
        labels.append(class_name)

    return paths, labels


all_paths, all_labels = read_training_paths()

class_names = sorted(set(all_labels))
class_to_index = {
    name: index for index, name in enumerate(class_names)
}

numeric_labels = np.array(
    [class_to_index[label] for label in all_labels],
    dtype=np.int32,
)

print(f"\nTraining images available: {len(all_paths)}")
print(f"Number of classes: {len(class_names)}")


# --------------------------------------------------
# 3. Create training and validation sets
# --------------------------------------------------

train_paths, val_paths, train_labels, val_labels = (
    train_test_split(
        all_paths,
        numeric_labels,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=numeric_labels,
    )
)

print(f"Training samples: {len(train_paths)}")
print(f"Validation samples: {len(val_paths)}")

with open(REPORT_DIR / "class_names.json", "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)


# --------------------------------------------------
# 4. Build the TensorFlow data pipeline
# --------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE


def load_and_preprocess_image(image_path, label):
    image_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image_bytes, channels=3)

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        antialias=True,
    )

    image = preprocess_input(image)

    return image, label


def create_dataset(paths, labels, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(paths), 5000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_and_preprocess_image,
        num_parallel_calls=AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset


train_dataset = create_dataset(
    train_paths,
    train_labels,
    training=True,
)

val_dataset = create_dataset(
    val_paths,
    val_labels,
)


# --------------------------------------------------
# 5. Handle class imbalance
# --------------------------------------------------

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_labels),
    y=train_labels,
)

class_weights = {
    int(class_index): float(weight)
    for class_index, weight in zip(
        np.unique(train_labels),
        weights,
    )
}


# --------------------------------------------------
# 6. Build MobileNetV2
# --------------------------------------------------

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
    ],
    name="data_augmentation",
)

base_model = MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
)

# Initially train only the new classification head.
base_model.trainable = False

inputs = layers.Input(
    shape=(*IMAGE_SIZE, 3),
    name="leaf_image",
)

x = augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax",
    name="disease_predictions",
)(x)

model = models.Model(
    inputs=inputs,
    outputs=outputs,
    name="LeafLens_MobileNetV2",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# --------------------------------------------------
# 7. Configure callbacks
# --------------------------------------------------

model_path = MODEL_DIR / "leaflens_mobilenetv2_best.keras"

callbacks = [
    ModelCheckpoint(
        filepath=str(model_path),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    CSVLogger(
        str(REPORT_DIR / "mobilenetv2_training.csv"),
        append=False,
    ),
]


# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

print("\nStarting training...")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)


# --------------------------------------------------
# 9. Evaluate on validation data
# --------------------------------------------------

print("\nEvaluating the best model on validation data...")

best_model = tf.keras.models.load_model(model_path)

val_loss, val_accuracy = best_model.evaluate(
    val_dataset,
    verbose=1,
)

metrics = {
    "model": "MobileNetV2",
    "image_size": list(IMAGE_SIZE),
    "batch_size": BATCH_SIZE,
    "epochs_requested": EPOCHS,
    "epochs_completed": len(history.history["loss"]),
    "training_samples": len(train_paths),
    "validation_samples": len(val_paths),
    "number_of_classes": len(class_names),
    "validation_loss": float(val_loss),
    "validation_accuracy": float(val_accuracy),
    "random_seed": SEED,
}

with open(
    REPORT_DIR / "mobilenetv2_metrics.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(metrics, file, indent=2)

print("\nTraining completed!")
print(f"Best model saved to: {model_path}")
print(f"Metrics saved to: {REPORT_DIR / 'mobilenetv2_metrics.json'}")
print(f"Training history saved to: {REPORT_DIR / 'mobilenetv2_training.csv'}")
print(f"Validation accuracy: {val_accuracy:.4f}")
print("\nThe official test set has NOT been used.")
