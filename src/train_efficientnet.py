
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "PlantVillage"
SPLIT_DIR = DATA_DIR / "splits"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

TRAIN_FILE = SPLIT_DIR / "color_train.txt"
CLASS_NAMES_FILE = REPORTS_DIR / "class_names.json"
MODEL_FILE = MODELS_DIR / "leaflens_efficientnetb0_best.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
SEED = 42

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

tf.keras.utils.set_random_seed(SEED)

print("TensorFlow version:", tf.__version__)
print("Project directory:", ROOT)


# --------------------------------------------------
# 2. Load training paths and labels
# --------------------------------------------------

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training split not found: {TRAIN_FILE}")

relative_paths = [
    line.strip()
    for line in TRAIN_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

image_paths = []
class_labels = []

for relative_path in relative_paths:
    parts = Path(relative_path).parts

    if len(parts) < 4:
        raise ValueError(f"Unexpected dataset path: {relative_path}")

    image_path = DATA_DIR / relative_path
    class_name = parts[2]

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_paths.append(str(image_path))
    class_labels.append(class_name)

class_names = sorted(set(class_labels))

# Keep class ordering identical to the MobileNetV2 experiment.
if CLASS_NAMES_FILE.exists():
    saved_class_names = json.loads(
        CLASS_NAMES_FILE.read_text(encoding="utf-8")
    )

    if class_names != saved_class_names:
        raise ValueError(
            "Class ordering differs from the MobileNetV2 experiment."
        )
else:
    raise FileNotFoundError(
        "reports/class_names.json is missing. "
        "Run the MobileNetV2 training script first."
    )

class_to_index = {
    class_name: index
    for index, class_name in enumerate(class_names)
}

labels = np.array(
    [class_to_index[name] for name in class_labels],
    dtype=np.int32,
)

image_paths = np.array(image_paths)

print("\nDataset loaded successfully.")
print("Training images:", len(image_paths))
print("Number of classes:", len(class_names))


# --------------------------------------------------
# 3. Create the same training/validation split
# --------------------------------------------------

train_paths, val_paths, y_train, y_val = train_test_split(
    image_paths,
    labels,
    test_size=0.15,
    random_state=SEED,
    stratify=labels,
)

print("\nTraining samples:", len(train_paths))
print("Validation samples:", len(val_paths))


# --------------------------------------------------
# 4. Image loading and preprocessing
# --------------------------------------------------

def load_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)

    # EfficientNetB0 in this TensorFlow/Keras version
    # includes its own input rescaling layers.
    # Keep pixel values in the 0–255 range.
    return image, label


def create_dataset(paths, labels_array, training=False):
    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels_array)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_dataset = create_dataset(
    train_paths,
    y_train,
    training=True,
)

val_dataset = create_dataset(
    val_paths,
    y_val,
    training=False,
)


# --------------------------------------------------
# 5. Calculate class weights
# --------------------------------------------------

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=y_train,
)

class_weights = {
    index: float(weight)
    for index, weight in enumerate(class_weights_array)
}


# --------------------------------------------------
# 6. Build EfficientNetB0
# --------------------------------------------------

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
    ],
    name="data_augmentation",
)

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

# Initially train only the classification head.
base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = augmentation(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)

outputs = tf.keras.layers.Dense(
    len(class_names),
    activation="softmax",
    name="predictions",
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="LeafLens_EfficientNetB0",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# --------------------------------------------------
# 7. Configure training callbacks
# --------------------------------------------------

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_FILE),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-6,
        verbose=1,
    ),
    tf.keras.callbacks.CSVLogger(
        str(REPORTS_DIR / "efficientnetb0_training.csv")
    ),
]


# --------------------------------------------------
# 8. Train the model
# --------------------------------------------------

print("\nStarting EfficientNetB0 training...")
print("This may take a while on a CPU.")

start_time = time.time()

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1,
)

training_seconds = time.time() - start_time


# --------------------------------------------------
# 9. Load best checkpoint and save metrics
# --------------------------------------------------

best_model = tf.keras.models.load_model(MODEL_FILE)

val_loss, val_accuracy = best_model.evaluate(
    val_dataset,
    verbose=1,
)

metrics = {
    "model": "EfficientNetB0",
    "dataset": "PlantVillage",
    "validation_samples": int(len(val_paths)),
    "number_of_classes": int(len(class_names)),
    "image_size": "224x224",
    "batch_size": BATCH_SIZE,
    "maximum_epochs": EPOCHS,
    "seed": SEED,
    "validation_loss": float(val_loss),
    "validation_accuracy": float(val_accuracy),
    "training_time_seconds": round(training_seconds, 2),
    "best_model_path": str(
        MODEL_FILE.relative_to(ROOT)
    ),
}

metrics_path = REPORTS_DIR / "efficientnetb0_metrics.json"

metrics_path.write_text(
    json.dumps(metrics, indent=4),
    encoding="utf-8",
)

print("\n" + "=" * 50)
print("EFFICIENTNETB0 TRAINING COMPLETE")
print("=" * 50)
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Validation loss:     {val_loss:.4f}")
print(f"Training time:       {training_seconds / 60:.2f} minutes")
print(f"Best model:          {MODEL_FILE}")
print(f"Metrics:             {metrics_path}")
