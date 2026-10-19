
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "PlantVillage"
TEST_FILE = DATA_DIR / "splits" / "color_test.txt"
REPORTS_DIR = ROOT / "reports"
CLASS_NAMES_FILE = REPORTS_DIR / "class_names.json"

IMAGE_SIZE = (224, 224)

MODEL_FILES = {
    "mobilenetv2": ROOT / "models" / "leaflens_mobilenetv2_best.keras",
    "efficientnetb0": ROOT / "models" / "leaflens_efficientnetb0_best.keras",
}


# --------------------------------------------------
# 2. Load trained model and class names
# --------------------------------------------------

def load_model_and_classes(model_name):
    model_path = MODEL_FILES[model_name]

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not CLASS_NAMES_FILE.exists():
        raise FileNotFoundError(
            f"Class names file not found: {CLASS_NAMES_FILE}"
        )

    with CLASS_NAMES_FILE.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = tf.keras.models.load_model(model_path)

    print("Model loaded:", model_name)
    print("Number of classes:", len(class_names))

    return model, class_names


# --------------------------------------------------
# 3. Build a Grad-CAM model
# --------------------------------------------------

def build_gradcam_model(model):
    # Locate the nested CNN backbone.
    backbone = None

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            if any(
                isinstance(inner, tf.keras.layers.Conv2D)
                for inner in layer.layers
            ):
                backbone = layer
                break

    if backbone is None:
        raise ValueError("Could not find the CNN backbone.")

    # Find the final convolutional layer inside the backbone.
    last_conv_layer = next(
        (
            layer
            for layer in reversed(backbone.layers)
            if isinstance(layer, tf.keras.layers.Conv2D)
        ),
        None,
    )

    if last_conv_layer is None:
        raise ValueError("Could not find the final convolutional layer.")

    print("Backbone:", backbone.name)
    print("Grad-CAM layer:", last_conv_layer.name)

    # Expose the final convolutional activations and backbone output.
    backbone_grad_model = tf.keras.Model(
        inputs=backbone.input,
        outputs=[
            last_conv_layer.output,
            backbone.output,
        ],
        name="backbone_gradcam_model",
    )

    # Rebuild the model's forward path from its original inputs.
    inputs = model.inputs
    x = inputs[0]

    backbone_index = model.layers.index(backbone)

    for layer in model.layers[1:backbone_index]:
        if isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=False)
        else:
            x = layer(x)

    conv_outputs, backbone_outputs = backbone_grad_model(
        x,
        training=False,
    )

    # Apply every classification layer after the backbone.
    x = backbone_outputs

    for layer in model.layers[backbone_index + 1:]:
        if isinstance(layer, tf.keras.layers.Dropout):
            x = layer(x, training=False)
        else:
            x = layer(x)

    predictions = x

    grad_model = tf.keras.Model(
        inputs=inputs,
        outputs=[conv_outputs, predictions],
        name="gradcam_model",
    )

    return grad_model


# --------------------------------------------------
# 4. Generate Grad-CAM heatmap
# --------------------------------------------------

def generate_gradcam(grad_model, input_tensor):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            input_tensor,
            training=False,
        )

        predicted_index = tf.argmax(
            predictions[0],
            output_type=tf.int32,
        )

        class_score = predictions[:, predicted_index]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Gradients could not be calculated. "
            "Check the CNN backbone and classification layers."
        )

    # Calculate the importance of each feature-map channel.
    channel_weights = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * channel_weights,
        axis=-1,
    )

    # Retain positive contributions.
    heatmap = tf.maximum(heatmap, 0)

    # Normalize heatmap values to the range 0–1.
    maximum = tf.reduce_max(heatmap)
    heatmap = heatmap / (
        maximum + tf.keras.backend.epsilon()
    )

    # Resize heatmap to the input image dimensions.
    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE,
    )

    return heatmap.numpy().squeeze(), predictions[0].numpy()


# --------------------------------------------------
# 5. Load and preprocess image
# --------------------------------------------------

def prepare_image(image_path, model_name):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    original_image = np.asarray(
        image,
        dtype=np.uint8,
    )

    image_array = original_image.astype(np.float32)

    # MobileNetV2 requires preprocessing to the -1 to 1 range.
    # EfficientNetB0 includes its own rescaling layers.
    if model_name == "mobilenetv2":
        image_array = (
            tf.keras.applications.mobilenet_v2.preprocess_input(
                image_array
            )
        )

    input_tensor = tf.convert_to_tensor(
        image_array[np.newaxis, ...],
        dtype=tf.float32,
    )

    return original_image, input_tensor


# --------------------------------------------------
# 6. Save visualization
# --------------------------------------------------

def save_visualization(
    original_image,
    heatmap,
    predicted_class,
    confidence,
    model_name,
    image_path,
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_name = (
        f"gradcam_{model_name}_{Path(image_path).stem}.png"
    )

    output_path = REPORTS_DIR / output_name

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_image)
    axes[0].set_title("Original Leaf Image")
    axes[0].axis("off")

    axes[1].imshow(
        heatmap,
        cmap="jet",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(original_image)
    axes[2].imshow(
        heatmap,
        cmap="jet",
        alpha=0.4,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Heatmap Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"{model_name} | {predicted_class} "
        f"| Confidence: {confidence:.2%}",
        fontsize=12,
    )

    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    return output_path


# --------------------------------------------------
# 7. Main execution
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for LeafLens AI."
    )

    parser.add_argument(
        "--model",
        choices=["mobilenetv2", "efficientnetb0"],
        default="mobilenetv2",
        help="Select the model to explain.",
    )

    parser.add_argument(
        "--image",
        default=None,
        help="Image path. Defaults to the first official test image.",
    )

    args = parser.parse_args()

    model, class_names = load_model_and_classes(args.model)

    # Select a test image if no custom path was provided.
    if args.image:
        image_path = Path(args.image)

        if not image_path.is_absolute():
            image_path = ROOT / image_path
    else:
        if not TEST_FILE.exists():
            raise FileNotFoundError(
                f"Official test split not found: {TEST_FILE}"
            )

        test_paths = [
            line.strip()
            for line in TEST_FILE.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if not test_paths:
            raise ValueError("The official test split is empty.")

        image_path = DATA_DIR / test_paths[0]

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    print("Input image:", image_path)

    original_image, input_tensor = prepare_image(
        image_path,
        args.model,
    )

    grad_model = build_gradcam_model(model)

    heatmap, predictions = generate_gradcam(
        grad_model,
        input_tensor,
    )

    predicted_index = int(np.argmax(predictions))
    predicted_class = class_names[predicted_index]
    confidence = float(predictions[predicted_index])

    output_path = save_visualization(
        original_image=original_image,
        heatmap=heatmap,
        predicted_class=predicted_class,
        confidence=confidence,
        model_name=args.model,
        image_path=image_path,
    )

    print("\nGrad-CAM generated successfully.")
    print("Predicted class:", predicted_class)
    print(f"Confidence: {confidence:.2%}")
    print("Visualization saved to:", output_path)


if __name__ == "__main__":
    main()
