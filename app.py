
import base64
import html
import io
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# =========================================================
# PATHS AND MODEL CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CLASS_FILE = REPORTS_DIR / "class_names.json"
GRADCAM_SCRIPT = ROOT / "src" / "gradcam.py"

MODELS = {
    "EfficientNetB0": {
        "key": "efficientnetb0",
        "path": MODELS_DIR / "leaflens_efficientnetb0_best.keras",
        "accuracy": "96.54%",
        "size": "16.73 MB",
        "description": "Higher test accuracy",
    },
    "MobileNetV2": {
        "key": "mobilenetv2",
        "path": MODELS_DIR / "leaflens_mobilenetv2_best.keras",
        "accuracy": "95.16%",
        "size": "9.69 MB",
        "description": "Smaller and faster in the tested CPU benchmark",
    },
}

st.set_page_config(
    page_title="LeafLens AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# IMAGE DISPLAY — BASE64 EMBEDDING
# =========================================================

def image_to_base64(image):
    """Convert a PIL image to an embedded PNG data URL."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def show_image(image, caption, height=390):
    """Display an image without relying on Streamlit's image route."""
    data_url = image_to_base64(image)
    safe_caption = html.escape(str(caption))

    st.markdown(
        f"""
        <div style="
            background:#FFFFFF;
            border:1px solid #DFE8DD;
            border-radius:16px;
            padding:14px;
            overflow:hidden;
        ">
            <img
                src="{data_url}"
                alt="{safe_caption}"
                style="
                    display:block;
                    width:100%;
                    height:{height}px;
                    object-fit:contain;
                    background:#F6F8F3;
                    border-radius:10px;
                "
            />
            <div style="
                padding-top:11px;
                color:#526558;
                font-size:13px;
                overflow-wrap:anywhere;
            ">{safe_caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    .stApp {
        background:#F5F7F1;
        color:#20392D;
        font-family:'DM Sans',sans-serif;
    }

    [data-testid="stSidebar"] {
        background:#173D32;
    }

    [data-testid="stSidebar"] * {
        color:#F2F7F0;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color:#173D32;
    }

    .brand {
        font-family:'Manrope',sans-serif;
        font-size:25px;
        font-weight:800;
        letter-spacing:-0.8px;
        color:white;
    }

    .brand-note {
        color:#C2D5C7;
        font-size:12px;
        margin-top:5px;
        margin-bottom:22px;
    }

    .hero {
        background:linear-gradient(120deg,#173D32,#2D6B4D);
        border-radius:22px;
        padding:35px;
        margin-bottom:24px;
    }

    .eyebrow {
        color:#C5E8B7;
        font-size:12px;
        font-weight:700;
        letter-spacing:2px;
        margin-bottom:12px;
    }

    .hero h1 {
        font-family:'Manrope',sans-serif;
        font-size:clamp(30px,4vw,46px);
        line-height:1.15;
        letter-spacing:-1.5px;
        font-weight:800;
        color:white;
        margin:0 0 13px 0;
    }

    .hero p {
        color:#E0EBE1;
        font-size:15px;
        line-height:1.8;
        max-width:730px;
        margin:0;
    }

    .section-title {
        font-family:'Manrope',sans-serif;
        color:#173D32;
        font-weight:800;
        font-size:23px;
        margin:18px 0 5px 0;
    }

    .section-note {
        color:#708073;
        font-size:14px;
        margin-bottom:17px;
    }

    .metric {
        background:white;
        border:1px solid #DFE8DD;
        border-radius:15px;
        padding:19px;
        min-height:105px;
    }

    .metric-label {
        font-size:11px;
        letter-spacing:1px;
        font-weight:700;
        color:#778579;
        margin-bottom:10px;
    }

    .metric-value {
        color:#173D32;
        font-family:'Manrope',sans-serif;
        font-size:22px;
        font-weight:800;
        overflow-wrap:anywhere;
    }

    .metric-note {
        font-size:12px;
        color:#778579;
        margin-top:6px;
    }

    .info-panel {
        background:white;
        border:1px solid #DFE8DD;
        border-radius:16px;
        padding:20px;
        margin:8px 0 18px 0;
    }

    .notice {
        background:#EAF2E5;
        border-left:4px solid #65965A;
        border-radius:8px;
        padding:15px;
        color:#405B44;
        font-size:13px;
        line-height:1.7;
        margin-top:18px;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background:#246447;
        color:white;
        border:0;
        border-radius:10px;
        font-weight:700;
        min-height:44px;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background:#173D32;
        color:white;
        border:0;
    }

    [data-testid="stFileUploader"] {
        background:white;
        border:1px dashed #B7CBB8;
        border-radius:13px;
        padding:12px;
    }

    .footer {
        text-align:center;
        color:#839084;
        font-size:12px;
        padding:28px 0 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# MODEL LOADING AND PREDICTION
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_data(show_spinner=False)
def load_classes(class_path):
    with open(class_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("class_names", list(data.values()))

    return list(data)


def predict(model, image, model_key, class_names):
    resized = image.convert("RGB").resize((224, 224))
    array = np.asarray(resized, dtype=np.float32)

    if model_key == "mobilenetv2":
        array = tf.keras.applications.mobilenet_v2.preprocess_input(array)

    # EfficientNetB0 applies rescaling internally.
    batch = np.expand_dims(array, axis=0)
    scores = model.predict(batch, verbose=0)[0]

    indices = np.argsort(scores)[::-1][:3]

    return [
        (class_names[int(i)], float(scores[i]) * 100)
        for i in indices
    ]


# =========================================================
# GRAD-CAM USING THE EXISTING SCRIPT
# =========================================================

def create_gradcam(uploaded_file, model_key):
    if not GRADCAM_SCRIPT.exists():
        raise FileNotFoundError(f"Grad-CAM script missing: {GRADCAM_SCRIPT}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(uploaded_file.name).name
    stem = Path(original_name).stem

    # Use a unique filename so two uploads with the same name
    # cannot accidentally reuse a previous Grad-CAM image.
    import uuid
    unique_stem = f"{stem}_{uuid.uuid4().hex[:8]}"
    temp_name = f"{unique_stem}{Path(original_name).suffix}"

    output_path = REPORTS_DIR / f"gradcam_{model_key}_{unique_stem}.png"

    with tempfile.TemporaryDirectory(prefix="leaflens_") as temp_dir:
        input_path = Path(temp_dir) / temp_name
        input_path.write_bytes(uploaded_file.getvalue())

        result = subprocess.run(
            [
                sys.executable,
                str(GRADCAM_SCRIPT),
                "--model", model_key,
                "--image", str(input_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                result.stderr.strip()
                or result.stdout.strip()
                or "Grad-CAM generation failed."
            )

        # The existing script names the output using the input image stem.
        if not output_path.exists():
            raise FileNotFoundError(
                f"Expected Grad-CAM output was not created: {output_path}\n"
                f"Script output:\n{result.stdout}"
            )

    return output_path


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown('<div class="brand">🌿 LeafLens AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-note">Plant health · Computer vision · Explainable AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### Model settings")

    selected_model = st.selectbox(
        "Classification model",
        list(MODELS.keys()),
        index=0,
    )

    model_info = MODELS[selected_model]

    st.markdown(
        f"""
        <div style="
            background:rgba(255,255,255,.08);
            border:1px solid rgba(255,255,255,.12);
            border-radius:13px;padding:15px;margin-top:10px;
        ">
            <div style="font-size:11px;color:#C5E8B7;letter-spacing:1px;">
                OFFICIAL TEST ACCURACY
            </div>
            <div style="font-size:29px;font-weight:800;margin:6px 0;">
                {model_info["accuracy"]}
            </div>
            <div style="font-size:12px;color:#D8E6DB;">
                {model_info["description"]}
            </div>
            <div style="font-size:12px;color:#D8E6DB;margin-top:8px;">
                Model size: {model_info["size"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workflow")
    st.markdown(
        "1. Upload a leaf image\n\n"
        "2. Predict its category\n\n"
        "3. Review top predictions\n\n"
        "4. Explore Grad-CAM"
    )

    st.markdown("---")
    st.caption(
        "Research prototype trained on PlantVillage. "
        "Results may differ on real-world field images."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">PLANT DISEASE CLASSIFICATION · 38 CLASSES</div>
        <h1>Understand what your leaf is telling you.</h1>
        <p>
            Classify a plant leaf with deep learning and explore which image
            regions influence the model's prediction through Grad-CAM.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-title">01 / Upload a leaf image</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Choose a clear JPG, JPEG, or PNG photograph.</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    key="leaflens_upload",
)

if uploaded is None:
    st.markdown(
        """
        <div class="info-panel" style="text-align:center;padding:35px 20px;">
            <div style="font-size:38px;">🌱</div>
            <div style="font-size:20px;font-weight:800;color:#173D32;margin-top:8px;">
                Your analysis starts here
            </div>
            <div style="font-size:13px;color:#708073;margin-top:8px;">
                Upload a leaf photograph to preview the image and run the model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    try:
        uploaded_image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")

        # Show uploaded image immediately, before model analysis.
        st.markdown("#### Uploaded image preview")
        show_image(uploaded_image, uploaded.name, height=350)

        st.caption(
            f"Image size: {uploaded_image.width} × {uploaded_image.height} pixels"
        )

        analyze = st.button(
            "🌿 Analyze leaf",
            type="primary",
            width="stretch",
        )

        if analyze:
            if not model_info["path"].exists():
                st.error(f"Model file not found: {model_info['path']}")
                st.stop()

            if not CLASS_FILE.exists():
                st.error(f"Class names file not found: {CLASS_FILE}")
                st.stop()

            with st.spinner("Running model prediction..."):
                model = load_model(str(model_info["path"]))
                class_names = load_classes(str(CLASS_FILE))

                if len(class_names) != int(model.output_shape[-1]):
                    st.error(
                        "Class names do not match the model output size. "
                        "Check reports/class_names.json."
                    )
                    st.stop()

                predictions = predict(
                    model,
                    uploaded_image,
                    model_info["key"],
                    class_names,
                )

            st.markdown("---")
            st.markdown(
                '<div class="section-title">02 / Prediction results</div>',
                unsafe_allow_html=True,
            )

            first_name, first_score = predictions[0]
            c1, c2, c3 = st.columns(3)

            metrics = [
                ("PREDICTED CLASS", first_name, "Highest model score"),
                ("MODEL SCORE", f"{first_score:.2f}%", "Softmax output"),
                ("AVAILABLE CLASSES", str(len(class_names)), "Classification categories"),
            ]

            for col, metric in zip([c1, c2, c3], metrics):
                with col:
                    st.markdown(
                        f"""
                        <div class="metric">
                            <div class="metric-label">{metric[0]}</div>
                            <div class="metric-value">{html.escape(metric[1])}</div>
                            <div class="metric-note">{metric[2]}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            st.markdown("### Top 3 predictions")
            for rank, (class_name, score) in enumerate(predictions, start=1):
                st.markdown(f"**{rank}. {class_name}** — {score:.2f}%")
                st.progress(min(max(score / 100, 0.0), 1.0))

            st.markdown("---")
            st.markdown(
                '<div class="section-title">03 / Visual explanation</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="section-note">Compare the original leaf with the Grad-CAM overlay.</div>',
                unsafe_allow_html=True,
            )

            try:
                with st.spinner("Generating Grad-CAM visualization..."):
                    gradcam_path = create_gradcam(
                        uploaded,
                        model_info["key"],
                    )

                gradcam_image = Image.open(gradcam_path).convert("RGB")

                left, right = st.columns(2, gap="large")

                with left:
                    st.markdown("#### Original image")
                    show_image(uploaded_image, "Uploaded leaf", height=390)

                with right:
                    st.markdown("#### Grad-CAM overlay")
                    show_image(
                        gradcam_image,
                        "Regions highlighted by Grad-CAM",
                        height=390,
                    )

                st.download_button(
                    "Download Grad-CAM image",
                    data=gradcam_path.read_bytes(),
                    file_name=gradcam_path.name,
                    mime="image/png",
                    width="stretch",
                )

            except Exception as error:
                st.error("Prediction succeeded, but Grad-CAM could not be generated.")
                st.code(str(error))

            st.markdown(
                """
                <div class="notice">
                    <b>Research limitation:</b> Model scores do not guarantee
                    real-world correctness. PlantVillage images are commonly
                    collected in controlled conditions. Verify findings with
                    field observations or a qualified agricultural expert.
                </div>
                """,
                unsafe_allow_html=True,
            )

    except Exception as error:
        st.error("Unable to open or process the uploaded image.")
        st.exception(error)


st.markdown(
    """
    <div class="footer">
        LeafLens AI · Deep learning for plant leaf classification · Research prototype
    </div>
    """,
    unsafe_allow_html=True,
)
