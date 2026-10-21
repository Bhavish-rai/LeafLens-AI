
import base64
import hashlib
import html
import io
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps, UnidentifiedImageError


# =========================================================
# CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CLASS_FILE = REPORTS_DIR / "class_names.json"
GRADCAM_SCRIPT = ROOT / "src" / "gradcam.py"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Load local development secrets without placing them in source code.
load_dotenv(ROOT / ".env")

IMAGE_SIZE = (224, 224)
MAX_UPLOAD_MB = 15

MODELS = {
    "EfficientNetB0": {
        "key": "efficientnetb0",
        "path": MODELS_DIR / "leaflens_efficientnetb0_best.keras",
        "accuracy": "96.54%",
        "size": "16.73 MB",
        "parameters": "4.10M",
        "latency": "58.68 ms/image",
        "description": "Higher measured test accuracy",
    },
    "MobileNetV2": {
        "key": "mobilenetv2",
        "path": MODELS_DIR / "leaflens_mobilenetv2_best.keras",
        "accuracy": "95.16%",
        "size": "9.69 MB",
        "parameters": "2.31M",
        "latency": "26.21 ms/image",
        "description": "Smaller and faster in the measured CPU benchmark",
    },
}

st.set_page_config(
    page_title="LeafLens AI | Plant Health Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# VISUAL DESIGN
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --green: #245B43;
        --deep-green: #173D2E;
        --mint: #E8F1E8;
        --canvas: #F5F7F2;
        --ink: #20382D;
        --muted: #738176;
        --border: #DFE7DC;
    }

    .stApp {
        background: var(--canvas);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stSidebar"] {
        background: var(--deep-green);
    }

    [data-testid="stSidebar"] * {
        color: #F3F7F1;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #173D2E;
    }

    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #D2E1D4;
    }

    .brand {
        font: 800 26px 'Manrope', sans-serif;
        letter-spacing: -1px;
        color: white;
    }

    .brand-subtitle {
        font-size: 12px;
        line-height: 1.7;
        color: #C9DCCB;
        margin-top: 5px;
    }

    .hero {
        background: linear-gradient(120deg, #173D2E 0%, #2D704D 100%);
        border-radius: 22px;
        padding: 36px;
        margin: 4px 0 26px;
        box-shadow: 0 10px 28px rgba(23, 61, 46, 0.09);
    }

    .eyebrow {
        color: #C7E7B8;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 13px;
    }

    .hero h1 {
        color: white;
        font: 800 clamp(30px, 4vw, 46px)/1.15 'Manrope', sans-serif;
        letter-spacing: -1.5px;
        margin: 0 0 15px;
    }

    .hero p {
        color: #E1EBE1;
        font-size: 15px;
        line-height: 1.8;
        max-width: 740px;
        margin: 0;
    }

    .section-title {
        font: 800 22px 'Manrope', sans-serif;
        color: var(--deep-green);
        margin: 18px 0 4px;
    }

    .section-note {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
        margin-bottom: 17px;
    }

    .panel {
        background: white;
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px;
        margin: 8px 0 16px;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 15px;
        padding: 19px;
        min-height: 120px;
        overflow-wrap: anywhere;
    }

    .metric-label {
        color: #778579;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: var(--deep-green);
        font: 800 21px 'Manrope', sans-serif;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: var(--muted);
        font-size: 11px;
        line-height: 1.6;
        margin-top: 7px;
    }

    .result-banner {
        background: #E8F2E5;
        border: 1px solid #D4E4CE;
        border-radius: 15px;
        padding: 19px;
        margin: 12px 0 18px;
    }

    .result-label {
        color: #55704F;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.2px;
    }

    .result-name {
        color: #204B32;
        font: 800 clamp(20px, 3vw, 29px) 'Manrope', sans-serif;
        margin-top: 7px;
        overflow-wrap: anywhere;
    }

    .notice {
        background: #EAF1E5;
        border-left: 4px solid #69965C;
        border-radius: 8px;
        padding: 15px;
        color: #425A43;
        font-size: 12px;
        line-height: 1.8;
        margin-top: 20px;
    }

    .footer {
        text-align: center;
        color: #839084;
        font-size: 11px;
        padding: 30px 0 12px;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: #246447;
        color: white;
        border: 1px solid #246447;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #173D2E;
        color: white;
        border-color: #173D2E;
    }

    [data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #B7CBB8;
        border-radius: 13px;
        padding: 12px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 25px 21px;
            border-radius: 17px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# IMAGE HELPERS
# =========================================================

def image_to_base64(image):
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def show_image(image, caption, height=360):
    """Embed images directly to avoid depending on Streamlit image routes."""
    data_url = image_to_base64(image)
    safe_caption = html.escape(str(caption))

    st.markdown(
        f"""
        <div style="
            background:#FFFFFF;
            border:1px solid #DFE7DC;
            border-radius:15px;
            padding:12px;
            overflow:hidden;">
            <img
                src="data:image/png;base64,{data_url}"
                alt="{safe_caption}"
                style="
                    display:block;
                    width:100%;
                    height:{height}px;
                    object-fit:contain;
                    background:#F6F8F3;
                    border-radius:9px;">
            <div style="
                padding-top:10px;
                color:#526558;
                font-size:12px;
                overflow-wrap:anywhere;">
                {safe_caption}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def image_fingerprint(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


# =========================================================
# MODEL LOADING
# =========================================================

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    return tf.keras.models.load_model(path, compile=False)


@st.cache_data(show_spinner=False)
def load_classes(class_path):
    with open(class_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        if "class_names" in data:
            data = data["class_names"]
        else:
            data = list(data.values())

    if not isinstance(data, list) or not data:
        raise ValueError("class_names.json must contain a non-empty class list.")

    return [str(name) for name in data]


# =========================================================
# PREDICTION
# =========================================================

def predict(model, image, model_key, class_names):
    """
    Keep preprocessing consistent with the training pipelines:
    MobileNetV2 uses preprocess_input.
    EfficientNetB0 performs rescaling inside its model.
    """
    resized = ImageOps.fit(
        image.convert("RGB"),
        IMAGE_SIZE,
        method=Image.Resampling.BILINEAR,
    )
    array = np.asarray(resized, dtype=np.float32)

    if model_key == "mobilenetv2":
        array = tf.keras.applications.mobilenet_v2.preprocess_input(array)

    scores = np.asarray(
        model.predict(np.expand_dims(array, axis=0), verbose=0)[0],
        dtype=np.float64,
    ).reshape(-1)

    if len(scores) != len(class_names):
        raise ValueError(
            f"Model returned {len(scores)} outputs but "
            f"{len(class_names)} class names were loaded."
        )

    if not np.all(np.isfinite(scores)):
        raise ValueError("Model produced invalid prediction values.")

    # Convert logits to probabilities only if output does not already
    # look like a probability distribution.
    looks_like_probabilities = (
        np.all(scores >= 0)
        and np.all(scores <= 1)
        and np.isclose(np.sum(scores), 1.0, atol=0.02)
    )

    if not looks_like_probabilities:
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / np.sum(exp_scores)

    top_indices = np.argsort(scores)[::-1][:3]

    return [
        (class_names[int(index)], float(scores[index]) * 100)
        for index in top_indices
    ]


# =========================================================
# GEMINI DISEASE ASSISTANT
# =========================================================

@st.cache_resource(show_spinner=False)
def get_gemini_client(api_key):
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=60000),
    )


def get_gemini_api_key():
    """Read the key from Streamlit secrets or the local .env environment."""
    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception:
        secret_key = ""

    return secret_key or os.getenv("GEMINI_API_KEY", "")


def readable_disease_name(class_name):
    """Convert PlantVillage class labels into a more readable name."""
    return str(class_name).replace("___", " — ").replace("_", " ")


def ask_gemini(prompt):
    """Generate a response using the same Gemini API used by the successful test."""
    api_key = get_gemini_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Add GEMINI_API_KEY to your project .env file."
        )

    client = get_gemini_client(api_key)
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=900),
        )
    except Exception as error:
        raise RuntimeError(
            f"Gemini request failed ({type(error).__name__}): {error}"
        ) from error

    answer = getattr(response, "text", None)
    if not answer or not str(answer).strip():
        raise RuntimeError(
            "Gemini returned an empty response. Check the model name, API key, "
            "and API quota, then try again."
        )
    return str(answer).strip()


def build_disease_guide_prompt(disease_name, model_name, model_score):
    return f"""
You are the educational plant-health assistant inside LeafLens AI, a research
prototype that classifies leaf images using a machine-learning model.

The model's top prediction is: {disease_name}
Selected classifier: {model_name}
Classifier output score: {model_score:.2f}%

Create a useful, beginner-friendly disease information guide in Markdown using
these headings:
## Disease overview
## Causes and how it spreads
## Common symptoms
## What you can do
## Prevention and plant care
## When to seek expert help

Requirements:
- Treat the label as a model prediction, not a confirmed diagnosis.
- Explain that symptoms can overlap with other diseases, pests, or nutrient issues.
- Give practical, low-risk steps first, such as isolating affected plants when
  appropriate, removing heavily affected material safely, cleaning tools, avoiding
  overhead watering when suitable, and improving airflow where relevant.
- Distinguish fungal, bacterial, viral, and non-infectious conditions accurately.
  Do not describe every disease as fungal.
- Recommend a pesticide or fungicide only when appropriate for this specific
  disease; avoid inventing product names, chemical doses, or application rates.
  Tell users to follow local agricultural guidance and the product label.
- Do not promise a cure. Explain that management depends on the crop, severity,
  and local conditions.
- If the label is a healthy class, explain that no disease was predicted and give
  general care advice rather than inventing a disease.
- Use clear, concise language. Do not claim you examined the image yourself.
"""


def build_followup_prompt(disease_name, guide, history, question):
    conversation = []
    for item in history[-8:]:
        role = "User" if item["role"] == "user" else "Assistant"
        conversation.append(f"{role}: {item['content']}")

    previous = "\n".join(conversation) if conversation else "No follow-up questions yet."
    return f"""
You are LeafLens AI's educational plant-health assistant.
The classifier's predicted label is "{disease_name}". This is not a confirmed
diagnosis. Use the following disease guide as context, but correct any obvious
inconsistency rather than inventing facts:

{guide}

Recent conversation:
{previous}

User's new question: {question}

Answer in simple language, directly and practically. Do not guarantee a cure,
invent chemical dosages, or replace local agricultural expertise. If the question
needs crop, location, or severity details to answer safely, say what is missing.
"""


# =========================================================
# GRAD-CAM
# =========================================================

def create_gradcam(image_bytes, original_filename, model_key):
    if not GRADCAM_SCRIPT.is_file():
        raise FileNotFoundError(f"Grad-CAM script missing: {GRADCAM_SCRIPT}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    original_name = Path(original_filename).name
    suffix = Path(original_name).suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png"}:
        suffix = ".png"

    unique_stem = f"{Path(original_name).stem}_{uuid.uuid4().hex[:8]}"
    input_filename = f"{unique_stem}{suffix}"
    output_path = REPORTS_DIR / f"gradcam_{model_key}_{unique_stem}.png"

    with tempfile.TemporaryDirectory(prefix="leaflens_") as temp_dir:
        input_path = Path(temp_dir) / input_filename
        input_path.write_bytes(image_bytes)

        result = subprocess.run(
            [
                sys.executable,
                str(GRADCAM_SCRIPT),
                "--model",
                model_key,
                "--image",
                str(input_path),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(details or "Grad-CAM generation failed.")

        if not output_path.is_file():
            raise FileNotFoundError(
                f"Expected Grad-CAM image was not created:\n{output_path}\n\n"
                f"Script output:\n{result.stdout}"
            )

    return output_path


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        '<div class="brand">🌿 LeafLens AI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="brand-subtitle">'
        'Plant health · Deep learning · Explainable AI'
        '</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### Model settings")

    selected_model = st.selectbox(
        "Classification model",
        options=list(MODELS.keys()),
        key="selected_model",
    )

    model_info = MODELS[selected_model]

    st.markdown(
        f"""
        <div style="
            background:rgba(255,255,255,.08);
            border:1px solid rgba(255,255,255,.14);
            border-radius:13px;
            padding:15px;
            margin-top:8px;">
            <div style="font-size:10px;color:#C7E7B8;letter-spacing:1px;">
                REPORTED TEST ACCURACY
            </div>
            <div style="font-size:29px;font-weight:800;margin:7px 0;">
                {model_info["accuracy"]}
            </div>
            <div style="font-size:12px;color:#E0EBE1;line-height:1.6;">
                {model_info["description"]}
            </div>
            <div style="font-size:12px;color:#D2E1D4;margin-top:10px;">
                Model size: {model_info["size"]}
            </div>
            <div style="font-size:12px;color:#D2E1D4;margin-top:4px;">
                Parameters: {model_info["parameters"]}
            </div>
            <div style="font-size:12px;color:#D2E1D4;margin-top:4px;">
                Measured CPU latency: {model_info["latency"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Workflow")
    st.markdown(
        "1. Upload a leaf image\n\n"
        "2. Select a model\n\n"
        "3. Analyze the prediction\n\n"
        "4. Review the top three classes\n\n"
        "5. Explore Grad-CAM"
    )

    st.divider()
    st.caption(
        "Research prototype trained on PlantVillage. "
        "Performance on real-world field photographs may differ."
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">
            PLANT DISEASE CLASSIFICATION · 38 CLASSES
        </div>
        <h1>Understand what your leaf is telling you.</h1>
        <p>
            Analyze a plant leaf using deep learning, compare the leading
            predicted classes, and inspect image regions highlighted by
            Grad-CAM to explore the model's visual attention.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">01 / Upload a leaf image</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-note">'
    'Choose a clear JPG, JPEG, or PNG image. For best results, '
    'use a well-lit image with the leaf clearly visible.'
    '</div>',
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Choose a leaf photograph",
    type=["jpg", "jpeg", "png"],
    key="leaflens_upload",
)

# Track the image and model used for the last successful analysis.
if uploaded is not None:
    image_bytes = uploaded.getvalue()
    current_fingerprint = image_fingerprint(image_bytes)
    current_analysis_key = f"{current_fingerprint}:{model_info['key']}"

    if len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Please upload an image smaller than {MAX_UPLOAD_MB} MB.")
        st.stop()

    try:
        uploaded_image = Image.open(io.BytesIO(image_bytes))
        uploaded_image = ImageOps.exif_transpose(uploaded_image).convert("RGB")
        uploaded_image.load()
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This image could not be opened. Please try another JPG or PNG.")
        st.stop()

    preview_col, details_col = st.columns([1.05, 0.95], gap="large")

    with preview_col:
        st.markdown("#### Image preview")
        show_image(uploaded_image, uploaded.name, height=330)

    with details_col:
        st.markdown("#### Image details")
        st.markdown(
            f"""
            <div class="panel">
                <div style="color:#738176;font-size:11px;font-weight:700;">
                    SELECTED MODEL
                </div>
                <div style="font:800 21px 'Manrope',sans-serif;
                            color:#173D2E;margin:6px 0 15px;">
                    {html.escape(selected_model)}
                </div>
                <div style="color:#738176;font-size:11px;font-weight:700;">
                    IMAGE DIMENSIONS
                </div>
                <div style="font-size:17px;font-weight:700;margin:6px 0 15px;">
                    {uploaded_image.width} × {uploaded_image.height} px
                </div>
                <div style="color:#738176;font-size:11px;font-weight:700;">
                    FILE SIZE
                </div>
                <div style="font-size:17px;font-weight:700;margin-top:6px;">
                    {len(image_bytes) / (1024 * 1024):.2f} MB
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        analyze = st.button(
            "🌿 Analyze leaf",
            type="primary",
            use_container_width=True,
        )

    if analyze:
        model_path = model_info["path"]

        if not model_path.is_file():
            st.error(f"Model file not found: {model_path}")
            st.stop()

        if not CLASS_FILE.is_file():
            st.error(f"Class names file not found: {CLASS_FILE}")
            st.stop()

        try:
            with st.spinner(f"Analyzing image with {selected_model}..."):
                model = load_model(str(model_path))
                class_names = load_classes(str(CLASS_FILE))
                predictions = predict(
                    model,
                    uploaded_image,
                    model_info["key"],
                    class_names,
                )

            st.session_state["leaflens_analysis"] = {
                "key": current_analysis_key,
                "predictions": predictions,
                "gradcam_path": None,
                "gradcam_error": None,
            }

        except Exception as error:
            st.error("Prediction failed. Check the model and class-name files.")
            st.exception(error)

    # Reuse the result on ordinary Streamlit reruns, but never show a result
    # belonging to a different image or model.
    analysis = st.session_state.get("leaflens_analysis")

    if analysis and analysis.get("key") == current_analysis_key:
        predictions = analysis["predictions"]
        predicted_class, top_score = predictions[0]

        st.divider()
        st.markdown(
            '<div class="section-title">02 / Prediction results</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-note">'
            'These are the model\'s highest-scoring classes, not a confirmed '
            'agricultural diagnosis.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="result-banner">
                <div class="result-label">TOP PREDICTED CLASS</div>
                <div class="result-name">{html.escape(predicted_class)}</div>
                <div style="color:#55704F;font-size:12px;margin-top:8px;">
                    Model score: {top_score:.2f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        metric_cols = st.columns(3)
        metric_data = [
            ("SELECTED MODEL", selected_model, "Classifier used"),
            ("TOP MODEL SCORE", f"{top_score:.2f}%", "Highest output score"),
            ("CLASS COUNT", str(len(load_classes(str(CLASS_FILE)))),
             "Available categories"),
        ]

        for col, (label, value, note) in zip(metric_cols, metric_data):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">{html.escape(label)}</div>
                        <div class="metric-value">{html.escape(value)}</div>
                        <div class="metric-note">{html.escape(note)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("### Top 3 predictions")

        chart_labels = []
        chart_scores = []

        for rank, (class_name, score) in enumerate(predictions, start=1):
            st.markdown(f"**{rank}. {class_name}**")
            st.progress(min(max(score / 100.0, 0.0), 1.0))
            st.caption(f"Model score: {score:.2f}%")
            chart_labels.append(class_name)
            chart_scores.append(score)

        chart_data = {
            "Class": chart_labels,
            "Score (%)": chart_scores,
        }
        st.bar_chart(
            chart_data,
            x="Class",
            y="Score (%)",
            horizontal=True,
        )

        st.caption(
            "Model scores are not guaranteed to be calibrated probabilities "
            "and should not be interpreted as the likelihood of disease "
            "in real-world conditions."
        )

        # -----------------------------------------------------
        # AI disease information and follow-up questions
        # -----------------------------------------------------
        st.divider()
        st.markdown(
            '<div class="section-title">03 / AI Disease Assistant</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-note">'
            'Learn about the predicted condition, its causes, symptoms, '
            'management, and prevention. This is educational guidance, not a '
            'confirmed diagnosis.'
            '</div>',
            unsafe_allow_html=True,
        )

        disease_name = readable_disease_name(predicted_class)
        ai_state_key = f"{current_analysis_key}:{predicted_class}"
        stored_ai_key = st.session_state.get("leaflens_ai_key")

        if stored_ai_key != ai_state_key:
            st.session_state["leaflens_ai_key"] = ai_state_key
            st.session_state["leaflens_ai_guide"] = None
            st.session_state["leaflens_ai_history"] = []

        if st.button(
            "🌱 Explain this disease with AI",
            use_container_width=True,
            key=f"ai_guide_{current_analysis_key}",
        ):
            guide_prompt = build_disease_guide_prompt(
                disease_name,
                selected_model,
                top_score,
            )
            try:
                with st.spinner("Gemini is preparing a plant-health guide..."):
                    guide_text = ask_gemini(guide_prompt)
                st.session_state["leaflens_ai_guide"] = guide_text
                st.session_state["leaflens_ai_history"] = []
            except Exception as error:
                st.error(
                    "The AI guide could not be generated. Your disease prediction "
                    "is still available."
                )
                st.caption(str(error))

        guide_text = st.session_state.get("leaflens_ai_guide")
        if guide_text:
            with st.container(border=True):
                st.markdown(guide_text)

            st.markdown("#### Ask a follow-up question")
            with st.form(key=f"ai_question_form_{current_analysis_key}"):
                followup_question = st.text_input(
                    "Your question",
                    placeholder="e.g. How can I prevent this from spreading?",
                    max_chars=500,
                )
                ask_question = st.form_submit_button(
                    "Ask Gemini",
                    use_container_width=True,
                )

            if ask_question:
                question = followup_question.strip()
                if not question:
                    st.warning("Enter a question first.")
                else:
                    history = st.session_state.get("leaflens_ai_history", [])
                    prompt = build_followup_prompt(
                        disease_name,
                        guide_text,
                        history,
                        question,
                    )
                    try:
                        with st.spinner("Gemini is preparing an answer..."):
                            answer = ask_gemini(prompt)
                        history.extend([
                            {"role": "user", "content": question},
                            {"role": "assistant", "content": answer},
                        ])
                        st.session_state["leaflens_ai_history"] = history[-12:]
                    except Exception as error:
                        st.error(
                            "Gemini could not answer that question. "
                            "Your prediction and disease guide are still available."
                        )
                        st.caption(str(error))

            for item in st.session_state.get("leaflens_ai_history", []):
                if item["role"] == "user":
                    st.markdown(f"**You asked:** {item['content']}")
                else:
                    st.markdown("**LeafLens AI:**")
                    st.markdown(item["content"])

            st.caption(
                "AI-generated educational information may be incomplete. "
                "Confirm diagnosis and treatment choices with a qualified "
                "agricultural expert, especially before applying chemicals."
            )

        st.divider()
        st.markdown(
            '<div class="section-title">04 / Explainable AI</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-note">'
            'Grad-CAM highlights image regions that influence the model\'s '
            'class-specific output. The heatmap is an explanation aid, '
            'not proof that a disease is present.'
            '</div>',
            unsafe_allow_html=True,
        )

        if st.button(
            "Generate Grad-CAM explanation",
            use_container_width=True,
            key=f"gradcam_{current_analysis_key}",
        ):
            try:
                with st.spinner("Generating Grad-CAM visualization..."):
                    gradcam_path = create_gradcam(
                        image_bytes,
                        uploaded.name,
                        model_info["key"],
                    )

                st.session_state["leaflens_analysis"]["gradcam_path"] = str(
                    gradcam_path
                )
                st.session_state["leaflens_analysis"]["gradcam_error"] = None

            except Exception as error:
                st.session_state["leaflens_analysis"]["gradcam_error"] = str(error)

        analysis = st.session_state.get("leaflens_analysis", {})
        gradcam_path_string = analysis.get("gradcam_path")

        if gradcam_path_string:
            gradcam_path = Path(gradcam_path_string)

            if gradcam_path.is_file():
                try:
                    gradcam_image = Image.open(gradcam_path).convert("RGB")
                    original_col, heatmap_col = st.columns(2, gap="large")

                    with original_col:
                        st.markdown("#### Original leaf")
                        show_image(uploaded_image, "Original uploaded image")

                    with heatmap_col:
                        st.markdown("#### Grad-CAM visualization")
                        show_image(
                            gradcam_image,
                            "Model-focused regions",
                        )

                    st.download_button(
                        "Download Grad-CAM image",
                        data=gradcam_path.read_bytes(),
                        file_name=gradcam_path.name,
                        mime="image/png",
                        use_container_width=True,
                    )
                except Exception as error:
                    st.error("The Grad-CAM image could not be displayed.")
                    st.exception(error)

        gradcam_error = analysis.get("gradcam_error")
        if gradcam_error:
            st.error("Grad-CAM generation failed.")
            st.code(gradcam_error)

        st.markdown(
            """
            <div class="notice">
                <b>Research limitation:</b> LeafLens AI is a research prototype
                trained on PlantVillage. Images captured in controlled settings
                may differ from field photographs in lighting, backgrounds,
                camera quality, and disease severity. Do not use this output
                alone to decide treatment. Confirm findings with field
                observations or a qualified agricultural expert.
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    # Clear stale results when the user removes the uploaded image.
    st.session_state.pop("leaflens_analysis", None)

    st.markdown(
        """
        <div class="panel" style="text-align:center;padding:35px 20px;">
            <div style="font-size:40px;">🌱</div>
            <div style="
                font:800 20px 'Manrope',sans-serif;
                color:#173D2E;margin-top:10px;">
                Your analysis starts here
            </div>
            <div style="
                color:#738176;font-size:13px;line-height:1.8;
                max-width:540px;margin:9px auto 0;">
                Upload a clear leaf image to inspect its predicted category,
                compare the leading model scores, and generate a Grad-CAM
                visualization.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
        LeafLens AI · Plant disease classification · Research prototype
    </div>
    """,
    unsafe_allow_html=True,
)
