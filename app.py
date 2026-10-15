import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications import efficientnet, mobilenet_v2, resnet50

ARTIFACTS = Path(__file__).parent / "artifacts"

# نفس دوال الـ preprocessing المستخدمة وقت التدريب في النوتبوك
PREPROCESS = {
    "mobilenetv2": mobilenet_v2.preprocess_input,
    "resnet50": resnet50.preprocess_input,
    "efficientnetb0": efficientnet.preprocess_input,
    "custom_cnn": lambda x: x / 255.0,
}

st.set_page_config(page_title="Fruit Classifier", page_icon="🍎", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_assets():
    model = keras.models.load_model(ARTIFACTS / "fruit_model.keras", compile=False)
    class_names = json.loads((ARTIFACTS / "class_names.json").read_text())
    config = json.loads((ARTIFACTS / "config.json").read_text())
    return model, class_names, config


@st.cache_data
def read_csv(name):
    p = ARTIFACTS / name
    return pd.read_csv(p, index_col=0) if p.exists() else None


def to_rgb(image: Image.Image) -> Image.Image:
    """الداتا ست خلفيتها بيضا، فالصور الشفافة بتتحط على أبيض مش أسود."""
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.alpha_composite(image)
        return bg.convert("RGB")
    return image.convert("RGB")


def prepare(image: Image.Image, size: int, preprocess) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    arr = tf.image.resize(arr, (size, size))
    arr = tf.cast(tf.round(arr), tf.float32).numpy()  # نفس load_img في النوتبوك
    return preprocess(arr)


def fruit_family(label: str) -> str:
    return re.sub(r"\s+\d+$", "", label)


try:
    model, class_names, config = load_assets()
except Exception as e:
    st.error(
        "Could not load the model. Make sure the `artifacts/` folder (next to app.py) "
        "contains `fruit_model.keras`, `class_names.json` and `config.json`."
    )
    st.exception(e)
    st.stop()

size = config["img_size"]
preprocess = PREPROCESS[config["backbone"]]

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Settings")
threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
top_k = st.sidebar.slider("Top-K predictions", 2, 10, 5)
st.sidebar.info(
    "Trained on Fruits-360: a single fruit on a white background, 100x100 px. "
    "Cluttered real-world photos or fruits outside the 70 classes may be misclassified."
)
st.sidebar.caption(
    f"Backbone: **{config['backbone']}**  \n"
    f"Test accuracy: **{config['test_accuracy']:.2%}**  \n"
    f"Test macro-F1: **{config['test_macro_f1']:.2%}**"
)

st.title("🍎 Fruit Classifier")
st.caption(f"{len(class_names)} classes · {config['backbone']} fine-tuned on Fruits-360")

tab_predict, tab_perf, tab_classes = st.tabs(["🔍 Predict", "📊 Model performance", "🍇 Classes"])

# ---------------------------------------------------------------- predict
with tab_predict:
    source = st.radio("Input", ["Upload images", "Camera"], horizontal=True)
    images = []
    if source == "Upload images":
        files = st.file_uploader(
            "Upload one or more fruit images", type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
        )
        for f in files or []:
            images.append((f.name, to_rgb(Image.open(f))))
    else:
        shot = st.camera_input("Take a photo of a fruit")
        if shot:
            images.append(("camera.jpg", to_rgb(Image.open(shot))))

    if not images:
        st.info("Upload an image (or take a photo) to get a prediction.")
    else:
        batch = np.stack([prepare(img, size, preprocess) for _, img in images])
        with st.spinner("Predicting..."):
            all_probs = model.predict(batch, verbose=0)

        rows = []
        for (name, img), probs in zip(images, all_probs):
            top = probs.argsort()[::-1][:top_k]
            best, conf = int(top[0]), float(probs[top[0]])
            rows.append({"file": name, "prediction": class_names[best],
                         "fruit": fruit_family(class_names[best]), "confidence": conf,
                         "status": "confident" if conf >= threshold else "uncertain"})

            st.divider()
            left, right = st.columns([1, 2])
            left.image(img, caption=name, width=260)
            with right:
                if conf >= threshold:
                    st.success(f"Prediction: **{class_names[best]}**")
                else:
                    st.warning(f"Uncertain. Closest match: **{class_names[best]}**")
                c1, c2 = st.columns(2)
                c1.metric("Confidence", f"{conf:.1%}")
                c2.metric("Fruit type", fruit_family(class_names[best]))
                st.progress(min(conf, 1.0))
                chart = pd.DataFrame({
                    "class": [class_names[i] for i in top],
                    "confidence": [float(probs[i]) for i in top],
                }).set_index("class")
                st.bar_chart(chart)

        if len(rows) > 1:
            st.divider()
            st.subheader("Batch summary")
            summary = pd.DataFrame(rows)
            st.dataframe(
                summary, hide_index=True,
                column_config={"confidence": st.column_config.ProgressColumn(
                    "confidence", min_value=0.0, max_value=1.0, format="%.1f%%")},
            )
            st.download_button("Download results (CSV)", summary.to_csv(index=False),
                               "predictions.csv", "text/csv")

# ---------------------------------------------------------------- performance
with tab_perf:
    metrics = read_csv("metrics_before_after.csv")
    comparison = read_csv("model_comparison.csv")
    report = read_csv("classification_report_after.csv")

    if metrics is None and comparison is None and report is None:
        st.info("Copy the CSV files from the notebook's `artifacts/` folder to see metrics here.")
    if metrics is not None:
        st.subheader("Before vs after fine-tuning")
        st.dataframe(metrics.style.format("{:.4f}"))
    if comparison is not None:
        st.subheader("Model comparison (validation)")
        st.dataframe(comparison.style.format("{:.4f}"))
        st.bar_chart(comparison[["accuracy", "f1"]])
    if report is not None:
        st.subheader("Weakest classes on the test set")
        rep = report.loc[[c for c in class_names if c in report.index]]
        st.dataframe(rep.sort_values("f1-score").head(10)[["precision", "recall", "f1-score", "support"]]
                     .style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                    "f1-score": "{:.3f}", "support": "{:.0f}"}))
        st.caption("Corn Husk and Eggplant are the weakest classes (often confused with "
                   "Dates, Pear Stone, Apple 9, Cactus fruit and Fig).")

# ---------------------------------------------------------------- classes
with tab_classes:
    q = st.text_input("Search classes")
    shown = [c for c in class_names if q.lower() in c.lower()]
    st.write(f"{len(shown)} of {len(class_names)} classes")
    st.dataframe(pd.DataFrame({"class": shown, "fruit": [fruit_family(c) for c in shown]}),
                 hide_index=True)
