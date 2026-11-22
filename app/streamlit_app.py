"""Trajectory demo: all inference is performed by the HTTP API."""

import hashlib
import os
from io import BytesIO

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from trajectory_mlops.ui_client import ApiClient, ApiError

st.set_page_config(page_title="Trajectory Classifier", page_icon="🚗", layout="centered")
st.title("Trajectory Classifier")
st.caption("Explore the five driving categories predicted from a trajectory window.")
st.markdown(
    "Upload a **PNG or JPEG trajectory image** produced by the project converter. "
    "Then select **Classify trajectory** to see all five class scores."
)

try:
    api = ApiClient(os.getenv("API_URL", "http://localhost:8000"))
    info = api.model_info()
    st.success(f"Model ready · version {info['model_version']}", icon="✅")
except ApiError as exc:
    info = None
    st.warning(str(exc))
    st.caption("Refresh this page once the API and its model are available.")

upload = st.file_uploader("Trajectory image", type=["png", "jpg", "jpeg"])
if upload is None:
    st.info("Choose a trajectory window to start.")
else:
    content = upload.getvalue()
    key = hashlib.sha256(content).hexdigest()
    if st.session_state.get("upload_key") != key:
        st.session_state.upload_key = key
        st.session_state.pop("prediction", None)
    valid = False
    try:
        if len(content) > (info["max_upload_bytes"] if info else 5 * 1024 * 1024):
            raise ValueError("The image exceeds the API upload limit.")
        with Image.open(BytesIO(content)) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise ValueError("Choose an actual PNG or JPEG image.")
            if image.width * image.height > (info["max_image_pixels"] if info else 4_000_000):
                raise ValueError("The image dimensions exceed the API pixel limit.")
            image.verify()
        st.image(content, caption=upload.name, width=400)
        valid = True
    except (ValueError, OSError, UnidentifiedImageError, Image.DecompressionBombError):
        st.error("This image is invalid or exceeds the upload limits. Choose another window.")

    if st.button("Classify trajectory", type="primary", disabled=not (info and valid)):
        st.session_state.pop("prediction", None)
        try:
            with st.spinner("Classifying the trajectory…"):
                result = api.predict(upload.name, content, upload.type)
            st.session_state.prediction = result.model_dump()
        except ApiError as exc:
            st.error(str(exc))

    if result := st.session_state.get("prediction"):
        left, right = st.columns(2)
        left.metric("Predicted class", result["label"])
        right.metric("Top class score", f"{result['confidence']:.1%}")
        scores = pd.DataFrame(
            [{"Class": name, "Score": score} for name, score in result["probabilities"].items()]
        ).sort_values("Score", ascending=False)
        st.bar_chart(scores, x="Class", y="Score", horizontal=True, sort=False)
        st.dataframe(
            scores,
            hide_index=True,
            column_config={
                "Score": st.column_config.ProgressColumn(
                    "Score", min_value=0, max_value=1, format="percent"
                )
            },
        )
        st.caption(f"Prediction served by model version {result['model_version']}.")
        if info and result["model_version"] != info["model_version"]:
            st.info("The API model changed. Classify again to refresh this result.")

with st.expander("How to interpret the scores"):
    st.write(
        "These are softmax scores, not calibrated confidence probabilities. "
        "The model always picks one of Safe, Safe Overall, Bad Weather, Tandem "
        "and Reckless. Unrelated images and unfamiliar trajectories can still "
        "receive a high score."
    )
    st.write(
        "The current model has limited results for minority classes. "
        "Review the evaluation and confusion matrix when comparing model versions."
    )
