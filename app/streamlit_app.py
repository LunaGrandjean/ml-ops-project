"""Interface légère de test : les prédictions viennent uniquement de l'API."""

import os

import requests
import streamlit as st

st.set_page_config(page_title="Trajectory Classifier", page_icon="🚗")
st.title("Classification de trajectoires")
st.caption("Safe · Safe Overall · Bad Weather · Tandem · Reckless")
api_url = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
image = st.file_uploader("Choisir une image de trajectoire", type=["png", "jpg", "jpeg"])
if image is not None:
    st.image(image, caption=image.name, width=400)
    if st.button("Classifier", type="primary"):
        try:
            response = requests.post(
                f"{api_url}/predict",
                files={"file": (image.name, image.getvalue(), image.type)},
                timeout=60,
            )
            response.raise_for_status()
            result = response.json()
            st.success(f"{result['label']} — score : {result['confidence']:.1%}")
            st.bar_chart(result["probabilities"])
            st.caption(f"Version du modèle : {result['model_version']}")
            st.caption("Le score softmax n'est pas une probabilité de confiance calibrée.")
        except requests.RequestException as exc:
            st.error(f"API indisponible ou image refusée : {exc}")
