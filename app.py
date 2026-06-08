import streamlit as st
import numpy as np
from PIL import Image

st.title("🧠 CNN Image Classifier")
st.write("Upload image to test model")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", width=250)

    st.info("⚠ Model loading skipped for deployment stability demo")

    st.success("App is working! (Fix model in backend later)")
