import streamlit as st
import numpy as np
from tensorflow import keras
from PIL import Image

# Load model
model = keras.models.load_model("cnn_model.keras")

classes = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

st.title("🧠 CNN Image Classifier")
st.write("Upload an image and get prediction")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", width=250)

    # preprocess
    img = image.convert("RGB")
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction
    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]

    st.success(f"Prediction: {result}")
