import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow import keras
model = keras.models.load_model("cnn_model.keras")

classes = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

st.title("🧠 CNN Image Classification App")
st.write("Upload an image and get prediction")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", width=250)

    img = image.resize((32,32))
    img = np.array(img)/255.0
    img = img.reshape(1,32,32,3)

    prediction = model.predict(img)
    result = classes[np.argmax(prediction)]

    st.success(f"Prediction: {result}")
