import gradio as gr
import numpy as np
from tensorflow import keras
from PIL import Image

# load model
model = keras.models.load_model("cnn_model.keras")

classes = ["airplane","automobile","bird","cat","deer",
           "dog","frog","horse","ship","truck"]

def predict(img):
    img = img.convert("RGB")
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    return classes[np.argmax(pred)]

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="CNN Image Classifier",
    description="Upload an image and get prediction"
)

demo.launch()
