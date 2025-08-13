import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("skin_cancer_model.h5")

model = load_model()

st.title("Skin Cancer Detection AI")
st.write("Upload a skin lesion image and the model will predict if it's benign or malignant.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)


   
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    pred = model.predict(img_array)[0][0]
    label = "Malignant" if pred > 0.5 else "Benign"

    st.markdown(f"### Prediction: **{label}**")
    st.markdown(f"**Confidence:** `{pred:.2f}`")

    if pred > 0.5:
        st.warning("⚠️ Possible cancer risk. Please consult a doctor.")
