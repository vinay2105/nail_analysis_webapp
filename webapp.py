import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import requests
import io

model = load_model('my_trained_model.keras')

class_names = [
    'acrall lentiginous melanoma',
    'blue finger',
    'onychogryphosis',
    'healthy nail',
    'clubbing',
    'pitting'
]

def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict(img):
    processed_img = preprocess_image(img)
    predictions = model.predict(processed_img)
    class_idx = np.argmax(predictions, axis=1)[0]
    return class_names[class_idx]

st.set_page_config(page_title="Nail Condition Classifier", layout="centered")
st.title("💅 Nail Condition Classifier")
st.markdown("Upload or paste an image URL of a nail to classify its condition.")

upload_option = st.radio("Choose input method:", ["Upload Image", "Image URL"])

image = None

if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
elif upload_option == "Image URL":
    image_url = st.text_input("Enter image URL")
    if image_url:
        try:
            response = requests.get(image_url)
            if response.status_code == 200:
                image = Image.open(io.BytesIO(response.content))
            else:
                st.error("Could not retrieve image from the URL.")
        except:
            st.error("Invalid URL or image cannot be opened.")

if image:
    st.image(image, caption="Input Image", use_column_width=True)
    if st.button("Predict"):
        with st.spinner("Classifying..."):
            try:
                prediction = predict(image)
                st.success(f"Predicted condition: **{prediction}**")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
