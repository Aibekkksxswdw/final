import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r"C:\Users\Айбек\final\model.h5")  # Replace with your model's path
    return model

model = load_model()

# Define a function to make predictions
def predict(image, model):
    image = image.resize((224, 224))  # Adjust based on your model input size
    image = np.array(image) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    prediction = model.predict(image)
    return prediction

# Streamlit interface
st.title("Sports Image Classification")
st.write("Upload an image to classify it.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("Classifying...")
    prediction = predict(image, model)
    st.write(f"Prediction: {np.argmax(prediction)}")  # Replace with your class labels

