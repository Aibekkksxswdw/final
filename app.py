import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Cache the model loading to avoid reloading on every interaction
@st.cache(allow_output_mutation=True)
def load_model():
    try:
        model_path = r"C:\Users\Айбек\final\model.h5"  # Update the path if needed
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((224, 224))  # Adjust size as per your model's input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Load the model
model = load_model()
if model is None:
    st.stop()

# Streamlit app interface
st.title("Image Classification with TensorFlow")
st.write("Upload an image to classify.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        # Open the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess the image
        input_data = preprocess_image(image)

        # Perform prediction
        predictions = model.predict(input_data)
        class_idx = np.argmax(predictions)
        confidence = np.max(predictions) * 100

        # Display the result
        st.write(f"Predicted Class: {class_idx}")
        st.write(f"Confidence: {confidence:.2f}%")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
