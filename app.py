import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Загрузка модели
model = tf.keras.models.load_model('model.h5')

# Заголовок
st.title("Sports Image Classification")

# Загружаем изображение
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Открытие изображения
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    
    # Преобразование изображения для подачи в модель
    img = img.resize((224, 224))  # Если нужно, укажи размер для модели
    img_array = np.array(img) / 255.0  # Нормализация, если необходимо
    img_array = np.expand_dims(img_array, axis=0)  # Изменение формы для модели
    
    # Прогноз
    predictions = model.predict(img_array)
    
    # Результаты
    st.write(f'Predictions: {predictions}')
