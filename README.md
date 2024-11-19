# Sports Image Classification Model

## Description
This project focuses on building a deep learning model to classify sports images into various categories. The model leverages the power of transfer learning with the pre-trained **VGG16** architecture, followed by additional fully connected layers for classification. The model is trained on a custom sports dataset and includes a future deployment solution using **Streamlit**, which allows users to upload images and classify them in real-time.

## Features
- **Image Preprocessing**: Uses data augmentation (shear, zoom, horizontal flip) to improve model generalization.
- **Transfer Learning**: Employs a pre-trained VGG16 model to extract features, which are further processed through fully connected layers.
- **Model Evaluation**: Evaluation metrics like accuracy, precision, recall, and F1-score are used to assess model performance.
- **Deployment with Streamlit**: The trained model will be deployed as an interactive web application, allowing users to classify sports images.

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- scikit-learn
- numpy
- matplotlib
- streamlit

