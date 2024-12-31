import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
from tensorflow.keras.models import load_model
from PIL import Image
st.write("""# Plant Disesase Predictor Using Python and VGG19""")
file_id ="1lJcCJFr1mmj-XYmHmpsFG8LvqkTfWAgq"
url = f'https://drive.google.com/uc?export=download&id={file_id}'
def load_model_from_drive(url):
    # Download the model from Google Drive using gdown
    output_path = './model.keras'  # Temporary path to store the model file
    gdown.download(url, output_path, quiet=False)

    # Load the model from the file
    model = tf.keras.models.load_model(output_path)
    return model
try:
    model = load_model_from_drive(url)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

class_labels = {
    0: "healthy",
    1: "multiple_diseases",
    2: "Rust Disease",
    3: "Scab Disease"
}


# Load a test image
uploaded_image = st.file_uploader("Upload Image of Infected Plant")
if uploaded_image is not None:
    # Load the image using PIL
    img = Image.open(uploaded_image)
    st.image(img,caption="Uploaded Image",use_container_width=True)
    img = tf.keras.preprocessing.image.load_img(uploaded_image, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]  # Get the index of the predicted class

    # Map index to class label
    predicted_class_label = class_labels[predicted_class_index]

    st.write(f"The Plant may Have **{predicted_class_label}**")
    print(f"Predicted class label: {predicted_class_label}")