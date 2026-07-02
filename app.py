import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Load Model 
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("best_teeth_PretrainedModel.h5")
    return model

model = load_model()

CLASS_NAMES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

#Preprocess Function
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

#Streamlit UI
st.title("🦷 Teeth Classification App")
st.write("Upload an image of teeth and the model will classify it into one of 7 categories.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # Prediction
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    st.write("### 🟢 Prediction Result:")
    st.write(f"**Class:** {CLASS_NAMES[pred_class]}")
    st.write(f"**Confidence:** {confidence:.2f}")