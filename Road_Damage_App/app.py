import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("Road_Damage_App/road_damage_model_final.keras")

# Class labels (CHANGE if your order is different)
class_names = ['pothole', 'crack', 'manhole']

st.title("🚧 Road Damage Detection System")

uploaded_file = st.file_uploader("Upload a road image", type=["jpg", "png", "jpeg"])

confidence = np.max(prediction) * 100
st.write(f"Confidence: {confidence:.2f}%")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Resize to match training size
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")

    if predicted_class == "pothole":
        st.warning("Recommended Action: Immediate repair required.")
    elif predicted_class == "crack":
        st.warning("Recommended Action: Monitor and schedule maintenance.")
    else:
        st.warning("Recommended Action: Inspect drainage area.")


