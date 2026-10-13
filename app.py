
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path

# -----------------------------
# Page Settings
# -----------------------------
st.set_page_config(
    page_title="Fruit & Vegetable Classifier",
    page_icon="🍎",
    layout="centered"
)

st.title("🍎 Fruit & Vegetable Classifier")
st.write("Upload a fruit or vegetable image and predict its class.")

# -----------------------------
# Model File Path
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "fruits_and_vegitable_model.keras"

# -----------------------------
# Class Names
# Ye order training dataset ke folders ke order
# ke bilkul same hona chahiye.
# -----------------------------
class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper',
    'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant',
    'garlic', 'ginger', 'grapes', 'jalepeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion',
    'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans',
    'spinach', 'sweetcorn', 'sweetpotato', 'tomato',
    'turnip', 'watermelon'
]

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model load nahi hua: {e}")
    st.stop()

st.success("Model successfully loaded!")

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Image yahan drag-and-drop karein ya browse karein",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Image open karo
    image = Image.open(uploaded_file).convert("RGB")

    # Uploaded image show karo
    st.image(image, caption="Uploaded Image", width=300)

    # Predict button
    if st.button("🔍 Predict", use_container_width=True):

        with st.spinner("Prediction ho rahi hai..."):

            # Model ke input ke mutabiq image resize karo
            img_height = model.input_shape[1]
            img_width = model.input_shape[2]

            image_resized = image.resize((img_width, img_height))

            # Image ko array mein convert karo
            img_array = np.array(image_resized, dtype=np.float32)

            # Batch dimension add karo
            img_array = np.expand_dims(img_array, axis=0)

            # Prediction
            predictions = model.predict(img_array, verbose=0)[0]

            # Logits ko probabilities mein convert karo
            scores = tf.nn.softmax(predictions).numpy()

            # Highest confidence wali class
            predicted_index = int(np.argmax(scores))
            predicted_class = class_names[predicted_index]
            confidence = float(scores[predicted_index]) * 100

        # -----------------------------
        # Show Results
        # -----------------------------
        st.divider()
        st.subheader("Prediction Result")

        st.success(f"Predicted Class: {predicted_class.title()}")
        st.metric("Confidence", f"{confidence:.2f}%")

        # Top 3 predictions
        st.subheader("Top Predictions")

        top_indices = np.argsort(scores)[-3:][::-1]

        for index in top_indices:
            st.write(
                f"{class_names[index].title()}: "
                f"{scores[index] * 100:.2f}%"
            )
            st.progress(float(scores[index]))

