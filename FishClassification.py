import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, Model


# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_fish_model():
    model = load_model("my_VGG16_model.h5", compile=False, safe_mode=False)
    return model
 
   
model = load_fish_model()

# Define your class names (order must match training generator)
class_names = ['animal fish', 'animal fish bass', 'fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream', 'fish sea_food hourse_mackerel', 'fish sea_food red_mullet', 'fish sea_food red_sea_bream', 'fish sea_food sea_bass', 'fish sea_food shrimp', 'fish sea_food striped_red_mullet', 'fish sea_food trout']  # <-- change to your actual classes

# -----------------------------
# App Layout
# -----------------------------
st.title("🐟 Fish Classification App")
st.write("Upload a fish image to predict its category using our trained VGG16 model.")

uploaded_files = st.file_uploader(
    "Upload fish image(s)", type=['jpg','jpeg','png'], accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Show image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img = Image.open(uploaded_file).convert('RGB')
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = np.max(preds)

        # Show result
        st.markdown(f"### 🏷️ Prediction: **{predicted_class}**")
        st.progress(float(confidence))
        st.write("Confidence Scores:")
        for cls, score in zip(class_names, preds[0]):
            st.write(f"• {cls}: {score*100:.2f}%")
