import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import openai
import requests
from io import BytesIO
import base64
import numpy as np
from utils import (
    load_baseline_model,
    generate_baseline_caption,
    load_image,
    preprocess_image
)

# Configuration
st.set_page_config(page_title="Image Caption Generator", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Loading
@st.cache_resource
def load_models():
    return load_baseline_model()

encoder, decoder, vocab = load_models()

# Helper Functions
def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

# UI Layout
st.title("Image Caption Generator")
st.markdown("Upload an image or paste a URL to generate captions")

# Image Input
col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
with col2:
    image_url = st.text_input("Or enter image URL")

# Example Images
example_images = {
    "Beach": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
    "Dog": "https://images.unsplash.com/photo-1561037404-61cd46aa615b",
    "Food": "https://images.unsplash.com/photo-1565958011703-72f8583c2708"
}

if not uploaded_file and not image_url:
    selected = st.selectbox("Try an example:", list(example_images.keys()))
    image_url = example_images[selected]

# Process Image
image = None
if uploaded_file:
    image = load_image(uploaded_file)
elif image_url:
    image = load_image(image_url)

if image:
    st.image(image, caption="Input Image", use_column_width=True)
    
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            image_tensor = preprocess_image(image, device)
            result = generate_baseline_caption(
                image_tensor=image_tensor,
                encoder=encoder,
                decoder=decoder,
                vocab=vocab
            )
            
            st.subheader("Generated Caption")
            st.success(result["caption"])
            st.write(f"Confidence: {result['confidence']:.2f}")

# Footer
st.divider()
st.markdown("""
**Model Details**:
- Encoder: ResNet50
- Decoder: LSTM with Attention
- Vocabulary Size: {vocab_size}
""".format(vocab_size=len(vocab)))
