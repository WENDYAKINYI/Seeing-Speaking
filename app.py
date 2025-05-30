import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import openai
import requests
from io import BytesIO
import numpy as np
from utils import (
    load_baseline_model,
    generate_baseline_caption,
    enhance_with_openai,
    load_image,
    preprocess_image
)

# --- Configuration ---
st.set_page_config(page_title="Caption Showdown", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
@st.cache_resource
def load_models():
    return load_baseline_model()

# --- UI Layout ---
st.title("Vision to Text: Baseline 🆚 OpenAI")
st.caption("Compare a custom CNN-RNN model against OpenAI GPT-3.5")

encoder, decoder, vocab = load_models()
# Sidebar
with st.sidebar:
    st.header("Settings")
    openai_enabled = st.toggle("Enable OpenAI", True)
    st.divider()
    st.markdown("""
    **Model Details**  
    - Baseline: ResNet50 + Attention LSTM  
    - OpenAI: GPT-3.5 Turbo  
    [View Baseline Model](https://huggingface.co/weakyy/image-captioning-baseline-model)
    """)
    st.write("Model Status:", 
    f"Encoder: {'Loaded' if encoder else 'Failed'}",
    f"Decoder: {'Loaded' if decoder else 'Failed'}",
    f"Vocab Size: {len(vocab['word2idx']) if vocab else 0}")

# Main Content
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
example_images = {
    "Beach": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600",
    "Dog": "https://images.unsplash.com/photo-1561037404-61cd46aa615b?w=600",
    "Food": "https://images.unsplash.com/photo-1565958011703-72f8583c2708?w=600"
}
selected = st.selectbox("Or try an example:", list(example_images.keys()))

# Process image
if uploaded_file:
    image = load_image(uploaded_file)
else:
    image = load_image(example_images[selected])

if image:
    # Display image
    st.image(image, caption="Input Image", use_container_width=True)
    
    # Preprocess image
    image_tensor = preprocess_image(image, device)  # This defines image_tensor
    
    # Load models
    encoder, decoder, vocab = load_models()
    
    # Generate captions
    col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Baseline Model")
    with st.spinner("Generating baseline caption..."):
        baseline_result = generate_baseline_caption(
                image_tensor=image_tensor,
                encoder=encoder,
                decoder=decoder,
                vocab=vocab,
                beam_size=3
            )
        st.success(baseline_result["caption"])
        st.caption(f"Confidence: {baseline_result['confidence']:.0%}")
        
        # Feedback
        st.write("Rate this caption:")
        if st.button("👍", key="like_baseline"):
            st.toast("Thanks for your feedback!")
        st.button("👎", key="dislike_baseline")

with col2:
    st.subheader("✨ OpenAI Enhanced")
    if openai_enabled and 'openai_key' in st.secrets:
        with st.spinner("Enhancing with GPT-3.5..."):
            enhanced = enhance_with_openai(baseline_result["caption"])
            st.success(enhanced)
            
            # Feedback
            st.write("Rate this enhancement:")
            if st.button("👍", key="like_openai"):
                st.toast("Thanks for your feedback!")
            st.button("👎", key="dislike_openai")
    elif not openai_enabled:
        st.info("Enable OpenAI in sidebar")
    else:
        st.error("Add OpenAI key in secrets.toml")

# # Performance Metrics
# with st.expander("📊 Performance Comparison"):
#     st.table({
#         "Model": ["Baseline", "OpenAI Enhanced"],
#         "BLEU-4": [0.42, 0.61],
#         "Inference Time": ["1.2s", "2.8s"]
#     })

# Footer
st.divider()
st.caption("""
⚡ **Pro Tip**: Click images to zoom | 
[GitHub Repo](https://github.com/your-repo) | 
[Model Card](https://huggingface.co/weakyy/image-captioning-baseline-model)
""")
