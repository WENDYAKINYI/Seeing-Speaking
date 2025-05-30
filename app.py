import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import openai
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Caption Showdown", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Fixed Model Loading ---
@st.cache_resource
def load_baseline_model():
    model_files = {
        "encoder": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/encoder.pth",
        "decoder": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/decoder.pth",
        "word2idx": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/word2idx.pkl",
        "idx2word": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/idx2word.pkl"
    }

    # Safe download function
    def download_file(url):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            return BytesIO(response.content)
        except Exception as e:
            st.error(f"Download failed: {str(e)}")
            return None
    
    # Initialize models
    encoder = EncoderCNN().eval().to(device)
    decoder = DecoderRNN(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=10004  # Update with your actual vocab size
    ).eval().to(device)
    
    # Load weights
    enc_file = download_file(model_files["encoder"])
    dec_file = download_file(model_files["decoder"])
    if enc_file and dec_file:
        encoder.load_state_dict(torch.load(enc_file, map_location=device))
        decoder.load_state_dict(torch.load(dec_file, map_location=device))
    
    # Load vocab
    vocab = {
        "word2idx": torch.load(download_file(model_files["word2idx"]), map_location='cpu'),
        "idx2word": torch.load(download_file(model_files["idx2word"]), map_location='cpu')
    }
    
    return encoder, decoder, vocab

# --- Fixed Image Loading ---
def load_image(image_source):
    try:
        if isinstance(image_source, str):  # URL case
            response = requests.get(image_source, stream=True)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        else:  # Uploaded file case
            return Image.open(image_source).convert("RGB")
    except Exception as e:
        st.error(f"Image loading failed: {str(e)}")
        return None

# --- UI Components ---
st.title("🆚 Caption Showdown")
st.caption("Compare our custom CNN-RNN model against GPT-3.5 enhancements")

# Example images (using direct URLs that work with PIL)
example_images = {
    "Beach": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=600&auto=format",
    "Dog": "https://images.unsplash.com/photo-1561037404-61cd46aa615b?w=600&auto=format",
    "Food": "https://images.unsplash.com/photo-1565958011703-72f8583c2708?w=600&auto=format"
}

# Image selection
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
selected = st.selectbox("Or try an example:", list(example_images.keys()))

# Load image
image = None
if uploaded_file:
    image = load_image(uploaded_file)
else:
    image = load_image(example_images[selected])

if image:
    st.image(image, caption="Input Image", use_container_width=True)  # Fixed deprecation
    
    # Model comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🧠 Baseline Model")
        with st.spinner("Generating caption..."):
            encoder, decoder, vocab = load_baseline_model()
            image_tensor = preprocess_image(image).to(device)
            caption = generate_caption(image_tensor, encoder, decoder, vocab)
            st.success(caption)
            
            st.write("Rate this caption:")
            if st.button("👍", key="like_baseline"):
                st.toast("Thanks for your feedback!")
            st.button("👎", key="dislike_baseline")

    with col2:
        st.subheader("✨ OpenAI Enhanced")
        if 'openai_key' in st.secrets:
            with st.spinner("Enhancing with GPT-3.5..."):
                enhanced = enhance_with_openai(caption)
                st.success(enhanced)
                
                st.write("Rate this enhancement:")
                if st.button("👍", key="like_openai"):
                    st.toast("Thanks for your feedback!")
                st.button("👎", key="dislike_openai")
        else:
            st.warning("Add OpenAI key in secrets.toml to enable")

# Footer
st.divider()
st.caption("""
⚡ [GitHub Repo](https://github.com/your-repo) | 
[Model Card](https://huggingface.co/weakyy/image-captioning-baseline-model)
""")
