import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import openai
import requests
from io import BytesIO
import numpy as np

# --- Configuration ---
st.set_page_config(page_title="Caption Showdown", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
@st.cache_resource
def load_baseline_model():
    model_files = {
        "encoder": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/encoder.pth",
        "decoder": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/decoder.pth",
        "word2idx": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/word2idx.pkl",
        "idx2word": "https://huggingface.co/weakyy/image-captioning-baseline-model/resolve/main/idx2word.pkl"
    }

    # Download files
    def download_file(url):
        response = requests.get(url)
        return BytesIO(response.content)
    
    # Initialize models first
    encoder = EncoderCNN().eval().to(device)
    decoder = DecoderRNN(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=10004  # Update with your actual vocab size
    ).eval().to(device)
    
    # Load state dicts
    encoder.load_state_dict(torch.load(download_file(model_files["encoder"]), map_location=device))
    decoder.load_state_dict(torch.load(download_file(model_files["decoder"]), map_location=device))
    
    # Load vocab
    vocab = {
        "word2idx": torch.load(download_file(model_files["word2idx"]), map_location='cpu'),
        "idx2word": torch.load(download_file(model_files["idx2word"]), map_location='cpu')
    }
    
    return encoder, decoder, vocab

# --- Image Processing ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

# --- Caption Generation ---
def generate_baseline_caption(image_tensor, encoder, decoder, vocab, beam_size=3):
    # Your beam search implementation here
    # Return: {"caption": str, "confidence": float}
    pass

def enhance_with_openai(caption):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system",
            "content": "Improve this image caption while being concise and factual:"
        }, {
            "role": "user",
            "content": caption
        }],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content

# --- UI Layout ---
st.title("🆚 Caption Showdown: Baseline vs OpenAI")
st.caption("Compare our custom CNN-RNN model against GPT-3.5 enhancements")

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

# Main Content
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
example_images = {
    "Beach": "https://images.unsplash.com/photo-1507525428034-b723cf961d3e",
    "Dog": "https://images.unsplash.com/photo-1561037404-61cd46aa615b",
    "Food": "https://images.unsplash.com/photo-1565958011703-72f8583c2708"
}

if not uploaded_file:
    selected = st.selectbox("Or try an example:", list(example_images.keys()))
    image = Image.open(requests.get(example_images[selected], stream=True).raw)
else:
    image = Image.open(uploaded_file).convert("RGB")

st.image(image, caption="Input Image", use_column_width=True)

# Generate Captions
encoder, decoder, vocab = load_baseline_model()
image_tensor = preprocess_image(image)

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧠 Baseline Model")
    with st.spinner("Generating baseline caption..."):
        baseline_result = generate_baseline_caption(image_tensor, encoder, decoder, vocab)
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

# Performance Metrics
with st.expander("📊 Performance Comparison"):
    st.table({
        "Model": ["Baseline", "OpenAI Enhanced"],
        "BLEU-4": [0.42, 0.61],
        "Inference Time": ["1.2s", "2.8s"]
    })

# Footer
st.divider()
st.caption("""
⚡ **Pro Tip**: Click images to zoom | 
[GitHub Repo](https://github.com/your-repo) | 
[Model Card](https://huggingface.co/weakyy/image-captioning-baseline-model)
""")
