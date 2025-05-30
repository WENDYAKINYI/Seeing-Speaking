import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pickle
from model import EncoderCNN, DecoderRNN
from utils import clean_caption, generate_caption
import openai

# --- Configuration ---
st.set_page_config(page_title="Image Captioning Comparison", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Loading ---
@st.cache_resource
def load_models():
    encoder = EncoderCNN().eval().to(device)
    decoder = DecoderRNN(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=len(word2idx)
    ).eval().to(device)
    
    encoder.load_state_dict(torch.load("baseline_model/encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("baseline_model/decoder.pth", map_location=device))
    return encoder, decoder

# --- Load Vocab ---
with open("baseline_model/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("baseline_model/idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

# --- Image Transforms ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
])

# --- GPT-3 Enhancement ---
def enhance_caption(caption):
    if 'openai_key' not in st.secrets:
        return None
        
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{
            "role": "system", 
            "content": "Improve this image caption keeping the original meaning:"
        }, {
            "role": "user",
            "content": caption
        }],
        temperature=0.7,
        max_tokens=100
    )
    return response.choices[0].message.content

# --- Feedback System ---
def log_feedback(model_type, rating):
    with open("feedback.log", "a") as f:
        f.write(f"{model_type},{rating}\n")

# --- Streamlit UI ---
st.title("📸 Image Captioning Demo")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Generate captions
    encoder, decoder = load_models()
    tensor_image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        encoder_out = encoder(tensor_image)
        caption = generate_caption(encoder_out, decoder, word2idx, idx2word, beam_size=3)  # Added beam search
    
    # Display with feedback
    st.subheader("Generated Caption")
    st.write(caption)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👍 Like"):
            log_feedback("baseline", 1)
            st.success("Thanks for your feedback!")
    with col2:
        if st.button("👎 Dislike"):
            log_feedback("baseline", 0)
            st.error("We'll improve!")
    
    # GPT-3 Enhancement Toggle
    if st.checkbox("Enhance with AI"):
        with st.spinner("Generating enhanced version..."):
            enhanced = enhance_caption(caption)
            if enhanced:
                st.subheader("Enhanced Version")
                st.write(enhanced)
