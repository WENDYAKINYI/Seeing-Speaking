import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pickle
import openai
from model import EncoderCNN, DecoderRNN
from utils import generate_caption_from_model, extract_keywords_from_image  # assume these exist

# --- Configuration ---
st.set_page_config(page_title="Image Captioning | Baseline vs GPT", layout="wide")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Vocabulary ---
with open("baseline_model/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)
with open("baseline_model/idx2word.pkl", "rb") as f:
    idx2word = pickle.load(f)

# --- Load Baseline Models ---
@st.cache_resource
def load_baseline_models():
    encoder = EncoderCNN().eval().to(device)
    decoder = DecoderRNN(attention_dim=256, embed_dim=256, decoder_dim=512, vocab_size=len(word2idx)).eval().to(device)
    encoder.load_state_dict(torch.load("baseline_model/encoder.pth", map_location=device))
    decoder.load_state_dict(torch.load("baseline_model/decoder.pth", map_location=device))
    return encoder, decoder

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- GPT API-Based Captioning ---
def gpt_generate_caption(keywords):
    prompt = f"Generate a vivid and natural sentence describing an image with the following content: {keywords}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a visual scene captioning expert."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8,
        max_tokens=100
    )
    return response.choices[0].message.content

# --- App Title ---
st.title("📸 Image Captioning: Baseline CNN+RNN vs GPT-4")
st.markdown("""
Upload an image and compare captions from a trained CNN+RNN baseline model and the GPT-4 captioning engine.
""")

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.divider()

    col1, col2 = st.columns(2)

    # Load and generate baseline caption
    with col1:
        st.subheader("🔍 Baseline Model (CNN + RNN)")
        encoder, decoder = load_baseline_models()
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            encoder_out = encoder(img_tensor)
            baseline_caption = generate_caption_from_model(encoder_out, decoder, word2idx, idx2word)
        st.write(baseline_caption)

    # Generate GPT caption
    with col2:
        st.subheader("🤖 GPT-4 Enhanced Caption")
        keywords = extract_keywords_from_image(image)  # placeholder function
        gpt_caption = gpt_generate_caption(keywords)
        st.write(gpt_caption)

    st.divider()
    if st.button("🔁 Refresh with New Image"):
        st.experimental_rerun()
