import streamlit as st
from PIL import Image
from baseline_model import generate_baseline_caption
from clip_gpt2_model import generate_clip_gpt2_caption

st.set_page_config(page_title="Vision to Text", layout="centered")
st.title("🖼️ Image Captioning Comparison: Baseline CNN+RNN vs CLIP+GPT-2")
st.markdown("### Upload an image and compare captions from a traditional CNN+RNN model versus a multimodal CLIP + GPT-2 model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.markdown("---")
    st.markdown("### Generating Captions...")

    # Generate captions from both models
    with st.spinner('Baseline model thinking...'):
        baseline_caption = generate_baseline_caption(image)

    with st.spinner('CLIP + GPT-2 model thinking...'):
        clip_gpt2_caption = generate_clip_gpt2_caption(image)

    # Display side-by-side
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 📘 Baseline (CNN+RNN+Attention)")
        st.success(baseline_caption)
    with col2:
        st.markdown("#### 🤖 CLIP + GPT-2")
        st.success(clip_gpt2_caption)

    st.markdown("---")
    st.button("Try another image", type="primary")
else:
    st.markdown("Upload a JPG or PNG file to begin.")
