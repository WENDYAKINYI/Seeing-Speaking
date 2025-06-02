import torch
import pickle
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from huggingface_hub import hf_hub_download
from torchvision import transforms
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def download_file_from_hf(filename):
    return hf_hub_download(
        repo_id="weakyy/image-captioning-baseline-model",
        filename=filename,
        repo_type="model"
    )

def load_baseline_model():
    # Load vocabulary first
    vocab_path = download_file_from_hf("vocab.pkl")
    with open(vocab_path, "rb") as f:
        vocab = pickle.load(f)
    
    # Initialize models
    encoder = EncoderCNN().to(device)
    decoder = DecoderRNN(vocab_size=len(vocab)).to(device)
    
    # Load weights
    encoder.load_state_dict(
        torch.load(download_file_from_hf("encoder.pth"), map_location=device),
        strict=False
    )
    decoder.load_state_dict(
        torch.load(download_file_from_hf("decoder.pth"), map_location=device),
        strict=False
    )
    
    return encoder, decoder, vocab

def generate_baseline_caption(image_tensor, encoder, decoder, vocab, beam_size=3, max_len=20):
    encoder.eval()
    decoder.eval()
    
    # Extract features
    features = encoder(image_tensor)
    encoder_out = features.unsqueeze(1)
    encoder_dim = encoder_out.size(-1)
    
    # Beam search setup
    k = beam_size
    encoder_out = encoder_out.expand(k, -1, -1)
    
    # Initialize
    seq = torch.full((k, 1), vocab['<start>'], dtype=torch.long, device=device)
    top_k_scores = torch.zeros(k, 1, device=device)
    complete_seqs = []
    complete_seqs_scores = []
    
    # Initialize LSTM state
    h, c = decoder.init_hidden_state(encoder_out.mean(1))
    
    for step in range(max_len):
        embeddings = decoder.embed(seq[:, -1]).unsqueeze(1)
        lstm_out, (h, c) = decoder.lstm(embeddings, (h, c))
        scores = decoder.linear(lstm_out.squeeze(1))
        scores = F.log_softmax(scores, dim=1)
        scores = top_k_scores.expand_as(scores) + scores
        
        if step == 0:
            top_k_scores, top_k_words = scores[0].topk(k, dim=0)
        else:
            top_k_scores, top_k_words = scores.view(-1).topk(k, dim=0)
            
        prev_word_inds = top_k_words // len(vocab)
        next_word_inds = top_k_words % len(vocab)
        
        seq = torch.cat([seq[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
        
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) 
                          if next_word != vocab['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds)
        
        if complete_inds:
            complete_seqs.extend(seq[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())
        k -= len(complete_inds)
        
        if k == 0:
            break
            
        seq = seq[incomplete_inds]
        h = h[0][prev_word_inds[incomplete_inds]].unsqueeze(0)
        c = c[0][prev_word_inds[incomplete_inds]].unsqueeze(0)
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
    
    if not complete_seqs:
        complete_seqs = seq.tolist()
        complete_seqs_scores = top_k_scores.tolist()
        
    best_idx = np.argmax(complete_seqs_scores)
    caption_ids = complete_seqs[best_idx]
    
    caption_words = []
    for word_id in caption_ids:
        word = vocab.get(word_id, '<unk>')
        if word not in ['<start>', '<end>', '<pad>']:
            caption_words.append(word)
        if word == '<end>':
            break
    
    confidence = min(float(np.exp(np.max(complete_seqs_scores))), 1.0)
    
    return {
        "caption": ' '.join(caption_words),
        "confidence": confidence
    }

def load_image(image_source):
    try:
        if isinstance(image_source, str):
            if image_source.startswith(('http:', 'https:')):
                response = requests.get(image_source, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content)).convert('RGB')
            else:
                return Image.open(image_source).convert('RGB')
        else:
            return Image.open(image_source).convert('RGB')
    except Exception as e:
        print(f"Image load failed: {str(e)}")
        return None

def preprocess_image(image, device='cpu'):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)
