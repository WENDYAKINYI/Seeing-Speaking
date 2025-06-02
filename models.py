# models.py

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # Remove the final FC layer
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images).squeeze()  # [B, 2048, 1, 1] → [B, 2048]
        features = self.linear(features)
        features = self.bn(features)
        return features  # [B, embed_size]

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        features = features.unsqueeze(1).expand(-1, embeddings.size(1), -1)
        inputs = torch.cat((features, embeddings), 2)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.linear(self.dropout(lstm_out))
        return outputs

    def init_hidden_state(self, encoder_out):
        batch_size = encoder_out.size(0)
        hidden = torch.zeros(1, batch_size, self.lstm.hidden_size).to(encoder_out.device)
        cell = torch.zeros(1, batch_size, self.lstm.hidden_size).to(encoder_out.device)
        return hidden, cell
