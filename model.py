import torch
import torch.nn as nn

class StressRNN(nn.Module):
  def __init__(self, input_size=2, hidden_size=64, num_layers=1, num_classes=1):
    super(StressRNN, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, num_classes)

  def forward(self, x):
    out, (hn, cn) = self.lstm(x)
    out = out[:, -1, :]
    out = self.fc(out)
    return out

class StressTransformer(nn.Module):
  def __init__(self, input_dim = 1, emb_dim = 128, num_heads = 4, num_layers = 2, hidden_dim = 256, num_classes = 1):
    super(StressTransformer, self).__init__()

    self.embedding = nn.Linear(input_dim, emb_dim)
    self.encoder_layer = nn.TransformerEncoderLayer(d_model = emb_dim, nhead = num_heads, dim_feedforward = hidden_dim)
    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers = num_layers)
    self.fc = nn.Linear(emb_dim, num_classes)

  def forward(self, x):
    x = self.embedding(x)
    # x = x.unsqueeze(1)
    x = x.permute(1, 0, 2)
    x = self.transformer_encoder(x)
    x = x.mean(dim = 0)
    x = self.fc(x)
    return x