import pandas as pd
import numpy as np
import pymysql
from sqlalchemy import create_engine
import joblib
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
from model import StressRNN, StressTransformer

import os

DB_USER = os.getenv('DB_USER', 'root')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')
DB_HOST = os.getenv('DB_HOST', '127.0.0.1')
DB_PORT = os.getenv('DB_PORT', '3306')
DB_NAME = os.getenv('DB_NAME', 'stress_db')

_auth = f'{DB_USER}:{DB_PASSWORD}' if DB_PASSWORD else DB_USER
engine = create_engine(f'mysql+pymysql://{_auth}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

with engine.connect() as conn:
  print('Connected successfully!')

eda_data = pd.read_sql('SELECT * FROM eda_data', engine)
bvp_data = pd.read_sql('SELECT * FROM bvp_data', engine)
label_data = pd.read_sql('SELECT * FROM label_data', engine)

print(eda_data.head())
print(bvp_data.head())
print(label_data.head())

merged_data = pd.merge(eda_data, bvp_data, on = ['id', 'subject_id'], how = 'inner')
merged_data = pd.merge(merged_data, label_data, on = ['id', 'subject_id'], how = 'inner')

print(merged_data.head())

scaler = MinMaxScaler()
scaler.fit_transform(merged_data['eda'])
joblib.dump('scaler_eda.pkl')

scaler = MinMaxScaler()
scaler.fit_transform(merged_data['bvp'])
joblib.dump('scaler_bvp.pkl')

# def sliding_window(data, window_size, stride = 1):
#   windows = []
#   labels = []

#   for subject_id, group in data.groupby('subject_id'):
#     for start in range(0, len(group) - window_size , stride):
#       end = start + window_size
#       window = group.iloc[start: end]
#       windows.append(window[['eda', 'bvp']].values)
#       labels.append(group.iloc[end-1]['labels'])


#   return np.array(windows), np.array(labels)

# window_size = 64 * 5
# X, y = sliding_window(merged_data, window_size)

# print(X.shape, y.shape)

def sliding_window(data, window_size, stride=1):
  windows = []
  labels = []

  for start in range(0, len(data) - window_size, stride):
    end = start + window_size
    window = data[start: end]
    windows.append(window[['eda', 'bvp']])
    labels.append(data.iloc[end - 1]['labels'])

  return np.array(windows), np.array(labels)

class StressDataset(Dataset):
  def __init__(self, X, y):
    self.X = torch.tensor(X, dtype = torch.float32)
    self.y = torch.tensor(y, dtype = torch.long)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    return self.X[idx], self.y[idx]

window_size = 64 * 5
models = {}
final_model = None

for subject_id in merged_data['subject_id'].unique():
  print(f'subject_id {subject_id}')
  subject_data = merged_data[merged_data['subject_id'] == subject_id]
  
  X, y = sliding_window(subject_data, window_size)

  # X_tensor = torch.tensor(X, dtype=torch.float32)
  # y_tensor = torch.tensor(y, dtype=torch.float32)

  train_dataset = StressDataset(X, y)
  train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False)

  input_dim = 2
  hidden_dim = 64
  num_layers = 2
  num_classes = 2
  emb_dim = 128
  num_heads = 4

  if final_model is None:
    # model = StressRNN(input_dim, hidden_dim, num_layers, num_classes)
    model = StressTransformer(input_dim = input_dim, emb_dim = emb_dim, num_heads = num_heads, num_layers = num_layers, hidden_dim = hidden_dim, num_classes = num_classes)
  else:
    model.load_state_dict(torch.load('fine_tuned_model.pt'))

  criterion = nn.CrossEntropyLoss() # nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  print(f'Training model for subject_id {subject_id}')

  num_epochs = 10
  model.train()
  for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()


    print(f'Epoch [{epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')
  
  torch.save(model.state_dict(), 'fine_tuned_model.pt')
  final_model = model

models[subject_id] = model
print(f'Finished training model for subject_id {subject_id}')