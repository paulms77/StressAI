from model import StressTransformer
import torch
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from io import BytesIO
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from fastapi.middleware.cors import CORSMiddleware

model_path = os.getenv('STRESS_MODEL_PATH', 'fine_tuned_model.pt')

input_dim = 2
hidden_dim = 64
num_layers = 2
num_classes = 2
emb_dim = 128
num_heads = 4

model = StressTransformer(input_dim = input_dim, emb_dim = emb_dim, num_heads = num_heads, num_layers = num_layers, hidden_dim = hidden_dim, num_classes = num_classes)

model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

app = FastAPI(title = 'Stress Detection API')

app.add_middleware(
  CORSMiddleware,
  allow_origins=['*'],
  allow_credentials = True,
  allow_methods=['*'],
  allow_headers=['*']
)

class InDataset(BaseModel):
  # 인터페이스 데이터
  EDA: float
  BVP: float

# 라우터
# # post 방식
# @app.post('/predict', status_code=200)
# async def predict(x: InDataset):
#   x_df = pd.DataFrame([[x.EDA, x.BVP]])
#   X_input = torch.tensor(x_df).float()
  
#   with torch.no_grad():
#     output = model(X_input)

#     prediction = torch.argmax(output, dim = 1).item()

#   return {'prediction': prediction}

def sliding_window(data, window_size, stride=1):
  windows = []

  for start in range(0, len(data) - window_size, stride):
    end = start + window_size
    window = data[start: end]
    windows.append(window[['eda', 'bvp']])

  return np.array(windows)

@app.post('/upload_and_predict')
async def upload_and_predict(file: UploadFile = File(...)):
  contents = await file.read()
  df = pd.read_csv(BytesIO(contents))

  if df.empty:
    return {"error": "CSV 파일이 비어 있습니다."}

  if 'eda' not in df.columns or 'bvp' not in df.columns:
    return {'error': 'CSV 파일에 "EDA"와 "BVP" 컬럼이 필요합니다.'}

  # X_input = torch.tensor(df[['eda', 'bvp']].values).float()

  window_size = 64 * 5
  X = sliding_window(df[['eda', 'bvp']], window_size)

  X_input = torch.tensor(X, dtype=torch.float32)

  # X_input = X_input.unsqueeze(0)

  predictions = []

  for x_input in X_input:
    with torch.no_grad():
      prediction = model(x_input.unsqueeze(0))
      predictions.append(prediction)

  return {'eda': df['eda'].tolist(), 'bvp': df['bvp'].tolist(), 'predictions': predictions}

# get 방식
@app .get('/')
async def root():
  return {'message': 'online'}