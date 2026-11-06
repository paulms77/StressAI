import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import sklearn
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
from model import StressTransformer
import torch.nn as nn
import os
import torch.nn.functional as F

# 모델 로드
model_path = os.getenv('STRESS_MODEL_PATH', 'fine_tuned_model.pt')
input_dim = 2
hidden_dim = 256 # 64
num_layers = 2
num_classes = 2
emb_dim = 128
num_heads = 4

device = torch.device("cpu")
model = StressTransformer(input_dim, emb_dim, num_heads, num_layers, hidden_dim, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 실시간 데이터를 저장할 리스트
data = []
stress_values = []
stress_class_values = []
timestamps = []
eda_values = []
bvp_values = []
accuracies = []
losses = []
current_index = 0  # 현재 위치 추적

# Dash App 설정
app = dash.Dash(__name__)
server = app.server

# dark_theme = {
#     'backgroudColor': '#121212',
#     'textColor': '#ffffff',
#     'cardColor': '#1e1e1e',
#     'borderRadius': '10px',
#     'padding': '10px',
#     'margin': '10px',
# }

app.layout = html.Div(
    style={'backgroundColor': '#121212', 'color': 'white', 'padding': '20px', 'fontFamily': 'Arial'},
    children=[
        html.H1("Live Stress Detection System", style={'textAlign': 'center', 'fontSize': '32px'}),

        html.Div(
            [
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('CSV 파일 업로드', style={
                        'backgroundColor': '#1DB954',  # 초록색 Spotify 느낌
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 24px',
                        'fontSize': '18px',
                        'cursor': 'pointer',
                        'borderRadius': '8px',
                        'display': 'inline-block'
                    }),
                    multiple=False
                ),
                html.Span(id='file-name', style={'marginLeft': '15px', 'fontSize': '18px'})
            ],
            style={'textAlign': 'center', 'marginBottom': '20px'}
        ),

        html.Div([
            html.Div([dcc.Graph(id='eda-graph')], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([dcc.Graph(id='bvp-graph')], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([dcc.Graph(id='stress-graph')], style={'width': '33%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),

        html.Div([
            html.Div([dcc.Graph(id='accuracy-graph')], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
            html.Div([dcc.Graph(id='loss-graph')], style={'width': '50%', 'display': 'inline-block', 'padding': '10px'}),
        ], style={'display': 'flex', 'justifyContent': 'space-between'}),

        dcc.Interval(
            id='interval-update',
            interval=1000,  # 1초마다 업데이트
            n_intervals=0
        )
    ]
)

stress_events = set()

### 📌 업로드된 CSV 데이터를 `data` 리스트에 저장
@app.callback(
    Output('upload-data', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
)
def load_data(contents, filename):
    global data, current_index, stress_values, stress_class_values, timestamps, labels, stress_events
    if contents is None:
        return 'CSV File Upload'
    
    df = pd.read_csv(filename)
    data = df[['eda', 'bvp']].values.tolist()  # 필요한 컬럼만 추출하여 리스트로 변환
    labels = df['labels'].values.tolist()
    current_index = 0  # 데이터 시작 위치 초기화
    stress_values = []  # 그래프 초기화
    stress_class_values = []
    timestamps = []  # 타임스탬프 초기화
    eda_values = []
    bvp_values = []
    return f'Loaded: {filename}'

### 📌 실시간 그래프 업데이트 (루프마다 반영)
@app.callback(
    [Output('eda-graph', 'figure'),
    Output('bvp-graph', 'figure'),
    Output('stress-graph', 'figure'),
    Output('accuracy-graph', 'figure'),
    Output('loss-graph', 'figure')],
    Input('interval-update', 'n_intervals')
)
def update_graph(n_intervals):
    global data, stress_values, stress_class_values, timestamps, current_index
    global eda_values, bvp_values, accuracies, losses, labels

    if len(data) < 320*2:  # 최소 5초(64Hz x 5) 데이터가 있어야 함
        return go.Figure(), go.Figure(), go.Figure(), go.Figure(), go.Figure()

    window_size = 64 * 10  # 10초 동안의 데이터
    stride = 64

    data = np.array(data)

    try:
        scaler = MinMaxScaler()
        # scaler = pickle.load('scaler_eda.pkl')
        data[:, 0] = scaler.fit_transform(data[:, 0].reshape(-1, 1)).flatten()
    except Exception as e:
        print(f'Error loading scaler_eda.pkl: {e}')

    try:
        scaler = MinMaxScaler()
        # scaler = pickle.load('scaler_bvp.pkl')
        data[:, 1] = scaler.fit_transform(data[:, 1].reshape(-1, 1)).flatten()
    except Exception as e:
        print(f'Error loading scaler_bvp.pkl: {e}')

    # 현재 인덱스의 윈도우만 처리
    if current_index + window_size <= len(data):
        window = np.array(data[current_index: current_index + window_size])

        # 모델 입력 데이터 변환
        X = torch.tensor(window[:, :2], dtype=torch.float32).unsqueeze(0)
        logits = model(X)
        pred = logits.detach().numpy()[0]
        probabilities = torch.softmax(logits, dim=1).detach().numpy()[0]
        pred_class = np.argmax(probabilities)
        
        # 새로운 데이터 추가
        stress_values.append(probabilities[1])  # 스트레스 확률
        stress_class_values.append(pred_class) # 스트레스 여부
        timestamps.append(current_index)  # 현재 시간
        eda_values.append(window[:, 0].mean())  # EDA 평균값
        bvp_values.append(window[:, 1].mean())  # BVP 평균값
        
        actual_label = int(labels[current_index + window_size - 1])
        print('actual_label', actual_label)
        print('pred_label', pred_class)
        print(probabilities)
        accuracy = int(actual_label == pred_class)
        # loss = -np.log(probabilities[int(actual_label)] + 1e-10)
        actual_label_tensor = torch.tensor([actual_label], dtype=torch.long)
        loss = F.cross_entropy(logits, actual_label_tensor)

        accuracies.append(accuracy)
        losses.append(loss.detach().numpy())

        current_index += stride  # 다음 윈도우로 이동

    # 실시간 그래프 업데이트
    fig_eda = go.Figure()
    fig_eda.add_trace(go.Scatter(x=timestamps, y=eda_values, mode='lines+markers', name='EDA'))
    fig_eda.update_layout(title='Real-time EDA Signal', xaxis_title='Time', yaxis_title='EDA', font=dict(color='black'))

    fig_bvp = go.Figure()
    fig_bvp.add_trace(go.Scatter(x=timestamps, y=bvp_values, mode='lines+markers', name='BVP Signal'))
    fig_bvp.update_layout(title='Real-time BVP Signal', xaxis_title='Time', yaxis_title='BVP', font=dict(color='black'))

    fig_stress = go.Figure()
    fig_stress.add_trace(go.Scatter(x=timestamps, y=stress_values, mode='lines+markers', name='Stress Detection'))
    
    print(stress_class_values)
    print('='*10)
    print(stress_class_values[-1])

    if stress_class_values[-1] == 1:
        stress_events.add(timestamps[-1])
    
    for event in sorted(stress_events):
        fig_stress.add_vline(x=event, line=dict(color='red', dash='dash'), annotation_position='top right')

    fig_stress.update_layout(title='Real-time Stress Monitoring', xaxis_title='Time', yaxis_title='Stress Detection', font=dict(color='black'))

    fig_accuracy = go.Figure([go.Scatter(x=timestamps, y=accuracies, mode='lines+markers', name='Accuracy', line=dict(color='green'))])
    fig_accuracy.update_layout(title='Model Accuracy', xaxis_title='Time', yaxis_title='Accuracy')
    
    fig_loss = go.Figure([go.Scatter(x=timestamps, y=losses, mode='lines+markers', name='Loss', line=dict(color='green'))])
    fig_loss.update_layout(title='Model Loss', xaxis_title='Time', yaxis_title='Loss')

    return fig_eda, fig_bvp, fig_stress, fig_accuracy, fig_loss

if __name__ == '__main__':
    app.run_server(debug=True)