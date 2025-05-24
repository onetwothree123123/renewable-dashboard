import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import requests
import math
from datetime import datetime

# 1. 샘플 날씨 데이터 생성 (168시간 = 7일치)
np.random.seed(42)
hours = 168
temperature = 10 + 10 * np.sin(np.linspace(0, 3 * np.pi, hours)) + np.random.normal(0, 1, hours)
wind_speed = 3 + 2 * np.cos(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 0.5, hours)
sunshine = np.clip(6 * np.sin(np.linspace(0, 2 * np.pi, hours)) + 6, 0, 12)

# 발전량(W) = 풍속 기반 공식 + 태양광 계수 + noise
power_output = (
    0.5 * 1.225 * 10.0 * (wind_speed ** 3) * 0.3
    + sunshine * 50
    + np.random.normal(0, 100, hours)
)

df_sample = pd.DataFrame({
    "temperature": temperature,
    "wind_speed": wind_speed,
    "sunshine_duration": sunshine,
    "power_output": power_output
})

# 2. PyTorch Dataset 구성
SEQ_LEN = 24

class WeatherPowerDataset(Dataset):
    def __init__(self, df, seq_len=24):
        self.inputs = []
        self.targets = []
        data = df[["temperature", "wind_speed", "sunshine_duration"]].values
        labels = df["power_output"].values
        self.X_mean = data.mean(axis=0)
        self.X_std = data.std(axis=0)
        self.y_mean = labels.mean()
        self.y_std = labels.std()

        data = (data - self.X_mean) / self.X_std
        labels = (labels - self.y_mean) / self.y_std

        for i in range(len(df) - seq_len):
            self.inputs.append(data[i:i+seq_len])
            self.targets.append(labels[i+seq_len])
        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

    def denormalize(self, y_tensor):
        return y_tensor * self.y_std + self.y_mean

# 3. LSTM 모델 정의
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# 4. 학습 루프
model = LSTMRegressor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = WeatherPowerDataset(df_sample, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

EPOCHS = 100
losses = []

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x_batch, y_batch in dataloader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    losses.append(epoch_loss / len(dataloader))

# 5. 예측 테스트 (학습용 전체 데이터 예측)
model.eval()
X_all = torch.tensor(df_sample[["temperature", "wind_speed", "sunshine_duration"]].values, dtype=torch.float32)
X_seq = torch.stack([X_all[i:i+SEQ_LEN] for i in range(len(df_sample)-SEQ_LEN)])
with torch.no_grad():
    y_pred = model(X_seq).squeeze().numpy()

y_true = df_sample["power_output"].values[SEQ_LEN:]

# 결과 시각화
plt.figure(figsize=(12,5))
plt.plot(y_true, label='True Power Output')
plt.plot(y_pred, label='Predicted Power Output')
plt.legend()
plt.title("🔋 발전량 예측: 샘플 시계열 기반 LSTM")
plt.xlabel("시간 Index")
plt.ylabel("Watt")
plt.grid(True)
plt.tight_layout()
plt.show()

# 6. 실시간 예보 기반 예측 함수 (Open-Meteo 활용)
def get_forecast_and_predict(model, dataset, latitude=37.5665, longitude=126.9780, hours=24):
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,cloudcover,windspeed_10m",
        "timezone": "Asia/Seoul",
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception("API 요청 실패")

    data = response.json()["hourly"]
    temperature = data["temperature_2m"][:hours]
    windspeed = data["windspeed_10m"][:hours]
    cloudcover = data["cloudcover"][:hours]

    sunshine_duration = [max(0, 12 - (c / 100) * 12) for c in cloudcover]  # 태양광 계산 대체

    X_input = np.array([
        [temperature[i], windspeed[i], sunshine_duration[i]]
        for i in range(hours)
    ])

    X_input = (X_input - dataset.X_mean) / dataset.X_std
    SEQ_LEN = 24
    sequences = np.array([X_input[i:i+SEQ_LEN] for i in range(len(X_input) - SEQ_LEN)])
    X_tensor = torch.tensor(sequences, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).squeeze().numpy()
        y_pred = dataset.denormalize(torch.tensor(y_pred)).numpy()

    return y_pred
