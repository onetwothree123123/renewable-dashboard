import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 샘플 날씨 데이터 생성 (168시간 = 7일치)
np.random.seed(42)
hours = 168
temperature = 10 + 10 * np.sin(np.linspace(0, 3 * np.pi, hours)) + np.random.normal(0, 1, hours)
wind_speed = 3 + 2 * np.cos(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 0.5, hours)
sunshine = np.clip(6 * np.sin(np.linspace(0, 2 * np.pi, hours)) + 6, 0, 12)

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

        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# 3. LSTM 모델 정의
class LSTMRegressor(nn.Module):
    def __init__(self, input_size=3, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])

# 4. 학습 실행
model = LSTMRegressor()
dataset = WeatherPowerDataset(df_sample, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 100
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for x_batch, y_batch in dataloader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"[Epoch {epoch+1}] Loss: {epoch_loss / len(dataloader):.4f}")

# 5. 모델 저장
torch.save(model.state_dict(), "lstm_power_model.pth")
torch.save({
    'X_mean': dataset.X_mean,
    'X_std': dataset.X_std,
    'y_mean': dataset.y_mean,
    'y_std': dataset.y_std
}, "normalization_stats.pt")

print("✅ 모델 및 정규화 통계 저장 완료")
