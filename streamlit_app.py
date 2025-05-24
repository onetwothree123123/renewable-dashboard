import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import requests
from lstm_power_regression import LSTMRegressor

# 예측 함수 정의
class NormalizationStats:
    def __init__(self, x_mean, x_std, y_mean, y_std):
        self.X_mean = x_mean
        self.X_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def normalize(self, X):
        return (X - self.X_mean) / self.X_std

    def denormalize(self, y_tensor):
        return y_tensor * self.y_std + self.y_mean

def get_forecast_and_predict(model, stats, latitude=37.5665, longitude=126.9780, hours=24):
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

    sunshine_duration = [max(0, 12 - (c / 100) * 12) for c in cloudcover]

    X_input = np.array([
        [temperature[i], windspeed[i], sunshine_duration[i]]
        for i in range(hours)
    ])

    X_input = stats.normalize(X_input)
    SEQ_LEN = 24

    if len(X_input) < SEQ_LEN:
        raise ValueError(f"예측을 위해 최소 {SEQ_LEN}개의 시계열 데이터가 필요하지만, {len(X_input)}개만 수신되었습니다.")

    sequences = np.array([X_input[i:i+SEQ_LEN] for i in range(len(X_input) - SEQ_LEN)])

    if sequences.shape[0] == 0:
        raise ValueError("기상 데이터가 충분하지 않아 입력 시퀀스를 생성할 수 없습니다.")

    X_tensor = torch.tensor(sequences, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_tensor).squeeze().numpy()
        y_pred = stats.denormalize(torch.tensor(y_pred)).numpy()

    return y_pred

# 모델 및 통계 로드
model = LSTMRegressor()
model.load_state_dict(torch.load("lstm_power_model.pth", map_location=torch.device('cpu')))
model.eval()

stats_dict = torch.load("normalization_stats.pt")
stats = NormalizationStats(
    stats_dict['X_mean'],
    stats_dict['X_std'],
    stats_dict['y_mean'],
    stats_dict['y_std']
)

# Streamlit 대시보드 구성
st.set_page_config(page_title="AI 신재생에너지 예측", layout="wide")
st.title("🔋 신재생에너지 발전량 예측 대시보드")

st.markdown("""
이 대시보드는 **Open-Meteo API**로부터 실시간 기상 데이터를 가져오고,
학습된 **LSTM 모델**을 활용해 향후 발전량을 예측합니다.

- 입력: 기온, 풍속, 구름량
- 출력: 예측 발전량 (태양광 + 풍력)
""")

if st.button("📡 실시간 기상 예보 기반 발전량 예측 시작"):
    try:
        predictions = get_forecast_and_predict(model, stats)
        st.subheader("📈 24시간 예측 발전량 (Watt)")
        st.line_chart(predictions)
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")

st.markdown("""
---
**사용된 모델:** LSTM (Long Short-Term Memory)\
**기상 API:** [Open-Meteo](https://open-meteo.com/)
""")
