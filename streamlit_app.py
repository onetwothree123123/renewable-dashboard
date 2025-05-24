import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import requests
import math
from datetime import datetime
from lstm_power_regression import LSTMRegressor, WeatherPowerDataset

# 실시간 예측 함수 정의
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

# 모델 및 데이터셋 로드
model = LSTMRegressor()
model.load_state_dict(torch.load("lstm_power_model.pth", map_location=torch.device('cpu')))
model.eval()

dataset = torch.load("weather_dataset.pt", map_location=torch.device('cpu'))

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
        predictions = get_forecast_and_predict(model, dataset)
        st.subheader("📈 24시간 예측 발전량 (Watt)")
        st.line_chart(predictions)
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")

st.markdown("""
---
**사용된 모델:** LSTM (Long Short-Term Memory)\
**기상 API:** [Open-Meteo](https://open-meteo.com/)
""")
