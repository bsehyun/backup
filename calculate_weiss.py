import numpy as np
import pandas as pd

# 예시 데이터
data = {
    "temp_C": [20, 25, 30],          # 수온
    "pressure_hPa": [1010, 1005, 1000],  # 현지기압
    "humidity_percent": [50, 70, 90],    # 상대습도
    "wind_m_s": [2, 4, 1],           # 풍속
    "solar_W_m2": [200, 800, 400],   # 일사량
    "precip_mm": [0, 5, 0]           # 강수량
}

df = pd.DataFrame(data)

# 상수
A1, A2, A3, A4 = -173.4292, 249.6339, 143.3483, -21.8492
P0 = 1013.25  # 기준압력 hPa

def compute_DO(temp_C, pressure_hPa, humidity_percent, wind_m_s, solar_W_m2, precip_mm):
    # 1. DO 포화농도 계산 (Weiss + 수증기압)
    T_K = temp_C + 273.15
    ln_C = A1 + A2*(100/T_K) + A3*np.log(T_K/100) + A4*(T_K/100)
    C_ml_per_L = np.exp(ln_C)
    C_mg_per_L = C_ml_per_L * 1.429

    Es = 6.1078 * 10**((7.5*temp_C)/(temp_C + 237.3))  # 수증기압 계산 hPa
    Pv = Es * humidity_percent / 100
    PO2 = (pressure_hPa - Pv) * 0.2095
    DO_sat = C_mg_per_L * (PO2 / P0)

    # 2. 산소전달 계수 kLa (간단 모델, 풍속과 교반, 일사량, 강수 반영)
    # 풍속 + 강수량으로 표면 교반 고려, 일사량으로 광합성 고려
    kLa_base = 0.5  # 기본값, 현장 측정 필요
    kLa_wind = 0.1 * wind_m_s
    kLa_rain = 0.05 * (precip_mm > 0)
    kLa_solar = 0.0005 * solar_W_m2  # 광합성 영향
    kLa_total = kLa_base + kLa_wind + kLa_rain + kLa_solar

    # 3. DO 실제값 예측 (포화농도 기반 단순 모델)
    # dDO/dt = kLa * (DO_sat - DO_actual), 여기서는 초기 DO = DO_sat*0.8 가정
    DO_actual = 0.8 * DO_sat + kLa_total * (DO_sat - 0.8 * DO_sat)  # 단일 시간 스텝 가정

    return DO_sat, DO_actual, kLa_total

# 적용
results = df.apply(lambda row: compute_DO(row['temp_C'], row['pressure_hPa'], row['humidity_percent'],
                                          row['wind_m_s'], row['solar_W_m2'], row['precip_mm']), axis=1)

df[['DO_sat_mg_L','DO_pred_mg_L','kLa']] = pd.DataFrame(results.tolist(), index=df.index)
print(df)
