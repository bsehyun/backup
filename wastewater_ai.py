#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
폐수 정제 AI 솔루션
목표: 유입량과 COD를 기반으로 blower 송풍량 자동 조절

변수 정의:
- aeration: 하루에 한 번 실측한 DO, MLSS, 유입 COD 값 dataframe
- sensor: 1분에 한 번 찍힌 DO, 유입량, blower 정보들 dataframe

조건:
1. 실측이 센서보다 정확하다
2. 송풍량은 목표값백분율로 정해지는데, 94, 96, 98 실험본밖에 없다
3. 목표값백분율을 바꾸는 데에 드는 cost가 크기 때문에 며칠에 한 번씩밖에 바꾸지 않는다
4. input은 실측이 아니라 센서 데이터다
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def create_sample_data():
    """샘플 데이터 생성 (실제 데이터로 교체 필요)"""
    np.random.seed(42)
    
    # aeration 데이터 (하루에 한 번 실측)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    aeration = pd.DataFrame({
        'date': dates,
        'measured_do': np.random.normal(2.5, 0.3, 30),  # 실측 DO
        'mlss': np.random.normal(3000, 200, 30),         # MLSS
        'inflow_cod': np.random.normal(150, 20, 30),     # 유입 COD
        'inflow_rate': np.random.normal(1000, 100, 30)   # 유입량
    })
    
    # sensor 데이터 (1분에 한 번)
    sensor_times = pd.date_range('2024-01-01', periods=30*24*60, freq='1min')
    sensor = pd.DataFrame({
        'timestamp': sensor_times,
        'sensor_do': np.random.normal(2.3, 0.5, len(sensor_times)),      # 센서 DO
        'sensor_inflow_rate': np.random.normal(1000, 150, len(sensor_times)), # 센서 유입량
        'blower_suction_flow': np.random.normal(500, 50, len(sensor_times)),   # blower 흡입 풍량
        'temperature': np.random.normal(25, 3, len(sensor_times)),            # 온도
        'filter_pressure_diff': np.random.normal(0.5, 0.1, len(sensor_times)), # 필터 차압
        'blower_output_flow': np.random.normal(480, 60, len(sensor_times))     # 송풍량
    })
    
    return aeration, sensor

def preprocess_data(aeration, sensor):
    """데이터 전처리 및 통합"""
    # sensor 데이터를 일별로 집계
    sensor['date'] = sensor['timestamp'].dt.date
    daily_sensor = sensor.groupby('date').agg({
        'sensor_do': ['mean', 'std'],
        'sensor_inflow_rate': ['mean', 'std'],
        'blower_suction_flow': ['mean', 'std'],
        'temperature': ['mean', 'std'],
        'filter_pressure_diff': ['mean', 'std'],
        'blower_output_flow': ['mean', 'std']
    }).reset_index()
    
    # 컬럼명 정리
    daily_sensor.columns = ['date', 'sensor_do_mean', 'sensor_do_std', 
                           'sensor_inflow_mean', 'sensor_inflow_std',
                           'blower_suction_mean', 'blower_suction_std',
                           'temp_mean', 'temp_std',
                           'filter_pressure_mean', 'filter_pressure_std',
                           'blower_output_mean', 'blower_output_std']
    
    # aeration과 sensor 데이터 통합
    aeration['date'] = pd.to_datetime(aeration['date']).dt.date
    merged_data = pd.merge(aeration, daily_sensor, on='date', how='inner')
    
    return merged_data

def calculate_target_do(inflow_rate, cod, mlss, temperature=25):
    """
    유입량과 COD를 기반으로 목표 DO 계산
    
    Parameters:
    - inflow_rate: 유입량 (m³/day)
    - cod: 유입 COD (mg/L)
    - mlss: MLSS (mg/L)
    - temperature: 온도 (°C)
    
    Returns:
    - target_do: 목표 DO (mg/L)
    """
    # 기본 목표 DO (온도 보정)
    base_target_do = 2.0  # 기본 목표 DO
    
    # 온도 보정 (온도가 높을수록 산소 용해도 감소)
    temp_factor = 1 + 0.02 * (temperature - 25)
    
    # COD 부하에 따른 보정
    cod_factor = 1 + 0.001 * (cod - 150)  # COD가 높을수록 더 많은 산소 필요
    
    # 유입량 부하에 따른 보정
    flow_factor = 1 + 0.0001 * (inflow_rate - 1000)  # 유입량이 높을수록 더 많은 산소 필요
    
    # MLSS에 따른 보정
    mlss_factor = 1 + 0.0001 * (mlss - 3000)  # MLSS가 높을수록 더 많은 산소 필요
    
    target_do = base_target_do * temp_factor * cod_factor * flow_factor * mlss_factor
    
    # 목표 DO 범위 제한 (1.5 ~ 4.0 mg/L)
    target_do = np.clip(target_do, 1.5, 4.0)
    
    return target_do

def determine_blower_percentage(do_diff_ratio, current_flow, temperature):
    """
    DO 차이를 기반으로 송풍량 백분율 결정
    
    Parameters:
    - do_diff_ratio: (목표 DO - 현재 DO) / 목표 DO
    - current_flow: 현재 유입량
    - temperature: 온도
    
    Returns:
    - blower_percentage: 94, 96, 98 중 하나
    """
    
    # DO 차이가 클수록 높은 백분율 필요
    if do_diff_ratio > 0.2:  # DO가 목표보다 20% 이상 낮음
        return 98
    elif do_diff_ratio > 0.1:  # DO가 목표보다 10-20% 낮음
        return 96
    elif do_diff_ratio > -0.1:  # DO가 목표 범위 내
        return 94
    else:  # DO가 목표보다 높음 (과도한 폭기)
        return 94  # 최소 백분율로 조절

def get_current_status(sensor_data, aeration_data):
    """현재 상태 정보 반환"""
    latest_sensor = sensor_data.iloc[-1]
    latest_aeration = aeration_data.iloc[-1]
    
    current_status = {
        'current_do': latest_sensor['sensor_do'],
        'current_inflow': latest_sensor['sensor_inflow_rate'],
        'current_temp': latest_sensor['temperature'],
        'current_blower_output': latest_sensor['blower_output_flow'],
        'measured_do': latest_aeration['measured_do'],
        'measured_cod': latest_aeration['inflow_cod'],
        'measured_mlss': latest_aeration['mlss']
    }
    
    return current_status

def calculate_control_action(current_status):
    """현재 상태를 기반으로 제어 액션 결정"""
    # 목표 DO 계산
    target_do = calculate_target_do(
        current_status['current_inflow'],
        current_status['measured_cod'],
        current_status['measured_mlss'],
        current_status['current_temp']
    )
    
    # DO 차이 계산
    do_diff_ratio = (target_do - current_status['current_do']) / target_do
    
    # 송풍량 백분율 결정
    blower_percentage = determine_blower_percentage(
        do_diff_ratio, 
        current_status['current_inflow'], 
        current_status['current_temp']
    )
    
    control_action = {
        'target_do': target_do,
        'do_diff': target_do - current_status['current_do'],
        'do_diff_ratio': do_diff_ratio,
        'blower_percentage': blower_percentage,
        'recommended_action': f"송풍량을 {blower_percentage}%로 설정하세요"
    }
    
    return control_action

def evaluate_system_performance(merged_data):
    """시스템 성능 평가"""
    # DO 제어 정확도 계산
    merged_data['do_control_error'] = abs(merged_data['measured_do'] - merged_data['target_do'])
    avg_control_error = merged_data['do_control_error'].mean()
    max_control_error = merged_data['do_control_error'].max()
    
    # 송풍량 백분율 변경 빈도 분석
    merged_data['blower_change'] = merged_data['blower_percentage'].diff().abs()
    change_frequency = (merged_data['blower_change'] > 0).sum()
    total_days = len(merged_data)
    
    performance_metrics = {
        'avg_control_error': avg_control_error,
        'max_control_error': max_control_error,
        'change_frequency': change_frequency,
        'total_days': total_days,
        'avg_change_interval': total_days / change_frequency if change_frequency > 0 else float('inf')
    }
    
    return performance_metrics

def visualize_results(merged_data):
    """결과 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. DO 제어 성능
    axes[0,0].plot(merged_data['date'], merged_data['measured_do'], 'o-', label='실측 DO', linewidth=2)
    axes[0,0].plot(merged_data['date'], merged_data['target_do'], 's-', label='목표 DO', linewidth=2)
    axes[0,0].fill_between(merged_data['date'], 
                           merged_data['target_do'] - 0.5, 
                           merged_data['target_do'] + 0.5, 
                           alpha=0.3, label='허용 범위')
    axes[0,0].set_xlabel('날짜')
    axes[0,0].set_ylabel('DO (mg/L)')
    axes[0,0].set_title('DO 제어 성능')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 송풍량 백분율 변화
    axes[0,1].plot(merged_data['date'], merged_data['blower_percentage'], 'o-', linewidth=2)
    axes[0,1].set_xlabel('날짜')
    axes[0,1].set_ylabel('송풍량 백분율 (%)')
    axes[0,1].set_title('송풍량 백분율 변화')
    axes[0,1].set_ylim(93, 99)
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 제어 오차 분포
    axes[1,0].hist(merged_data['do_control_error'], bins=10, alpha=0.7, edgecolor='black')
    axes[1,0].axvline(merged_data['do_control_error'].mean(), color='red', linestyle='--', 
                       label=f'평균: {merged_data["do_control_error"].mean():.3f}')
    axes[1,0].set_xlabel('DO 제어 오차 (mg/L)')
    axes[1,0].set_ylabel('빈도')
    axes[1,0].set_title('DO 제어 오차 분포')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 유입량 vs DO 관계
    scatter = axes[1,1].scatter(merged_data['inflow_rate'], merged_data['measured_do'], 
                                c=merged_data['blower_percentage'], cmap='viridis', s=50)
    axes[1,1].set_xlabel('유입량 (m³/day)')
    axes[1,1].set_ylabel('실측 DO (mg/L)')
    axes[1,1].set_title('유입량 vs DO (색상: 송풍량 백분율)')
    plt.colorbar(scatter, ax=axes[1,1], label='송풍량 백분율 (%)')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """메인 실행 함수"""
    print("=== 폐수 정제 AI 솔루션 ===\n")
    
    # 1. 데이터 생성
    print("1. 데이터 생성 중...")
    aeration, sensor = create_sample_data()
    print(f"   - Aeration 데이터: {len(aeration)}일")
    print(f"   - Sensor 데이터: {len(sensor)}개 측정값")
    
    # 2. 데이터 전처리
    print("\n2. 데이터 전처리 중...")
    merged_data = preprocess_data(aeration, sensor)
    print(f"   - 통합된 데이터: {merged_data.shape}")
    
    # 3. 목표 DO 계산
    print("\n3. 목표 DO 계산 중...")
    merged_data['target_do'] = calculate_target_do(
        merged_data['inflow_rate'], 
        merged_data['inflow_cod'], 
        merged_data['mlss'], 
        merged_data['temp_mean']
    )
    
    # 4. 송풍량 백분율 결정
    print("\n4. 송풍량 백분율 결정 중...")
    merged_data['do_diff'] = merged_data['target_do'] - merged_data['measured_do']
    merged_data['do_diff_ratio'] = merged_data['do_diff'] / merged_data['target_do']
    
    merged_data['blower_percentage'] = merged_data.apply(
        lambda row: determine_blower_percentage(
            row['do_diff_ratio'], 
            row['inflow_rate'], 
            row['temp_mean']
        ), axis=1
    )
    
    # 5. 실시간 제어 시뮬레이션
    print("\n5. 실시간 제어 시뮬레이션...")
    current_status = get_current_status(sensor, aeration)
    control_action = calculate_control_action(current_status)
    
    print("   현재 상태:")
    for key, value in current_status.items():
        print(f"     {key}: {value:.2f}")
    
    print("\n   제어 액션:")
    for key, value in control_action.items():
        if isinstance(value, float):
            print(f"     {key}: {value:.3f}")
        else:
            print(f"     {key}: {value}")
    
    # 6. 성능 평가
    print("\n6. 성능 평가 중...")
    performance_metrics = evaluate_system_performance(merged_data)
    
    print(f"   평균 DO 제어 오차: {performance_metrics['avg_control_error']:.3f} mg/L")
    print(f"   최대 DO 제어 오차: {performance_metrics['max_control_error']:.3f} mg/L")
    print(f"   송풍량 변경 빈도: {performance_metrics['change_frequency']}번")
    print(f"   평균 변경 간격: {performance_metrics['avg_change_interval']:.1f}일")
    
    # 7. 결과 시각화
    print("\n7. 결과 시각화...")
    visualize_results(merged_data)
    
    # 8. 시스템 요약
    print("\n=== 시스템 요약 ===")
    print("1. 시스템 구성:")
    print("   - 목표 DO 계산: 유입량, COD, MLSS, 온도 기반")
    print("   - 송풍량 제어: 94%, 96%, 98% 중 선택")
    print("   - 실시간 모니터링: 센서 데이터 기반")
    
    print("\n2. 주요 특징:")
    print(f"   - 평균 DO 제어 오차: {performance_metrics['avg_control_error']:.3f} mg/L")
    print(f"   - 송풍량 변경 빈도: {performance_metrics['avg_change_interval']:.1f}일에 한 번")
    print("   - 데이터 활용: 실측 데이터로 모델 보정, 센서 데이터로 실시간 제어")
    
    print("\n3. 권장사항:")
    print("   - 실측 데이터와 센서 데이터의 정기적 교정 필요")
    print("   - 송풍량 변경 시 안정화 시간 고려")
    print("   - 계절적 변화에 따른 모델 재보정 필요")
    print("   - 에너지 효율성과 처리 효율성의 균형 유지")

if __name__ == "__main__":
    main()
