#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
폐수 정제 AI 솔루션 - ML 모델 학습
송풍량 제어 모델과 통합 제어 모델 학습
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def create_training_data():
    """
    학습용 데이터 생성 (실제 데이터로 교체 필요)
    """
    np.random.seed(42)
    
    # 30일간의 데이터 생성
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    
    # 기본 변수들 생성
    data = pd.DataFrame({
        'date': dates,
        # 실측 데이터 (하루에 한 번)
        'measured_do': np.random.normal(2.5, 0.3, 30),
        'mlss': np.random.normal(3000, 200, 30),
        'inflow_cod': np.random.normal(150, 20, 30),
        'inflow_rate': np.random.normal(1000, 100, 30),
        
        # 센서 데이터 (일별 평균)
        'sensor_do_mean': np.random.normal(2.3, 0.5, 30),
        'sensor_do_std': np.random.uniform(0.1, 0.3, 30),
        'sensor_inflow_mean': np.random.normal(1000, 150, 30),
        'sensor_inflow_std': np.random.uniform(50, 200, 30),
        'blower_suction_mean': np.random.normal(500, 50, 30),
        'blower_suction_std': np.random.uniform(20, 80, 30),
        'temp_mean': np.random.normal(25, 3, 30),
        'temp_std': np.random.uniform(1, 5, 30),
        'filter_pressure_mean': np.random.normal(0.5, 0.1, 30),
        'filter_pressure_std': np.random.uniform(0.05, 0.15, 30),
        'blower_output_mean': np.random.normal(480, 60, 30),
        'blower_output_std': np.random.uniform(30, 90, 30),
    })
    
    return data

def create_target_variables(data):
    """
    목표 변수 생성
    """
    # 1. 목표 DO 계산 (기존 로직 사용)
    data['target_do'] = calculate_target_do_original(
        data['inflow_rate'], 
        data['inflow_cod'], 
        data['mlss'], 
        data['temp_mean']
    )
    
    # 2. DO 차이 계산
    data['do_diff'] = data['target_do'] - data['measured_do']
    data['do_diff_ratio'] = data['do_diff'] / data['target_do']
    
    # 3. 송풍량 백분율 결정 (기존 로직 사용)
    data['blower_percentage'] = data.apply(
        lambda row: determine_blower_percentage_original(
            row['do_diff_ratio'], 
            row['inflow_rate'], 
            row['temp_mean']
        ), axis=1
    )
    
    # 4. 제어 성공 여부 (DO가 목표 범위 내인지)
    data['control_success'] = (abs(data['do_diff']) <= 0.5).astype(int)
    
    # 5. 에너지 효율성 (송풍량 대비 DO 개선도)
    data['energy_efficiency'] = data['do_diff'] / (data['blower_output_mean'] / 100)
    
    return data

def calculate_target_do_original(inflow_rate, cod, mlss, temperature=25):
    """기존 하드코딩된 목표 DO 계산 (라벨 생성용)"""
    base_target_do = 2.0
    temp_factor = 1 + 0.02 * (temperature - 25)
    cod_factor = 1 + 0.001 * (cod - 150)
    flow_factor = 1 + 0.0001 * (inflow_rate - 1000)
    mlss_factor = 1 + 0.0001 * (mlss - 3000)
    
    target_do = base_target_do * temp_factor * cod_factor * flow_factor * mlss_factor
    return np.clip(target_do, 1.5, 4.0)

def determine_blower_percentage_original(do_diff_ratio, current_flow, temperature):
    """기존 하드코딩된 송풍량 제어 (라벨 생성용)"""
    if do_diff_ratio > 0.2:
        return 98
    elif do_diff_ratio > 0.1:
        return 96
    elif do_diff_ratio > -0.1:
        return 94
    else:
        return 94

def prepare_features(data):
    """
    ML 모델 학습을 위한 특성 준비
    """
    # 기본 특성들
    basic_features = [
        'inflow_rate', 'inflow_cod', 'mlss',
        'sensor_do_mean', 'sensor_do_std',
        'sensor_inflow_mean', 'sensor_inflow_std',
        'blower_suction_mean', 'blower_suction_std',
        'temp_mean', 'temp_std',
        'filter_pressure_mean', 'filter_pressure_std',
        'blower_output_mean', 'blower_output_std'
    ]
    
    # 파생 특성들
    data['cod_load'] = data['inflow_rate'] * data['inflow_cod'] / 1000  # COD 부하
    data['temp_factor'] = 1 + 0.02 * (data['temp_mean'] - 25)  # 온도 보정
    data['flow_variability'] = data['sensor_inflow_std'] / data['sensor_inflow_mean']  # 유입량 변동성
    data['do_variability'] = data['sensor_do_std'] / data['sensor_do_mean']  # DO 변동성
    data['blower_efficiency'] = data['blower_output_mean'] / data['blower_suction_mean']  # Blower 효율성
    
    # 시계열 특성들 (과거 데이터 활용)
    for lag in [1, 2, 3]:
        data[f'do_lag_{lag}'] = data['measured_do'].shift(lag)
        data[f'inflow_lag_{lag}'] = data['inflow_rate'].shift(lag)
        data[f'cod_lag_{lag}'] = data['inflow_cod'].shift(lag)
    
    # 이동평균 특성들
    for window in [3, 7]:
        data[f'do_ma_{window}'] = data['measured_do'].rolling(window=window).mean()
        data[f'inflow_ma_{window}'] = data['inflow_rate'].rolling(window=window).mean()
        data[f'cod_ma_{window}'] = data['inflow_cod'].rolling(window=window).mean()
    
    # 상호작용 특성들
    data['cod_temp_interaction'] = data['inflow_cod'] * data['temp_mean']
    data['flow_temp_interaction'] = data['inflow_rate'] * data['temp_mean']
    data['mlss_cod_interaction'] = data['mlss'] * data['inflow_cod']
    
    # 추가 특성들
    derived_features = [
        'cod_load', 'temp_factor', 'flow_variability', 'do_variability', 'blower_efficiency',
        'do_lag_1', 'do_lag_2', 'do_lag_3', 'inflow_lag_1', 'inflow_lag_2', 'inflow_lag_3',
        'cod_lag_1', 'cod_lag_2', 'cod_lag_3', 'do_ma_3', 'do_ma_7', 'inflow_ma_3', 'inflow_ma_7',
        'cod_ma_3', 'cod_ma_7', 'cod_temp_interaction', 'flow_temp_interaction', 'mlss_cod_interaction'
    ]
    
    all_features = basic_features + derived_features
    
    # 결측값 처리
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    return data, all_features

def train_target_do_model(data, features):
    """
    목표 DO 예측 모델 학습
    """
    print("=== 목표 DO 예측 모델 학습 ===")
    
    # 특성과 타겟 준비
    X = data[features].dropna()
    y = data['target_do'].dropna()
    
    # 인덱스 맞추기
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델들 정의
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Linear Regression': LinearRegression(),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    
    # 모델 비교
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        score = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"{name}:")
        print(f"  R² Score: {score:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print()
        
        if score > best_score:
            best_score = score
            best_model = model
    
    # 최적 모델 저장
    joblib.dump(best_model, 'target_do_model.pkl')
    joblib.dump(scaler, 'target_do_scaler.pkl')
    
    return best_model, scaler, X_test_scaled, y_test

def train_blower_control_model(data, features):
    """
    송풍량 제어 모델 학습
    """
    print("=== 송풍량 제어 모델 학습 ===")
    
    # 특성과 타겟 준비
    X = data[features].dropna()
    y = data['blower_percentage'].dropna()
    
    # 인덱스 맞추기
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 모델들 정의
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVC': SVC(kernel='rbf', random_state=42),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    best_model = None
    best_score = -np.inf
    
    # 모델 비교
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        score = model.score(X_test_scaled, y_test)
        
        print(f"{name}:")
        print(f"  Accuracy: {score:.4f}")
        print("  Classification Report:")
        print(classification_report(y_test, y_pred))
        print()
        
        if score > best_score:
            best_score = score
            best_model = model
    
    # 최적 모델 저장
    joblib.dump(best_model, 'blower_control_model.pkl')
    joblib.dump(scaler, 'blower_control_scaler.pkl')
    
    return best_model, scaler, X_test_scaled, y_test

def train_integrated_control_model(data, features):
    """
    통합 제어 모델 학습 (다중 출력)
    """
    print("=== 통합 제어 모델 학습 ===")
    
    # 특성과 타겟 준비
    X = data[features].dropna()
    y_target_do = data['target_do'].dropna()
    y_blower_percentage = data['blower_percentage'].dropna()
    y_control_success = data['control_success'].dropna()
    
    # 인덱스 맞추기
    common_index = X.index.intersection(y_target_do.index).intersection(y_blower_percentage.index).intersection(y_control_success.index)
    X = X.loc[common_index]
    y_target_do = y_target_do.loc[common_index]
    y_blower_percentage = y_blower_percentage.loc[common_index]
    y_control_success = y_control_success.loc[common_index]
    
    # 데이터 분할
    X_train, X_test, y_train_target, y_test_target, y_train_blower, y_test_blower, y_train_success, y_test_success = train_test_split(
        X, y_target_do, y_blower_percentage, y_control_success, test_size=0.2, random_state=42, stratify=y_control_success
    )
    
    # 특성 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 통합 모델 (Random Forest 기반)
    integrated_model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # 목표 DO 예측
    integrated_model.fit(X_train_scaled, y_train_target)
    target_do_pred = integrated_model.predict(X_test_scaled)
    target_do_score = r2_score(y_test_target, target_do_pred)
    
    print(f"통합 모델 - 목표 DO 예측 R² Score: {target_do_score:.4f}")
    
    # 송풍량 제어 (분류)
    blower_model = RandomForestClassifier(n_estimators=100, random_state=42)
    blower_model.fit(X_train_scaled, y_train_blower)
    blower_pred = blower_model.predict(X_test_scaled)
    blower_score = blower_model.score(X_test_scaled, y_test_blower)
    
    print(f"통합 모델 - 송풍량 제어 Accuracy: {blower_score:.4f}")
    
    # 제어 성공 예측
    success_model = RandomForestClassifier(n_estimators=100, random_state=42)
    success_model.fit(X_train_scaled, y_train_success)
    success_pred = success_model.predict(X_test_scaled)
    success_score = success_model.score(X_test_scaled, y_test_success)
    
    print(f"통합 모델 - 제어 성공 예측 Accuracy: {success_score:.4f}")
    
    # 모델들 저장
    joblib.dump(integrated_model, 'integrated_target_do_model.pkl')
    joblib.dump(blower_model, 'integrated_blower_model.pkl')
    joblib.dump(success_model, 'integrated_success_model.pkl')
    joblib.dump(scaler, 'integrated_scaler.pkl')
    
    return integrated_model, blower_model, success_model, scaler

def evaluate_models(data, features):
    """
    모델 성능 평가 및 시각화
    """
    print("=== 모델 성능 평가 ===")
    
    # 목표 DO 모델 평가
    target_do_model = joblib.load('target_do_model.pkl')
    target_do_scaler = joblib.load('target_do_scaler.pkl')
    
    X = data[features].dropna()
    y = data['target_do'].dropna()
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    
    X_scaled = target_do_scaler.transform(X)
    y_pred = target_do_model.predict(X_scaled)
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 실제 vs 예측
    axes[0,0].scatter(y, y_pred, alpha=0.6)
    axes[0,0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0,0].set_xlabel('실제 목표 DO')
    axes[0,0].set_ylabel('예측 목표 DO')
    axes[0,0].set_title('목표 DO 예측 성능')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 예측 오차 분포
    error = y_pred - y
    axes[0,1].hist(error, bins=20, alpha=0.7, edgecolor='black')
    axes[0,1].axvline(0, color='red', linestyle='--')
    axes[0,1].set_xlabel('예측 오차')
    axes[0,1].set_ylabel('빈도')
    axes[0,1].set_title('목표 DO 예측 오차 분포')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 시간별 예측 성능
    axes[1,0].plot(y.index, y, 'o-', label='실제', linewidth=2)
    axes[1,0].plot(y.index, y_pred, 's-', label='예측', linewidth=2)
    axes[1,0].set_xlabel('날짜')
    axes[1,0].set_ylabel('목표 DO (mg/L)')
    axes[1,0].set_title('시간별 목표 DO 예측')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 특성 중요도 (Random Forest인 경우)
    if hasattr(target_do_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': target_do_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = feature_importance.head(10)
        axes[1,1].barh(range(len(top_features)), top_features['importance'])
        axes[1,1].set_yticks(range(len(top_features)))
        axes[1,1].set_yticklabels(top_features['feature'])
        axes[1,1].set_xlabel('중요도')
        axes[1,1].set_title('특성 중요도 (상위 10개)')
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 성능 지표 출력
    print(f"목표 DO 예측 R² Score: {r2_score(y, y_pred):.4f}")
    print(f"목표 DO 예측 RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

def main():
    """
    메인 실행 함수
    """
    print("=== ML 모델 학습 시작 ===\n")
    
    # 1. 데이터 생성
    print("1. 학습 데이터 생성 중...")
    data = create_training_data()
    print(f"   생성된 데이터: {data.shape}")
    
    # 2. 목표 변수 생성
    print("\n2. 목표 변수 생성 중...")
    data = create_target_variables(data)
    print("   목표 변수 생성 완료")
    
    # 3. 특성 준비
    print("\n3. 특성 준비 중...")
    data, features = prepare_features(data)
    print(f"   준비된 특성 수: {len(features)}")
    
    # 4. 모델 학습
    print("\n4. 모델 학습 중...")
    
    # 목표 DO 예측 모델
    target_do_model, target_do_scaler, _, _ = train_target_do_model(data, features)
    
    # 송풍량 제어 모델
    blower_model, blower_scaler, _, _ = train_blower_control_model(data, features)
    
    # 통합 제어 모델
    integrated_model, integrated_blower_model, integrated_success_model, integrated_scaler = train_integrated_control_model(data, features)
    
    # 5. 모델 평가
    print("\n5. 모델 평가 중...")
    evaluate_models(data, features)
    
    print("\n=== ML 모델 학습 완료 ===")
    print("저장된 모델들:")
    print("- target_do_model.pkl")
    print("- blower_control_model.pkl")
    print("- integrated_target_do_model.pkl")
    print("- integrated_blower_model.pkl")
    print("- integrated_success_model.pkl")
    print("- 각 모델에 대응하는 scaler 파일들")

if __name__ == "__main__":
    main() 
