#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
폐수 정제 AI 솔루션 - ML 모델 통합
학습된 ML 모델을 실제 시스템에 적용
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class MLWastewaterController:
    """
    ML 모델 기반 폐수 정제 제어 시스템
    """
    
    def __init__(self):
        """ML 모델 로드"""
        try:
            # 개별 모델들
            self.target_do_model = joblib.load('target_do_model.pkl')
            self.target_do_scaler = joblib.load('target_do_scaler.pkl')
            self.blower_control_model = joblib.load('blower_control_model.pkl')
            self.blower_control_scaler = joblib.load('blower_control_scaler.pkl')
            
            # 통합 모델들
            self.integrated_target_do_model = joblib.load('integrated_target_do_model.pkl')
            self.integrated_blower_model = joblib.load('integrated_blower_model.pkl')
            self.integrated_success_model = joblib.load('integrated_success_model.pkl')
            self.integrated_scaler = joblib.load('integrated_scaler.pkl')
            
            print("ML 모델 로드 완료")
            
        except FileNotFoundError:
            print("ML 모델 파일이 없습니다. 먼저 ml_model_training.py를 실행하세요.")
            self.target_do_model = None
            self.target_do_scaler = None
            self.blower_control_model = None
            self.blower_control_scaler = None
            self.integrated_target_do_model = None
            self.integrated_blower_model = None
            self.integrated_success_model = None
            self.integrated_scaler = None
    
    def prepare_features(self, current_data, historical_data=None):
        """
        ML 모델 입력을 위한 특성 준비
        """
        # 기본 특성들
        features = {
            'inflow_rate': current_data.get('inflow_rate', 1000),
            'inflow_cod': current_data.get('inflow_cod', 150),
            'mlss': current_data.get('mlss', 3000),
            'sensor_do_mean': current_data.get('sensor_do', 2.3),
            'sensor_do_std': current_data.get('sensor_do_std', 0.2),
            'sensor_inflow_mean': current_data.get('sensor_inflow', 1000),
            'sensor_inflow_std': current_data.get('sensor_inflow_std', 100),
            'blower_suction_mean': current_data.get('blower_suction', 500),
            'blower_suction_std': current_data.get('blower_suction_std', 50),
            'temp_mean': current_data.get('temperature', 25),
            'temp_std': current_data.get('temp_std', 2),
            'filter_pressure_mean': current_data.get('filter_pressure', 0.5),
            'filter_pressure_std': current_data.get('filter_pressure_std', 0.1),
            'blower_output_mean': current_data.get('blower_output', 480),
            'blower_output_std': current_data.get('blower_output_std', 60)
        }
        
        # 파생 특성들
        features['cod_load'] = features['inflow_rate'] * features['inflow_cod'] / 1000
        features['temp_factor'] = 1 + 0.02 * (features['temp_mean'] - 25)
        features['flow_variability'] = features['sensor_inflow_std'] / features['sensor_inflow_mean']
        features['do_variability'] = features['sensor_do_std'] / features['sensor_do_mean']
        features['blower_efficiency'] = features['blower_output_mean'] / features['blower_suction_mean']
        
        # 과거 데이터가 있는 경우 시계열 특성 추가
        if historical_data is not None and len(historical_data) > 0:
            for lag in [1, 2, 3]:
                if len(historical_data) >= lag:
                    features[f'do_lag_{lag}'] = historical_data.iloc[-lag]['measured_do']
                    features[f'inflow_lag_{lag}'] = historical_data.iloc[-lag]['inflow_rate']
                    features[f'cod_lag_{lag}'] = historical_data.iloc[-lag]['inflow_cod']
                else:
                    features[f'do_lag_{lag}'] = features['sensor_do_mean']
                    features[f'inflow_lag_{lag}'] = features['inflow_rate']
                    features[f'cod_lag_{lag}'] = features['inflow_cod']
            
            # 이동평균 계산
            if len(historical_data) >= 3:
                features['do_ma_3'] = historical_data['measured_do'].tail(3).mean()
                features['inflow_ma_3'] = historical_data['inflow_rate'].tail(3).mean()
                features['cod_ma_3'] = historical_data['inflow_cod'].tail(3).mean()
            else:
                features['do_ma_3'] = features['sensor_do_mean']
                features['inflow_ma_3'] = features['inflow_rate']
                features['cod_ma_3'] = features['inflow_cod']
            
            if len(historical_data) >= 7:
                features['do_ma_7'] = historical_data['measured_do'].tail(7).mean()
                features['inflow_ma_7'] = historical_data['inflow_rate'].tail(7).mean()
                features['cod_ma_7'] = historical_data['inflow_cod'].tail(7).mean()
            else:
                features['do_ma_7'] = features['sensor_do_mean']
                features['inflow_ma_7'] = features['inflow_rate']
                features['cod_ma_7'] = features['inflow_cod']
        else:
            # 과거 데이터가 없는 경우 현재 값으로 채움
            for lag in [1, 2, 3]:
                features[f'do_lag_{lag}'] = features['sensor_do_mean']
                features[f'inflow_lag_{lag}'] = features['inflow_rate']
                features[f'cod_lag_{lag}'] = features['inflow_cod']
            
            features['do_ma_3'] = features['sensor_do_mean']
            features['inflow_ma_3'] = features['inflow_rate']
            features['cod_ma_3'] = features['inflow_cod']
            features['do_ma_7'] = features['sensor_do_mean']
            features['inflow_ma_7'] = features['inflow_rate']
            features['cod_ma_7'] = features['inflow_cod']
        
        # 상호작용 특성들
        features['cod_temp_interaction'] = features['inflow_cod'] * features['temp_mean']
        features['flow_temp_interaction'] = features['inflow_rate'] * features['temp_mean']
        features['mlss_cod_interaction'] = features['mlss'] * features['inflow_cod']
        
        return features
    
    def predict_target_do_ml(self, current_data, historical_data=None):
        """
        ML 모델을 사용한 목표 DO 예측
        """
        if self.target_do_model is None:
            return None
        
        # 특성 준비
        features = self.prepare_features(current_data, historical_data)
        feature_values = list(features.values())
        
        # 스케일링 및 예측
        features_scaled = self.target_do_scaler.transform([feature_values])
        predicted_do = self.target_do_model.predict(features_scaled)[0]
        
        # 안전 범위 내 제한
        return np.clip(predicted_do, 1.5, 4.0)
    
    def predict_blower_percentage_ml(self, current_data, historical_data=None):
        """
        ML 모델을 사용한 송풍량 백분율 예측
        """
        if self.blower_control_model is None:
            return None
        
        # 특성 준비
        features = self.prepare_features(current_data, historical_data)
        feature_values = list(features.values())
        
        # 스케일링 및 예측
        features_scaled = self.blower_control_scaler.transform([feature_values])
        predicted_percentage = self.blower_control_model.predict(features_scaled)[0]
        
        # 94, 96, 98 중 가장 가까운 값으로 매핑
        options = [94, 96, 98]
        return min(options, key=lambda x: abs(x - predicted_percentage))
    
    def integrated_control_ml(self, current_data, historical_data=None):
        """
        통합 ML 모델을 사용한 제어 결정
        """
        if self.integrated_target_do_model is None:
            return None
        
        # 특성 준비
        features = self.prepare_features(current_data, historical_data)
        feature_values = list(features.values())
        
        # 스케일링
        features_scaled = self.integrated_scaler.transform([feature_values])
        
        # 통합 모델 예측
        target_do = self.integrated_target_do_model.predict(features_scaled)[0]
        blower_percentage = self.integrated_blower_model.predict(features_scaled)[0]
        control_success_prob = self.integrated_success_model.predict_proba(features_scaled)[0][1]
        
        # 송풍량 백분율을 옵션으로 매핑
        options = [94, 96, 98]
        blower_percentage = min(options, key=lambda x: abs(x - blower_percentage))
        
        return {
            'target_do': np.clip(target_do, 1.5, 4.0),
            'blower_percentage': blower_percentage,
            'control_success_probability': control_success_prob,
            'recommended_action': f"송풍량을 {blower_percentage}%로 설정하세요 (성공 확률: {control_success_prob:.2%})"
        }
    
    def compare_models(self, current_data, historical_data=None):
        """
        기존 하드코딩된 모델과 ML 모델 비교
        """
        # 기존 하드코딩된 모델
        def calculate_target_do_original(inflow_rate, cod, mlss, temperature=25):
            base_target_do = 2.0
            temp_factor = 1 + 0.02 * (temperature - 25)
            cod_factor = 1 + 0.001 * (cod - 150)
            flow_factor = 1 + 0.0001 * (inflow_rate - 1000)
            mlss_factor = 1 + 0.0001 * (mlss - 3000)
            target_do = base_target_do * temp_factor * cod_factor * flow_factor * mlss_factor
            return np.clip(target_do, 1.5, 4.0)
        
        def determine_blower_percentage_original(do_diff_ratio, current_flow, temperature):
            if do_diff_ratio > 0.2:
                return 98
            elif do_diff_ratio > 0.1:
                return 96
            elif do_diff_ratio > -0.1:
                return 94
            else:
                return 94
        
        # 기존 모델 결과
        original_target_do = calculate_target_do_original(
            current_data.get('inflow_rate', 1000),
            current_data.get('inflow_cod', 150),
            current_data.get('mlss', 3000),
            current_data.get('temperature', 25)
        )
        
        do_diff_ratio = (original_target_do - current_data.get('sensor_do', 2.3)) / original_target_do
        original_blower_percentage = determine_blower_percentage_original(
            do_diff_ratio,
            current_data.get('inflow_rate', 1000),
            current_data.get('temperature', 25)
        )
        
        # ML 모델 결과
        ml_target_do = self.predict_target_do_ml(current_data, historical_data)
        ml_blower_percentage = self.predict_blower_percentage_ml(current_data, historical_data)
        integrated_result = self.integrated_control_ml(current_data, historical_data)
        
        comparison = {
            'original_model': {
                'target_do': original_target_do,
                'blower_percentage': original_blower_percentage,
                'do_diff_ratio': do_diff_ratio
            },
            'ml_models': {
                'target_do_prediction': ml_target_do,
                'blower_percentage_prediction': ml_blower_percentage,
                'integrated_result': integrated_result
            }
        }
        
        return comparison

def create_sample_current_data():
    """샘플 현재 데이터 생성"""
    return {
        'sensor_do': 2.1,
        'inflow_rate': 1100,
        'inflow_cod': 160,
        'mlss': 3200,
        'temperature': 26,
        'sensor_inflow': 1050,
        'blower_suction': 520,
        'blower_output': 500,
        'filter_pressure': 0.52,
        'sensor_do_std': 0.15,
        'sensor_inflow_std': 120,
        'blower_suction_std': 45,
        'temp_std': 2.5,
        'filter_pressure_std': 0.08,
        'blower_output_std': 55
    }

def create_sample_historical_data():
    """샘플 과거 데이터 생성"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    
    historical_data = pd.DataFrame({
        'date': dates,
        'measured_do': np.random.normal(2.4, 0.3, 10),
        'inflow_rate': np.random.normal(1050, 100, 10),
        'inflow_cod': np.random.normal(155, 15, 10),
        'mlss': np.random.normal(3100, 200, 10),
        'temperature': np.random.normal(25.5, 2, 10)
    })
    
    return historical_data

def main():
    """
    ML 모델 통합 시스템 테스트
    """
    print("=== ML 모델 통합 시스템 테스트 ===\n")
    
    # 1. ML 컨트롤러 초기화
    print("1. ML 컨트롤러 초기화...")
    controller = MLWastewaterController()
    
    if controller.target_do_model is None:
        print("ML 모델이 로드되지 않았습니다. 먼저 ml_model_training.py를 실행하세요.")
        return
    
    # 2. 샘플 데이터 생성
    print("\n2. 샘플 데이터 생성...")
    current_data = create_sample_current_data()
    historical_data = create_sample_historical_data()
    
    print("현재 상태:")
    for key, value in current_data.items():
        print(f"  {key}: {value}")
    
    # 3. 모델 비교
    print("\n3. 모델 비교...")
    comparison = controller.compare_models(current_data, historical_data)
    
    print("=== 모델 비교 결과 ===")
    print("\n기존 하드코딩된 모델:")
    print(f"  목표 DO: {comparison['original_model']['target_do']:.3f} mg/L")
    print(f"  송풍량 백분율: {comparison['original_model']['blower_percentage']}%")
    print(f"  DO 차이 비율: {comparison['original_model']['do_diff_ratio']:.3f}")
    
    print("\nML 모델:")
    print(f"  목표 DO 예측: {comparison['ml_models']['target_do_prediction']:.3f} mg/L")
    print(f"  송풍량 백분율 예측: {comparison['ml_models']['blower_percentage_prediction']}%")
    
    if comparison['ml_models']['integrated_result']:
        integrated = comparison['ml_models']['integrated_result']
        print(f"  통합 모델 목표 DO: {integrated['target_do']:.3f} mg/L")
        print(f"  통합 모델 송풍량 백분율: {integrated['blower_percentage']}%")
        print(f"  제어 성공 확률: {integrated['control_success_probability']:.2%}")
        print(f"  권장 액션: {integrated['recommended_action']}")
    
    # 4. 시각화
    print("\n4. 결과 시각화...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 목표 DO 비교
    models = ['기존 모델', 'ML 모델', '통합 모델']
    target_dos = [
        comparison['original_model']['target_do'],
        comparison['ml_models']['target_do_prediction'],
        comparison['ml_models']['integrated_result']['target_do'] if comparison['ml_models']['integrated_result'] else 0
    ]
    
    axes[0,0].bar(models, target_dos, color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[0,0].set_ylabel('목표 DO (mg/L)')
    axes[0,0].set_title('목표 DO 비교')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 송풍량 백분율 비교
    blower_percentages = [
        comparison['original_model']['blower_percentage'],
        comparison['ml_models']['blower_percentage_prediction'],
        comparison['ml_models']['integrated_result']['blower_percentage'] if comparison['ml_models']['integrated_result'] else 0
    ]
    
    axes[0,1].bar(models, blower_percentages, color=['#ff9999', '#66b3ff', '#99ff99'])
    axes[0,1].set_ylabel('송풍량 백분율 (%)')
    axes[0,1].set_title('송풍량 백분율 비교')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 현재 상태 vs 목표
    current_do = current_data['sensor_do']
    target_dos_plot = [comparison['original_model']['target_do'], 
                       comparison['ml_models']['target_do_prediction'],
                       comparison['ml_models']['integrated_result']['target_do'] if comparison['ml_models']['integrated_result'] else 0]
    
    axes[1,0].bar(['현재 DO'] + models, [current_do] + target_dos_plot, 
                   color=['#ffcc99'] + ['#ff9999', '#66b3ff', '#99ff99'])
    axes[1,0].set_ylabel('DO (mg/L)')
    axes[1,0].set_title('현재 DO vs 목표 DO')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 제어 성공 확률 (통합 모델만)
    if comparison['ml_models']['integrated_result']:
        success_prob = comparison['ml_models']['integrated_result']['control_success_probability']
        axes[1,1].pie([success_prob, 1-success_prob], 
                      labels=['성공 확률', '실패 확률'], 
                      autopct='%1.1f%%', 
                      colors=['#99ff99', '#ff9999'])
        axes[1,1].set_title('제어 성공 확률 (통합 모델)')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== ML 모델 통합 시스템 테스트 완료 ===")

if __name__ == "__main__":
    main() 
