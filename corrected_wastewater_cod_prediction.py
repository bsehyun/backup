#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
올바른 폐수 처리 폭기조 COD 예측 시스템
480개 데이터 포인트를 실제로 활용하는 방법
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class CorrectedWastewaterCODPredictor:
    """
    올바른 폐수 처리 COD 예측 클래스
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
        self.feature_generator = None
        
    def create_synthetic_data(self, n_samples=160):
        """
        실제 폐수 처리 현장을 시뮬레이션한 데이터 생성
        """
        print("=== 폐수 처리 현장 데이터 생성 ===\n")
        
        np.random.seed(42)
        
        # 시간 인덱스 (3초마다 데이터 수집)
        time_index = pd.date_range('2024-01-01', periods=n_samples, freq='3S')
        
        # 기본 환경 변수들
        temperature = 25 + 5 * np.sin(np.arange(n_samples) * 2 * np.pi / (24 * 1200)) + np.random.normal(0, 1, n_samples)
        ph = 7.0 + 0.5 * np.sin(np.arange(n_samples) * np.pi / 600) + np.random.normal(0, 0.1, n_samples)
        dissolved_oxygen = 2.0 + 1.0 * np.sin(np.arange(n_samples) * np.pi / 800) + np.random.normal(0, 0.2, n_samples)
        
        # 유입량 관련 변수들
        flow_rate = 1000 + 200 * np.sin(np.arange(n_samples) * np.pi / 1000) + np.random.normal(0, 50, n_samples)
        conductivity = 800 + 100 * np.sin(np.arange(n_samples) * np.pi / 1200) + np.random.normal(0, 20, n_samples)
        
        # 슬러지 관련 변수들
        mlss = 3000 + 500 * np.sin(np.arange(n_samples) * np.pi / 1500) + np.random.normal(0, 100, n_samples)
        sludge_volume = 200 + 50 * np.sin(np.arange(n_samples) * np.pi / 2000) + np.random.normal(0, 10, n_samples)
        
        # 각 폭기조별 특성 (약간의 차이)
        # 폭기조 1 (첫 번째)
        aeration_tank1_do = dissolved_oxygen + np.random.normal(0, 0.1, n_samples)
        aeration_tank1_temp = temperature + np.random.normal(0, 0.2, n_samples)
        aeration_tank1_ph = ph + np.random.normal(0, 0.05, n_samples)
        
        # 폭기조 2 (중간 - 타겟)
        aeration_tank2_do = dissolved_oxygen + np.random.normal(0, 0.1, n_samples)
        aeration_tank2_temp = temperature + np.random.normal(0, 0.2, n_samples)
        aeration_tank2_ph = ph + np.random.normal(0, 0.05, n_samples)
        
        # 폭기조 3 (마지막)
        aeration_tank3_do = dissolved_oxygen + np.random.normal(0, 0.1, n_samples)
        aeration_tank3_temp = temperature + np.random.normal(0, 0.2, n_samples)
        aeration_tank3_ph = ph + np.random.normal(0, 0.05, n_samples)
        
        # COD 값 생성 (실제로는 하루에 한 번 측정)
        base_cod = 150 + 30 * np.sin(np.arange(n_samples) * np.pi / 1000) + np.random.normal(0, 10, n_samples)
        
        cod_tank1 = base_cod + np.random.normal(0, 5, n_samples)
        cod_tank2 = base_cod + np.random.normal(0, 5, n_samples)  # 타겟
        cod_tank3 = base_cod + np.random.normal(0, 5, n_samples)
        
        # 데이터프레임 생성
        data = pd.DataFrame({
            'timestamp': time_index,
            'temperature': temperature,
            'ph': ph,
            'flow_rate': flow_rate,
            'conductivity': conductivity,
            'mlss': mlss,
            'sludge_volume': sludge_volume,
            
            # 폭기조 1 센서 데이터
            'tank1_do': aeration_tank1_do,
            'tank1_temp': aeration_tank1_temp,
            'tank1_ph': aeration_tank1_ph,
            'cod_tank1': cod_tank1,
            
            # 폭기조 2 센서 데이터 (타겟)
            'tank2_do': aeration_tank2_do,
            'tank2_temp': aeration_tank2_temp,
            'tank2_ph': aeration_tank2_ph,
            'cod_tank2': cod_tank2,  # 예측 대상
            
            # 폭기조 3 센서 데이터
            'tank3_do': aeration_tank3_do,
            'tank3_temp': aeration_tank3_temp,
            'tank3_ph': aeration_tank3_ph,
            'cod_tank3': cod_tank3,
        })
        
        print(f"생성된 데이터 크기: {data.shape}")
        print(f"시간 범위: {data['timestamp'].min()} ~ {data['timestamp'].max()}")
        print(f"총 데이터 포인트: {len(data)}")
        
        return data
    
    def create_480_datapoints(self, data):
        """
        160개 데이터를 480개로 확장하는 올바른 방법
        각 시간점에서 3개의 폭기조를 각각 독립적인 데이터 포인트로 만듦
        """
        print("\n=== 480개 데이터 포인트 생성 ===\n")
        
        # 각 폭기조를 독립적인 데이터 포인트로 변환
        expanded_data = []
        
        for idx, row in data.iterrows():
            # 폭기조 1 데이터 포인트
            tank1_data = {
                'timestamp': row['timestamp'],
                'tank_id': 1,
                'target_cod': row['cod_tank1'],  # 각 폭기조의 COD가 타겟
                'temperature': row['temperature'],
                'ph': row['ph'],
                'flow_rate': row['flow_rate'],
                'conductivity': row['conductivity'],
                'mlss': row['mlss'],
                'sludge_volume': row['sludge_volume'],
                'do': row['tank1_do'],
                'temp': row['tank1_temp'],
                'ph_tank': row['tank1_ph'],
                # 다른 폭기조의 센서 정보 (COD 제외)
                'tank2_do': row['tank2_do'],
                'tank2_temp': row['tank2_temp'],
                'tank2_ph': row['tank2_ph'],
                'tank3_do': row['tank3_do'],
                'tank3_temp': row['tank3_temp'],
                'tank3_ph': row['tank3_ph'],
            }
            expanded_data.append(tank1_data)
            
            # 폭기조 2 데이터 포인트
            tank2_data = {
                'timestamp': row['timestamp'],
                'tank_id': 2,
                'target_cod': row['cod_tank2'],
                'temperature': row['temperature'],
                'ph': row['ph'],
                'flow_rate': row['flow_rate'],
                'conductivity': row['conductivity'],
                'mlss': row['mlss'],
                'sludge_volume': row['sludge_volume'],
                'do': row['tank2_do'],
                'temp': row['tank2_temp'],
                'ph_tank': row['tank2_ph'],
                # 다른 폭기조의 센서 정보 (COD 제외)
                'tank1_do': row['tank1_do'],
                'tank1_temp': row['tank1_temp'],
                'tank1_ph': row['tank1_ph'],
                'tank3_do': row['tank3_do'],
                'tank3_temp': row['tank3_temp'],
                'tank3_ph': row['tank3_ph'],
            }
            expanded_data.append(tank2_data)
            
            # 폭기조 3 데이터 포인트
            tank3_data = {
                'timestamp': row['timestamp'],
                'tank_id': 3,
                'target_cod': row['cod_tank3'],
                'temperature': row['temperature'],
                'ph': row['ph'],
                'flow_rate': row['flow_rate'],
                'conductivity': row['conductivity'],
                'mlss': row['mlss'],
                'sludge_volume': row['sludge_volume'],
                'do': row['tank3_do'],
                'temp': row['tank3_temp'],
                'ph_tank': row['tank3_ph'],
                # 다른 폭기조의 센서 정보 (COD 제외)
                'tank1_do': row['tank1_do'],
                'tank1_temp': row['tank1_temp'],
                'tank1_ph': row['tank1_ph'],
                'tank2_do': row['tank2_do'],
                'tank2_temp': row['tank2_temp'],
                'tank2_ph': row['tank2_ph'],
            }
            expanded_data.append(tank3_data)
        
        expanded_df = pd.DataFrame(expanded_data)
        
        print(f"확장된 데이터 크기: {expanded_df.shape}")
        print(f"실제 데이터 포인트: {len(expanded_df)}")
        print(f"폭기조별 데이터 분포:")
        print(expanded_df['tank_id'].value_counts().sort_index())
        
        return expanded_df
    
    def auto_generate_features(self, data):
        """
        자동화된 특성 생성 시스템
        """
        print("\n=== 자동화된 특성 생성 ===\n")
        
        # 기본 특성들 (자동으로 찾기)
        base_features = ['temperature', 'ph', 'flow_rate', 'conductivity', 'mlss', 'sludge_volume']
        tank_features = ['do', 'temp', 'ph_tank']
        other_tank_features = [col for col in data.columns if col.startswith('tank') and col != 'tank_id']
        
        # 시간 관련 특성 자동 생성
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        
        # 주기적 특성 자동 생성
        periodic_features = []
        for feature, period in [('hour', 24), ('day_of_week', 7), ('month', 12)]:
            data[f'{feature}_sin'] = np.sin(2 * np.pi * data[feature] / period)
            data[f'{feature}_cos'] = np.cos(2 * np.pi * data[feature] / period)
            periodic_features.extend([f'{feature}_sin', f'{feature}_cos'])
        
        # 폭기조 간 차이 특성 자동 생성
        diff_features = []
        for tank_feat in tank_features:
            for other_tank in [1, 2, 3]:
                if other_tank != data['tank_id'].iloc[0]:  # 현재 폭기조가 아닌 경우
                    other_col = f'tank{other_tank}_{tank_feat.split("_")[0]}'
                    if other_col in data.columns:
                        diff_col = f'{tank_feat}_diff_tank{other_tank}'
                        data[diff_col] = data[tank_feat] - data[other_col]
                        diff_features.append(diff_col)
        
        # 상호작용 특성 자동 생성
        interaction_features = []
        for i, feat1 in enumerate(tank_features):
            for feat2 in tank_features[i+1:]:
                interaction_col = f'{feat1}_{feat2}_interaction'
                data[interaction_col] = data[feat1] * data[feat2]
                interaction_features.append(interaction_col)
        
        # 환경 변수와의 상호작용
        for env_feat in base_features:
            for tank_feat in tank_features:
                interaction_col = f'{env_feat}_{tank_feat}_interaction'
                data[interaction_col] = data[env_feat] * data[tank_feat]
                interaction_features.append(interaction_col)
        
        # 이동 평균 특성 자동 생성
        ma_features = []
        for window in [3, 5, 10]:
            for tank_feat in tank_features:
                ma_col = f'{tank_feat}_ma_{window}'
                data[ma_col] = data.groupby('tank_id')[tank_feat].rolling(
                    window=window, min_periods=1).mean().reset_index(0, drop=True)
                ma_features.append(ma_col)
        
        # 표준편차 특성 자동 생성
        std_features = []
        for window in [3, 5]:
            for tank_feat in tank_features:
                std_col = f'{tank_feat}_std_{window}'
                data[std_col] = data.groupby('tank_id')[tank_feat].rolling(
                    window=window, min_periods=1).std().reset_index(0, drop=True)
                std_features.append(std_col)
        
        # 비율 특성 자동 생성
        ratio_features = []
        for i, feat1 in enumerate(tank_features):
            for feat2 in tank_features[i+1:]:
                ratio_col = f'{feat1}_{feat2}_ratio'
                data[ratio_col] = data[feat1] / (data[feat2] + 1e-8)
                ratio_features.append(ratio_col)
        
        # 최종 특성 리스트 자동 생성
        all_features = (base_features + tank_features + other_tank_features + 
                       periodic_features + diff_features + interaction_features + 
                       ma_features + std_features + ratio_features)
        
        # 결측값 처리
        data[all_features] = data[all_features].fillna(0)
        
        X = data[all_features]
        y = data['target_cod']
        
        print(f"자동 생성된 특성 개수: {len(all_features)}")
        print(f"입력 데이터 크기: {X.shape}")
        print(f"타겟 데이터 크기: {y.shape}")
        print(f"실제 데이터 포인트: {len(X)}")
        
        return X, y, all_features
    
    def train_models(self, X, y):
        """
        다양한 모델 훈련 및 비교
        """
        print("\n=== 모델 훈련 및 비교 ===\n")
        
        # 데이터 분할 (시간 순서 유지)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # 모델 정의
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # 모델 훈련 및 평가
        results = {}
        
        for name, model in models.items():
            print(f"훈련 중: {name}")
            
            # 모델 훈련
            model.fit(X_train, y_train)
            
            # 예측
            y_pred = model.predict(X_test)
            
            # 평가 지표
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'y_pred': y_pred
            }
            
            print(f"  MSE: {mse:.2f}")
            print(f"  MAE: {mae:.2f}")
            print(f"  R²: {r2:.3f}")
            print()
        
        # 최고 성능 모델 선택
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.best_model = results[best_model_name]['model']
        
        print(f"최고 성능 모델: {best_model_name}")
        print(f"R² Score: {results[best_model_name]['r2']:.3f}")
        
        return results, X_test, y_test
    
    def analyze_feature_importance(self, X, feature_names):
        """
        특성 중요도 분석
        """
        print("\n=== 특성 중요도 분석 ===\n")
        
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            importance = np.abs(self.best_model.coef_)
        else:
            print("이 모델은 특성 중요도를 제공하지 않습니다.")
            return
        
        # 특성 중요도 데이터프레임 생성
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = feature_importance_df
        
        print("상위 15개 중요 특성:")
        print(feature_importance_df.head(15))
        
        return feature_importance_df
    
    def visualize_results(self, data, results, X_test, y_test):
        """
        결과 시각화
        """
        print("\n=== 결과 시각화 ===\n")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 실제 vs 예측 비교 (최고 성능 모델)
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        y_pred_best = results[best_model_name]['y_pred']
        
        axes[0,0].scatter(y_test, y_pred_best, alpha=0.6)
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('실제 COD 값')
        axes[0,0].set_ylabel('예측 COD 값')
        axes[0,0].set_title(f'{best_model_name} - 실제 vs 예측')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 시간별 COD 변화
        test_indices = X_test.index
        axes[0,1].plot(test_indices, y_test.values, 'o-', label='실제', linewidth=2)
        axes[0,1].plot(test_indices, y_pred_best, 's-', label='예측', linewidth=2)
        axes[0,1].set_xlabel('시간 인덱스')
        axes[0,1].set_ylabel('COD (mg/L)')
        axes[0,1].set_title('시간별 COD 변화')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 모델 성능 비교
        model_names = list(results.keys())
        r2_scores = [results[name]['r2'] for name in model_names]
        
        bars = axes[0,2].bar(model_names, r2_scores, alpha=0.7)
        axes[0,2].set_ylabel('R² Score')
        axes[0,2].set_title('모델 성능 비교')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3)
        
        # 최고 성능 모델 하이라이트
        best_idx = np.argmax(r2_scores)
        bars[best_idx].set_color('red')
        
        # 4. 특성 중요도 (상위 10개)
        if self.feature_importance is not None:
            top_features = self.feature_importance.head(10)
            axes[1,0].barh(top_features['feature'], top_features['importance'])
            axes[1,0].set_xlabel('중요도')
            axes[1,0].set_title('상위 10개 중요 특성')
            axes[1,0].grid(True, alpha=0.3)
        
        # 5. 예측 오차 분포
        residuals = y_test - y_pred_best
        axes[1,1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[1,1].axvline(0, color='red', linestyle='--', label='오차 없음')
        axes[1,1].set_xlabel('예측 오차')
        axes[1,1].set_ylabel('빈도')
        axes[1,1].set_title('예측 오차 분포')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. 폭기조별 COD 분포
        test_data = data.iloc[test_indices]
        tank_data = [test_data[test_data['tank_id'] == i]['target_cod'] for i in [1, 2, 3]]
        axes[1,2].boxplot(tank_data, labels=['폭기조 1', '폭기조 2', '폭기조 3'])
        axes[1,2].set_ylabel('COD (mg/L)')
        axes[1,2].set_title('폭기조별 COD 분포')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    메인 실행 함수
    """
    print("=== 올바른 폐수 처리 COD 예측 시스템 ===\n")
    
    # 1. 예측기 초기화
    predictor = CorrectedWastewaterCODPredictor()
    
    # 2. 데이터 생성
    data = predictor.create_synthetic_data(n_samples=160)
    
    # 3. 480개 데이터 포인트로 확장
    expanded_data = predictor.create_480_datapoints(data)
    
    # 4. 자동화된 특성 엔지니어링
    X, y, feature_names = predictor.auto_generate_features(expanded_data)
    
    # 5. 모델 훈련 및 비교
    results, X_test, y_test = predictor.train_models(X, y)
    
    # 6. 특성 중요도 분석
    feature_importance = predictor.analyze_feature_importance(X, feature_names)
    
    # 7. 결과 시각화
    predictor.visualize_results(expanded_data, results, X_test, y_test)
    
    # 8. 결론
    print("\n=== 결론 ===\n")
    print("✅ 올바른 접근 방식:")
    print("   - 160개 데이터를 480개로 실제 확장")
    print("   - 각 폭기조를 독립적인 데이터 포인트로 처리")
    print("   - COD 정보는 타겟으로만 사용, X에는 포함하지 않음")
    print("   - 자동화된 특성 생성 시스템 구현")
    
    print(f"\n📊 실제 데이터 활용:")
    print(f"   - 원본 데이터: 160개")
    print(f"   - 확장된 데이터: {len(X)}개")
    print(f"   - 특성 개수: {len(feature_names)}개")
    
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    print(f"\n🎯 모델 성능:")
    print(f"   - 최고 성능 모델: {best_model_name}")
    print(f"   - R² Score: {results[best_model_name]['r2']:.3f}")

if __name__ == "__main__":
    main() 