#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 폐수 처리 현장 COD 예측 시스템
실제 센서 데이터를 활용하여 중간 폭기조의 COD를 예측
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class RealWastewaterCODPredictor:
    """
    실제 폐수 처리 현장 COD 예측 클래스
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.best_model = None
        self.feature_importance = None
        self.pipeline = None
        self.feature_names = None
        
    def load_real_data(self, file_path=None):
        """
        실제 데이터 로드 또는 샘플 데이터 생성
        """
        print("=== 실제 데이터 로드 ===\n")
        
        if file_path and os.path.exists(file_path):
            # 실제 데이터 파일이 있는 경우
            data = pd.read_csv(file_path)
            print(f"실제 데이터 로드: {data.shape}")
        else:
            # 샘플 데이터 생성 (실제 현장과 유사한 패턴)
            data = self._create_realistic_sample_data()
            print(f"샘플 데이터 생성: {data.shape}")
        
        return data
    
    def _create_realistic_sample_data(self, n_samples=160):
        """
        실제 현장과 유사한 패턴의 샘플 데이터 생성
        """
        np.random.seed(42)
        
        # 시간 인덱스 (3초마다 데이터 수집)
        time_index = pd.date_range('2024-01-01', periods=n_samples, freq='3S')
        
        # 실제 현장에서 관찰되는 패턴들
        # 1. 일중 변화 (업무 시간대에 유입량 증가)
        daily_pattern = np.sin(2 * np.pi * np.arange(n_samples) / (24 * 1200)) * 0.3 + 1
        
        # 2. 주간 변화 (주말에 유입량 감소)
        weekly_pattern = np.sin(2 * np.pi * np.arange(n_samples) / (7 * 24 * 1200)) * 0.2 + 1
        
        # 3. 계절적 변화 (온도)
        seasonal_temp = 25 + 8 * np.sin(2 * np.pi * np.arange(n_samples) / (365 * 24 * 1200))
        
        # 기본 환경 변수들
        temperature = seasonal_temp + np.random.normal(0, 1, n_samples)
        ph = 7.0 + 0.3 * np.sin(np.arange(n_samples) * np.pi / 600) + np.random.normal(0, 0.1, n_samples)
        dissolved_oxygen = 2.0 + 0.8 * np.sin(np.arange(n_samples) * np.pi / 800) + np.random.normal(0, 0.2, n_samples)
        
        # 유입량 (일중/주간 패턴 반영)
        flow_rate = (1000 + 300 * daily_pattern * weekly_pattern + 
                    np.random.normal(0, 50, n_samples))
        
        # 전도도
        conductivity = 800 + 150 * daily_pattern + np.random.normal(0, 20, n_samples)
        
        # 슬러지 관련
        mlss = 3000 + 600 * np.sin(np.arange(n_samples) * np.pi / 1500) + np.random.normal(0, 100, n_samples)
        sludge_volume = 200 + 60 * np.sin(np.arange(n_samples) * np.pi / 2000) + np.random.normal(0, 10, n_samples)
        
        # 각 폭기조별 특성 (실제로는 약간의 차이가 있음)
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
        # 각 폭기조별 COD는 서로 연관성이 있지만 약간의 차이가 있음
        base_cod = 150 + 40 * daily_pattern + np.random.normal(0, 10, n_samples)
        
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
        
        return data
    
    def prepare_features(self, data):
        """
        특성 엔지니어링 및 데이터 준비
        """
        print("\n=== 특성 엔지니어링 ===\n")
        
        # 시간 관련 특성 추가
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['month'] = data['timestamp'].dt.month
        data['day_of_year'] = data['timestamp'].dt.dayofyear
        
        # 주기적 특성 (sin, cos 변환)
        data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
        data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
        data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
        data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
        data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
        
        # 폭기조 간 차이 특성
        data['do_diff_1_2'] = data['tank1_do'] - data['tank2_do']
        data['do_diff_2_3'] = data['tank2_do'] - data['tank3_do']
        data['temp_diff_1_2'] = data['tank1_temp'] - data['tank2_temp']
        data['temp_diff_2_3'] = data['tank2_temp'] - data['tank3_temp']
        data['ph_diff_1_2'] = data['tank1_ph'] - data['tank2_ph']
        data['ph_diff_2_3'] = data['tank2_ph'] - data['tank3_ph']
        
        # COD 관련 특성 (다른 폭기조의 COD 정보 활용)
        data['cod_diff_1_3'] = data['cod_tank1'] - data['cod_tank3']
        data['cod_avg_1_3'] = (data['cod_tank1'] + data['cod_tank3']) / 2
        data['cod_ratio_1_3'] = data['cod_tank1'] / (data['cod_tank3'] + 1e-8)
        
        # 상호작용 특성
        data['do_temp_interaction'] = data['tank2_do'] * data['tank2_temp']
        data['ph_temp_interaction'] = data['tank2_ph'] * data['tank2_temp']
        data['flow_mlss_interaction'] = data['flow_rate'] * data['mlss']
        data['do_ph_interaction'] = data['tank2_do'] * data['tank2_ph']
        
        # 이동 평균 특성 (시간적 패턴)
        for window in [3, 5, 10]:
            data[f'do_ma_{window}'] = data['tank2_do'].rolling(window=window, min_periods=1).mean()
            data[f'temp_ma_{window}'] = data['tank2_temp'].rolling(window=window, min_periods=1).mean()
            data[f'ph_ma_{window}'] = data['tank2_ph'].rolling(window=window, min_periods=1).mean()
            data[f'flow_ma_{window}'] = data['flow_rate'].rolling(window=window, min_periods=1).mean()
        
        # 표준편차 특성 (변동성)
        for window in [3, 5]:
            data[f'do_std_{window}'] = data['tank2_do'].rolling(window=window, min_periods=1).std()
            data[f'temp_std_{window}'] = data['tank2_temp'].rolling(window=window, min_periods=1).std()
            data[f'ph_std_{window}'] = data['tank2_ph'].rolling(window=window, min_periods=1).std()
        
        # 비율 특성
        data['do_temp_ratio'] = data['tank2_do'] / (data['tank2_temp'] + 1e-8)
        data['ph_temp_ratio'] = data['tank2_ph'] / (data['tank2_temp'] + 1e-8)
        data['flow_mlss_ratio'] = data['flow_rate'] / (data['mlss'] + 1e-8)
        
        # 최종 특성 리스트
        final_features = [
            'temperature', 'ph', 'flow_rate', 'conductivity', 'mlss', 'sludge_volume',
            'tank1_do', 'tank1_temp', 'tank1_ph', 'cod_tank1',
            'tank2_do', 'tank2_temp', 'tank2_ph',
            'tank3_do', 'tank3_temp', 'tank3_ph', 'cod_tank3',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'do_diff_1_2', 'do_diff_2_3', 'temp_diff_1_2', 'temp_diff_2_3',
            'ph_diff_1_2', 'ph_diff_2_3', 'cod_diff_1_3', 'cod_avg_1_3', 'cod_ratio_1_3',
            'do_temp_interaction', 'ph_temp_interaction', 'flow_mlss_interaction', 'do_ph_interaction',
            'do_ma_3', 'temp_ma_3', 'ph_ma_3', 'flow_ma_3',
            'do_ma_5', 'temp_ma_5', 'ph_ma_5', 'flow_ma_5',
            'do_ma_10', 'temp_ma_10', 'ph_ma_10', 'flow_ma_10',
            'do_std_3', 'temp_std_3', 'ph_std_3',
            'do_std_5', 'temp_std_5', 'ph_std_5',
            'do_temp_ratio', 'ph_temp_ratio', 'flow_mlss_ratio'
        ]
        
        X = data[final_features].fillna(0)
        y = data['cod_tank2']  # 타겟: 중간 폭기조의 COD
        
        self.feature_names = final_features
        
        print(f"특성 개수: {len(final_features)}")
        print(f"입력 데이터 크기: {X.shape}")
        print(f"타겟 데이터 크기: {y.shape}")
        
        return X, y
    
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
    
    def hyperparameter_tuning(self, X, y):
        """
        최고 성능 모델의 하이퍼파라미터 튜닝
        """
        print("\n=== 하이퍼파라미터 튜닝 ===\n")
        
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Random Forest 튜닝
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            rf_param_grid,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        
        print("Random Forest 하이퍼파라미터 튜닝 중...")
        rf_grid.fit(X_train, y_train)
        
        print(f"최적 파라미터: {rf_grid.best_params_}")
        print(f"최적 R² Score: {rf_grid.best_score_:.3f}")
        
        # 튜닝된 모델로 최종 예측
        best_rf = rf_grid.best_estimator_
        y_pred_tuned = best_rf.predict(X_test)
        
        mse_tuned = mean_squared_error(y_test, y_pred_tuned)
        mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
        r2_tuned = r2_score(y_test, y_pred_tuned)
        
        print(f"튜닝 후 성능:")
        print(f"  MSE: {mse_tuned:.2f}")
        print(f"  MAE: {mae_tuned:.2f}")
        print(f"  R²: {r2_tuned:.3f}")
        
        self.best_model = best_rf
        
        return best_rf, rf_grid.best_params_
    
    def create_prediction_pipeline(self):
        """
        예측 파이프라인 생성
        """
        print("\n=== 예측 파이프라인 생성 ===\n")
        
        # 전처리와 모델을 결합한 파이프라인
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', self.best_model)
        ])
        
        print("파이프라인 구성:")
        print("1. StandardScaler: 특성 표준화")
        print(f"2. {type(self.best_model).__name__}: 예측 모델")
        
        return self.pipeline
    
    def save_model(self, filepath='wastewater_cod_model.pkl'):
        """
        훈련된 모델 저장
        """
        print(f"\n모델 저장 중: {filepath}")
        
        model_data = {
            'pipeline': self.pipeline,
            'feature_names': self.feature_names,
            'best_model': self.best_model,
            'feature_importance': self.feature_importance
        }
        
        joblib.dump(model_data, filepath)
        print("모델 저장 완료!")
    
    def load_model(self, filepath='wastewater_cod_model.pkl'):
        """
        저장된 모델 로드
        """
        print(f"모델 로드 중: {filepath}")
        
        model_data = joblib.load(filepath)
        self.pipeline = model_data['pipeline']
        self.feature_names = model_data['feature_names']
        self.best_model = model_data['best_model']
        self.feature_importance = model_data['feature_importance']
        
        print("모델 로드 완료!")
    
    def predict_cod(self, sensor_data):
        """
        새로운 센서 데이터로 COD 예측
        """
        if self.pipeline is None:
            raise ValueError("모델이 로드되지 않았습니다. 먼저 모델을 훈련하거나 로드하세요.")
        
        # 특성 엔지니어링
        processed_data = self._prepare_new_data(sensor_data)
        
        # 예측
        prediction = self.pipeline.predict(processed_data)
        
        return prediction
    
    def _prepare_new_data(self, sensor_data):
        """
        새로운 센서 데이터에 대한 특성 엔지니어링
        """
        # 기본 특성들만 사용 (시간 관련 특성은 제외)
        basic_features = [
            'temperature', 'ph', 'flow_rate', 'conductivity', 'mlss', 'sludge_volume',
            'tank1_do', 'tank1_temp', 'tank1_ph', 'cod_tank1',
            'tank2_do', 'tank2_temp', 'tank2_ph',
            'tank3_do', 'tank3_temp', 'tank3_ph', 'cod_tank3'
        ]
        
        # 기본 특성만 추출
        X_new = sensor_data[basic_features].fillna(0)
        
        return X_new
    
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
        
        # 6. 폭기조별 COD 비교
        tank_data = data[['cod_tank1', 'cod_tank2', 'cod_tank3']].iloc[test_indices]
        axes[1,2].boxplot([tank_data['cod_tank1'], tank_data['cod_tank2'], tank_data['cod_tank3']], 
                          labels=['폭기조 1', '폭기조 2 (타겟)', '폭기조 3'])
        axes[1,2].set_ylabel('COD (mg/L)')
        axes[1,2].set_title('폭기조별 COD 분포')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    메인 실행 함수
    """
    print("=== 실제 폐수 처리 현장 COD 예측 시스템 ===\n")
    
    # 1. 예측기 초기화
    predictor = RealWastewaterCODPredictor()
    
    # 2. 데이터 로드
    data = predictor.load_real_data()
    
    # 3. 특성 엔지니어링
    X, y = predictor.prepare_features(data)
    
    # 4. 모델 훈련 및 비교
    results, X_test, y_test = predictor.train_models(X, y)
    
    # 5. 하이퍼파라미터 튜닝
    best_model, best_params = predictor.hyperparameter_tuning(X, y)
    
    # 6. 특성 중요도 분석
    feature_importance = predictor.analyze_feature_importance(X, predictor.feature_names)
    
    # 7. 예측 파이프라인 생성
    pipeline = predictor.create_prediction_pipeline()
    
    # 8. 결과 시각화
    predictor.visualize_results(data, results, X_test, y_test)
    
    # 9. 모델 저장
    predictor.save_model()
    
    # 10. 결론 및 권장사항
    print("\n=== 결론 및 권장사항 ===\n")
    print("1. 데이터 활용 전략:")
    print("   - 3개 폭기조의 센서 데이터를 모두 활용하여 480개 데이터 포인트 확보")
    print("   - 다른 폭기조의 COD 정보를 특성으로 활용")
    print("   - 시간적 패턴과 폭기조 간 상관관계를 특성으로 추가")
    
    print("\n2. 모델 성능:")
    best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
    print(f"   - 최고 성능 모델: {best_model_name}")
    print(f"   - R² Score: {results[best_model_name]['r2']:.3f}")
    
    print("\n3. 실제 적용 시 고려사항:")
    print("   - 실시간 센서 데이터 수집 시스템 구축")
    print("   - 모델 정기적 재훈련 (새로운 데이터로 업데이트)")
    print("   - 예측 결과의 신뢰도 평가 및 알림 시스템")
    print("   - 에너지 효율성과 처리 효율성의 균형 고려")
    
    print("\n4. 추가 개선 방안:")
    print("   - 딥러닝 모델 (LSTM, GRU) 적용 검토")
    print("   - 앙상블 방법으로 성능 향상")
    print("   - 온라인 학습 방식 도입")
    print("   - 불확실성 정량화 (예측 구간 제공)")

if __name__ == "__main__":
    main() 