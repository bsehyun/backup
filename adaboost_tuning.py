#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AdaBoost Regressor with Optuna Hyperparameter Tuning
Jupyter Notebook Cell-Compatible Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
import optuna
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ============================================================================
# 1. 샘플 데이터 생성 (실제 데이터가 없는 경우)
# ============================================================================

def create_sample_data(n_samples=1000, n_features=10, noise=0.1, random_state=42):
    """
    AdaBoost 튜닝을 위한 샘플 데이터 생성
    """
    np.random.seed(random_state)
    
    # 특성 생성
    X = np.random.randn(n_samples, n_features)
    
    # 비선형 타겟 생성 (AdaBoost가 잘 처리하는 패턴)
    y = (X[:, 0] ** 2 + 
         np.sin(X[:, 1]) + 
         X[:, 2] * X[:, 3] + 
         np.abs(X[:, 4]) + 
         np.random.normal(0, noise, n_samples))
    
    # 데이터프레임으로 변환
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"생성된 데이터: {df.shape}")
    print(f"특성 범위: {df[feature_names].min().min():.3f} ~ {df[feature_names].max().max():.3f}")
    print(f"타겟 범위: {df['target'].min():.3f} ~ {df['target'].max():.3f}")
    
    return df

# ============================================================================
# 2. Optuna를 사용한 AdaBoost 하이퍼파라미터 튜닝
# ============================================================================

def objective_function(trial, X_train, X_val, y_train, y_val, cv_folds=5):
    """
    Optuna 목적 함수 - AdaBoost 하이퍼파라미터 최적화
    """
    # 하이퍼파라미터 정의
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
        'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
        'random_state': 42
    }
    
    # 기본 추정기 하이퍼파라미터
    base_estimator_params = {
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'random_state': 42
    }
    
    # 기본 추정기 생성
    base_estimator = DecisionTreeRegressor(**base_estimator_params)
    
    # AdaBoost 모델 생성
    model = AdaBoostRegressor(
        base_estimator=base_estimator,
        **params
    )
    
    # 교차 검증으로 성능 평가
    cv_scores = cross_val_score(
        model, X_train, y_train, 
        cv=cv_folds, scoring='neg_mean_squared_error'
    )
    
    # 평균 MSE 반환 (음수로 반환되므로 양수로 변환)
    return -cv_scores.mean()

def optimize_adaboost_hyperparameters(X, y, n_trials=100, cv_folds=5, random_state=42):
    """
    Optuna를 사용한 AdaBoost 하이퍼파라미터 최적화
    """
    print("=== AdaBoost 하이퍼파라미터 최적화 시작 ===")
    
    # 데이터 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )
    
    # Optuna 스터디 생성
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(seed=random_state)
    )
    
    # 최적화 실행
    study.optimize(
        lambda trial: objective_function(trial, X_train, X_val, y_train, y_val, cv_folds),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    print(f"\n최적화 완료!")
    print(f"최적 하이퍼파라미터: {study.best_params}")
    print(f"최적 MSE: {study.best_value:.6f}")
    
    return study, X_train, X_val, y_train, y_val

# ============================================================================
# 3. 최적화된 모델 훈련 및 평가
# ============================================================================

def train_optimized_adaboost(study, X_train, X_val, y_train, y_val):
    """
    최적화된 하이퍼파라미터로 AdaBoost 모델 훈련
    """
    print("\n=== 최적화된 AdaBoost 모델 훈련 ===")
    
    # 최적 하이퍼파라미터 추출
    best_params = study.best_params
    
    # 기본 추정기 생성
    base_estimator = DecisionTreeRegressor(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    
    # AdaBoost 모델 생성
    best_model = AdaBoostRegressor(
        base_estimator=base_estimator,
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        loss=best_params['loss'],
        random_state=42
    )
    
    # 모델 훈련
    best_model.fit(X_train, y_train)
    
    # 예측
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    
    # 성능 평가
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"훈련 성능:")
    print(f"  MSE: {train_mse:.6f}")
    print(f"  MAE: {train_mae:.6f}")
    print(f"  R²: {train_r2:.6f}")
    
    print(f"\n검증 성능:")
    print(f"  MSE: {val_mse:.6f}")
    print(f"  MAE: {val_mae:.6f}")
    print(f"  R²: {val_r2:.6f}")
    
    return best_model, {
        'train': {'mse': train_mse, 'mae': train_mae, 'r2': train_r2},
        'val': {'mse': val_mse, 'mae': val_mae, 'r2': val_r2},
        'y_train_pred': y_train_pred,
        'y_val_pred': y_val_pred
    }

# ============================================================================
# 4. 시각화 및 분석 함수들
# ============================================================================

def plot_optimization_history(study):
    """
    Optuna 최적화 히스토리 시각화
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 최적화 히스토리
    optuna.visualization.matplotlib.plot_optimization_history(study, ax=ax1)
    ax1.set_title('Optimization History')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('MSE')
    
    # 하이퍼파라미터 중요도
    optuna.visualization.matplotlib.plot_param_importances(study, ax=ax2)
    ax2.set_title('Parameter Importances')
    
    plt.tight_layout()
    plt.show()

def plot_hyperparameter_relationships(study):
    """
    하이퍼파라미터 간 관계 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # n_estimators vs learning_rate
    optuna.visualization.matplotlib.plot_contour(
        study, params=['n_estimators', 'learning_rate'], ax=axes[0, 0]
    )
    axes[0, 0].set_title('n_estimators vs learning_rate')
    
    # max_depth vs learning_rate
    optuna.visualization.matplotlib.plot_contour(
        study, params=['max_depth', 'learning_rate'], ax=axes[0, 1]
    )
    axes[0, 1].set_title('max_depth vs learning_rate')
    
    # n_estimators vs max_depth
    optuna.visualization.matplotlib.plot_contour(
        study, params=['n_estimators', 'max_depth'], ax=axes[1, 0]
    )
    axes[1, 0].set_title('n_estimators vs max_depth')
    
    # min_samples_split vs min_samples_leaf
    optuna.visualization.matplotlib.plot_contour(
        study, params=['min_samples_split', 'min_samples_leaf'], ax=axes[1, 1]
    )
    axes[1, 1].set_title('min_samples_split vs min_samples_leaf')
    
    plt.tight_layout()
    plt.show()

def plot_model_performance(y_true, y_pred, title="Model Performance"):
    """
    모델 성능 시각화
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 실제 vs 예측
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('Actual Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Actual vs Predicted')
    axes[0].grid(True, alpha=0.3)
    
    # 잔차 플롯
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    # 잔차 히스토그램
    axes[2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Residuals')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Residual Distribution')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    특성 중요도 시각화
    """
    # AdaBoost의 특성 중요도는 기본 추정기의 평균
    importances = np.mean([tree.feature_importances_ for tree in model.estimators_], axis=0)
    
    # 중요도 순으로 정렬
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances (AdaBoost)')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()
    
    return importances, indices

# ============================================================================
# 5. 시간 시계열 데이터용 특별 함수
# ============================================================================

def optimize_adaboost_timeseries(X, y, n_trials=100, random_state=42):
    """
    시간 시계열 데이터용 AdaBoost 최적화
    """
    print("=== 시간 시계열 데이터용 AdaBoost 최적화 ===")
    
    # 시간 기반 분할
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # 시계열 교차 검증
    tscv = TimeSeriesSplit(n_splits=5)
    
    def timeseries_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0, log=True),
            'loss': trial.suggest_categorical('loss', ['linear', 'square', 'exponential']),
            'random_state': 42
        }
        
        base_estimator_params = {
            'max_depth': trial.suggest_int('max_depth', 1, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
        
        base_estimator = DecisionTreeRegressor(**base_estimator_params)
        model = AdaBoostRegressor(base_estimator=base_estimator, **params)
        
        # 시계열 교차 검증
        scores = []
        for train_idx, val_idx in tscv.split(X_train):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            y_pred = model.predict(X_fold_val)
            score = mean_squared_error(y_fold_val, y_pred)
            scores.append(score)
        
        return np.mean(scores)
    
    # 최적화 실행
    study = optuna.create_study(direction='minimize')
    study.optimize(timeseries_objective, n_trials=n_trials, show_progress_bar=True)
    
    return study, X_train, X_val, y_train, y_val

# ============================================================================
# 6. 메인 실행 함수
# ============================================================================

def run_adaboost_optuna_analysis(data=None, target_col='target', n_trials=100, 
                                cv_folds=5, random_state=42, is_timeseries=False):
    """
    AdaBoost Optuna 분석 전체 파이프라인 실행
    """
    print("=== AdaBoost Regressor with Optuna Tuning ===")
    
    # 데이터 준비
    if data is None:
        print("샘플 데이터 생성 중...")
        data = create_sample_data(n_samples=1000, n_features=10, random_state=random_state)
    
    # 특성과 타겟 분리
    if isinstance(data, pd.DataFrame):
        feature_cols = [col for col in data.columns if col != target_col]
        X = data[feature_cols]
        y = data[target_col]
    else:
        X = data
        y = target_col
    
    print(f"데이터 형태: {X.shape}")
    print(f"특성 수: {X.shape[1]}")
    print(f"샘플 수: {X.shape[0]}")
    
    # 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # 최적화 실행
    if is_timeseries:
        study, X_train, X_val, y_train, y_val = optimize_adaboost_timeseries(
            X_scaled, y, n_trials=n_trials, random_state=random_state
        )
    else:
        study, X_train, X_val, y_train, y_val = optimize_adaboost_hyperparameters(
            X_scaled, y, n_trials=n_trials, cv_folds=cv_folds, random_state=random_state
        )
    
    # 최적화된 모델 훈련
    best_model, performance = train_optimized_adaboost(
        study, X_train, X_val, y_train, y_val
    )
    
    # 시각화
    print("\n=== 시각화 생성 중 ===")
    
    # 최적화 히스토리
    plot_optimization_history(study)
    
    # 하이퍼파라미터 관계
    plot_hyperparameter_relationships(study)
    
    # 모델 성능
    plot_model_performance(y_val, performance['y_val_pred'], "Validation Performance")
    
    # 특성 중요도
    if hasattr(X_train, 'columns'):
        plot_feature_importance(best_model, X_train.columns)
    
    # 결과 요약
    print("\n=== 최종 결과 요약 ===")
    print(f"최적 하이퍼파라미터:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    print(f"\n검증 성능:")
    print(f"  MSE: {performance['val']['mse']:.6f}")
    print(f"  MAE: {performance['val']['mae']:.6f}")
    print(f"  R²: {performance['val']['r2']:.6f}")
    
    return {
        'study': study,
        'best_model': best_model,
        'performance': performance,
        'scaler': scaler,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val
    }

# ============================================================================
# 7. 사용 예시 (Jupyter Notebook Cell)
# ============================================================================

if __name__ == "__main__":
    # 예시 1: 기본 실행
    print("예시 1: 기본 AdaBoost Optuna 튜닝")
    results = run_adaboost_optuna_analysis(n_trials=50)
    
    # 예시 2: 시계열 데이터용
    print("\n예시 2: 시계열 데이터용 AdaBoost Optuna 튜닝")
    # 시계열 데이터가 있다면:
    # results_ts = run_adaboost_optuna_analysis(data=your_timeseries_data, is_timeseries=True)
























import optuna
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.datasets import fetch_california_housing
import numpy as np

# Sample data (replace with your dataset)
X, y = fetch_california_housing(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Objective function for Optuna
def objective(trial):
    # Base estimator parameters
    max_depth = trial.suggest_int("max_depth", 1, 10)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    
    # AdaBoost parameters
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 2.0, log=True)

    base_estimator = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    model = AdaBoostRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=42
    )

    # 5-fold cross-validation (negative MSE -> maximize score)
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
    return np.mean(scores)

# Run Optuna study
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=-1)

# Best parameters
print("Best Parameters:", study.best_params)
print("Best CV Score:", study.best_value)

# Train final model
best_params = study.best_params
final_base_estimator = DecisionTreeRegressor(
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    random_state=42
)
final_model = AdaBoostRegressor(
    estimator=final_base_estimator,
    n_estimators=best_params["n_estimators"],
    learning_rate=best_params["learning_rate"],
    random_state=42
)
final_model.fit(X_train, y_train)
print("Test R^2:", final_model.score(X_test, y_test))

