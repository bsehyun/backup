import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def ensemble_predict(X, y, model_a, model_b, threshold=0.5):
    """Ensemble 예측 수행"""
    pred_a = model_a.predict(X)
    pred_b = model_b.predict(X)
    
    # A 모델의 신뢰도가 threshold를 넘으면 A 사용, 아니면 B 사용
    confidence_a = np.abs(pred_a - y.mean()) / y.std()
    ensemble_pred = np.where(confidence_a > threshold, pred_a, pred_b)
    
    return ensemble_pred, pred_a, pred_b, confidence_a

def correlation_analysis(X, y):
    """주요 tag들의 상관관계 분석"""
    print("=== 태그 간 상관관계 분석 ===")
    
    # 전체 상관관계 히트맵
    plt.figure(figsize=(15, 12))
    correlation_matrix = X.corr()
    
    # 상관관계가 높은 상위 20개만 선택
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:  # 상관계수 0.3 이상
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                      correlation_matrix.columns[j], corr_val))
    
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # 상위 10개 상관관계 시각화
    if len(high_corr_pairs) > 0:
        top_pairs = high_corr_pairs[:10]
        pair_names = [f"{pair[0]} vs {pair[1]}" for pair in top_pairs]
        corr_values = [pair[2] for pair in top_pairs]
        
        plt.subplot(2, 2, 1)
        bars = plt.barh(range(len(pair_names)), corr_values, 
                       color=['red' if x < 0 else 'blue' for x in corr_values])
        plt.yticks(range(len(pair_names)), pair_names, fontsize=8)
        plt.xlabel('Correlation Coefficient')
        plt.title('Top 10 Tag Correlations')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Target과의 상관관계
        target_corr = X.corrwith(y).sort_values(key=abs, ascending=False)
        
        plt.subplot(2, 2, 2)
        top_target_corr = target_corr.head(10)
        bars = plt.barh(range(len(top_target_corr)), top_target_corr.values,
                       color=['red' if x < 0 else 'blue' for x in top_target_corr.values])
        plt.yticks(range(len(top_target_corr)), top_target_corr.index, fontsize=8)
        plt.xlabel('Correlation with Target')
        plt.title('Top 10 Tags Correlated with Target')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 전체 상관관계 히트맵 (상위 20개 태그만)
        plt.subplot(2, 2, 3)
        top_tags = target_corr.head(20).index
        corr_subset = correlation_matrix.loc[top_tags, top_tags]
        sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                   fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('Correlation Heatmap (Top 20 Tags)')
        
        # Target과의 상관관계 분포
        plt.subplot(2, 2, 4)
        plt.hist(target_corr.values, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Correlation with Target')
        plt.ylabel('Frequency')
        plt.title('Distribution of Target Correlations')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Target과 가장 상관관계가 높은 태그: {target_corr.index[0]} (상관계수: {target_corr.iloc[0]:.3f})")
        print(f"Target과 가장 상관관계가 낮은 태그: {target_corr.index[-1]} (상관계수: {target_corr.iloc[-1]:.3f})")
        
    return correlation_matrix, target_corr

def feature_importance_analysis(X, y, model_a, model_b):
    """모델별 feature importance 분석"""
    print("\n=== Feature Importance 분석 ===")
    
    # 모델 A의 feature importance
    if hasattr(model_a, 'feature_importances_'):
        importance_a = model_a.feature_importances_
    else:
        importance_a = np.abs(model_a.coef_) if hasattr(model_a, 'coef_') else np.ones(len(X.columns))
    
    # 모델 B의 feature importance
    if hasattr(model_b, 'feature_importances_'):
        importance_b = model_b.feature_importances_
    else:
        importance_b = np.abs(model_b.coef_) if hasattr(model_b, 'coef_') else np.ones(len(X.columns))
    
    # DataFrame 생성
    importance_df = pd.DataFrame({
        'Tag': X.columns.tolist(),
        'Model_A_Importance': importance_a,
        'Model_B_Importance': importance_b
    })
    
    # 상위 15개 태그 시각화
    plt.figure(figsize=(15, 10))
    
    # Model A Importance
    plt.subplot(2, 2, 1)
    top_a = importance_df.nlargest(15, 'Model_A_Importance')
    plt.barh(range(len(top_a)), top_a['Model_A_Importance'], color='skyblue')
    plt.yticks(range(len(top_a)), top_a['Tag'], fontsize=8)
    plt.xlabel('Importance')
    plt.title('Model A - Top 15 Feature Importance')
    
    # Model B Importance
    plt.subplot(2, 2, 2)
    top_b = importance_df.nlargest(15, 'Model_B_Importance')
    plt.barh(range(len(top_b)), top_b['Model_B_Importance'], color='lightcoral')
    plt.yticks(range(len(top_b)), top_b['Tag'], fontsize=8)
    plt.xlabel('Importance')
    plt.title('Model B - Top 15 Feature Importance')
    
    # Importance 비교
    plt.subplot(2, 2, 3)
    common_tags = set(top_a['Tag']) & set(top_b['Tag'])
    if common_tags:
        common_df = importance_df[importance_df['Tag'].isin(common_tags)]
        x_pos = np.arange(len(common_df))
        width = 0.35
        
        plt.bar(x_pos - width/2, common_df['Model_A_Importance'], width, 
               label='Model A', alpha=0.8)
        plt.bar(x_pos + width/2, common_df['Model_B_Importance'], width, 
               label='Model B', alpha=0.8)
        
        plt.xlabel('Tags')
        plt.ylabel('Importance')
        plt.title('Feature Importance Comparison')
        plt.xticks(x_pos, common_df['Tag'], rotation=45, ha='right', fontsize=8)
        plt.legend()
    
    # Importance 차이 분석
    plt.subplot(2, 2, 4)
    importance_df['Importance_Diff'] = importance_df['Model_A_Importance'] - importance_df['Model_B_Importance']
    top_diff = importance_df.nlargest(10, 'Importance_Diff')
    plt.barh(range(len(top_diff)), top_diff['Importance_Diff'], 
            color=['red' if x < 0 else 'blue' for x in top_diff['Importance_Diff']])
    plt.yticks(range(len(top_diff)), top_diff['Tag'], fontsize=8)
    plt.xlabel('Importance Difference (A - B)')
    plt.title('Top 10 Importance Differences')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return importance_df

def model_performance_comparison(X, y, model_a, model_b, threshold=0.5):
    """모델 성능 비교 분석"""
    print("\n=== 모델 성능 비교 분석 ===")
    
    # Ensemble 예측
    ensemble_pred, pred_a, pred_b, confidence_a = ensemble_predict(X, y, model_a, model_b, threshold)
    
    # 성능 지표 계산
    metrics = {}
    for name, pred in [('Model A', pred_a), ('Model B', pred_b), ('Ensemble', ensemble_pred)]:
        mse = mean_squared_error(y, pred)
        mae = mean_absolute_error(y, pred)
        r2 = r2_score(y, pred)
        metrics[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
    
    # 성능 비교 시각화
    plt.figure(figsize=(15, 10))
    
    # MSE 비교
    plt.subplot(2, 3, 1)
    model_names = list(metrics.keys())
    mse_values = [metrics[name]['MSE'] for name in model_names]
    bars = plt.bar(model_names, mse_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('Mean Squared Error')
    plt.title('MSE Comparison')
    plt.xticks(rotation=45)
    
    # MAE 비교
    plt.subplot(2, 3, 2)
    mae_values = [metrics[name]['MAE'] for name in model_names]
    bars = plt.bar(model_names, mae_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('Mean Absolute Error')
    plt.title('MAE Comparison')
    plt.xticks(rotation=45)
    
    # R² 비교
    plt.subplot(2, 3, 3)
    r2_values = [metrics[name]['R2'] for name in model_names]
    bars = plt.bar(model_names, r2_values, color=['skyblue', 'lightcoral', 'lightgreen'])
    plt.ylabel('R² Score')
    plt.title('R² Comparison')
    plt.xticks(rotation=45)
    
    # 예측값 vs 실제값 비교
    plt.subplot(2, 3, 4)
    plt.scatter(y, pred_a, alpha=0.6, label='Model A', s=20)
    plt.scatter(y, pred_b, alpha=0.6, label='Model B', s=20)
    plt.scatter(y, ensemble_pred, alpha=0.6, label='Ensemble', s=20)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual')
    plt.legend()
    
    # 잔차 분석
    plt.subplot(2, 3, 5)
    residuals_a = y - pred_a
    residuals_b = y - pred_b
    residuals_ensemble = y - ensemble_pred
    
    plt.scatter(pred_a, residuals_a, alpha=0.6, label='Model A', s=20)
    plt.scatter(pred_b, residuals_b, alpha=0.6, label='Model B', s=20)
    plt.scatter(ensemble_pred, residuals_ensemble, alpha=0.6, label='Ensemble', s=20)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    
    # 잔차 분포
    plt.subplot(2, 3, 6)
    plt.hist(residuals_a, alpha=0.7, label='Model A', bins=30)
    plt.hist(residuals_b, alpha=0.7, label='Model B', bins=30)
    plt.hist(residuals_ensemble, alpha=0.7, label='Ensemble', bins=30)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # 성능 지표 출력
    print("\n성능 지표:")
    for model_name, metric_dict in metrics.items():
        print(f"{model_name}:")
        for metric, value in metric_dict.items():
            print(f"  {metric}: {value:.4f}")
    
    return metrics, ensemble_pred, pred_a, pred_b, confidence_a

def time_lag_analysis(X, y):
    """Time lag 분석"""
    print("\n=== Time Lag 분석 ===")
    
    # Target과의 상관관계를 통한 lag 분석
    target_corr = X.corrwith(y).sort_values(key=abs, ascending=False)
    
    # 상위 10개 태그에 대해 lag 분석
    top_tags = target_corr.head(10).index
    
    plt.figure(figsize=(15, 10))
    
    for i, tag in enumerate(top_tags[:6]):  # 상위 6개만 시각화
        plt.subplot(2, 3, i+1)
        
        # 다양한 lag에 대한 상관관계 계산
        lags = range(-5, 6)  # -5부터 +5까지
        lag_correlations = []
        
        for lag in lags:
            if lag < 0:
                # 과거 값들
                lagged_tag = X[tag].shift(-lag)
                current_target = y.shift(lag)
            else:
                # 미래 값들
                lagged_tag = X[tag].shift(lag)
                current_target = y.shift(-lag)
            
            # NaN 제거
            valid_mask = ~(lagged_tag.isna() | current_target.isna())
            if valid_mask.sum() > 0:
                corr = lagged_tag[valid_mask].corr(current_target[valid_mask])
                lag_correlations.append(corr)
            else:
                lag_correlations.append(0)
        
        plt.plot(lags, lag_correlations, marker='o', linewidth=2, markersize=6)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.title(f'{tag} - Lag Analysis')
        plt.grid(True, alpha=0.3)
        
        # 최적 lag 표시
        optimal_lag = lags[np.argmax(np.abs(lag_correlations))]
        max_corr = max(lag_correlations, key=abs)
        plt.annotate(f'Optimal lag: {optimal_lag}\nCorr: {max_corr:.3f}', 
                    xy=(optimal_lag, max_corr), xytext=(0.7, 0.9),
                    textcoords='axes fraction', bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return target_corr

def ensemble_decision_analysis(X, y, model_a, model_b, threshold=0.5):
    """Ensemble 결정 분석"""
    print("\n=== Ensemble 결정 분석 ===")
    
    # 각 모델의 예측
    pred_a = model_a.predict(X)
    pred_b = model_b.predict(X)
    
    # A 모델의 신뢰도 계산
    confidence_a = np.abs(pred_a - y.mean()) / y.std()
    
    # Ensemble 결정
    use_model_a = confidence_a > threshold
    
    plt.figure(figsize=(15, 10))
    
    # 결정 분포
    plt.subplot(2, 3, 1)
    decision_counts = [np.sum(use_model_a), np.sum(~use_model_a)]
    plt.pie(decision_counts, labels=['Model A', 'Model B'], autopct='%1.1f%%',
            colors=['skyblue', 'lightcoral'])
    plt.title('Ensemble Decision Distribution')
    
    # 신뢰도 분포
    plt.subplot(2, 3, 2)
    plt.hist(confidence_a, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}')
    plt.xlabel('Model A Confidence')
    plt.ylabel('Frequency')
    plt.title('Model A Confidence Distribution')
    plt.legend()
    
    # 신뢰도 vs 예측 오차
    plt.subplot(2, 3, 3)
    error_a = np.abs(y - pred_a)
    error_b = np.abs(y - pred_b)
    
    plt.scatter(confidence_a[use_model_a], error_a[use_model_a], 
               alpha=0.6, label='Model A Used', color='blue')
    plt.scatter(confidence_a[~use_model_a], error_b[~use_model_a], 
               alpha=0.6, label='Model B Used', color='red')
    plt.axvline(x=threshold, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Model A Confidence')
    plt.ylabel('Absolute Error')
    plt.title('Confidence vs Error')
    plt.legend()
    
    # 모델별 예측값 분포
    plt.subplot(2, 3, 4)
    plt.hist(pred_a, alpha=0.7, label='Model A', bins=30, color='skyblue')
    plt.hist(pred_b, alpha=0.7, label='Model B', bins=30, color='lightcoral')
    plt.xlabel('Predicted Values')
    plt.ylabel('Frequency')
    plt.title('Prediction Distribution')
    plt.legend()
    
    # 예측값 차이 분석
    plt.subplot(2, 3, 5)
    pred_diff = pred_a - pred_b
    plt.hist(pred_diff, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Prediction Difference (A - B)')
    plt.ylabel('Frequency')
    plt.title('Prediction Difference Distribution')
    
    # 시간에 따른 결정 변화 (시계열 데이터가 있다고 가정)
    plt.subplot(2, 3, 6)
    if len(X) > 100:
        # 데이터가 충분히 많으면 시간에 따른 변화 시각화
        window_size = len(X) // 20
        decision_rate = []
        time_points = []
        
        for i in range(0, len(X) - window_size, window_size):
            window_decisions = use_model_a[i:i+window_size]
            decision_rate.append(np.mean(window_decisions))
            time_points.append(i + window_size//2)
        
        plt.plot(time_points, decision_rate, marker='o', linewidth=2)
        plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Time Point')
        plt.ylabel('Model A Usage Rate')
        plt.title('Model A Usage Over Time')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Model A 사용률: {np.mean(use_model_a):.2%}")
    print(f"Model B 사용률: {np.mean(~use_model_a):.2%}")
    print(f"평균 신뢰도: {np.mean(confidence_a):.3f}")
    
    return use_model_a, confidence_a

def comprehensive_eda(X, y, model_a, model_b, threshold=0.5):
    """종합 EDA 실행"""
    print("=== 제조 산업 센서 데이터 Ensemble 모델 EDA ===\n")
    
    # 1. 상관관계 분석
    correlation_matrix, target_corr = correlation_analysis(X, y)
    
    # 2. Feature Importance 분석
    importance_df = feature_importance_analysis(X, y, model_a, model_b)
    
    # 3. 모델 성능 비교
    metrics, ensemble_pred, pred_a, pred_b, confidence_a = model_performance_comparison(X, y, model_a, model_b, threshold)
    
    # 4. Time Lag 분석
    target_corr_lag = time_lag_analysis(X, y)
    
    # 5. Ensemble 결정 분석
    use_model_a, confidence_a = ensemble_decision_analysis(X, y, model_a, model_b, threshold)
    
    # 종합 결론
    print("\n=== 종합 결론 ===")
    print("1. 주요 발견사항:")
    print(f"   - Target과 가장 상관관계가 높은 태그: {target_corr.index[0]} (상관계수: {target_corr.iloc[0]:.3f})")
    print(f"   - Model A 사용률: {np.mean(use_model_a):.2%}")
    print(f"   - Model B 사용률: {np.mean(~use_model_a):.2%}")
    
    best_model = max(metrics.keys(), key=lambda x: metrics[x]['R2'])
    print(f"   - 최고 성능 모델: {best_model} (R²: {metrics[best_model]['R2']:.4f})")
    
    print("\n2. 권장사항:")
    if np.mean(use_model_a) > 0.7:
        print("   - Model A가 주로 사용되므로, Model A의 성능 개선에 집중")
    elif np.mean(use_model_a) < 0.3:
        print("   - Model B가 주로 사용되므로, Model B의 성능 개선에 집중")
    else:
        print("   - 두 모델이 균형있게 사용되므로, ensemble 전략이 효과적")
    
    # 중요 태그 추천
    top_tags = importance_df.nlargest(5, 'Model_A_Importance')['Tag'].tolist()
    print(f"   - 모니터링 우선순위 태그: {', '.join(top_tags[:3])}")
    
    return {
        'correlation_matrix': correlation_matrix,
        'target_correlation': target_corr,
        'importance_df': importance_df,
        'metrics': metrics,
        'ensemble_pred': ensemble_pred,
        'use_model_a': use_model_a,
        'confidence_a': confidence_a
    }

# 샘플 데이터 생성 함수
def create_sample_data(n_samples=1000, n_features=20):
    """테스트용 샘플 데이터 생성"""
    np.random.seed(42)
    
    # 센서 데이터 시뮬레이션
    X = pd.DataFrame(np.random.randn(n_samples, n_features),
                     columns=[f'Tag_{i:02d}' for i in range(1, n_features+1)])
    
    # Target 생성 (일부 태그와 상관관계 있게)
    y = (0.3 * X['Tag_01'] + 0.2 * X['Tag_05'] + 0.1 * X['Tag_10'] + 
         0.05 * X['Tag_15'] + np.random.randn(n_samples) * 0.1)
    
    return X, y

# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    X, y = create_sample_data()
    
    # 모델 생성 및 훈련
    model_a = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_b = GradientBoostingRegressor(n_estimators=100, random_state=43)
    
    model_a.fit(X, y)
    model_b.fit(X, y)
    
    # 종합 EDA 실행
    results = comprehensive_eda(X, y, model_a, model_b, threshold=0.5) 
