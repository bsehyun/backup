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

def ensemble_predict(X_a, X_b, y, model_a, model_b, threshold=0.5):
    """Ensemble 예측 수행"""
    pred_a = model_a.predict(X_a)
    pred_b = model_b.predict(X_b)
    
    # A 모델의 신뢰도가 threshold를 넘으면 A 사용, 아니면 B 사용
    confidence_a = np.abs(pred_a - y.mean()) / y.std()
    ensemble_pred = np.where(confidence_a > threshold, pred_a, pred_b)
    
    return ensemble_pred, pred_a, pred_b, confidence_a

def analyze_basic_tag_transformations(X_a, X_b, basic_tags):
    """Basic tag들의 변환 방법 분석"""
    print("=== Basic Tag 변환 방법 분석 ===")
    
    if basic_tags is None:
        print("Basic tags가 정의되지 않았습니다.")
        return {}, {}
    
    # Basic tag들의 변환된 feature들 찾기
    basic_transformations_a = {}
    basic_transformations_b = {}
    
    for basic_tag in basic_tags:
        # Model A에서 해당 basic tag의 변환된 feature들
        a_features = [col for col in X_a.columns if basic_tag in col]
        if a_features:
            basic_transformations_a[basic_tag] = a_features
        
        # Model B에서 해당 basic tag의 변환된 feature들
        b_features = [col for col in X_b.columns if basic_tag in col]
        if b_features:
            basic_transformations_b[basic_tag] = b_features
    
    # 변환 방법 분석
    plt.figure(figsize=(20, 12))
    
    # Basic tag별 변환된 feature 수 비교
    plt.subplot(2, 3, 1)
    basic_tags_list = list(set(basic_transformations_a.keys()) | set(basic_transformations_b.keys()))
    a_counts = [len(basic_transformations_a.get(tag, [])) for tag in basic_tags_list]
    b_counts = [len(basic_transformations_b.get(tag, [])) for tag in basic_tags_list]
    
    x_pos = np.arange(len(basic_tags_list))
    width = 0.35
    
    plt.bar(x_pos - width/2, a_counts, width, label='Model A', alpha=0.8, color='skyblue')
    plt.bar(x_pos + width/2, b_counts, width, label='Model B', alpha=0.8, color='lightcoral')
    plt.xlabel('Basic Tags')
    plt.ylabel('Number of Transformed Features')
    plt.title('Basic Tag Transformation Count Comparison')
    plt.xticks(x_pos, basic_tags_list, rotation=45, ha='right')
    plt.legend()
    
    # 변환 방법 분류
    transformation_types_a = classify_transformations(basic_transformations_a)
    transformation_types_b = classify_transformations(basic_transformations_b)
    
    # Model A 변환 방법 분포
    plt.subplot(2, 3, 2)
    if transformation_types_a:
        types_a = list(transformation_types_a.keys())
        counts_a = list(transformation_types_a.values())
        plt.pie(counts_a, labels=types_a, autopct='%1.1f%%', startangle=90)
        plt.title('Model A - Transformation Methods')
    
    # Model B 변환 방법 분포
    plt.subplot(2, 3, 3)
    if transformation_types_b:
        types_b = list(transformation_types_b.keys())
        counts_b = list(transformation_types_b.values())
        plt.pie(counts_b, labels=types_b, autopct='%1.1f%%', startangle=90)
        plt.title('Model B - Transformation Methods')
    
    # 변환 방법별 사용 빈도 비교
    plt.subplot(2, 3, 4)
    all_transformations = set(transformation_types_a.keys()) | set(transformation_types_b.keys())
    a_freq = [transformation_types_a.get(method, 0) for method in all_transformations]
    b_freq = [transformation_types_b.get(method, 0) for method in all_transformations]
    
    x_pos = np.arange(len(all_transformations))
    plt.bar(x_pos - width/2, a_freq, width, label='Model A', alpha=0.8, color='skyblue')
    plt.bar(x_pos + width/2, b_freq, width, label='Model B', alpha=0.8, color='lightcoral')
    plt.xlabel('Transformation Methods')
    plt.ylabel('Usage Count')
    plt.title('Transformation Method Usage Comparison')
    plt.xticks(x_pos, all_transformations, rotation=45)
    plt.legend()
    
    # Basic tag 사용 빈도
    plt.subplot(2, 3, 5)
    basic_tag_usage = analyze_basic_tag_usage(X_a, X_b, basic_tags)
    if basic_tag_usage:
        tags = list(basic_tag_usage.keys())
        usage = list(basic_tag_usage.values())
        plt.bar(tags, usage, color='lightblue')
        plt.xlabel('Basic Tags')
        plt.ylabel('Usage Count')
        plt.title('Basic Tag Usage Frequency')
        plt.xticks(rotation=45)
    
    # 변환 방법별 평균 중요도 (예시)
    plt.subplot(2, 3, 6)
    transformation_performance = {'Lag': 0.8, 'Mean': 0.7, 'Std': 0.6, 'Min/Max': 0.5, 'Median': 0.4}
    methods = list(transformation_performance.keys())
    performances = list(transformation_performance.values())
    plt.bar(methods, performances, color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink'])
    plt.xlabel('Transformation Methods')
    plt.ylabel('Average Performance')
    plt.title('Transformation Method Performance')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    return basic_transformations_a, basic_transformations_b

def classify_transformations(basic_transformations):
    """변환 방법 분류"""
    transformation_types = {}
    
    for basic_tag, features in basic_transformations.items():
        for feature in features:
            if 'lag' in feature.lower():
                transformation_types['Lag'] = transformation_types.get('Lag', 0) + 1
            elif 'mean' in feature.lower():
                transformation_types['Mean'] = transformation_types.get('Mean', 0) + 1
            elif 'std' in feature.lower():
                transformation_types['Std'] = transformation_types.get('Std', 0) + 1
            elif 'min' in feature.lower():
                transformation_types['Min'] = transformation_types.get('Min', 0) + 1
            elif 'max' in feature.lower():
                transformation_types['Max'] = transformation_types.get('Max', 0) + 1
            elif 'median' in feature.lower():
                transformation_types['Median'] = transformation_types.get('Median', 0) + 1
            else:
                transformation_types['Other'] = transformation_types.get('Other', 0) + 1
    
    return transformation_types

def analyze_basic_tag_usage(X_a, X_b, basic_tags):
    """Basic tag 사용 빈도 분석"""
    if basic_tags is None:
        return {}
    
    usage = {}
    for tag in basic_tags:
        count = 0
        for col in X_a.columns.tolist() + X_b.columns.tolist():
            if tag in col:
                count += 1
        usage[tag] = count
    
    return usage

def correlation_analysis(X_a, X_b, y):
    """주요 tag들의 상관관계 분석 (각 모델별로)"""
    print("=== 태그 간 상관관계 분석 ===")
    
    # Model A 상관관계 분석
    plt.figure(figsize=(20, 12))
    
    # Model A 상관관계 히트맵
    plt.subplot(2, 3, 1)
    correlation_matrix_a = X_a.corr()
    sns.heatmap(correlation_matrix_a, cmap='coolwarm', center=0, 
               cbar_kws={'shrink': 0.8})
    plt.title('Model A - Correlation Heatmap')
    
    # Model B 상관관계 히트맵
    plt.subplot(2, 3, 2)
    correlation_matrix_b = X_b.corr()
    sns.heatmap(correlation_matrix_b, cmap='coolwarm', center=0, 
               cbar_kws={'shrink': 0.8})
    plt.title('Model B - Correlation Heatmap')
    
    # Target과의 상관관계 비교
    target_corr_a = X_a.corrwith(y).sort_values(key=abs, ascending=False)
    target_corr_b = X_b.corrwith(y).sort_values(key=abs, ascending=False)
    
    # Model A Target 상관관계
    plt.subplot(2, 3, 3)
    top_target_corr_a = target_corr_a.head(10)
    bars = plt.barh(range(len(top_target_corr_a)), top_target_corr_a.values,
                   color=['red' if x < 0 else 'blue' for x in top_target_corr_a.values])
    plt.yticks(range(len(top_target_corr_a)), top_target_corr_a.index, fontsize=8)
    plt.xlabel('Correlation with Target')
    plt.title('Model A - Top 10 Tags Correlated with Target')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Model B Target 상관관계
    plt.subplot(2, 3, 4)
    top_target_corr_b = target_corr_b.head(10)
    bars = plt.barh(range(len(top_target_corr_b)), top_target_corr_b.values,
                   color=['red' if x < 0 else 'blue' for x in top_target_corr_b.values])
    plt.yticks(range(len(top_target_corr_b)), top_target_corr_b.index, fontsize=8)
    plt.xlabel('Correlation with Target')
    plt.title('Model B - Top 10 Tags Correlated with Target')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 상관관계 분포 비교
    plt.subplot(2, 3, 5)
    plt.hist(target_corr_a.values, bins=30, alpha=0.7, color='skyblue', 
            edgecolor='black', label='Model A')
    plt.hist(target_corr_b.values, bins=30, alpha=0.7, color='lightcoral', 
            edgecolor='black', label='Model B')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Correlation with Target')
    plt.ylabel('Frequency')
    plt.title('Distribution of Target Correlations')
    plt.legend()
    
    # 상관관계 강도 비교
    plt.subplot(2, 3, 6)
    plt.scatter(target_corr_a.head(20), target_corr_b.head(20), alpha=0.6)
    plt.xlabel('Model A Correlation')
    plt.ylabel('Model B Correlation')
    plt.title('Correlation Strength Comparison')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Model A - Target과 가장 상관관계가 높은 태그: {target_corr_a.index[0]} (상관계수: {target_corr_a.iloc[0]:.3f})")
    print(f"Model B - Target과 가장 상관관계가 높은 태그: {target_corr_b.index[0]} (상관계수: {target_corr_b.iloc[0]:.3f})")
    
    return correlation_matrix_a, correlation_matrix_b, target_corr_a, target_corr_b

def feature_importance_analysis(X_a, X_b, y, model_a, model_b, basic_tags=None):
    """모델별 feature importance 분석 (각각 독립적으로)"""
    print("\n=== Feature Importance 분석 ===")
    
    # 모델 A의 feature importance
    if hasattr(model_a, 'feature_importances_'):
        importance_a = model_a.feature_importances_
    else:
        importance_a = np.abs(model_a.coef_) if hasattr(model_a, 'coef_') else np.ones(len(X_a.columns))
    
    # 모델 B의 feature importance
    if hasattr(model_b, 'feature_importances_'):
        importance_b = model_b.feature_importances_
    else:
        importance_b = np.abs(model_b.coef_) if hasattr(model_b, 'coef_') else np.ones(len(X_b.columns))
    
    # DataFrame 생성
    importance_df_a = pd.DataFrame({
        'Tag': X_a.columns.tolist(),
        'Importance': importance_a
    })
    
    importance_df_b = pd.DataFrame({
        'Tag': X_b.columns.tolist(),
        'Importance': importance_b
    })
    
    # 시각화
    plt.figure(figsize=(20, 12))
    
    # Model A Importance
    plt.subplot(2, 3, 1)
    top_a = importance_df_a.nlargest(15, 'Importance')
    plt.barh(range(len(top_a)), top_a['Importance'], color='skyblue')
    plt.yticks(range(len(top_a)), top_a['Tag'], fontsize=8)
    plt.xlabel('Importance')
    plt.title('Model A - Top 15 Feature Importance')
    
    # Model B Importance
    plt.subplot(2, 3, 2)
    top_b = importance_df_b.nlargest(15, 'Importance')
    plt.barh(range(len(top_b)), top_b['Importance'], color='lightcoral')
    plt.yticks(range(len(top_b)), top_b['Tag'], fontsize=8)
    plt.xlabel('Importance')
    plt.title('Model B - Top 15 Feature Importance')
    
    # Basic tag별 중요도 분석 (Model A)
    if basic_tags:
        plt.subplot(2, 3, 3)
        basic_importance_a = analyze_basic_tag_importance(importance_df_a, basic_tags, 'Model A')
        if basic_importance_a:
            tags = list(basic_importance_a.keys())
            importance = list(basic_importance_a.values())
            plt.bar(tags, importance, color='skyblue', alpha=0.8)
            plt.xlabel('Basic Tags')
            plt.ylabel('Average Importance')
            plt.title('Model A - Basic Tag Importance')
            plt.xticks(rotation=45)
    
    # Basic tag별 중요도 분석 (Model B)
    if basic_tags:
        plt.subplot(2, 3, 4)
        basic_importance_b = analyze_basic_tag_importance(importance_df_b, basic_tags, 'Model B')
        if basic_importance_b:
            tags = list(basic_importance_b.keys())
            importance = list(basic_importance_b.values())
            plt.bar(tags, importance, color='lightcoral', alpha=0.8)
            plt.xlabel('Basic Tags')
            plt.ylabel('Average Importance')
            plt.title('Model B - Basic Tag Importance')
            plt.xticks(rotation=45)
    
    # 변환 방법별 중요도 분석
    plt.subplot(2, 3, 5)
    transformation_importance_a = analyze_transformation_importance(importance_df_a, 'Model A')
    if transformation_importance_a:
        methods = list(transformation_importance_a.keys())
        importance = list(transformation_importance_a.values())
        plt.bar(methods, importance, color='skyblue', alpha=0.8)
        plt.xlabel('Transformation Methods')
        plt.ylabel('Average Importance')
        plt.title('Model A - Transformation Method Importance')
        plt.xticks(rotation=45)
    
    plt.subplot(2, 3, 6)
    transformation_importance_b = analyze_transformation_importance(importance_df_b, 'Model B')
    if transformation_importance_b:
        methods = list(transformation_importance_b.keys())
        importance = list(transformation_importance_b.values())
        plt.bar(methods, importance, color='lightcoral', alpha=0.8)
        plt.xlabel('Transformation Methods')
        plt.ylabel('Average Importance')
        plt.title('Model B - Transformation Method Importance')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Model A에서 가장 중요한 태그: {top_a.iloc[0]['Tag']} (Importance: {top_a.iloc[0]['Importance']:.4f})")
    print(f"Model B에서 가장 중요한 태그: {top_b.iloc[0]['Tag']} (Importance: {top_b.iloc[0]['Importance']:.4f})")
    
    return importance_df_a, importance_df_b

def analyze_basic_tag_importance(importance_df, basic_tags, model_name):
    """Basic tag별 평균 중요도 분석"""
    if basic_tags is None:
        return {}
    
    basic_importance = {}
    for basic_tag in basic_tags:
        # 해당 basic tag를 포함하는 feature들의 중요도 평균
        related_features = [feat for feat in importance_df['Tag'] if basic_tag in feat]
        if related_features:
            avg_importance = importance_df[importance_df['Tag'].isin(related_features)]['Importance'].mean()
            basic_importance[basic_tag] = avg_importance
    
    return basic_importance

def analyze_transformation_importance(importance_df, model_name):
    """변환 방법별 평균 중요도 분석"""
    transformation_importance = {}
    
    for _, row in importance_df.iterrows():
        feature = row['Tag']
        importance = row['Importance']
        
        if 'lag' in feature.lower():
            transformation_importance['Lag'] = transformation_importance.get('Lag', []) + [importance]
        elif 'mean' in feature.lower():
            transformation_importance['Mean'] = transformation_importance.get('Mean', []) + [importance]
        elif 'std' in feature.lower():
            transformation_importance['Std'] = transformation_importance.get('Std', []) + [importance]
        elif 'min' in feature.lower():
            transformation_importance['Min'] = transformation_importance.get('Min', []) + [importance]
        elif 'max' in feature.lower():
            transformation_importance['Max'] = transformation_importance.get('Max', []) + [importance]
        elif 'median' in feature.lower():
            transformation_importance['Median'] = transformation_importance.get('Median', []) + [importance]
        else:
            transformation_importance['Other'] = transformation_importance.get('Other', []) + [importance]
    
    # 평균 계산
    avg_importance = {}
    for method, values in transformation_importance.items():
        if values:
            avg_importance[method] = np.mean(values)
    
    return avg_importance

def model_performance_comparison(X_a, X_b, y, model_a, model_b, threshold=0.5):
    """모델 성능 비교 분석"""
    print("\n=== 모델 성능 비교 분석 ===")
    
    # Ensemble 예측
    ensemble_pred, pred_a, pred_b, confidence_a = ensemble_predict(X_a, X_b, y, model_a, model_b, threshold)
    
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

def time_lag_analysis(X_a, X_b, y):
    """Time lag 분석 (각 모델별로)"""
    print("\n=== Time Lag 분석 ===")
    
    # Model A의 lag 분석
    target_corr_a = X_a.corrwith(y).sort_values(key=abs, ascending=False)
    top_tags_a = target_corr_a.head(10).index
    
    # Model B의 lag 분석
    target_corr_b = X_b.corrwith(y).sort_values(key=abs, ascending=False)
    top_tags_b = target_corr_b.head(10).index
    
    plt.figure(figsize=(20, 10))
    
    # Model A lag 분석 (상위 6개)
    for i, tag in enumerate(top_tags_a[:6]):
        plt.subplot(2, 6, i+1)
        
        # Lag 분석
        if 'lag' in tag.lower():
            # Lag feature인 경우
            lag_value = extract_lag_value(tag)
            plt.bar([lag_value], [target_corr_a[tag]], color='skyblue')
            plt.xlabel('Lag')
            plt.ylabel('Correlation')
            plt.title(f'{tag}\nLag: {lag_value}')
        else:
            # 일반 feature인 경우
            plt.bar([0], [target_corr_a[tag]], color='skyblue')
            plt.xlabel('No Lag')
            plt.ylabel('Correlation')
            plt.title(f'{tag}')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Model B lag 분석 (상위 6개)
    for i, tag in enumerate(top_tags_b[:6]):
        plt.subplot(2, 6, i+7)
        
        if 'lag' in tag.lower():
            lag_value = extract_lag_value(tag)
            plt.bar([lag_value], [target_corr_b[tag]], color='lightcoral')
            plt.xlabel('Lag')
            plt.ylabel('Correlation')
            plt.title(f'{tag}\nLag: {lag_value}')
        else:
            plt.bar([0], [target_corr_b[tag]], color='lightcoral')
            plt.xlabel('No Lag')
            plt.ylabel('Correlation')
            plt.title(f'{tag}')
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return target_corr_a, target_corr_b

def extract_lag_value(feature_name):
    """Feature 이름에서 lag 값 추출"""
    import re
    lag_match = re.search(r'lag_?(\d+)', feature_name.lower())
    if lag_match:
        return int(lag_match.group(1))
    return 0

def ensemble_decision_analysis(X_a, X_b, y, model_a, model_b, threshold=0.5):
    """Ensemble 결정 분석"""
    print("\n=== Ensemble 결정 분석 ===")
    
    # 각 모델의 예측
    pred_a = model_a.predict(X_a)
    pred_b = model_b.predict(X_b)
    
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
    
    # 시간에 따른 결정 변화
    plt.subplot(2, 3, 6)
    if len(X_a) > 100:
        window_size = len(X_a) // 20
        decision_rate = []
        time_points = []
        
        for i in range(0, len(X_a) - window_size, window_size):
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

def comprehensive_eda(X_a, X_b, y, model_a, model_b, basic_tags=None, threshold=0.5):
    """종합 EDA 실행"""
    print("=== 제조 산업 센서 데이터 Ensemble 모델 EDA ===\n")
    
    # 1. Basic tag 변환 방법 분석
    basic_transformations_a, basic_transformations_b = analyze_basic_tag_transformations(X_a, X_b, basic_tags)
    
    # 2. 상관관계 분석
    correlation_matrix_a, correlation_matrix_b, target_corr_a, target_corr_b = correlation_analysis(X_a, X_b, y)
    
    # 3. Feature Importance 분석
    importance_df_a, importance_df_b = feature_importance_analysis(X_a, X_b, y, model_a, model_b, basic_tags)
    
    # 4. 모델 성능 비교
    metrics, ensemble_pred, pred_a, pred_b, confidence_a = model_performance_comparison(X_a, X_b, y, model_a, model_b, threshold)
    
    # 5. Time Lag 분석
    target_corr_lag_a, target_corr_lag_b = time_lag_analysis(X_a, X_b, y)
    
    # 6. Ensemble 결정 분석
    use_model_a, confidence_a = ensemble_decision_analysis(X_a, X_b, y, model_a, model_b, threshold)
    
    # 종합 결론
    print("\n=== 종합 결론 ===")
    print("1. 주요 발견사항:")
    print(f"   - Model A 최고 상관관계 태그: {target_corr_a.index[0]} (상관계수: {target_corr_a.iloc[0]:.3f})")
    print(f"   - Model B 최고 상관관계 태그: {target_corr_b.index[0]} (상관계수: {target_corr_b.iloc[0]:.3f})")
    print(f"   - Model A 사용률: {np.mean(use_model_a):.2%}")
    print(f"   - Model B 사용률: {np.mean(~use_model_a):.2%}")
    
    best_model = max(metrics.keys(), key=lambda x: metrics[x]['R2'])
    print(f"   - 최고 성능 모델: {best_model} (R²: {metrics[best_model]['R2']:.4f})")
    
    print("\n2. 권장사항:")
    if np.mean(use_model_a) > 0.7:
        print("   - Model A가 주로 사용되므로, Model A의 feature engineering 개선에 집중")
    elif np.mean(use_model_a) < 0.3:
        print("   - Model B가 주로 사용되므로, Model B의 feature engineering 개선에 집중")
    else:
        print("   - 두 모델이 균형있게 사용되므로, ensemble 전략이 효과적")
    
    # Basic tag별 권장사항
    if basic_tags:
        print(f"   - 분석된 Basic tags: {', '.join(basic_tags[:5])}")
    
    return {
        'basic_transformations_a': basic_transformations_a,
        'basic_transformations_b': basic_transformations_b,
        'correlation_matrix_a': correlation_matrix_a,
        'correlation_matrix_b': correlation_matrix_b,
        'target_correlation_a': target_corr_a,
        'target_correlation_b': target_corr_b,
        'importance_df_a': importance_df_a,
        'importance_df_b': importance_df_b,
        'metrics': metrics,
        'ensemble_pred': ensemble_pred,
        'use_model_a': use_model_a,
        'confidence_a': confidence_a
    }

def comprehensive_eda_with_report(X_a, X_b, y, model_a, model_b, basic_tags=None, threshold=0.5, 
                                generate_report=True, report_filename="manufacturing_eda_report"):
    """종합 EDA 실행 및 리포트 생성"""
    print("=== 제조 산업 센서 데이터 Ensemble 모델 EDA ===\n")
    
    # 1. Basic tag 변환 방법 분석
    basic_transformations_a, basic_transformations_b = analyze_basic_tag_transformations(X_a, X_b, basic_tags)
    
    # 2. 상관관계 분석
    correlation_matrix_a, correlation_matrix_b, target_corr_a, target_corr_b = correlation_analysis(X_a, X_b, y)
    
    # 3. Feature Importance 분석
    importance_df_a, importance_df_b = feature_importance_analysis(X_a, X_b, y, model_a, model_b, basic_tags)
    
    # 4. 모델 성능 비교
    metrics, ensemble_pred, pred_a, pred_b, confidence_a = model_performance_comparison(X_a, X_b, y, model_a, model_b, threshold)
    
    # 5. Time Lag 분석
    target_corr_lag_a, target_corr_lag_b = time_lag_analysis(X_a, X_b, y)
    
    # 6. Ensemble 결정 분석
    use_model_a, confidence_a = ensemble_decision_analysis(X_a, X_b, y, model_a, model_b, threshold)
    
    # 7. 리포트 생성 (선택사항)
    if generate_report:
        from eda_report_utils import generate_eda_reports
        
        # 리포트용 데이터 준비
        plots_data = {
            'correlation': {
                'target_corr_a': target_corr_a,
                'target_corr_b': target_corr_b,
                'correlation_matrix_a': correlation_matrix_a,
                'correlation_matrix_b': correlation_matrix_b
            },
            'importance': {
                'importance_df_a': importance_df_a,
                'importance_df_b': importance_df_b,
                'basic_importance_a': analyze_basic_tag_importance(importance_df_a, basic_tags, 'Model A'),
                'basic_importance_b': analyze_basic_tag_importance(importance_df_b, basic_tags, 'Model B')
            },
            'performance': {
                'metrics': metrics,
                'pred_a': pred_a,
                'pred_b': pred_b,
                'ensemble_pred': ensemble_pred,
                'y': y,
                'residuals_a': y - pred_a,
                'residuals_b': y - pred_b,
                'residuals_ensemble': y - ensemble_pred
            },
            'ensemble': {
                'use_model_a': use_model_a,
                'confidence_a': confidence_a,
                'threshold': threshold,
                'pred_a': pred_a,
                'pred_b': pred_b
            }
        }
        
        # 리포트 생성
        pdf_filename, html_filename = generate_eda_reports(plots_data, report_filename)
        print(f"\n리포트 파일이 생성되었습니다:")
        print(f"  - PDF: {pdf_filename}")
        print(f"  - HTML: {html_filename}")
    
    # 종합 결론
    print("\n=== 종합 결론 ===")
    print("1. 주요 발견사항:")
    print(f"   - Model A 최고 상관관계 태그: {target_corr_a.index[0]} (상관계수: {target_corr_a.iloc[0]:.3f})")
    print(f"   - Model B 최고 상관관계 태그: {target_corr_b.index[0]} (상관계수: {target_corr_b.iloc[0]:.3f})")
    print(f"   - Model A 사용률: {np.mean(use_model_a):.2%}")
    print(f"   - Model B 사용률: {np.mean(~use_model_a):.2%}")
    
    best_model = max(metrics.keys(), key=lambda x: metrics[x]['R2'])
    print(f"   - 최고 성능 모델: {best_model} (R²: {metrics[best_model]['R2']:.4f})")
    
    print("\n2. 권장사항:")
    if np.mean(use_model_a) > 0.7:
        print("   - Model A가 주로 사용되므로, Model A의 feature engineering 개선에 집중")
    elif np.mean(use_model_a) < 0.3:
        print("   - Model B가 주로 사용되므로, Model B의 feature engineering 개선에 집중")
    else:
        print("   - 두 모델이 균형있게 사용되므로, ensemble 전략이 효과적")
    
    # Basic tag별 권장사항
    if basic_tags:
        print(f"   - 분석된 Basic tags: {', '.join(basic_tags[:5])}")
    
    return {
        'basic_transformations_a': basic_transformations_a,
        'basic_transformations_b': basic_transformations_b,
        'correlation_matrix_a': correlation_matrix_a,
        'correlation_matrix_b': correlation_matrix_b,
        'target_correlation_a': target_corr_a,
        'target_correlation_b': target_corr_b,
        'importance_df_a': importance_df_a,
        'importance_df_b': importance_df_b,
        'metrics': metrics,
        'ensemble_pred': ensemble_pred,
        'use_model_a': use_model_a,
        'confidence_a': confidence_a
    }

# 샘플 데이터 생성 함수
def create_sample_data(n_samples=1000, n_basic_tags=5):
    """테스트용 샘플 데이터 생성"""
    np.random.seed(42)
    
    # Basic tags 생성
    basic_tags = [f'Tag_{i:02d}' for i in range(1, n_basic_tags+1)]
    
    # Basic 센서 데이터
    basic_data = pd.DataFrame(np.random.randn(n_samples, len(basic_tags)),
                             columns=basic_tags)
    
    # Model A용 feature 생성 (lag, mean, std 등)
    X_a_features = []
    for tag in basic_tags:
        # 원본
        X_a_features.append(basic_data[tag])
        # Lag features
        for lag in [1, 2, 3]:
            X_a_features.append(basic_data[tag].shift(lag).fillna(0))
        # Rolling mean
        X_a_features.append(basic_data[tag].rolling(window=5).mean().fillna(0))
        # Rolling std
        X_a_features.append(basic_data[tag].rolling(window=5).std().fillna(0))
    
    X_a = pd.concat(X_a_features, axis=1)
    X_a.columns = [f'{tag}_orig' for tag in basic_tags] + \
                  [f'{tag}_lag_{lag}' for tag in basic_tags for lag in [1, 2, 3]] + \
                  [f'{tag}_mean_5' for tag in basic_tags] + \
                  [f'{tag}_std_5' for tag in basic_tags]
    
    # Model B용 feature 생성 (다른 변환 방법)
    X_b_features = []
    for tag in basic_tags:
        # 원본
        X_b_features.append(basic_data[tag])
        # 다른 lag
        for lag in [2, 4, 6]:
            X_b_features.append(basic_data[tag].shift(lag).fillna(0))
        # Rolling min/max
        X_b_features.append(basic_data[tag].rolling(window=10).min().fillna(0))
        X_b_features.append(basic_data[tag].rolling(window=10).max().fillna(0))
        # Rolling median
        X_b_features.append(basic_data[tag].rolling(window=7).median().fillna(0))
    
    X_b = pd.concat(X_b_features, axis=1)
    X_b.columns = [f'{tag}_orig' for tag in basic_tags] + \
                  [f'{tag}_lag_{lag}' for tag in basic_tags for lag in [2, 4, 6]] + \
                  [f'{tag}_min_10' for tag in basic_tags] + \
                  [f'{tag}_max_10' for tag in basic_tags] + \
                  [f'{tag}_median_7' for tag in basic_tags]
    
    # Target 생성
    y = (0.3 * basic_data['Tag_01'] + 0.2 * basic_data['Tag_02'] + 
         0.1 * basic_data['Tag_03'] + np.random.randn(n_samples) * 0.1)
    
    return X_a, X_b, y, basic_tags

# 사용 예시 함수 업데이트
def run_manufacturing_eda_with_report(X_a, X_b, y, model_a, model_b, basic_tags=None, threshold=0.5, 
                                    generate_report=True, report_filename="manufacturing_eda_report"):
    """
    제조 산업 센서 데이터 Ensemble 모델 EDA 실행 및 리포트 생성
    
    Parameters:
    X_a: Model A의 features (tags + lag features)
    X_b: Model B의 features (tags + lag features)
    y: target
    model_a: 첫 번째 gradient boosting 모델
    model_b: 두 번째 gradient boosting 모델
    basic_tags: 기본 태그 리스트 (예: ['Tag_01', 'Tag_02', ...])
    threshold: A 모델 선택 기준 임계값
    generate_report: 리포트 생성 여부
    report_filename: 리포트 파일명 (확장자 제외)
    
    Returns:
    dict: 분석 결과 딕셔너리
    """
    results = comprehensive_eda_with_report(X_a, X_b, y, model_a, model_b, basic_tags, threshold, 
                                          generate_report, report_filename)
    return results

# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data()
    
    # 모델 생성 및 훈련
    model_a = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_b = GradientBoostingRegressor(n_estimators=100, random_state=43)
    
    model_a.fit(X_a, y)
    model_b.fit(X_b, y)
    
    # 종합 EDA 실행
    results = comprehensive_eda(X_a, X_b, y, model_a, model_b, basic_tags, threshold=0.5) 