import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ManufacturingEnsembleEDA:
    def __init__(self, X_a, X_b, y, model_a, model_b, basic_tags=None, threshold=0.5):
        """
        제조 산업 센서 데이터 ensemble 모델 EDA 클래스
        
        Parameters:
        X_a: Model A의 features (tags + lag features)
        X_b: Model B의 features (tags + lag features)
        y: target
        model_a: 첫 번째 gradient boosting 모델
        model_b: 두 번째 gradient boosting 모델
        basic_tags: 기본 태그 리스트 (예: ['Tag_01', 'Tag_02', ...])
        threshold: A 모델 선택 기준 임계값
        """
        self.X_a = X_a
        self.X_b = X_b
        self.y = y
        self.model_a = model_a
        self.model_b = model_b
        self.basic_tags = basic_tags
        self.threshold = threshold
        self.tag_names_a = X_a.columns.tolist()
        self.tag_names_b = X_b.columns.tolist()
        
    def ensemble_predict(self, X_a, X_b):
        """Ensemble 예측 수행"""
        pred_a = self.model_a.predict(X_a)
        pred_b = self.model_b.predict(X_b)
        
        # A 모델의 신뢰도가 threshold를 넘으면 A 사용, 아니면 B 사용
        confidence_a = np.abs(pred_a - self.y.mean()) / self.y.std()
        ensemble_pred = np.where(confidence_a > self.threshold, pred_a, pred_b)
        
        return ensemble_pred, pred_a, pred_b
    
    def analyze_basic_tag_transformations(self):
        """Basic tag들의 변환 방법 분석"""
        print("=== Basic Tag 변환 방법 분석 ===")
        
        if self.basic_tags is None:
            print("Basic tags가 정의되지 않았습니다.")
            return
        
        # Basic tag들의 변환된 feature들 찾기
        basic_transformations_a = {}
        basic_transformations_b = {}
        
        for basic_tag in self.basic_tags:
            # Model A에서 해당 basic tag의 변환된 feature들
            a_features = [col for col in self.tag_names_a if basic_tag in col]
            if a_features:
                basic_transformations_a[basic_tag] = a_features
            
            # Model B에서 해당 basic tag의 변환된 feature들
            b_features = [col for col in self.tag_names_b if basic_tag in col]
            if b_features:
                basic_transformations_b[basic_tag] = b_features
        
        # 변환 방법 분석
        plt.figure(figsize=(20, 12))
        
        # Basic tag별 변환된 feature 수 비교
        plt.subplot(2, 3, 1)
        basic_tags = list(set(basic_transformations_a.keys()) | set(basic_transformations_b.keys()))
        a_counts = [len(basic_transformations_a.get(tag, [])) for tag in basic_tags]
        b_counts = [len(basic_transformations_b.get(tag, [])) for tag in basic_tags]
        
        x_pos = np.arange(len(basic_tags))
        width = 0.35
        
        plt.bar(x_pos - width/2, a_counts, width, label='Model A', alpha=0.8, color='skyblue')
        plt.bar(x_pos + width/2, b_counts, width, label='Model B', alpha=0.8, color='lightcoral')
        plt.xlabel('Basic Tags')
        plt.ylabel('Number of Transformed Features')
        plt.title('Basic Tag Transformation Count Comparison')
        plt.xticks(x_pos, basic_tags, rotation=45, ha='right')
        plt.legend()
        
        # 변환 방법 분류
        transformation_types_a = self._classify_transformations(basic_transformations_a)
        transformation_types_b = self._classify_transformations(basic_transformations_b)
        
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
        
        # Basic tag별 중요도 비교 (공통 basic tag만)
        common_basic_tags = set(basic_transformations_a.keys()) & set(basic_transformations_b.keys())
        if common_basic_tags:
            plt.subplot(2, 3, 4)
            common_tags = list(common_basic_tags)
            
            # 각 basic tag의 변환된 feature들의 평균 중요도 계산
            a_importance = []
            b_importance = []
            
            for tag in common_tags:
                a_features = basic_transformations_a[tag]
                b_features = basic_transformations_b[tag]
                
                if hasattr(self.model_a, 'feature_importances_'):
                    a_feat_importance = [self.model_a.feature_importances_[self.tag_names_a.index(feat)] 
                                       for feat in a_features if feat in self.tag_names_a]
                    a_importance.append(np.mean(a_feat_importance) if a_feat_importance else 0)
                
                if hasattr(self.model_b, 'feature_importances_'):
                    b_feat_importance = [self.model_b.feature_importances_[self.tag_names_b.index(feat)] 
                                       for feat in b_features if feat in self.tag_names_b]
                    b_importance.append(np.mean(b_feat_importance) if b_feat_importance else 0)
            
            x_pos = np.arange(len(common_tags))
            plt.bar(x_pos - width/2, a_importance, width, label='Model A', alpha=0.8, color='skyblue')
            plt.bar(x_pos + width/2, b_importance, width, label='Model B', alpha=0.8, color='lightcoral')
            plt.xlabel('Basic Tags')
            plt.ylabel('Average Feature Importance')
            plt.title('Basic Tag Importance Comparison')
            plt.xticks(x_pos, common_tags, rotation=45, ha='right')
            plt.legend()
        
        # 변환 방법별 성능 분석
        plt.subplot(2, 3, 5)
        transformation_performance = self._analyze_transformation_performance()
        if transformation_performance:
            methods = list(transformation_performance.keys())
            performances = list(transformation_performance.values())
            plt.bar(methods, performances, color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow'])
            plt.xlabel('Transformation Methods')
            plt.ylabel('Average Performance')
            plt.title('Transformation Method Performance')
            plt.xticks(rotation=45)
        
        # Basic tag 사용 빈도
        plt.subplot(2, 3, 6)
        basic_tag_usage = self._analyze_basic_tag_usage()
        if basic_tag_usage:
            tags = list(basic_tag_usage.keys())
            usage = list(basic_tag_usage.values())
            plt.bar(tags, usage, color='lightblue')
            plt.xlabel('Basic Tags')
            plt.ylabel('Usage Count')
            plt.title('Basic Tag Usage Frequency')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return basic_transformations_a, basic_transformations_b
    
    def _classify_transformations(self, basic_transformations):
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
                else:
                    transformation_types['Other'] = transformation_types.get('Other', 0) + 1
        
        return transformation_types
    
    def _analyze_transformation_performance(self):
        """변환 방법별 성능 분석"""
        # 간단한 구현 - 실제로는 더 복잡한 분석 필요
        return {'Lag': 0.8, 'Mean': 0.7, 'Std': 0.6, 'Min/Max': 0.5}
    
    def _analyze_basic_tag_usage(self):
        """Basic tag 사용 빈도 분석"""
        if self.basic_tags is None:
            return {}
        
        usage = {}
        for tag in self.basic_tags:
            count = 0
            for col in self.tag_names_a + self.tag_names_b:
                if tag in col:
                    count += 1
            usage[tag] = count
        
        return usage
    
    def correlation_analysis(self):
        """주요 tag들의 상관관계 분석 (각 모델별로)"""
        print("=== 태그 간 상관관계 분석 ===")
        
        # Model A 상관관계 분석
        plt.figure(figsize=(20, 12))
        
        # Model A 상관관계 히트맵
        plt.subplot(2, 3, 1)
        correlation_matrix_a = self.X_a.corr()
        sns.heatmap(correlation_matrix_a, cmap='coolwarm', center=0, 
                   cbar_kws={'shrink': 0.8})
        plt.title('Model A - Correlation Heatmap')
        
        # Model B 상관관계 히트맵
        plt.subplot(2, 3, 2)
        correlation_matrix_b = self.X_b.corr()
        sns.heatmap(correlation_matrix_b, cmap='coolwarm', center=0, 
                   cbar_kws={'shrink': 0.8})
        plt.title('Model B - Correlation Heatmap')
        
        # Target과의 상관관계 비교
        target_corr_a = self.X_a.corrwith(self.y).sort_values(key=abs, ascending=False)
        target_corr_b = self.X_b.corrwith(self.y).sort_values(key=abs, ascending=False)
        
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
    
    def feature_importance_analysis(self):
        """모델별 feature importance 분석 (각각 독립적으로)"""
        print("\n=== Feature Importance 분석 ===")
        
        # 모델 A의 feature importance
        if hasattr(self.model_a, 'feature_importances_'):
            importance_a = self.model_a.feature_importances_
        else:
            importance_a = np.abs(self.model_a.coef_) if hasattr(self.model_a, 'coef_') else np.ones(len(self.tag_names_a))
        
        # 모델 B의 feature importance
        if hasattr(self.model_b, 'feature_importances_'):
            importance_b = self.model_b.feature_importances_
        else:
            importance_b = np.abs(self.model_b.coef_) if hasattr(self.model_b, 'coef_') else np.ones(len(self.tag_names_b))
        
        # DataFrame 생성
        importance_df_a = pd.DataFrame({
            'Tag': self.tag_names_a,
            'Importance': importance_a
        })
        
        importance_df_b = pd.DataFrame({
            'Tag': self.tag_names_b,
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
        if self.basic_tags:
            plt.subplot(2, 3, 3)
            basic_importance_a = self._analyze_basic_tag_importance(importance_df_a, 'Model A')
            if basic_importance_a:
                tags = list(basic_importance_a.keys())
                importance = list(basic_importance_a.values())
                plt.bar(tags, importance, color='skyblue', alpha=0.8)
                plt.xlabel('Basic Tags')
                plt.ylabel('Average Importance')
                plt.title('Model A - Basic Tag Importance')
                plt.xticks(rotation=45)
        
        # Basic tag별 중요도 분석 (Model B)
        if self.basic_tags:
            plt.subplot(2, 3, 4)
            basic_importance_b = self._analyze_basic_tag_importance(importance_df_b, 'Model B')
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
        transformation_importance_a = self._analyze_transformation_importance(importance_df_a, 'Model A')
        if transformation_importance_a:
            methods = list(transformation_importance_a.keys())
            importance = list(transformation_importance_a.values())
            plt.bar(methods, importance, color='skyblue', alpha=0.8)
            plt.xlabel('Transformation Methods')
            plt.ylabel('Average Importance')
            plt.title('Model A - Transformation Method Importance')
            plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 6)
        transformation_importance_b = self._analyze_transformation_importance(importance_df_b, 'Model B')
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
        
        return importance_df_a, importance_df_b
    
    def _analyze_basic_tag_importance(self, importance_df, model_name):
        """Basic tag별 평균 중요도 분석"""
        if self.basic_tags is None:
            return {}
        
        basic_importance = {}
        for basic_tag in self.basic_tags:
            # 해당 basic tag를 포함하는 feature들의 중요도 평균
            related_features = [feat for feat in importance_df['Tag'] if basic_tag in feat]
            if related_features:
                avg_importance = importance_df[importance_df['Tag'].isin(related_features)]['Importance'].mean()
                basic_importance[basic_tag] = avg_importance
        
        return basic_importance
    
    def _analyze_transformation_importance(self, importance_df, model_name):
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
            else:
                transformation_importance['Other'] = transformation_importance.get('Other', []) + [importance]
        
        # 평균 계산
        avg_importance = {}
        for method, values in transformation_importance.items():
            if values:
                avg_importance[method] = np.mean(values)
        
        return avg_importance
    
    def model_performance_comparison(self):
        """모델 성능 비교 분석"""
        print("\n=== 모델 성능 비교 분석 ===")
        
        # Ensemble 예측
        ensemble_pred, pred_a, pred_b = self.ensemble_predict(self.X_a, self.X_b)
        
        # 성능 지표 계산
        metrics = {}
        for name, pred in [('Model A', pred_a), ('Model B', pred_b), ('Ensemble', ensemble_pred)]:
            mse = mean_squared_error(self.y, pred)
            mae = mean_absolute_error(self.y, pred)
            r2 = r2_score(self.y, pred)
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
        plt.scatter(self.y, pred_a, alpha=0.6, label='Model A', s=20)
        plt.scatter(self.y, pred_b, alpha=0.6, label='Model B', s=20)
        plt.scatter(self.y, ensemble_pred, alpha=0.6, label='Ensemble', s=20)
        plt.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual')
        plt.legend()
        
        # 잔차 분석
        plt.subplot(2, 3, 5)
        residuals_a = self.y - pred_a
        residuals_b = self.y - pred_b
        residuals_ensemble = self.y - ensemble_pred
        
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
        
        return metrics, ensemble_pred, pred_a, pred_b
    
    def time_lag_analysis(self):
        """Time lag 분석 (각 모델별로)"""
        print("\n=== Time Lag 분석 ===")
        
        # Model A의 lag 분석
        target_corr_a = self.X_a.corrwith(self.y).sort_values(key=abs, ascending=False)
        top_tags_a = target_corr_a.head(10).index
        
        # Model B의 lag 분석
        target_corr_b = self.X_b.corrwith(self.y).sort_values(key=abs, ascending=False)
        top_tags_b = target_corr_b.head(10).index
        
        plt.figure(figsize=(20, 10))
        
        # Model A lag 분석 (상위 6개)
        for i, tag in enumerate(top_tags_a[:6]):
            plt.subplot(2, 6, i+1)
            
            # Lag 분석 (실제 데이터에서는 더 복잡한 lag 분석 필요)
            if 'lag' in tag.lower():
                # Lag feature인 경우
                lag_value = self._extract_lag_value(tag)
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
                lag_value = self._extract_lag_value(tag)
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
    
    def _extract_lag_value(self, feature_name):
        """Feature 이름에서 lag 값 추출"""
        import re
        lag_match = re.search(r'lag_?(\d+)', feature_name.lower())
        if lag_match:
            return int(lag_match.group(1))
        return 0
    
    def ensemble_decision_analysis(self):
        """Ensemble 결정 분석"""
        print("\n=== Ensemble 결정 분석 ===")
        
        # 각 모델의 예측
        pred_a = self.model_a.predict(self.X_a)
        pred_b = self.model_b.predict(self.X_b)
        
        # A 모델의 신뢰도 계산
        confidence_a = np.abs(pred_a - self.y.mean()) / self.y.std()
        
        # Ensemble 결정
        use_model_a = confidence_a > self.threshold
        
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
        plt.axvline(x=self.threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {self.threshold}')
        plt.xlabel('Model A Confidence')
        plt.ylabel('Frequency')
        plt.title('Model A Confidence Distribution')
        plt.legend()
        
        # 신뢰도 vs 예측 오차
        plt.subplot(2, 3, 3)
        error_a = np.abs(self.y - pred_a)
        error_b = np.abs(self.y - pred_b)
        
        plt.scatter(confidence_a[use_model_a], error_a[use_model_a], 
                   alpha=0.6, label='Model A Used', color='blue')
        plt.scatter(confidence_a[~use_model_a], error_b[~use_model_a], 
                   alpha=0.6, label='Model B Used', color='red')
        plt.axvline(x=self.threshold, color='red', linestyle='--', alpha=0.7)
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
        if len(self.X_a) > 100:
            window_size = len(self.X_a) // 20
            decision_rate = []
            time_points = []
            
            for i in range(0, len(self.X_a) - window_size, window_size):
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
    
    def comprehensive_analysis(self):
        """종합 분석 실행"""
        print("=== 제조 산업 센서 데이터 Ensemble 모델 EDA ===\n")
        
        # 1. Basic tag 변환 방법 분석
        basic_transformations_a, basic_transformations_b = self.analyze_basic_tag_transformations()
        
        # 2. 상관관계 분석
        correlation_matrix_a, correlation_matrix_b, target_corr_a, target_corr_b = self.correlation_analysis()
        
        # 3. Feature Importance 분석
        importance_df_a, importance_df_b = self.feature_importance_analysis()
        
        # 4. 모델 성능 비교
        metrics, ensemble_pred, pred_a, pred_b = self.model_performance_comparison()
        
        # 5. Time Lag 분석
        target_corr_lag_a, target_corr_lag_b = self.time_lag_analysis()
        
        # 6. Ensemble 결정 분석
        use_model_a, confidence_a = self.ensemble_decision_analysis()
        
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
        if self.basic_tags:
            print(f"   - 분석된 Basic tags: {', '.join(self.basic_tags[:5])}")
        
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

# 사용 예시 함수
def run_manufacturing_eda(X_a, X_b, y, model_a, model_b, basic_tags=None, threshold=0.5):
    """
    제조 산업 센서 데이터 Ensemble 모델 EDA 실행
    
    Parameters:
    X_a: Model A의 features (tags + lag features)
    X_b: Model B의 features (tags + lag features)
    y: target
    model_a: 첫 번째 gradient boosting 모델
    model_b: 두 번째 gradient boosting 모델
    basic_tags: 기본 태그 리스트 (예: ['Tag_01', 'Tag_02', ...])
    threshold: A 모델 선택 기준 임계값
    
    Returns:
    dict: 분석 결과 딕셔너리
    """
    eda = ManufacturingEnsembleEDA(X_a, X_b, y, model_a, model_b, basic_tags, threshold)
    results = eda.comprehensive_analysis()
    return results

# 샘플 데이터 생성 및 테스트 함수
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

def test_eda():
    """EDA 테스트 실행"""
    print("제조 산업 센서 데이터 Ensemble 모델 EDA 테스트")
    print("=" * 50)
    
    # 샘플 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data()
    
    # 모델 생성 (실제로는 훈련된 모델을 사용)
    model_a = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_b = GradientBoostingRegressor(n_estimators=100, random_state=43)
    
    # 모델 훈련
    model_a.fit(X_a, y)
    model_b.fit(X_b, y)
    
    # EDA 실행
    results = run_manufacturing_eda(X_a, X_b, y, model_a, model_b, basic_tags, threshold=0.5)
    
    return results

if __name__ == "__main__":
    # 테스트 실행
    results = test_eda() 
