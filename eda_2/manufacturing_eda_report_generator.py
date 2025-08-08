import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# PDF 생성을 위한 라이브러리
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from datetime import datetime
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ManufacturingEDAReportGenerator:
    def __init__(self, X_a, X_b, y, model_a, model_b, basic_tags=None, threshold=0.5):
        """
        제조 산업 센서 데이터 Ensemble 모델 EDA 리포트 생성기
        
        Parameters:
        X_a: Model A의 features
        X_b: Model B의 features
        y: target
        model_a: 첫 번째 gradient boosting 모델
        model_b: 두 번째 gradient boosting 모델
        basic_tags: 기본 태그 리스트
        threshold: A 모델 선택 기준 임계값
        """
        self.X_a = X_a
        self.X_b = X_b
        self.y = y
        self.model_a = model_a
        self.model_b = model_b
        self.basic_tags = basic_tags
        self.threshold = threshold
        self.results = {}
        
    def generate_pdf_report(self, filename="manufacturing_eda_report.pdf"):
        """PDF 리포트 생성"""
        with PdfPages(filename) as pdf:
            # 1. 커버 페이지
            self._create_cover_page(pdf)
            
            # 2. 요약 페이지
            self._create_summary_page(pdf)
            
            # 3. Basic Tag 변환 분석
            self._create_basic_tag_analysis_page(pdf)
            
            # 4. 상관관계 분석
            self._create_correlation_analysis_page(pdf)
            
            # 5. Feature Importance 분석
            self._create_feature_importance_page(pdf)
            
            # 6. 모델 성능 비교
            self._create_performance_comparison_page(pdf)
            
            # 7. Time Lag 분석
            self._create_time_lag_analysis_page(pdf)
            
            # 8. Ensemble 결정 분석
            self._create_ensemble_decision_page(pdf)
            
            # 9. 결론 및 권장사항
            self._create_conclusion_page(pdf)
            
        print(f"PDF 리포트가 생성되었습니다: {filename}")
        
    def _create_cover_page(self, pdf):
        """커버 페이지 생성"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # 제목
        title = "제조 산업 센서 데이터\nEnsemble 모델 EDA 리포트"
        ax.text(0.5, 0.8, title, fontsize=24, fontweight='bold', 
                ha='center', va='center', transform=ax.transAxes)
        
        # 생성 정보
        info_text = f"""
        생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        
        데이터 정보:
        - Model A Features: {self.X_a.shape[1]}개
        - Model B Features: {self.X_b.shape[1]}개
        - 샘플 수: {self.X_a.shape[0]}개
        - Basic Tags: {len(self.basic_tags) if self.basic_tags else 0}개
        
        분석 내용:
        • Basic Tag 변환 방법 분석
        • 상관관계 분석
        • Feature Importance 분석
        • 모델 성능 비교
        • Time Lag 분석
        • Ensemble 결정 분석
        """
        
        ax.text(0.5, 0.4, info_text, fontsize=12, ha='center', va='center',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightblue", alpha=0.3))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_summary_page(self, pdf):
        """요약 페이지 생성"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # 성능 요약
        ensemble_pred, pred_a, pred_b, confidence_a = self._ensemble_predict()
        metrics = self._calculate_metrics(pred_a, pred_b, ensemble_pred)
        
        summary_text = f"""
        📊 모델 성능 요약
        
        Model A:
        - R² Score: {metrics['Model A']['R2']:.4f}
        - MSE: {metrics['Model A']['MSE']:.4f}
        - MAE: {metrics['Model A']['MAE']:.4f}
        
        Model B:
        - R² Score: {metrics['Model B']['R2']:.4f}
        - MSE: {metrics['Model B']['MSE']:.4f}
        - MAE: {metrics['Model B']['MAE']:.4f}
        
        Ensemble:
        - R² Score: {metrics['Ensemble']['R2']:.4f}
        - MSE: {metrics['Ensemble']['MSE']:.4f}
        - MAE: {metrics['Ensemble']['MAE']:.4f}
        
        🎯 Ensemble 사용률
        - Model A 사용률: {np.mean(confidence_a > self.threshold):.2%}
        - Model B 사용률: {np.mean(confidence_a <= self.threshold):.2%}
        
        🏆 최고 성능 모델: {max(metrics.keys(), key=lambda x: metrics[x]['R2'])}
        """
        
        ax.text(0.05, 0.95, summary_text, fontsize=12, ha='left', va='top',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightgreen", alpha=0.3))
        
        # 주요 발견사항
        findings_text = f"""
        🔍 주요 발견사항
        
        • Model A와 B는 서로 다른 feature engineering 전략 사용
        • Basic tags: {', '.join(self.basic_tags[:5]) if self.basic_tags else 'N/A'}
        • Ensemble threshold: {self.threshold}
        • 평균 신뢰도: {np.mean(confidence_a):.3f}
        
        📈 개선 포인트
        • Feature engineering 최적화
        • Threshold 조정 실험
        • 새로운 변환 방법 도입 검토
        """
        
        ax.text(0.05, 0.4, findings_text, fontsize=12, ha='left', va='top',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightyellow", alpha=0.3))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_basic_tag_analysis_page(self, pdf):
        """Basic Tag 변환 분석 페이지"""
        if not self.basic_tags:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Basic Tag 변환 방법 분석', fontsize=16, fontweight='bold')
        
        # Basic tag별 변환된 feature 수 비교
        basic_transformations_a = {}
        basic_transformations_b = {}
        
        for basic_tag in self.basic_tags:
            a_features = [col for col in self.X_a.columns if basic_tag in col]
            b_features = [col for col in self.X_b.columns if basic_tag in col]
            if a_features:
                basic_transformations_a[basic_tag] = a_features
            if b_features:
                basic_transformations_b[basic_tag] = b_features
        
        # 변환 수 비교
        ax1 = axes[0, 0]
        basic_tags_list = list(set(basic_transformations_a.keys()) | set(basic_transformations_b.keys()))
        a_counts = [len(basic_transformations_a.get(tag, [])) for tag in basic_tags_list]
        b_counts = [len(basic_transformations_b.get(tag, [])) for tag in basic_tags_list]
        
        x_pos = np.arange(len(basic_tags_list))
        width = 0.35
        
        ax1.bar(x_pos - width/2, a_counts, width, label='Model A', alpha=0.8, color='skyblue')
        ax1.bar(x_pos + width/2, b_counts, width, label='Model B', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Basic Tags')
        ax1.set_ylabel('Number of Transformed Features')
        ax1.set_title('Basic Tag Transformation Count')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(basic_tags_list, rotation=45, ha='right')
        ax1.legend()
        
        # 변환 방법 분류
        def classify_transformations(basic_transformations):
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
        
        transformation_types_a = classify_transformations(basic_transformations_a)
        transformation_types_b = classify_transformations(basic_transformations_b)
        
        # Model A 변환 방법 분포
        ax2 = axes[0, 1]
        if transformation_types_a:
            types_a = list(transformation_types_a.keys())
            counts_a = list(transformation_types_a.values())
            ax2.pie(counts_a, labels=types_a, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Model A - Transformation Methods')
        
        # Model B 변환 방법 분포
        ax3 = axes[1, 0]
        if transformation_types_b:
            types_b = list(transformation_types_b.keys())
            counts_b = list(transformation_types_b.values())
            ax3.pie(counts_b, labels=types_b, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Model B - Transformation Methods')
        
        # 변환 방법별 사용 빈도 비교
        ax4 = axes[1, 1]
        all_transformations = set(transformation_types_a.keys()) | set(transformation_types_b.keys())
        a_freq = [transformation_types_a.get(method, 0) for method in all_transformations]
        b_freq = [transformation_types_b.get(method, 0) for method in all_transformations]
        
        x_pos = np.arange(len(all_transformations))
        ax4.bar(x_pos - width/2, a_freq, width, label='Model A', alpha=0.8, color='skyblue')
        ax4.bar(x_pos + width/2, b_freq, width, label='Model B', alpha=0.8, color='lightcoral')
        ax4.set_xlabel('Transformation Methods')
        ax4.set_ylabel('Usage Count')
        ax4.set_title('Transformation Method Usage')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(all_transformations, rotation=45)
        ax4.legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_correlation_analysis_page(self, pdf):
        """상관관계 분석 페이지"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('상관관계 분석', fontsize=16, fontweight='bold')
        
        # Target과의 상관관계 비교
        target_corr_a = self.X_a.corrwith(self.y).sort_values(key=abs, ascending=False)
        target_corr_b = self.X_b.corrwith(self.y).sort_values(key=abs, ascending=False)
        
        # Model A Top 10 상관관계
        ax1 = axes[0, 0]
        top_target_corr_a = target_corr_a.head(10)
        bars = ax1.barh(range(len(top_target_corr_a)), top_target_corr_a.values,
                       color=['red' if x < 0 else 'blue' for x in top_target_corr_a.values])
        ax1.set_yticks(range(len(top_target_corr_a)))
        ax1.set_yticklabels(top_target_corr_a.index, fontsize=8)
        ax1.set_xlabel('Correlation with Target')
        ax1.set_title('Model A - Top 10 Correlations')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Model B Top 10 상관관계
        ax2 = axes[0, 1]
        top_target_corr_b = target_corr_b.head(10)
        bars = ax2.barh(range(len(top_target_corr_b)), top_target_corr_b.values,
                       color=['red' if x < 0 else 'blue' for x in top_target_corr_b.values])
        ax2.set_yticks(range(len(top_target_corr_b)))
        ax2.set_yticklabels(top_target_corr_b.index, fontsize=8)
        ax2.set_xlabel('Correlation with Target')
        ax2.set_title('Model B - Top 10 Correlations')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 상관관계 분포 비교
        ax3 = axes[0, 2]
        ax3.hist(target_corr_a.values, bins=30, alpha=0.7, color='skyblue', 
                edgecolor='black', label='Model A')
        ax3.hist(target_corr_b.values, bins=30, alpha=0.7, color='lightcoral', 
                edgecolor='black', label='Model B')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Correlation with Target')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Correlation Distribution')
        ax3.legend()
        
        # 상관관계 히트맵 (Model A)
        ax4 = axes[1, 0]
        correlation_matrix_a = self.X_a.corr()
        sns.heatmap(correlation_matrix_a.iloc[:10, :10], cmap='coolwarm', center=0, 
                   cbar_kws={'shrink': 0.8}, ax=ax4)
        ax4.set_title('Model A - Correlation Heatmap (Top 10)')
        
        # 상관관계 히트맵 (Model B)
        ax5 = axes[1, 1]
        correlation_matrix_b = self.X_b.corr()
        sns.heatmap(correlation_matrix_b.iloc[:10, :10], cmap='coolwarm', center=0, 
                   cbar_kws={'shrink': 0.8}, ax=ax5)
        ax5.set_title('Model B - Correlation Heatmap (Top 10)')
        
        # 상관관계 강도 비교
        ax6 = axes[1, 2]
        ax6.scatter(target_corr_a.head(20), target_corr_b.head(20), alpha=0.6)
        ax6.set_xlabel('Model A Correlation')
        ax6.set_ylabel('Model B Correlation')
        ax6.set_title('Correlation Strength Comparison')
        ax6.axhline(y=0, color='red', linestyle='--', alpha=0.3)
        ax6.axvline(x=0, color='red', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_feature_importance_page(self, pdf):
        """Feature Importance 분석 페이지"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Feature Importance 분석', fontsize=16, fontweight='bold')
        
        # Feature importance 계산
        importance_a = self.model_a.feature_importances_
        importance_b = self.model_b.feature_importances_
        
        importance_df_a = pd.DataFrame({
            'Tag': self.X_a.columns.tolist(),
            'Importance': importance_a
        })
        
        importance_df_b = pd.DataFrame({
            'Tag': self.X_b.columns.tolist(),
            'Importance': importance_b
        })
        
        # Model A Top 15 Importance
        ax1 = axes[0, 0]
        top_a = importance_df_a.nlargest(15, 'Importance')
        ax1.barh(range(len(top_a)), top_a['Importance'], color='skyblue')
        ax1.set_yticks(range(len(top_a)))
        ax1.set_yticklabels(top_a['Tag'], fontsize=8)
        ax1.set_xlabel('Importance')
        ax1.set_title('Model A - Top 15 Feature Importance')
        
        # Model B Top 15 Importance
        ax2 = axes[0, 1]
        top_b = importance_df_b.nlargest(15, 'Importance')
        ax2.barh(range(len(top_b)), top_b['Importance'], color='lightcoral')
        ax2.set_yticks(range(len(top_b)))
        ax2.set_yticklabels(top_b['Tag'], fontsize=8)
        ax2.set_xlabel('Importance')
        ax2.set_title('Model B - Top 15 Feature Importance')
        
        # Basic tag별 중요도 분석 (Model A)
        if self.basic_tags:
            ax3 = axes[0, 2]
            basic_importance_a = {}
            for basic_tag in self.basic_tags:
                related_features = [feat for feat in importance_df_a['Tag'] if basic_tag in feat]
                if related_features:
                    avg_importance = importance_df_a[importance_df_a['Tag'].isin(related_features)]['Importance'].mean()
                    basic_importance_a[basic_tag] = avg_importance
            
            if basic_importance_a:
                tags = list(basic_importance_a.keys())
                importance = list(basic_importance_a.values())
                ax3.bar(tags, importance, color='skyblue', alpha=0.8)
                ax3.set_xlabel('Basic Tags')
                ax3.set_ylabel('Average Importance')
                ax3.set_title('Model A - Basic Tag Importance')
                ax3.tick_params(axis='x', rotation=45)
        
        # Basic tag별 중요도 분석 (Model B)
        if self.basic_tags:
            ax4 = axes[1, 0]
            basic_importance_b = {}
            for basic_tag in self.basic_tags:
                related_features = [feat for feat in importance_df_b['Tag'] if basic_tag in feat]
                if related_features:
                    avg_importance = importance_df_b[importance_df_b['Tag'].isin(related_features)]['Importance'].mean()
                    basic_importance_b[basic_tag] = avg_importance
            
            if basic_importance_b:
                tags = list(basic_importance_b.keys())
                importance = list(basic_importance_b.values())
                ax4.bar(tags, importance, color='lightcoral', alpha=0.8)
                ax4.set_xlabel('Basic Tags')
                ax4.set_ylabel('Average Importance')
                ax4.set_title('Model B - Basic Tag Importance')
                ax4.tick_params(axis='x', rotation=45)
        
        # 변환 방법별 중요도 분석
        def analyze_transformation_importance(importance_df):
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
            
            avg_importance = {}
            for method, values in transformation_importance.items():
                if values:
                    avg_importance[method] = np.mean(values)
            return avg_importance
        
        transformation_importance_a = analyze_transformation_importance(importance_df_a)
        transformation_importance_b = analyze_transformation_importance(importance_df_b)
        
        # Model A 변환 방법별 중요도
        ax5 = axes[1, 1]
        if transformation_importance_a:
            methods = list(transformation_importance_a.keys())
            importance = list(transformation_importance_a.values())
            ax5.bar(methods, importance, color='skyblue', alpha=0.8)
            ax5.set_xlabel('Transformation Methods')
            ax5.set_ylabel('Average Importance')
            ax5.set_title('Model A - Transformation Method Importance')
            ax5.tick_params(axis='x', rotation=45)
        
        # Model B 변환 방법별 중요도
        ax6 = axes[1, 2]
        if transformation_importance_b:
            methods = list(transformation_importance_b.keys())
            importance = list(transformation_importance_b.values())
            ax6.bar(methods, importance, color='lightcoral', alpha=0.8)
            ax6.set_xlabel('Transformation Methods')
            ax6.set_ylabel('Average Importance')
            ax6.set_title('Model B - Transformation Method Importance')
            ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_performance_comparison_page(self, pdf):
        """모델 성능 비교 페이지"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('모델 성능 비교', fontsize=16, fontweight='bold')
        
        # Ensemble 예측
        ensemble_pred, pred_a, pred_b, confidence_a = self._ensemble_predict()
        metrics = self._calculate_metrics(pred_a, pred_b, ensemble_pred)
        
        # 성능 지표 비교
        model_names = list(metrics.keys())
        mse_values = [metrics[name]['MSE'] for name in model_names]
        mae_values = [metrics[name]['MAE'] for name in model_names]
        r2_values = [metrics[name]['R2'] for name in model_names]
        
        # MSE 비교
        ax1 = axes[0, 0]
        bars = ax1.bar(model_names, mse_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('MSE Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # MAE 비교
        ax2 = axes[0, 1]
        bars = ax2.bar(model_names, mae_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('MAE Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # R² 비교
        ax3 = axes[0, 2]
        bars = ax3.bar(model_names, r2_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # 예측값 vs 실제값 비교
        ax4 = axes[1, 0]
        ax4.scatter(self.y, pred_a, alpha=0.6, label='Model A', s=20)
        ax4.scatter(self.y, pred_b, alpha=0.6, label='Model B', s=20)
        ax4.scatter(self.y, ensemble_pred, alpha=0.6, label='Ensemble', s=20)
        ax4.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], 'r--', lw=2)
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title('Predicted vs Actual')
        ax4.legend()
        
        # 잔차 분석
        ax5 = axes[1, 1]
        residuals_a = self.y - pred_a
        residuals_b = self.y - pred_b
        residuals_ensemble = self.y - ensemble_pred
        
        ax5.scatter(pred_a, residuals_a, alpha=0.6, label='Model A', s=20)
        ax5.scatter(pred_b, residuals_b, alpha=0.6, label='Model B', s=20)
        ax5.scatter(ensemble_pred, residuals_ensemble, alpha=0.6, label='Ensemble', s=20)
        ax5.axhline(y=0, color='r', linestyle='--')
        ax5.set_xlabel('Predicted Values')
        ax5.set_ylabel('Residuals')
        ax5.set_title('Residual Plot')
        ax5.legend()
        
        # 잔차 분포
        ax6 = axes[1, 2]
        ax6.hist(residuals_a, alpha=0.7, label='Model A', bins=30)
        ax6.hist(residuals_b, alpha=0.7, label='Model B', bins=30)
        ax6.hist(residuals_ensemble, alpha=0.7, label='Ensemble', bins=30)
        ax6.set_xlabel('Residuals')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Residual Distribution')
        ax6.legend()
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_time_lag_analysis_page(self, pdf):
        """Time Lag 분석 페이지"""
        fig, axes = plt.subplots(2, 6, figsize=(20, 8))
        fig.suptitle('Time Lag 분석', fontsize=16, fontweight='bold')
        
        # Model A의 lag 분석
        target_corr_a = self.X_a.corrwith(self.y).sort_values(key=abs, ascending=False)
        top_tags_a = target_corr_a.head(10).index
        
        # Model B의 lag 분석
        target_corr_b = self.X_b.corrwith(self.y).sort_values(key=abs, ascending=False)
        top_tags_b = target_corr_b.head(10).index
        
        # Model A lag 분석 (상위 6개)
        for i, tag in enumerate(top_tags_a[:6]):
            ax = axes[0, i]
            
            if 'lag' in tag.lower():
                import re
                lag_match = re.search(r'lag_?(\d+)', tag.lower())
                lag_value = int(lag_match.group(1)) if lag_match else 0
                ax.bar([lag_value], [target_corr_a[tag]], color='skyblue')
                ax.set_xlabel('Lag')
                ax.set_ylabel('Correlation')
                ax.set_title(f'{tag}\\nLag: {lag_value}')
            else:
                ax.bar([0], [target_corr_a[tag]], color='skyblue')
                ax.set_xlabel('No Lag')
                ax.set_ylabel('Correlation')
                ax.set_title(f'{tag}')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Model B lag 분석 (상위 6개)
        for i, tag in enumerate(top_tags_b[:6]):
            ax = axes[1, i]
            
            if 'lag' in tag.lower():
                import re
                lag_match = re.search(r'lag_?(\d+)', tag.lower())
                lag_value = int(lag_match.group(1)) if lag_match else 0
                ax.bar([lag_value], [target_corr_b[tag]], color='lightcoral')
                ax.set_xlabel('Lag')
                ax.set_ylabel('Correlation')
                ax.set_title(f'{tag}\\nLag: {lag_value}')
            else:
                ax.bar([0], [target_corr_b[tag]], color='lightcoral')
                ax.set_xlabel('No Lag')
                ax.set_ylabel('Correlation')
                ax.set_title(f'{tag}')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_ensemble_decision_page(self, pdf):
        """Ensemble 결정 분석 페이지"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Ensemble 결정 분석', fontsize=16, fontweight='bold')
        
        # 각 모델의 예측
        pred_a = self.model_a.predict(self.X_a)
        pred_b = self.model_b.predict(self.X_b)
        
        # A 모델의 신뢰도 계산
        confidence_a = np.abs(pred_a - self.y.mean()) / self.y.std()
        
        # Ensemble 결정
        use_model_a = confidence_a > self.threshold
        
        # 결정 분포
        ax1 = axes[0, 0]
        decision_counts = [np.sum(use_model_a), np.sum(~use_model_a)]
        ax1.pie(decision_counts, labels=['Model A', 'Model B'], autopct='%1.1f%%',
                colors=['skyblue', 'lightcoral'])
        ax1.set_title('Ensemble Decision Distribution')
        
        # 신뢰도 분포
        ax2 = axes[0, 1]
        ax2.hist(confidence_a, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(x=self.threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {self.threshold}')
        ax2.set_xlabel('Model A Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Model A Confidence Distribution')
        ax2.legend()
        
        # 신뢰도 vs 예측 오차
        ax3 = axes[0, 2]
        error_a = np.abs(self.y - pred_a)
        error_b = np.abs(self.y - pred_b)
        
        ax3.scatter(confidence_a[use_model_a], error_a[use_model_a], 
                   alpha=0.6, label='Model A Used', color='blue')
        ax3.scatter(confidence_a[~use_model_a], error_b[~use_model_a], 
                   alpha=0.6, label='Model B Used', color='red')
        ax3.axvline(x=self.threshold, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Model A Confidence')
        ax3.set_ylabel('Absolute Error')
        ax3.set_title('Confidence vs Error')
        ax3.legend()
        
        # 모델별 예측값 분포
        ax4 = axes[1, 0]
        ax4.hist(pred_a, alpha=0.7, label='Model A', bins=30, color='skyblue')
        ax4.hist(pred_b, alpha=0.7, label='Model B', bins=30, color='lightcoral')
        ax4.set_xlabel('Predicted Values')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Distribution')
        ax4.legend()
        
        # 예측값 차이 분석
        ax5 = axes[1, 1]
        pred_diff = pred_a - pred_b
        ax5.hist(pred_diff, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax5.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax5.set_xlabel('Prediction Difference (A - B)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Prediction Difference Distribution')
        
        # 시간에 따른 결정 변화
        ax6 = axes[1, 2]
        if len(self.X_a) > 100:
            window_size = len(self.X_a) // 20
            decision_rate = []
            time_points = []
            
            for i in range(0, len(self.X_a) - window_size, window_size):
                window_decisions = use_model_a[i:i+window_size]
                decision_rate.append(np.mean(window_decisions))
                time_points.append(i + window_size//2)
            
            ax6.plot(time_points, decision_rate, marker='o', linewidth=2)
            ax6.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
            ax6.set_xlabel('Time Point')
            ax6.set_ylabel('Model A Usage Rate')
            ax6.set_title('Model A Usage Over Time')
            ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _create_conclusion_page(self, pdf):
        """결론 및 권장사항 페이지"""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')
        
        # 성능 요약
        ensemble_pred, pred_a, pred_b, confidence_a = self._ensemble_predict()
        metrics = self._calculate_metrics(pred_a, pred_b, ensemble_pred)
        
        conclusion_text = f"""
        📊 분석 결과 요약
        
        🏆 성능 비교:
        • Model A R²: {metrics['Model A']['R2']:.4f}
        • Model B R²: {metrics['Model B']['R2']:.4f}
        • Ensemble R²: {metrics['Ensemble']['R2']:.4f}
        
        🎯 Ensemble 사용률:
        • Model A 사용률: {np.mean(confidence_a > self.threshold):.2%}
        • Model B 사용률: {np.mean(confidence_a <= self.threshold):.2%}
        
        📈 주요 발견사항:
        • {max(metrics.keys(), key=lambda x: metrics[x]['R2'])}가 최고 성능
        • 평균 신뢰도: {np.mean(confidence_a):.3f}
        • Basic tags: {', '.join(self.basic_tags[:5]) if self.basic_tags else 'N/A'}
        
        🔧 권장사항:
        • Feature engineering 최적화
        • Threshold 조정 실험 ({self.threshold} → 0.3~0.7 범위에서 테스트)
        • 새로운 변환 방법 도입 검토
        • 실시간 성능 모니터링 시스템 구축
        • 정기적인 모델 재훈련 계획 수립
        
        📋 다음 단계:
        1. 운영 환경에서의 성능 검증
        2. 새로운 센서 데이터에 대한 모델 적응성 테스트
        3. Ensemble 전략의 실시간 효과성 모니터링
        4. 추가 feature engineering 실험
        """
        
        ax.text(0.05, 0.95, conclusion_text, fontsize=12, ha='left', va='top',
                transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
                facecolor="lightblue", alpha=0.3))
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    def _ensemble_predict(self):
        """Ensemble 예측 수행"""
        pred_a = self.model_a.predict(self.X_a)
        pred_b = self.model_b.predict(self.X_b)
        
        # A 모델의 신뢰도가 threshold를 넘으면 A 사용, 아니면 B 사용
        confidence_a = np.abs(pred_a - self.y.mean()) / self.y.std()
        ensemble_pred = np.where(confidence_a > self.threshold, pred_a, pred_b)
        
        return ensemble_pred, pred_a, pred_b, confidence_a
    
    def _calculate_metrics(self, pred_a, pred_b, ensemble_pred):
        """성능 지표 계산"""
        metrics = {}
        for name, pred in [('Model A', pred_a), ('Model B', pred_b), ('Ensemble', ensemble_pred)]:
            mse = mean_squared_error(self.y, pred)
            mae = mean_absolute_error(self.y, pred)
            r2 = r2_score(self.y, pred)
            metrics[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
        return metrics

# 사용 예시 함수
def generate_eda_report(X_a, X_b, y, model_a, model_b, basic_tags=None, threshold=0.5, filename="manufacturing_eda_report.pdf"):
    """
    EDA 리포트 PDF 생성
    
    Parameters:
    X_a: Model A의 features
    X_b: Model B의 features
    y: target
    model_a: 첫 번째 gradient boosting 모델
    model_b: 두 번째 gradient boosting 모델
    basic_tags: 기본 태그 리스트
    threshold: A 모델 선택 기준 임계값
    filename: PDF 파일명
    
    Returns:
    str: 생성된 PDF 파일 경로
    """
    generator = ManufacturingEDAReportGenerator(X_a, X_b, y, model_a, model_b, basic_tags, threshold)
    generator.generate_pdf_report(filename)
    return filename

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

def test_report_generation():
    """리포트 생성 테스트"""
    print("제조 산업 센서 데이터 Ensemble 모델 EDA 리포트 생성 테스트")
    print("=" * 60)
    
    # 샘플 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data()
    
    # 모델 생성 및 훈련
    model_a = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_b = GradientBoostingRegressor(n_estimators=100, random_state=43)
    
    model_a.fit(X_a, y)
    model_b.fit(X_b, y)
    
    # 리포트 생성
    filename = generate_eda_report(X_a, X_b, y, model_a, model_b, basic_tags, threshold=0.5)
    
    print(f"리포트가 생성되었습니다: {filename}")
    return filename

if __name__ == "__main__":
    # 테스트 실행
    filename = test_report_generation()
