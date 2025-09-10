
"""
ACN 수율 최적화 - 핵심 변수 관계 분석
수율에 영향을 주는 핵심 변수들(spec a, b, c, input, output) 간의 관계를 통계적으로 분석

Author: AI Assistant
Date: 2024
Purpose: 수율 최적화를 위한 핵심 인자 파악
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class YieldCoreAnalyzer:
    """수율 핵심 변수 분석 클래스"""
    
    def __init__(self, data):
        """
        Args:
            data (pd.DataFrame): 분석할 데이터프레임
        """
        self.data = data.copy()
        self.core_vars = ['spec_a', 'spec_b', 'spec_c', 'input_amount', 'output_amount', 'yield']
        self.scaler = StandardScaler()
        
    def prepare_data(self):
        """데이터 전처리 및 핵심 변수 추출"""
        print("=== 데이터 전처리 ===")
        
        # 핵심 변수 존재 여부 확인
        missing_vars = [var for var in self.core_vars if var not in self.data.columns]
        if missing_vars:
            print(f"경고: 다음 변수들이 데이터에 없습니다: {missing_vars}")
            available_vars = [var for var in self.core_vars if var in self.data.columns]
            self.core_vars = available_vars
        
        # 핵심 변수만 추출
        self.core_data = self.data[self.core_vars].copy()
        
        # 결측치 확인
        print("\n결측치 현황:")
        missing_info = self.core_data.isnull().sum()
        print(missing_info[missing_info > 0])
        
        # 결측치 제거
        initial_count = len(self.core_data)
        self.core_data = self.core_data.dropna()
        final_count = len(self.core_data)
        print(f"\n결측치 제거: {initial_count} → {final_count} (제거: {initial_count - final_count})")
        
        # 수율 계산 (output/input)
        if 'input_amount' in self.core_data.columns and 'output_amount' in self.core_data.columns:
            self.core_data['calculated_yield'] = (self.core_data['output_amount'] / 
                                                 self.core_data['input_amount']) * 100
            
            # 기존 yield와 계산된 yield 비교
            if 'yield' in self.core_data.columns:
                yield_diff = abs(self.core_data['yield'] - self.core_data['calculated_yield'])
                print(f"\n수율 차이 통계:")
                print(f"평균 차이: {yield_diff.mean():.2f}%")
                print(f"최대 차이: {yield_diff.max():.2f}%")
                print(f"표준편차: {yield_diff.std():.2f}%")
        
        print(f"\n최종 분석 데이터 크기: {self.core_data.shape}")
        return self.core_data
    
    def basic_statistics(self):
        """기초 통계 분석"""
        print("\n=== 기초 통계 분석 ===")
        
        # 기술통계
        print("\n1. 기술통계:")
        desc_stats = self.core_data.describe()
        print(desc_stats.round(3))
        
        # 변동계수 (CV)
        print("\n2. 변동계수 (CV = std/mean):")
        cv_stats = (self.core_data.std() / self.core_data.mean() * 100).round(2)
        print(cv_stats)
        
        # 왜도와 첨도
        print("\n3. 분포 특성:")
        skewness = self.core_data.skew().round(3)
        kurtosis = self.core_data.kurtosis().round(3)
        dist_stats = pd.DataFrame({
            'Skewness': skewness,
            'Kurtosis': kurtosis
        })
        print(dist_stats)
        
        return desc_stats, cv_stats, dist_stats
    
    def correlation_analysis(self):
        """상관관계 분석"""
        print("\n=== 상관관계 분석 ===")
        
        # 수치형 변수만 선택
        numeric_vars = self.core_data.select_dtypes(include=[np.number]).columns
        corr_data = self.core_data[numeric_vars]
        
        # 1. 피어슨 상관계수
        print("\n1. 피어슨 상관계수:")
        pearson_corr = corr_data.corr()
        print(pearson_corr.round(3))
        
        # 2. 스피어만 상관계수
        print("\n2. 스피어만 상관계수:")
        spearman_corr = corr_data.corr(method='spearman')
        print(spearman_corr.round(3))
        
        # 3. 수율과의 상관관계 (절댓값 기준 정렬)
        yield_col = 'yield' if 'yield' in corr_data.columns else 'calculated_yield'
        if yield_col in corr_data.columns:
            yield_corr = corr_data.corr()[yield_col].abs().sort_values(ascending=False)
            print(f"\n3. {yield_col}과의 상관관계 (절댓값 기준):")
            print(yield_corr.round(3))
        
        return pearson_corr, spearman_corr
    
    def statistical_tests(self):
        """통계적 유의성 검정"""
        print("\n=== 통계적 유의성 검정 ===")
        
        yield_col = 'yield' if 'yield' in self.core_data.columns else 'calculated_yield'
        if yield_col not in self.core_data.columns:
            print("수율 데이터가 없어 검정을 수행할 수 없습니다.")
            return None
        
        results = []
        
        for col in self.core_data.columns:
            if col != yield_col:
                # 피어슨 상관계수 검정
                pearson_r, pearson_p = pearsonr(self.core_data[col], self.core_data[yield_col])
                
                # 스피어만 상관계수 검정
                spearman_r, spearman_p = spearmanr(self.core_data[col], self.core_data[yield_col])
                
                # 켄달 타우 검정
                kendall_tau, kendall_p = kendalltau(self.core_data[col], self.core_data[yield_col])
                
                results.append({
                    'Variable': col,
                    'Pearson_r': pearson_r,
                    'Pearson_p': pearson_p,
                    'Spearman_r': spearman_r,
                    'Spearman_p': spearman_p,
                    'Kendall_tau': kendall_tau,
                    'Kendall_p': kendall_p
                })
        
        results_df = pd.DataFrame(results)
        print("\n상관계수 검정 결과:")
        print(results_df.round(4))
        
        # 유의한 상관관계 (p < 0.05)
        significant_results = results_df[
            (results_df['Pearson_p'] < 0.05) | 
            (results_df['Spearman_p'] < 0.05) | 
            (results_df['Kendall_p'] < 0.05)
        ]
        
        print(f"\n유의한 상관관계 (p < 0.05): {len(significant_results)}개")
        if len(significant_results) > 0:
            print(significant_results[['Variable', 'Pearson_r', 'Pearson_p', 'Spearman_r', 'Spearman_p']].round(4))
        
        return results_df
    
    def regression_analysis(self):
        """회귀 분석"""
        print("\n=== 회귀 분석 ===")
        
        yield_col = 'yield' if 'yield' in self.core_data.columns else 'calculated_yield'
        if yield_col not in self.core_data.columns:
            print("수율 데이터가 없어 회귀분석을 수행할 수 없습니다.")
            return None
        
        # 독립변수와 종속변수 분리
        X = self.core_data.drop(columns=[yield_col])
        y = self.core_data[yield_col]
        
        # 데이터 분할
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 1. 다중선형회귀
        print("\n1. 다중선형회귀:")
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        
        # 예측
        y_pred_train = lr_model.predict(X_train)
        y_pred_test = lr_model.predict(X_test)
        
        # 성능 평가
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        print(f"Train R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE: {test_rmse:.4f}")
        
        # 회귀계수
        coefficients = pd.DataFrame({
            'Variable': X.columns,
            'Coefficient': lr_model.coef_,
            'Abs_Coefficient': np.abs(lr_model.coef_)
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\n회귀계수 (절댓값 기준 정렬):")
        print(coefficients.round(4))
        
        # 2. 랜덤포레스트
        print("\n2. 랜덤포레스트:")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # 예측
        y_pred_rf_train = rf_model.predict(X_train)
        y_pred_rf_test = rf_model.predict(X_test)
        
        # 성능 평가
        rf_train_r2 = r2_score(y_train, y_pred_rf_train)
        rf_test_r2 = r2_score(y_test, y_pred_rf_test)
        rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
        rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
        
        print(f"Train R²: {rf_train_r2:.4f}")
        print(f"Test R²: {rf_test_r2:.4f}")
        print(f"Train RMSE: {rf_train_rmse:.4f}")
        print(f"Test RMSE: {rf_test_rmse:.4f}")
        
        # 특성 중요도
        feature_importance = pd.DataFrame({
            'Variable': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n특성 중요도:")
        print(feature_importance.round(4))
        
        return {
            'linear_model': lr_model,
            'rf_model': rf_model,
            'linear_coefficients': coefficients,
            'rf_importance': feature_importance,
            'performance': {
                'linear': {'train_r2': train_r2, 'test_r2': test_r2, 'train_rmse': train_rmse, 'test_rmse': test_rmse},
                'rf': {'train_r2': rf_train_r2, 'test_r2': rf_test_r2, 'train_rmse': rf_train_rmse, 'test_rmse': rf_test_rmse}
            }
        }
    
    def pca_analysis(self):
        """주성분 분석"""
        print("\n=== 주성분 분석 (PCA) ===")
        
        # 수율 제외한 변수들로 PCA 수행
        yield_col = 'yield' if 'yield' in self.core_data.columns else 'calculated_yield'
        pca_data = self.core_data.drop(columns=[yield_col])
        
        # 데이터 표준화
        pca_data_scaled = self.scaler.fit_transform(pca_data)
        
        # PCA 수행
        pca = PCA()
        pca_result = pca.fit_transform(pca_data_scaled)
        
        # 설명 분산 비율
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        print("주성분별 설명 분산 비율:")
        for i, (var_ratio, cum_var_ratio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
            print(f"PC{i+1}: {var_ratio:.4f} (누적: {cum_var_ratio:.4f})")
        
        # 주성분 계수
        pca_components = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(pca.components_.shape[0])],
            index=pca_data.columns
        )
        
        print("\n주성분 계수:")
        print(pca_components.round(4))
        
        # 주성분과 수율의 상관관계
        if yield_col in self.core_data.columns:
            pca_yield_corr = []
            for i in range(pca_result.shape[1]):
                corr, p_value = pearsonr(pca_result[:, i], self.core_data[yield_col])
                pca_yield_corr.append({'PC': f'PC{i+1}', 'Correlation': corr, 'P_value': p_value})
            
            pca_yield_df = pd.DataFrame(pca_yield_corr)
            print(f"\n주성분과 {yield_col}의 상관관계:")
            print(pca_yield_df.round(4))
        
        return {
            'pca': pca,
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance_ratio': cumulative_variance_ratio,
            'components': pca_components,
            'pca_result': pca_result
        }
    
    def create_visualizations(self):
        """시각화 생성"""
        print("\n=== 시각화 생성 ===")
        
        # 1. 상관관계 히트맵
        plt.figure(figsize=(12, 10))
        
        # 상관관계 계산
        corr_matrix = self.core_data.corr()
        
        # 히트맵
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('핵심 변수 상관관계 히트맵', fontsize=16, pad=20)
        plt.tight_layout()
        plt.show()
        
        # 2. 수율과 각 변수의 산점도
        yield_col = 'yield' if 'yield' in self.core_data.columns else 'calculated_yield'
        if yield_col in self.core_data.columns:
            other_vars = [col for col in self.core_data.columns if col != yield_col]
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.ravel()
            
            for i, var in enumerate(other_vars[:6]):  # 최대 6개 변수
                if i < len(axes):
                    axes[i].scatter(self.core_data[var], self.core_data[yield_col], alpha=0.6)
                    axes[i].set_xlabel(var)
                    axes[i].set_ylabel(yield_col)
                    axes[i].set_title(f'{var} vs {yield_col}')
                    
                    # 상관계수 표시
                    corr, p_val = pearsonr(self.core_data[var], self.core_data[yield_col])
                    axes[i].text(0.05, 0.95, f'r={corr:.3f}\np={p_val:.3f}', 
                               transform=axes[i].transAxes, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 빈 subplot 제거
            for i in range(len(other_vars), len(axes)):
                fig.delaxes(axes[i])
            
            plt.suptitle('수율과 핵심 변수들의 산점도', fontsize=16)
            plt.tight_layout()
            plt.show()
        
        # 3. 분포 히스토그램
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, var in enumerate(self.core_data.columns[:6]):
            if i < len(axes):
                axes[i].hist(self.core_data[var], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_xlabel(var)
                axes[i].set_ylabel('빈도')
                axes[i].set_title(f'{var} 분포')
                
                # 통계 정보 표시
                mean_val = self.core_data[var].mean()
                std_val = self.core_data[var].std()
                axes[i].axvline(mean_val, color='red', linestyle='--', label=f'평균: {mean_val:.2f}')
                axes[i].axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7)
                axes[i].axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
                axes[i].legend()
        
        # 빈 subplot 제거
        for i in range(len(self.core_data.columns), len(axes)):
            fig.delaxes(axes[i])
        
        plt.suptitle('핵심 변수들의 분포', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def generate_report(self):
        """종합 분석 리포트 생성"""
        print("\n" + "="*60)
        print("ACN 수율 최적화 - 핵심 변수 관계 분석 리포트")
        print("="*60)
        
        # 데이터 전처리
        self.prepare_data()
        
        # 기초 통계
        desc_stats, cv_stats, dist_stats = self.basic_statistics()
        
        # 상관관계 분석
        pearson_corr, spearman_corr = self.correlation_analysis()
        
        # 통계적 검정
        test_results = self.statistical_tests()
        
        # 회귀 분석
        regression_results = self.regression_analysis()
        
        # PCA 분석
        pca_results = self.pca_analysis()
        
        # 시각화
        self.create_visualizations()
        
        # 종합 결론
        print("\n" + "="*60)
        print("종합 결론 및 권장사항")
        print("="*60)
        
        yield_col = 'yield' if 'yield' in self.core_data.columns else 'calculated_yield'
        
        if yield_col in self.core_data.columns:
            # 수율과 가장 상관관계가 높은 변수들
            yield_corr = self.core_data.corr()[yield_col].abs().sort_values(ascending=False)
            top_correlations = yield_corr.drop(yield_col).head(3)
            
            print(f"\n1. {yield_col}과 가장 상관관계가 높은 변수들:")
            for var, corr in top_correlations.items():
                print(f"   - {var}: {corr:.3f}")
        
        if regression_results:
            # 회귀분석 결과 요약
            print(f"\n2. 회귀분석 성능:")
            print(f"   - 선형회귀 R²: {regression_results['performance']['linear']['test_r2']:.3f}")
            print(f"   - 랜덤포레스트 R²: {regression_results['performance']['rf']['test_r2']:.3f}")
            
            # 가장 중요한 변수들
            print(f"\n3. 수율에 가장 중요한 변수들 (랜덤포레스트 기준):")
            top_features = regression_results['rf_importance'].head(3)
            for _, row in top_features.iterrows():
                print(f"   - {row['Variable']}: {row['Importance']:.3f}")
        
        if pca_results:
            # PCA 결과 요약
            print(f"\n4. 주성분 분석:")
            print(f"   - 첫 번째 주성분 설명 분산: {pca_results['explained_variance_ratio'][0]:.3f}")
            print(f"   - 첫 두 주성분 누적 설명 분산: {pca_results['cumulative_variance_ratio'][1]:.3f}")
        
        print(f"\n5. 권장사항:")
        print(f"   - 수율 최적화를 위해 상관관계가 높은 변수들의 모니터링 강화")
        print(f"   - 회귀분석 결과를 바탕으로 수율 예측 모델 구축")
        print(f"   - 주성분 분석을 통한 차원 축소 및 노이즈 제거")
        print(f"   - 추가적인 공정 변수와의 관계 분석 필요")
        
        return {
            'descriptive_stats': desc_stats,
            'correlations': {'pearson': pearson_corr, 'spearman': spearman_corr},
            'statistical_tests': test_results,
            'regression': regression_results,
            'pca': pca_results
        }

def main():
    """메인 실행 함수"""
    print("ACN 수율 최적화 - 핵심 변수 관계 분석")
    print("="*50)
    
    # 샘플 데이터 생성 (실제 데이터로 대체 필요)
    np.random.seed(42)
    n_samples = 100
    
    sample_data = pd.DataFrame({
        'spec_a': np.random.normal(85, 5, n_samples),
        'spec_b': np.random.normal(92, 3, n_samples),
        'spec_c': np.random.normal(78, 4, n_samples),
        'input_amount': np.random.normal(1000, 100, n_samples),
        'output_amount': np.random.normal(850, 80, n_samples),
        'yield': np.random.normal(85, 5, n_samples)
    })
    
    # 분석기 생성 및 실행
    analyzer = YieldCoreAnalyzer(sample_data)
    results = analyzer.generate_report()
    
    return analyzer, results

if __name__ == "__main__":
    analyzer, results = main()
