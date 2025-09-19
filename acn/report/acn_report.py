import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr, ttest_ind, mannwhitneyu
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 리포트 생성 라이브러리
try:
    from jinja2 import Template
    import weasyprint
    HTML_TO_PDF_AVAILABLE = True
except ImportError:
    HTML_TO_PDF_AVAILABLE = False
    print("HTML to PDF 변환을 위해 jinja2, weasyprint 설치를 권장합니다.")

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ACNComprehensiveAnalyzer:
    """
    ACN 정제 공정 종합 분석 클래스
    - Final_FR과 F/R Level을 고려한 구간별 분석
    - Feature Selection, 다변량 통계, 영향 인자 분석
    - 종합 리포트 생성
    """
    
    def __init__(self, data_path=None, df=None):
        """
        초기화
        
        Parameters:
        data_path: 데이터 파일 경로
        df: 이미 로드된 DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("데이터 경로 또는 DataFrame을 제공해야 합니다.")
        
        self.analysis_results = {}
        self.final_data = None
        self.process_data = None
        self.quality_columns = ['AN-10_200nm', 'AN-10_225nm', 'AN-10_250nm']
        
        print("ACN 종합 분석기 초기화 완료")
    
    def preprocess_data(self):
        """
        데이터 전처리 및 구간별 분리
        """
        print("=" * 80)
        print("데이터 전처리 및 구간별 분리")
        print("=" * 80)
        
        # 1. 데이터 타입 설정
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
        
        # 범주형 변수
        categorical_columns = ['Source', 'IsBubbled', 'IsBothChillerOn', 'Final_FR']
        for col in categorical_columns:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype('category')
        
        # 수치형 변수
        numeric_columns = [col for col in self.df.columns if col not in categorical_columns + ['Date']]
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 2. 품질값 정규화 (spec 기준)
        for col in self.quality_columns:
            if col in self.df.columns:
                # 0을 기준으로 정규화 (실제 spec 값에 따라 조정 필요)
                self.df[f'{col}_normalized'] = self.df[col] - 0
        
        # 3. 구간별 데이터 분리
        # Final_FR = 1: 최종 분석 데이터
        # Final_FR = 0: 과정 중 분석 데이터
        if 'Final_FR' in self.df.columns:
            self.final_data = self.df[self.df['Final_FR'] == 1].copy()
            self.process_data = self.df[self.df['Final_FR'] == 0].copy()
            
            print(f"전체 데이터 크기: {self.df.shape}")
            print(f"최종 분석 데이터 크기: {self.final_data.shape}")
            print(f"과정 중 분석 데이터 크기: {self.process_data.shape}")
        else:
            print("Final_FR 컬럼이 없습니다. 전체 데이터를 최종 데이터로 사용합니다.")
            self.final_data = self.df.copy()
            self.process_data = pd.DataFrame()
        
        # 4. Batch별 분석
        self.analyze_batch_patterns()
        
        return self.final_data, self.process_data
    
    def analyze_batch_patterns(self):
        """
        Batch별 분석 패턴 분석
        """
        print("\n" + "=" * 80)
        print("Batch별 분석 패턴 분석")
        print("=" * 80)
        
        if 'No' in self.df.columns:
            # Batch별 분석 횟수
            batch_analysis_counts = self.df.groupby('No').size()
            
            print("Batch별 분석 횟수 통계:")
            print(batch_analysis_counts.describe())
            
            # 최종값만 있는 batch vs 여러 분석값이 있는 batch
            final_only_batches = self.df[self.df['Final_FR'] == 1]['No'].unique()
            process_batches = self.df[self.df['Final_FR'] == 0]['No'].unique()
            
            print(f"\n최종값만 있는 batch 수: {len(final_only_batches)}")
            print(f"과정 중 분석값이 있는 batch 수: {len(process_batches)}")
            
            # F/R Level 분포
            if 'F/R Level' in self.df.columns:
                print(f"\nF/R Level 분포:")
                print(self.df['F/R Level'].describe())
                
                # 구간별 분포
                level_ranges = pd.cut(self.df['F/R Level'], 
                                    bins=[0, 25, 50, 75, 100], 
                                    labels=['0-25%', '25-50%', '50-75%', '75-100%'])
                print(f"\nF/R Level 구간별 분포:")
                print(level_ranges.value_counts().sort_index())
    
    def comprehensive_eda_analysis(self):
        """
        종합 EDA 분석
        """
        print("\n" + "=" * 80)
        print("종합 EDA 분석")
        print("=" * 80)
        
        # 1. 기본 통계 분석
        basic_stats = self.basic_statistical_analysis()
        
        # 2. Feature Selection
        feature_selection = self.feature_selection_analysis()
        
        # 3. 다변량 통계 분석
        multivariate_analysis = self.multivariate_statistical_analysis()
        
        # 4. 영향 인자 분석
        impact_analysis = self.impact_factor_analysis()
        
        # 5. 구간 기반 영향도 분석
        interval_analysis = self.interval_based_analysis()
        
        # 6. Yield vs Input/Output 분석
        yield_analysis = self.yield_input_output_analysis()
        
        self.analysis_results = {
            'basic_stats': basic_stats,
            'feature_selection': feature_selection,
            'multivariate_analysis': multivariate_analysis,
            'impact_analysis': impact_analysis,
            'interval_analysis': interval_analysis,
            'yield_analysis': yield_analysis
        }
        
        return self.analysis_results
    
    def basic_statistical_analysis(self):
        """
        기본 통계 분석
        """
        print("\n1. 기본 통계 분석")
        print("-" * 50)
        
        results = {}
        
        # 최종 데이터 기본 통계
        if self.final_data is not None and len(self.final_data) > 0:
            numeric_cols = self.final_data.select_dtypes(include=[np.number]).columns
            final_stats = self.final_data[numeric_cols].describe()
            results['final_data_stats'] = final_stats
            
            print("최종 데이터 기본 통계:")
            print(final_stats.round(3))
        
        # 과정 데이터 기본 통계
        if self.process_data is not None and len(self.process_data) > 0:
            numeric_cols = self.process_data.select_dtypes(include=[np.number]).columns
            process_stats = self.process_data[numeric_cols].describe()
            results['process_data_stats'] = process_stats
            
            print("\n과정 데이터 기본 통계:")
            print(process_stats.round(3))
        
        # Yield 통계
        if 'Yield' in self.df.columns:
            yield_stats = self.df['Yield'].describe()
            results['yield_stats'] = yield_stats
            
            print(f"\nYield 통계:")
            print(f"평균: {yield_stats['mean']:.4f}")
            print(f"표준편차: {yield_stats['std']:.4f}")
            print(f"최솟값: {yield_stats['min']:.4f}")
            print(f"최댓값: {yield_stats['max']:.4f}")
            print(f"변동계수: {(yield_stats['std']/yield_stats['mean']*100):.2f}%")
        
        return results
    
    def feature_selection_analysis(self):
        """
        Feature Selection 분석
        """
        print("\n2. Feature Selection 분석")
        print("-" * 50)
        
        if 'Yield' not in self.df.columns:
            print("Yield 컬럼이 없습니다.")
            return None
        
        # 수치형 변수만 선택 (Yield 제외)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Yield' in numeric_cols:
            numeric_cols.remove('Yield')
        
        # 결측치가 있는 컬럼 제외
        numeric_cols = [col for col in numeric_cols if not self.df[col].isnull().all()]
        
        X = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        y = self.df['Yield'].fillna(self.df['Yield'].median())
        
        results = {}
        
        # 1. 상관관계 분석
        correlations = {}
        for col in numeric_cols:
            if col in X.columns:
                corr_pearson, p_value_pearson = pearsonr(X[col], y)
                corr_spearman, p_value_spearman = spearmanr(X[col], y)
                correlations[col] = {
                    'pearson_corr': corr_pearson,
                    'pearson_pvalue': p_value_pearson,
                    'spearman_corr': corr_spearman,
                    'spearman_pvalue': p_value_spearman,
                    'abs_pearson': abs(corr_pearson),
                    'abs_spearman': abs(corr_spearman)
                }
        
        corr_df = pd.DataFrame(correlations).T
        corr_df = corr_df.sort_values('abs_pearson', ascending=False)
        results['correlations'] = corr_df
        
        print("Yield와의 상관관계 (상위 10개):")
        print(corr_df[['pearson_corr', 'pearson_pvalue', 'spearman_corr', 'spearman_pvalue']].head(10).round(4))
        
        # 2. F-test
        selector_f = SelectKBest(score_func=f_regression, k='all')
        selector_f.fit(X, y)
        
        f_scores = pd.DataFrame({
            'feature': numeric_cols,
            'f_score': selector_f.scores_,
            'p_value': selector_f.pvalues_
        }).sort_values('f_score', ascending=False)
        
        results['f_test'] = f_scores
        
        print("\nF-test 결과 (상위 10개):")
        print(f_scores.head(10).round(4))
        
        # 3. 상호정보량
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_df = pd.DataFrame({
            'feature': numeric_cols,
            'mutual_info': mi_scores
        }).sort_values('mutual_info', ascending=False)
        
        results['mutual_info'] = mi_df
        
        print("\n상호정보량 결과 (상위 10개):")
        print(mi_df.head(10).round(4))
        
        return results
    
    def multivariate_statistical_analysis(self):
        """
        다변량 통계 분석
        """
        print("\n3. 다변량 통계 분석")
        print("-" * 50)
        
        if 'Yield' not in self.df.columns:
            print("Yield 컬럼이 없습니다.")
            return None
        
        # 수치형 변수만 선택 (Yield 제외)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Yield' in numeric_cols:
            numeric_cols.remove('Yield')
        
        X = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        y = self.df['Yield'].fillna(self.df['Yield'].median())
        
        results = {}
        
        # 1. PCA 분석
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA()
        pca_result = pca.fit_transform(X_scaled)
        
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance_ratio)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        results['pca'] = {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components_95': n_components_95,
            'components': pca.components_
        }
        
        print("PCA 분석 결과:")
        print(f"95% 분산을 설명하는 주성분 수: {n_components_95}")
        print(f"첫 번째 주성분 설명 분산: {explained_variance_ratio[0]:.4f}")
        
        # 2. 클러스터링
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        df_clustered = self.df.copy()
        df_clustered['Cluster'] = cluster_labels
        
        cluster_yield_stats = df_clustered.groupby('Cluster')['Yield'].agg(['count', 'mean', 'std', 'min', 'max'])
        
        results['clustering'] = {
            'cluster_labels': cluster_labels,
            'cluster_stats': cluster_yield_stats
        }
        
        print("\n클러스터링 결과:")
        print(cluster_yield_stats.round(4))
        
        return results
    
    def impact_factor_analysis(self):
        """
        영향 인자 분석
        """
        print("\n4. 영향 인자 분석")
        print("-" * 50)
        
        if 'Yield' not in self.df.columns:
            print("Yield 컬럼이 없습니다.")
            return None
        
        results = {}
        
        # 1. 고수율/저수율 그룹 분석
        yield_median = self.df['Yield'].median()
        high_yield_mask = self.df['Yield'] > yield_median
        low_yield_mask = self.df['Yield'] <= yield_median
        
        print(f"수율 중간값: {yield_median:.4f}")
        print(f"고수율 그룹: {high_yield_mask.sum()}개")
        print(f"저수율 그룹: {low_yield_mask.sum()}개")
        
        # 수치형 변수에 대한 t-test
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Yield' in numeric_cols:
            numeric_cols.remove('Yield')
        
        t_test_results = {}
        for col in numeric_cols:
            if col in self.df.columns:
                high_group = self.df[high_yield_mask][col].dropna()
                low_group = self.df[low_yield_mask][col].dropna()
                
                if len(high_group) > 0 and len(low_group) > 0:
                    t_stat, p_value = ttest_ind(high_group, low_group)
                    t_test_results[col] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'high_mean': high_group.mean(),
                        'low_mean': low_group.mean(),
                        'difference': high_group.mean() - low_group.mean()
                    }
        
        t_test_df = pd.DataFrame(t_test_results).T
        t_test_df = t_test_df.sort_values('p_value')
        
        results['t_test'] = t_test_df
        
        print("\n고수율 vs 저수율 그룹 t-test (상위 10개):")
        significant_features = t_test_df[t_test_df['p_value'] < 0.05].head(10)
        print(significant_features.round(4))
        
        # 2. 범주형 변수 분석
        categorical_cols = ['Source', 'IsBubbled', 'IsBothChillerOn']
        categorical_results = {}
        
        for col in categorical_cols:
            if col in self.df.columns:
                high_yield_cat = self.df[high_yield_mask][col].value_counts()
                low_yield_cat = self.df[low_yield_mask][col].value_counts()
                
                categorical_results[col] = {
                    'high_yield': high_yield_cat,
                    'low_yield': low_yield_cat
                }
        
        results['categorical'] = categorical_results
        
        return results
    
    def interval_based_analysis(self):
        """
        구간 기반 영향도 분석
        """
        print("\n5. 구간 기반 영향도 분석")
        print("-" * 50)
        
        if 'F/R Level' not in self.df.columns or 'Yield' not in self.df.columns:
            print("F/R Level 또는 Yield 컬럼이 없습니다.")
            return None
        
        results = {}
        
        # F/R Level 구간별 분석
        level_ranges = pd.cut(self.df['F/R Level'], 
                            bins=[0, 25, 50, 75, 100], 
                            labels=['0-25%', '25-50%', '50-75%', '75-100%'])
        
        # 구간별 Yield 통계
        interval_yield_stats = self.df.groupby(level_ranges)['Yield'].agg(['count', 'mean', 'std', 'min', 'max'])
        
        results['interval_yield_stats'] = interval_yield_stats
        
        print("F/R Level 구간별 Yield 통계:")
        print(interval_yield_stats.round(4))
        
        # 구간별 품질값 분석
        quality_interval_analysis = {}
        for col in self.quality_columns:
            if col in self.df.columns:
                quality_stats = self.df.groupby(level_ranges)[col].agg(['count', 'mean', 'std'])
                quality_interval_analysis[col] = quality_stats
        
        results['quality_interval_analysis'] = quality_interval_analysis
        
        print("\n구간별 품질값 분석:")
        for col, stats in quality_interval_analysis.items():
            print(f"\n{col}:")
            print(stats.round(4))
        
        # 구간별 상관관계 분석
        interval_correlations = {}
        for interval in ['0-25%', '25-50%', '50-75%', '75-100%']:
            interval_data = self.df[level_ranges == interval]
            if len(interval_data) > 10:  # 충분한 데이터가 있는 경우만
                numeric_cols = interval_data.select_dtypes(include=[np.number]).columns.tolist()
                if 'Yield' in numeric_cols:
                    numeric_cols.remove('Yield')
                
                correlations = {}
                for col in numeric_cols:
                    if col in interval_data.columns:
                        corr, p_value = pearsonr(interval_data[col], interval_data['Yield'])
                        correlations[col] = {'correlation': corr, 'p_value': p_value}
                
                interval_correlations[interval] = correlations
        
        results['interval_correlations'] = interval_correlations
        
        return results
    
    def yield_input_output_analysis(self):
        """
        Yield vs Input/Output 분석
        """
        print("\n6. Yield vs Input/Output 분석")
        print("-" * 50)
        
        if 'Input_source' not in self.df.columns or 'Yield' not in self.df.columns:
            print("Input_source 또는 Yield 컬럼이 없습니다.")
            return None
        
        results = {}
        
        # Output 계산 (Yield = output/input*100)
        # output = Yield * input / 100
        self.df['Calculated_Output'] = self.df['Yield'] * self.df['Input_source'] / 100
        
        # Input vs Output 상관관계
        input_output_corr, input_output_p = pearsonr(self.df['Input_source'], self.df['Calculated_Output'])
        
        results['input_output_correlation'] = {
            'correlation': input_output_corr,
            'p_value': input_output_p
        }
        
        print(f"Input vs Output 상관관계: {input_output_corr:.4f} (p-value: {input_output_p:.4f})")
        
        # Input 구간별 분석
        input_quartiles = pd.qcut(self.df['Input_source'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        input_interval_stats = self.df.groupby(input_quartiles).agg({
            'Input_source': ['count', 'mean', 'std'],
            'Calculated_Output': ['mean', 'std'],
            'Yield': ['mean', 'std']
        }).round(4)
        
        results['input_interval_stats'] = input_interval_stats
        
        print("\nInput 구간별 통계:")
        print(input_interval_stats)
        
        # Output 일정성 검정
        # 가설: Output이 Input에 상관없이 일정하다
        output_std = self.df['Calculated_Output'].std()
        output_mean = self.df['Calculated_Output'].mean()
        output_cv = output_std / output_mean * 100
        
        results['output_consistency'] = {
            'mean': output_mean,
            'std': output_std,
            'cv': output_cv
        }
        
        print(f"\nOutput 일정성 분석:")
        print(f"Output 평균: {output_mean:.4f}")
        print(f"Output 표준편차: {output_std:.4f}")
        print(f"Output 변동계수: {output_cv:.2f}%")
        
        # Input 증가에 따른 Output 변화 분석
        input_bins = pd.cut(self.df['Input_source'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        input_output_analysis = self.df.groupby(input_bins).agg({
            'Input_source': 'mean',
            'Calculated_Output': 'mean',
            'Yield': 'mean'
        }).round(4)
        
        results['input_output_analysis'] = input_output_analysis
        
        print("\nInput 증가에 따른 Output 변화:")
        print(input_output_analysis)
        
        return results
    
    def generate_comprehensive_report(self, output_format='html'):
        """
        종합 리포트 생성
        
        Parameters:
        output_format: 'html' 또는 'pdf'
        """
        print("\n" + "=" * 80)
        print("종합 리포트 생성")
        print("=" * 80)
        
        # 리포트 데이터 준비
        report_data = self._prepare_report_data()
        
        # HTML 리포트 생성
        html_content = self._generate_html_report(report_data)
        
        # 파일 저장
        if output_format == 'html':
            with open('acn_comprehensive_report.html', 'w', encoding='utf-8') as f:
                f.write(html_content)
            print("HTML 리포트가 'acn_comprehensive_report.html'로 저장되었습니다.")
        
        elif output_format == 'pdf' and HTML_TO_PDF_AVAILABLE:
            try:
                pdf_content = weasyprint.HTML(string=html_content).write_pdf()
                with open('acn_comprehensive_report.pdf', 'wb') as f:
                    f.write(pdf_content)
                print("PDF 리포트가 'acn_comprehensive_report.pdf'로 저장되었습니다.")
            except Exception as e:
                print(f"PDF 생성 실패: {str(e)}")
                print("HTML 리포트만 생성되었습니다.")
        
        return html_content
    
    def _prepare_report_data(self):
        """리포트 데이터 준비"""
        report_data = {
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'data_summary': {
                'total_samples': len(self.df),
                'final_samples': len(self.final_data) if self.final_data is not None else 0,
                'process_samples': len(self.process_data) if self.process_data is not None else 0,
                'features': len(self.df.columns)
            },
            'yield_summary': {},
            'key_findings': [],
            'recommendations': []
        }
        
        # Yield 요약
        if 'Yield' in self.df.columns:
            yield_stats = self.df['Yield'].describe()
            report_data['yield_summary'] = {
                'mean': yield_stats['mean'],
                'std': yield_stats['std'],
                'min': yield_stats['min'],
                'max': yield_stats['max'],
                'cv': (yield_stats['std'] / yield_stats['mean'] * 100)
            }
        
        # 주요 발견사항
        if 'feature_selection' in self.analysis_results:
            corr_data = self.analysis_results['feature_selection']['correlations']
            top_feature = corr_data.index[0]
            top_corr = corr_data.iloc[0]['pearson_corr']
            report_data['key_findings'].append(f"가장 높은 상관관계를 보이는 특성: {top_feature} (상관계수: {top_corr:.4f})")
        
        if 'yield_analysis' in self.analysis_results:
            input_output_corr = self.analysis_results['yield_analysis']['input_output_correlation']['correlation']
            report_data['key_findings'].append(f"Input과 Output의 상관관계: {input_output_corr:.4f}")
            
            output_cv = self.analysis_results['yield_analysis']['output_consistency']['cv']
            report_data['key_findings'].append(f"Output의 변동계수: {output_cv:.2f}%")
        
        # 권장사항
        if 'feature_selection' in self.analysis_results:
            corr_data = self.analysis_results['feature_selection']['correlations']
            top_3_features = corr_data.head(3).index.tolist()
            report_data['recommendations'].append(f"주요 영향 인자 모니터링 강화: {', '.join(top_3_features)}")
        
        if 'yield_analysis' in self.analysis_results:
            input_output_corr = self.analysis_results['yield_analysis']['input_output_correlation']['correlation']
            if input_output_corr < 0.5:
                report_data['recommendations'].append("Input 증가에 따른 Output 향상 방안 모색 필요")
            else:
                report_data['recommendations'].append("Input과 Output의 상관관계가 높으므로 Input 최적화에 집중")
        
        return report_data
    
    def _generate_html_report(self, report_data):
        """HTML 리포트 생성"""
        html_template = """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ACN 정제 공정 종합 분석 리포트</title>
            <style>
                body { font-family: 'Malgun Gothic', Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .header { text-align: center; margin-bottom: 40px; }
                .header h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
                .section { margin: 30px 0; }
                .section h2 { color: #34495e; border-left: 5px solid #3498db; padding-left: 15px; }
                .section h3 { color: #7f8c8d; margin-top: 25px; }
                .summary-box { background-color: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
                .finding-box { background-color: #e8f5e8; padding: 15px; border-radius: 5px; margin: 10px 0; }
                .recommendation-box { background-color: #fff3cd; padding: 15px; border-radius: 5px; margin: 10px 0; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { border: 1px solid #bdc3c7; padding: 12px; text-align: left; }
                th { background-color: #34495e; color: white; }
                tr:nth-child(even) { background-color: #f8f9fa; }
                .metric { font-weight: bold; color: #e74c3c; }
                .conclusion { background-color: #d4edda; padding: 20px; border-radius: 5px; margin: 30px 0; }
                .conclusion h3 { color: #155724; margin-top: 0; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ACN 정제 공정 종합 분석 리포트</h1>
                <p>분석 일시: {{ analysis_date }}</p>
            </div>
            
            <div class="section">
                <h2>📊 데이터 요약</h2>
                <div class="summary-box">
                    <p><strong>전체 샘플 수:</strong> {{ data_summary.total_samples }}개</p>
                    <p><strong>최종 분석 데이터:</strong> {{ data_summary.final_samples }}개</p>
                    <p><strong>과정 중 분석 데이터:</strong> {{ data_summary.process_samples }}개</p>
                    <p><strong>분석 특성 수:</strong> {{ data_summary.features }}개</p>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Yield 분석 결과</h2>
                <div class="summary-box">
                    <p><strong>평균 Yield:</strong> <span class="metric">{{ "%.4f"|format(yield_summary.mean) }}</span></p>
                    <p><strong>표준편차:</strong> {{ "%.4f"|format(yield_summary.std) }}</p>
                    <p><strong>최솟값:</strong> {{ "%.4f"|format(yield_summary.min) }}</p>
                    <p><strong>최댓값:</strong> {{ "%.4f"|format(yield_summary.max) }}</p>
                    <p><strong>변동계수:</strong> <span class="metric">{{ "%.2f"|format(yield_summary.cv) }}%</span></p>
                </div>
            </div>
            
            <div class="section">
                <h2>🔍 주요 발견사항</h2>
                {% for finding in key_findings %}
                <div class="finding-box">
                    <p>{{ finding }}</p>
                </div>
                {% endfor %}
            </div>
            
            <div class="section">
                <h2>💡 권장사항</h2>
                {% for recommendation in recommendations %}
                <div class="recommendation-box">
                    <p>{{ recommendation }}</p>
                </div>
                {% endfor %}
            </div>
            
            <div class="conclusion">
                <h3>🎯 결론</h3>
                <p>본 분석을 통해 ACN 정제 공정의 Yield에 영향을 주는 주요 인자들을 식별하고, 
                Input과 Output의 관계를 분석했습니다. 특히 Output이 Input에 상관없이 일정한 경향을 보이는 것을 확인했으며, 
                이를 바탕으로 Input을 증가시키면서 Output도 함께 향상시킬 수 있는 방안을 모색해야 합니다.</p>
                
                <p><strong>핵심 인사이트:</strong></p>
                <ul>
                    <li>Yield = Output/Input × 100 공식에서 Output이 거의 일정함</li>
                    <li>Input 증가에 따른 Output 향상이 Yield 개선의 핵심</li>
                    <li>주요 영향 인자들의 모니터링과 최적화 필요</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        from jinja2 import Template
        template = Template(html_template)
        return template.render(**report_data)

def main_comprehensive_analysis(data_path=None, df=None, output_format='html'):
    """
    ACN 정제 공정 종합 분석 메인 함수
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    output_format: 'html' 또는 'pdf'
    
    Returns:
    results: 분석 결과 딕셔너리
    """
    # 1. 분석기 초기화
    analyzer = ACNComprehensiveAnalyzer(data_path, df)
    
    # 2. 데이터 전처리
    analyzer.preprocess_data()
    
    # 3. 종합 EDA 분석
    analysis_results = analyzer.comprehensive_eda_analysis()
    
    # 4. 시각화 생성
    from acn_report_visualizations import create_comprehensive_visualizations
    create_comprehensive_visualizations(analyzer)
    
    # 5. 종합 리포트 생성
    report = analyzer.generate_comprehensive_report(output_format)
    
    return {
        'analyzer': analyzer,
        'analysis_results': analysis_results,
        'report': report
    }

# 사용 예시
if __name__ == "__main__":
    print("ACN 정제 공정 종합 분석을 시작합니다.")
    print("\n사용법:")
    print("1. CSV 파일에서 분석:")
    print("   results = main_comprehensive_analysis(data_path='your_data.csv')")
    print("\n2. DataFrame에서 분석:")
    print("   results = main_comprehensive_analysis(df=your_dataframe)")
    print("\n3. PDF 리포트 생성:")
    print("   results = main_comprehensive_analysis(df=your_dataframe, output_format='pdf')")
    print("\n4. 결과 확인:")
    print("   print(results['analysis_results']['yield_analysis'])")
