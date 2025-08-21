"""
시계열 데이터 EDA (Exploratory Data Analysis) 도구
대용량 고차원 시계열 데이터의 첫 번째 데이터 성질 분석을 위한 종합 도구
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap
import os

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class TimeseriesEDA:
    """
    시계열 데이터 EDA 클래스
    """
    
    def __init__(self, data_path=None, df=None):
        """
        초기화
        
        Args:
            data_path (str): 데이터 파일 경로
            df (pd.DataFrame): 직접 전달된 데이터프레임
        """
        if df is not None:
            self.df = df
        elif data_path is not None:
            self.df = self._load_data(data_path)
        else:
            raise ValueError("data_path 또는 df 중 하나를 제공해야 합니다.")
        
        self.results = {}
        self._setup_analysis()
    
    def _load_data(self, data_path):
        """데이터 로드"""
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.xlsx'):
            return pd.read_excel(data_path)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다.")
    
    def _setup_analysis(self):
        """분석 설정"""
        print("=== 시계열 데이터 EDA 시작 ===")
        print(f"데이터 크기: {self.df.shape}")
        print(f"메모리 사용량: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"차원 수: {self.df.shape[1]}")
        print(f"시계열 길이: {self.df.shape[0]}")
        print(f"데이터 타입 분포:\n{self.df.dtypes.value_counts()}")
    
    def analyze_first_timestep(self, save_plots=True, output_dir='eda_results'):
        """
        첫 번째 시점 데이터 분석
        
        Args:
            save_plots (bool): 플롯 저장 여부
            output_dir (str): 결과 저장 디렉토리
        """
        print("\n=== 첫 번째 시점 데이터 분석 ===")
        
        # 결과 디렉토리 생성
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        first_row = self.df.iloc[0]
        
        # 기본 통계
        stats = {
            'mean': first_row.mean(),
            'std': first_row.std(),
            'min': first_row.min(),
            'max': first_row.max(),
            'median': first_row.median(),
            'missing_count': first_row.isnull().sum(),
            'zero_count': (first_row == 0).sum()
        }
        
        print(f"첫 번째 시점: {self.df.index[0] if hasattr(self.df.index, 'name') else '인덱스 0'}")
        print(f"평균: {stats['mean']:.4f}")
        print(f"표준편차: {stats['std']:.4f}")
        print(f"최솟값: {stats['min']:.4f}")
        print(f"최댓값: {stats['max']:.4f}")
        print(f"중앙값: {stats['median']:.4f}")
        print(f"결측값: {stats['missing_count']}")
        print(f"0값: {stats['zero_count']}")
        
        # 시각화
        self._plot_first_timestep_distribution(first_row, save_plots, output_dir)
        
        self.results['first_timestep'] = stats
        return stats
    
    def _plot_first_timestep_distribution(self, first_row, save_plots, output_dir):
        """첫 번째 시점 분포 시각화"""
        plt.figure(figsize=(15, 5))
        
        # 히스토그램
        plt.subplot(1, 3, 1)
        plt.hist(first_row.dropna(), bins=30, alpha=0.7, color='skyblue')
        plt.title('첫 번째 시점 값 분포')
        plt.xlabel('값')
        plt.ylabel('빈도')
        
        # 박스플롯
        plt.subplot(1, 3, 2)
        plt.boxplot(first_row.dropna())
        plt.title('첫 번째 시점 박스플롯')
        plt.ylabel('값')
        
        # 상위 값들
        plt.subplot(1, 3, 3)
        sorted_values = first_row.sort_values(ascending=False)
        plt.bar(range(20), sorted_values.head(20))
        plt.title('첫 번째 시점 상위 20개 값')
        plt.xlabel('순위')
        plt.ylabel('값')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/first_timestep_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_dimensions(self, save_plots=True, output_dir='eda_results'):
        """
        차원 분석 (PCA 등)
        
        Args:
            save_plots (bool): 플롯 저장 여부
            output_dir (str): 결과 저장 디렉토리
        """
        print("\n=== 차원 분석 ===")
        
        if self.df.shape[1] <= 100:
            print("차원이 100 이하로 PCA 분석을 건너뜁니다.")
            return
        
        # 수치형 데이터만 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 10:
            print("수치형 변수가 10개 미만으로 PCA 분석을 건너뜁니다.")
            return
        
        # 데이터 전처리
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.df[numeric_cols])
        
        # PCA 분석
        pca = PCA()
        pca.fit(scaled_data)
        
        # 설명 분산 비율
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"95% 분산을 설명하는 주성분 수: {n_components_95}")
        print(f"전체 변수 대비: {n_components_95/len(numeric_cols)*100:.1f}%")
        
        # 시각화
        plt.figure(figsize=(12, 4))
        
        # 누적 설명 분산
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% 설명 분산')
        plt.xlabel('주성분 수')
        plt.ylabel('누적 설명 분산 비율')
        plt.title('PCA 누적 설명 분산')
        plt.legend()
        
        # 첫 번째 시점의 주성분 값
        first_scaled = scaler.transform(self.df.iloc[0:1][numeric_cols])
        first_pca = pca.transform(first_scaled)
        
        plt.subplot(1, 2, 2)
        plt.bar(range(1, 21), first_pca[0, :20])
        plt.xlabel('주성분')
        plt.ylabel('값')
        plt.title('첫 번째 시점의 상위 20개 주성분')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/dimension_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['dimension_analysis'] = {
            'n_components_95': n_components_95,
            'reduction_ratio': n_components_95/len(numeric_cols),
            'explained_variance': cumulative_variance
        }
    
    def analyze_variable_characteristics(self):
        """변수별 특성 분석"""
        print("\n=== 변수별 특성 분석 ===")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        first_numeric = self.df.iloc[0][numeric_cols]
        
        # 값 크기별 분류
        large_values = first_numeric[first_numeric > first_numeric.quantile(0.9)]
        small_values = first_numeric[first_numeric < first_numeric.quantile(0.1)]
        zero_values = first_numeric[first_numeric == 0]
        
        print(f"큰 값(상위 10%): {len(large_values)}개 변수")
        print(f"작은 값(하위 10%): {len(small_values)}개 변수")
        print(f"0값: {len(zero_values)}개 변수")
        
        # 상위/하위 값 변수명
        print(f"\n상위 10개 큰 값 변수:")
        for var, val in large_values.head(10).items():
            print(f"  {var}: {val:.4f}")
        
        print(f"\n하위 10개 작은 값 변수:")
        for var, val in small_values.head(10).items():
            print(f"  {var}: {val:.4f}")
        
        self.results['variable_characteristics'] = {
            'large_values': large_values,
            'small_values': small_values,
            'zero_values': zero_values
        }
    
    def analyze_initial_timeseries(self, n_periods=100, save_plots=True, output_dir='eda_results'):
        """
        초기 시계열 구간 분석
        
        Args:
            n_periods (int): 분석할 초기 구간 길이
            save_plots (bool): 플롯 저장 여부
            output_dir (str): 결과 저장 디렉토리
        """
        print(f"\n=== 초기 시계열 구간 분석 (처음 {n_periods}개 시점) ===")
        
        initial_period = self.df.head(n_periods)
        initial_stats = initial_period.describe()
        
        print(f"평균 범위: {initial_stats.loc['mean'].min():.4f} ~ {initial_stats.loc['mean'].max():.4f}")
        print(f"표준편차 범위: {initial_stats.loc['std'].min():.4f} ~ {initial_stats.loc['std'].max():.4f}")
        
        # 변동성 높은 변수들
        high_variance_vars = initial_stats.loc['std'].sort_values(ascending=False).head(10)
        print(f"\n변동성이 높은 상위 10개 변수:")
        for var, std_val in high_variance_vars.items():
            print(f"  {var}: {std_val:.4f}")
        
        # 시계열 시각화
        plt.figure(figsize=(15, 10))
        for i, var in enumerate(high_variance_vars.head(5).index):
            plt.subplot(5, 1, i+1)
            plt.plot(initial_period.index, initial_period[var])
            plt.title(f'{var} - 초기 구간 시계열')
            plt.ylabel('값')
            if i == 4:
                plt.xlabel('시간')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/initial_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['initial_timeseries'] = {
            'high_variance_vars': high_variance_vars,
            'initial_stats': initial_stats
        }
    
    def analyze_correlations(self, save_plots=True, output_dir='eda_results'):
        """상관관계 분석"""
        print("\n=== 상관관계 분석 ===")
        
        # 상관관계 계산
        correlation_matrix = self.df.corr()
        
        # 첫 번째 시점과의 상관관계
        first_row_corr = correlation_matrix.iloc[0].abs().sort_values(ascending=False)
        high_corr_vars = first_row_corr[first_row_corr > 0.5].head(10)
        
        print(f"첫 번째 시점과 상관관계가 높은 변수들 (|r| > 0.5):")
        for var, corr in high_corr_vars.items():
            print(f"  {var}: {corr:.4f}")
        
        # 상관관계 히트맵
        plt.figure(figsize=(12, 10))
        top_vars = first_row_corr.head(20).index
        corr_subset = correlation_matrix.loc[top_vars, top_vars]
        
        sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title('첫 번째 시점과 상관관계 높은 변수들의 상관관계 히트맵')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['correlations'] = {
            'high_corr_vars': high_corr_vars,
            'correlation_matrix': correlation_matrix
        }
    
    def cluster_variables(self, n_clusters=5, save_plots=True, output_dir='eda_results'):
        """
        변수 클러스터링
        
        Args:
            n_clusters (int): 클러스터 수
            save_plots (bool): 플롯 저장 여부
            output_dir (str): 결과 저장 디렉토리
        """
        print(f"\n=== 변수 클러스터링 (클러스터 수: {n_clusters}) ===")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < n_clusters * 2:
            print("변수 수가 너무 적어 클러스터링을 건너뜁니다.")
            return
        
        # 첫 번째 시점 데이터로 클러스터링
        first_numeric_data = self.df.iloc[0][numeric_cols].values.reshape(-1, 1)
        
        # 데이터 전처리
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(first_numeric_data)
        
        # 클러스터링
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(scaled_data)
        
        # 클러스터별 특성 분석
        cluster_df = pd.DataFrame({
            'variable': numeric_cols,
            'value': self.df.iloc[0][numeric_cols],
            'cluster': cluster_labels
        })
        
        print(f"클러스터별 특성:")
        for cluster in range(n_clusters):
            cluster_vars = cluster_df[cluster_df['cluster'] == cluster]
            print(f"\n클러스터 {cluster} ({len(cluster_vars)}개 변수):")
            print(f"  평균값: {cluster_vars['value'].mean():.4f}")
            print(f"  표준편차: {cluster_vars['value'].std():.4f}")
            print(f"  값 범위: {cluster_vars['value'].min():.4f} ~ {cluster_vars['value'].max():.4f}")
        
        # 시각화
        plt.figure(figsize=(12, 5))
        
        # 엘보우 메소드
        plt.subplot(1, 2, 1)
        inertias = []
        K_range = range(2, min(11, len(scaled_data)//10 + 1))
        
        for k in K_range:
            kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans_temp.fit(scaled_data)
            inertias.append(kmeans_temp.inertia_)
        
        plt.plot(K_range, inertias, 'bo-')
        plt.xlabel('클러스터 수')
        plt.ylabel('Inertia')
        plt.title('엘보우 메소드')
        
        # 클러스터별 분포
        plt.subplot(1, 2, 2)
        for cluster in range(n_clusters):
            cluster_data = cluster_df[cluster_df['cluster'] == cluster]['value']
            plt.hist(cluster_data, alpha=0.7, label=f'Cluster {cluster}', bins=20)
        
        plt.xlabel('첫 번째 시점 값')
        plt.ylabel('빈도')
        plt.title('클러스터별 값 분포')
        plt.legend()
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/variable_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self.results['clustering'] = {
            'cluster_df': cluster_df,
            'n_clusters': n_clusters
        }
    
    def generate_summary_report(self, output_dir='eda_results'):
        """종합 요약 보고서 생성"""
        print("\n=== 종합 요약 보고서 ===")
        
        summary = {
            '데이터 크기': f"{self.df.shape[0]} 시점 × {self.df.shape[1]} 변수",
            '첫 번째 시점 평균': f"{self.df.iloc[0].mean():.4f}",
            '첫 번째 시점 표준편차': f"{self.df.iloc[0].std():.4f}",
            '첫 번째 시점 범위': f"{self.df.iloc[0].min():.4f} ~ {self.df.iloc[0].max():.4f}",
            '결측값 비율': f"{self.df.iloc[0].isnull().sum() / len(self.df.iloc[0]) * 100:.2f}%",
            '0값 비율': f"{(self.df.iloc[0] == 0).sum() / len(self.df.iloc[0]) * 100:.2f}%",
            '극값 비율(상위 1%)': f"{(self.df.iloc[0] > self.df.iloc[0].quantile(0.99)).sum() / len(self.df.iloc[0]) * 100:.2f}%"
        }
        
        print("데이터 특성 요약:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        print(f"\n주요 인사이트:")
        print("1. 첫 번째 시점의 데이터 분포 특성")
        print("2. 변수들 간의 상관관계 패턴")
        print("3. 클러스터링을 통한 변수 그룹 특성")
        print("4. 시계열 변동성 패턴")
        print("5. 차원 축소를 통한 주요 패턴")
        
        # 결과를 파일로 저장
        os.makedirs(output_dir, exist_ok=True)
        
        # 요약 보고서 저장
        with open(f'{output_dir}/summary_report.txt', 'w', encoding='utf-8') as f:
            f.write("시계열 데이터 EDA 종합 보고서\n")
            f.write("=" * 50 + "\n\n")
            
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n주요 인사이트:\n")
            f.write("1. 첫 번째 시점의 데이터 분포 특성\n")
            f.write("2. 변수들 간의 상관관계 패턴\n")
            f.write("3. 클러스터링을 통한 변수 그룹 특성\n")
            f.write("4. 시계열 변동성 패턴\n")
            f.write("5. 차원 축소를 통한 주요 패턴\n")
        
        # 결과 딕셔너리 저장
        import json
        with open(f'{output_dir}/analysis_results.json', 'w', encoding='utf-8') as f:
            # JSON 직렬화 가능한 형태로 변환
            json_results = {}
            for key, value in self.results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: str(v) if not isinstance(v, (int, float, str, bool)) else v 
                                       for k, v in value.items()}
                else:
                    json_results[key] = str(value)
            
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n결과가 '{output_dir}' 디렉토리에 저장되었습니다.")
    
    def run_full_analysis(self, output_dir='eda_results', save_plots=True):
        """
        전체 분석 실행
        
        Args:
            output_dir (str): 결과 저장 디렉토리
            save_plots (bool): 플롯 저장 여부
        """
        print("시계열 데이터 EDA 전체 분석을 시작합니다...")
        
        # 1. 첫 번째 시점 분석
        self.analyze_first_timestep(save_plots, output_dir)
        
        # 2. 차원 분석
        self.analyze_dimensions(save_plots, output_dir)
        
        # 3. 변수 특성 분석
        self.analyze_variable_characteristics()
        
        # 4. 초기 시계열 분석
        self.analyze_initial_timeseries(save_plots=save_plots, output_dir=output_dir)
        
        # 5. 상관관계 분석
        self.analyze_correlations(save_plots, output_dir)
        
        # 6. 변수 클러스터링
        self.cluster_variables(save_plots=save_plots, output_dir=output_dir)
        
        # 7. 종합 보고서 생성
        self.generate_summary_report(output_dir)
        
        print("\n전체 분석이 완료되었습니다!")


# 사용 예시 함수
def example_usage():
    """사용 예시"""
    
    # 1. 데이터 로드
    # eda = TimeseriesEDA(data_path='your_data.csv')
    
    # 2. 또는 직접 데이터프레임 전달
    # eda = TimeseriesEDA(df=your_dataframe)
    
    # 3. 전체 분석 실행
    # eda.run_full_analysis(output_dir='my_eda_results')
    
    # 4. 개별 분석 실행
    # eda.analyze_first_timestep()
    # eda.analyze_dimensions()
    # eda.analyze_correlations()
    
    print("TimeseriesEDA 클래스 사용 예시:")
    print("1. eda = TimeseriesEDA(data_path='your_data.csv')")
    print("2. eda.run_full_analysis(output_dir='results')")
    print("3. 개별 분석: eda.analyze_first_timestep() 등")


if __name__ == "__main__":
    example_usage()
