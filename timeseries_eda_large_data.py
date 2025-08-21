"""
대용량 시계열 데이터 EDA (Exploratory Data Analysis) 도구
메모리 효율적이고 성능 최적화된 버전
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
import umap
import os
import gc
import psutil
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class LargeTimeseriesEDA:
    """
    대용량 시계열 데이터 EDA 클래스
    메모리 효율적이고 성능 최적화된 버전
    """
    
    def __init__(self, data_path=None, df=None, chunk_size=10000, max_memory_gb=8):
        """
        초기화
        
        Args:
            data_path (str): 데이터 파일 경로
            df (pd.DataFrame): 직접 전달된 데이터프레임
            chunk_size (int): 청크 크기
            max_memory_gb (float): 최대 메모리 사용량 (GB)
        """
        self.chunk_size = chunk_size
        self.max_memory_gb = max_memory_gb
        self.results = {}
        
        if df is not None:
            self.df = df
            self.is_large_data = self._check_if_large_data()
        elif data_path is not None:
            self.df = self._load_data_optimized(data_path)
            self.is_large_data = self._check_if_large_data()
        else:
            raise ValueError("data_path 또는 df 중 하나를 제공해야 합니다.")
        
        self._setup_analysis()
    
    def _check_if_large_data(self):
        """대용량 데이터 여부 확인"""
        if self.df is None:
            return False
        
        # 메모리 사용량 계산 (GB)
        memory_gb = self.df.memory_usage(deep=True).sum() / (1024**3)
        total_elements = self.df.shape[0] * self.df.shape[1]
        
        # 대용량 데이터 기준: 1GB 이상 또는 1천만 개 이상의 요소
        return memory_gb > 1.0 or total_elements > 10_000_000
    
    def _load_data_optimized(self, data_path):
        """메모리 효율적인 데이터 로드"""
        print(f"데이터 로드 중: {data_path}")
        
        # 파일 크기 확인
        file_size_gb = os.path.getsize(data_path) / (1024**3)
        print(f"파일 크기: {file_size_gb:.2f} GB")
        
        if file_size_gb > self.max_memory_gb:
            print("대용량 파일 감지. 청크 단위로 로드합니다.")
            return self._load_large_file(data_path)
        else:
            return self._load_small_file(data_path)
    
    def _load_small_file(self, data_path):
        """작은 파일 로드"""
        if data_path.endswith('.csv'):
            return pd.read_csv(data_path)
        elif data_path.endswith('.parquet'):
            return pd.read_parquet(data_path)
        elif data_path.endswith('.xlsx'):
            return pd.read_excel(data_path)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다.")
    
    def _load_large_file(self, data_path):
        """대용량 파일 청크 단위 로드"""
        if data_path.endswith('.csv'):
            # 첫 번째 청크만 로드하여 구조 파악
            first_chunk = pd.read_csv(data_path, nrows=self.chunk_size)
            print(f"첫 번째 청크 로드 완료: {first_chunk.shape}")
            return first_chunk
        else:
            raise ValueError("대용량 파일은 현재 CSV 형식만 지원합니다.")
    
    def _setup_analysis(self):
        """분석 설정"""
        print("=== 대용량 시계열 데이터 EDA 시작 ===")
        print(f"데이터 크기: {self.df.shape}")
        print(f"메모리 사용량: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        print(f"대용량 데이터 여부: {self.is_large_data}")
        
        if self.is_large_data:
            print("대용량 데이터 최적화 모드로 실행됩니다.")
            self._optimize_data_types()
    
    def _optimize_data_types(self):
        """데이터 타입 최적화로 메모리 사용량 감소"""
        print("데이터 타입 최적화 중...")
        
        initial_memory = self.df.memory_usage(deep=True).sum() / 1024**2
        
        # 수치형 컬럼 최적화
        for col in self.df.select_dtypes(include=[np.number]).columns:
            col_min = self.df[col].min()
            col_max = self.df[col].max()
            
            # 정수형 최적화
            if self.df[col].dtype == 'int64':
                if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                    self.df[col] = self.df[col].astype(np.int8)
                elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                    self.df[col] = self.df[col].astype(np.int16)
                elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                    self.df[col] = self.df[col].astype(np.int32)
            
            # 실수형 최적화
            elif self.df[col].dtype == 'float64':
                if self.df[col].isnull().sum() == 0:  # 결측값이 없는 경우
                    self.df[col] = self.df[col].astype(np.float32)
        
        final_memory = self.df.memory_usage(deep=True).sum() / 1024**2
        print(f"메모리 최적화: {initial_memory:.2f} MB → {final_memory:.2f} MB ({(1-final_memory/initial_memory)*100:.1f}% 감소)")
    
    def preprocess_large_data(self, sample_ratio=0.1, resample_freq=None, drop_columns=None):
        """
        대용량 데이터 전처리
        
        Args:
            sample_ratio (float): 샘플링 비율 (0.1 = 10%)
            resample_freq (str): 리샘플링 주기 ('1H', '1D' 등)
            drop_columns (list): 제거할 컬럼 리스트
        """
        print("=== 대용량 데이터 전처리 ===")
        
        original_shape = self.df.shape
        print(f"원본 데이터 크기: {original_shape}")
        
        # 1. 불필요한 컬럼 제거
        if drop_columns:
            self.df = self.df.drop(columns=drop_columns)
            print(f"컬럼 제거 후: {self.df.shape}")
        
        # 2. 샘플링 (대용량 데이터인 경우)
        if self.is_large_data and sample_ratio < 1.0:
            print(f"데이터 샘플링: {sample_ratio*100:.1f}%")
            self.df = self.df.sample(frac=sample_ratio, random_state=42)
            print(f"샘플링 후: {self.df.shape}")
        
        # 3. 리샘플링 (시계열 데이터인 경우)
        if resample_freq and self.df.index.dtype == 'datetime64[ns]':
            print(f"시계열 리샘플링: {resample_freq}")
            self.df = self.df.resample(resample_freq).mean()
            print(f"리샘플링 후: {self.df.shape}")
        
        # 4. 메모리 최적화
        self._optimize_data_types()
        
        final_shape = self.df.shape
        print(f"전처리 완료: {original_shape} → {final_shape}")
        print(f"데이터 크기 감소: {(1-final_shape[0]*final_shape[1]/(original_shape[0]*original_shape[1]))*100:.1f}%")
    
    def analyze_first_timestep_optimized(self, save_plots=True, output_dir='eda_results'):
        """
        첫 번째 시점 최적화 분석
        """
        print("\n=== 첫 번째 시점 최적화 분석 ===")
        
        if save_plots:
            os.makedirs(output_dir, exist_ok=True)
        
        first_row = self.df.iloc[0]
        
        # 기본 통계 (메모리 효율적)
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
        
        # 시각화 (대용량 데이터는 샘플링)
        if self.is_large_data:
            self._plot_first_timestep_sampled(first_row, save_plots, output_dir)
        else:
            self._plot_first_timestep_full(first_row, save_plots, output_dir)
        
        self.results['first_timestep'] = stats
        return stats
    
    def _plot_first_timestep_sampled(self, first_row, save_plots, output_dir):
        """샘플링된 첫 번째 시점 시각화"""
        # 대용량 데이터는 샘플링하여 시각화
        sample_size = min(10000, len(first_row))
        sampled_data = first_row.sample(n=sample_size, random_state=42)
        
        plt.figure(figsize=(15, 5))
        
        # 히스토그램
        plt.subplot(1, 3, 1)
        plt.hist(sampled_data.dropna(), bins=50, alpha=0.7, color='skyblue')
        plt.title(f'첫 번째 시점 값 분포 (샘플: {sample_size:,}개)')
        plt.xlabel('값')
        plt.ylabel('빈도')
        
        # 박스플롯
        plt.subplot(1, 3, 2)
        plt.boxplot(sampled_data.dropna())
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
            plt.savefig(f'{output_dir}/first_timestep_analysis_sampled.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_first_timestep_full(self, first_row, save_plots, output_dir):
        """전체 첫 번째 시점 시각화"""
        plt.figure(figsize=(15, 5))
        
        # 히스토그램
        plt.subplot(1, 3, 1)
        plt.hist(first_row.dropna(), bins=50, alpha=0.7, color='skyblue')
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
    
    def analyze_dimensions_optimized(self, save_plots=True, output_dir='eda_results'):
        """
        차원 분석 최적화 버전
        """
        print("\n=== 차원 분석 (최적화) ===")
        
        if self.df.shape[1] <= 100:
            print("차원이 100 이하로 PCA 분석을 건너뜁니다.")
            return
        
        # 수치형 데이터만 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 10:
            print("수치형 변수가 10개 미만으로 PCA 분석을 건너뜁니다.")
            return
        
        # 대용량 데이터는 IncrementalPCA 사용
        if self.is_large_data:
            self._pca_incremental(numeric_cols, save_plots, output_dir)
        else:
            self._pca_standard(numeric_cols, save_plots, output_dir)
    
    def _pca_incremental(self, numeric_cols, save_plots, output_dir):
        """IncrementalPCA를 사용한 차원 분석"""
        print("IncrementalPCA 사용 (대용량 데이터)")
        
        # 데이터 전처리
        scaler = StandardScaler()
        
        # 청크 단위로 스케일링
        scaled_data = []
        for chunk in tqdm(self.df[numeric_cols].groupby(np.arange(len(self.df)) // self.chunk_size), 
                         desc="데이터 스케일링"):
            chunk_data = chunk[1]
            scaled_chunk = scaler.fit_transform(chunk_data)
            scaled_data.append(scaled_chunk)
        
        scaled_data = np.vstack(scaled_data)
        
        # IncrementalPCA
        ipca = IncrementalPCA(n_components=min(50, len(numeric_cols)))
        
        # 청크 단위로 학습
        for i in tqdm(range(0, len(scaled_data), self.chunk_size), desc="PCA 학습"):
            chunk = scaled_data[i:i+self.chunk_size]
            ipca.partial_fit(chunk)
        
        # 설명 분산 비율
        cumulative_variance = np.cumsum(ipca.explained_variance_ratio_)
        n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        
        print(f"95% 분산을 설명하는 주성분 수: {n_components_95}")
        print(f"전체 변수 대비: {n_components_95/len(numeric_cols)*100:.1f}%")
        
        # 시각화
        self._plot_pca_results(cumulative_variance, n_components_95, save_plots, output_dir, "incremental")
        
        self.results['dimension_analysis'] = {
            'n_components_95': n_components_95,
            'reduction_ratio': n_components_95/len(numeric_cols),
            'explained_variance': cumulative_variance,
            'method': 'IncrementalPCA'
        }
    
    def _pca_standard(self, numeric_cols, save_plots, output_dir):
        """표준 PCA를 사용한 차원 분석"""
        print("표준 PCA 사용")
        
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
        self._plot_pca_results(cumulative_variance, n_components_95, save_plots, output_dir, "standard")
        
        self.results['dimension_analysis'] = {
            'n_components_95': n_components_95,
            'reduction_ratio': n_components_95/len(numeric_cols),
            'explained_variance': cumulative_variance,
            'method': 'StandardPCA'
        }
    
    def _plot_pca_results(self, cumulative_variance, n_components_95, save_plots, output_dir, method):
        """PCA 결과 시각화"""
        plt.figure(figsize=(12, 4))
        
        # 누적 설명 분산
        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance)
        plt.axhline(y=0.95, color='r', linestyle='--', label='95% 설명 분산')
        plt.xlabel('주성분 수')
        plt.ylabel('누적 설명 분산 비율')
        plt.title(f'PCA 누적 설명 분산 ({method})')
        plt.legend()
        
        # 주성분별 설명 분산
        plt.subplot(1, 2, 2)
        explained_variance = np.diff(cumulative_variance, prepend=0)
        plt.bar(range(1, 21), explained_variance[:20])
        plt.xlabel('주성분')
        plt.ylabel('설명 분산 비율')
        plt.title(f'상위 20개 주성분 설명 분산 ({method})')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/dimension_analysis_{method}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_variable_characteristics_optimized(self):
        """변수 특성 분석 최적화 버전"""
        print("\n=== 변수별 특성 분석 (최적화) ===")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        first_numeric = self.df.iloc[0][numeric_cols]
        
        # 값 크기별 분류
        large_values = first_numeric[first_numeric > first_numeric.quantile(0.9)]
        small_values = first_numeric[first_numeric < first_numeric.quantile(0.1)]
        zero_values = first_numeric[first_numeric == 0]
        
        print(f"큰 값(상위 10%): {len(large_values)}개 변수")
        print(f"작은 값(하위 10%): {len(small_values)}개 변수")
        print(f"0값: {len(zero_values)}개 변수")
        
        # 상위/하위 값 변수명 (최대 10개만 표시)
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
    
    def analyze_initial_timeseries_optimized(self, n_periods=100, save_plots=True, output_dir='eda_results'):
        """
        초기 시계열 구간 분석 최적화 버전
        """
        print(f"\n=== 초기 시계열 구간 분석 (처음 {n_periods}개 시점) ===")
        
        # 대용량 데이터는 샘플링
        if self.is_large_data and n_periods > 1000:
            sample_periods = min(n_periods, 1000)
            print(f"대용량 데이터로 인해 {sample_periods}개 시점만 분석합니다.")
            initial_period = self.df.head(sample_periods)
        else:
            initial_period = self.df.head(n_periods)
        
        initial_stats = initial_period.describe()
        
        print(f"평균 범위: {initial_stats.loc['mean'].min():.4f} ~ {initial_stats.loc['mean'].max():.4f}")
        print(f"표준편차 범위: {initial_stats.loc['std'].min():.4f} ~ {initial_stats.loc['std'].max():.4f}")
        
        # 변동성 높은 변수들
        high_variance_vars = initial_stats.loc['std'].sort_values(ascending=False).head(10)
        print(f"\n변동성이 높은 상위 10개 변수:")
        for var, std_val in high_variance_vars.items():
            print(f"  {var}: {std_val:.4f}")
        
        # 시계열 시각화 (최대 5개 변수)
        self._plot_initial_timeseries(initial_period, high_variance_vars, save_plots, output_dir)
        
        self.results['initial_timeseries'] = {
            'high_variance_vars': high_variance_vars,
            'initial_stats': initial_stats
        }
    
    def _plot_initial_timeseries(self, initial_period, high_variance_vars, save_plots, output_dir):
        """초기 시계열 시각화"""
        n_vars = min(5, len(high_variance_vars))
        
        plt.figure(figsize=(15, 3*n_vars))
        for i, var in enumerate(high_variance_vars.head(n_vars).index):
            plt.subplot(n_vars, 1, i+1)
            
            # 대용량 데이터는 다운샘플링
            if len(initial_period) > 1000:
                plot_data = initial_period[var].iloc[::len(initial_period)//1000]
                plt.plot(plot_data.index, plot_data.values, alpha=0.7)
                plt.title(f'{var} - 초기 구간 시계열 (다운샘플링)')
            else:
                plt.plot(initial_period.index, initial_period[var])
                plt.title(f'{var} - 초기 구간 시계열')
            
            plt.ylabel('값')
            if i == n_vars - 1:
                plt.xlabel('시간')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/initial_timeseries.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_correlations_optimized(self, save_plots=True, output_dir='eda_results'):
        """상관관계 분석 최적화 버전"""
        print("\n=== 상관관계 분석 (최적화) ===")
        
        # 대용량 데이터는 샘플링하여 상관관계 계산
        if self.is_large_data:
            sample_size = min(10000, len(self.df))
            print(f"대용량 데이터로 인해 {sample_size:,}개 행을 샘플링하여 상관관계를 계산합니다.")
            sample_df = self.df.sample(n=sample_size, random_state=42)
            correlation_matrix = sample_df.corr()
        else:
            correlation_matrix = self.df.corr()
        
        # 첫 번째 시점과의 상관관계
        first_row_corr = correlation_matrix.iloc[0].abs().sort_values(ascending=False)
        high_corr_vars = first_row_corr[first_row_corr > 0.5].head(10)
        
        print(f"첫 번째 시점과 상관관계가 높은 변수들 (|r| > 0.5):")
        for var, corr in high_corr_vars.items():
            print(f"  {var}: {corr:.4f}")
        
        # 상관관계 히트맵 (최대 20개 변수)
        self._plot_correlation_heatmap(correlation_matrix, first_row_corr, save_plots, output_dir)
        
        self.results['correlations'] = {
            'high_corr_vars': high_corr_vars,
            'correlation_matrix': correlation_matrix
        }
    
    def _plot_correlation_heatmap(self, correlation_matrix, first_row_corr, save_plots, output_dir):
        """상관관계 히트맵 시각화"""
        n_vars = min(20, len(first_row_corr))
        top_vars = first_row_corr.head(n_vars).index
        corr_subset = correlation_matrix.loc[top_vars, top_vars]
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
        plt.title(f'첫 번째 시점과 상관관계 높은 변수들의 상관관계 히트맵 (상위 {n_vars}개)')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f'{output_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def cluster_variables_optimized(self, n_clusters=5, save_plots=True, output_dir='eda_results'):
        """
        변수 클러스터링 최적화 버전
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
        
        # 대용량 데이터는 MiniBatchKMeans 사용
        if self.is_large_data:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        else:
            from sklearn.cluster import KMeans
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
        self._plot_clustering_results(cluster_df, n_clusters, save_plots, output_dir)
        
        self.results['clustering'] = {
            'cluster_df': cluster_df,
            'n_clusters': n_clusters
        }
    
    def _plot_clustering_results(self, cluster_df, n_clusters, save_plots, output_dir):
        """클러스터링 결과 시각화"""
        plt.figure(figsize=(12, 5))
        
        # 엘보우 메소드 (최대 10개 클러스터)
        plt.subplot(1, 2, 1)
        max_k = min(10, len(cluster_df) // 10)
        inertias = []
        K_range = range(2, max_k + 1)
        
        for k in K_range:
            from sklearn.cluster import MiniBatchKMeans
            kmeans_temp = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=1000)
            kmeans_temp.fit(cluster_df[['value']].values)
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
    
    def generate_summary_report_optimized(self, output_dir='eda_results'):
        """종합 요약 보고서 생성 (최적화)"""
        print("\n=== 종합 요약 보고서 (최적화) ===")
        
        summary = {
            '데이터 크기': f"{self.df.shape[0]:,} 시점 × {self.df.shape[1]:,} 변수",
            '메모리 사용량': f"{self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
            '첫 번째 시점 평균': f"{self.df.iloc[0].mean():.4f}",
            '첫 번째 시점 표준편차': f"{self.df.iloc[0].std():.4f}",
            '첫 번째 시점 범위': f"{self.df.iloc[0].min():.4f} ~ {self.df.iloc[0].max():.4f}",
            '결측값 비율': f"{self.df.iloc[0].isnull().sum() / len(self.df.iloc[0]) * 100:.2f}%",
            '0값 비율': f"{(self.df.iloc[0] == 0).sum() / len(self.df.iloc[0]) * 100:.2f}%",
            '대용량 데이터 여부': str(self.is_large_data)
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
        with open(f'{output_dir}/summary_report_optimized.txt', 'w', encoding='utf-8') as f:
            f.write("대용량 시계열 데이터 EDA 종합 보고서\n")
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
        with open(f'{output_dir}/analysis_results_optimized.json', 'w', encoding='utf-8') as f:
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
    
    def run_full_analysis_optimized(self, output_dir='eda_results', save_plots=True):
        """
        전체 분석 실행 (최적화)
        """
        print("대용량 시계열 데이터 EDA 전체 분석을 시작합니다...")
        
        # 메모리 사용량 모니터링
        initial_memory = psutil.virtual_memory().percent
        print(f"초기 메모리 사용량: {initial_memory:.1f}%")
        
        # 1. 첫 번째 시점 분석
        self.analyze_first_timestep_optimized(save_plots, output_dir)
        
        # 2. 차원 분석
        self.analyze_dimensions_optimized(save_plots, output_dir)
        
        # 3. 변수 특성 분석
        self.analyze_variable_characteristics_optimized()
        
        # 4. 초기 시계열 분석
        self.analyze_initial_timeseries_optimized(save_plots=save_plots, output_dir=output_dir)
        
        # 5. 상관관계 분석
        self.analyze_correlations_optimized(save_plots, output_dir)
        
        # 6. 변수 클러스터링
        self.cluster_variables_optimized(save_plots=save_plots, output_dir=output_dir)
        
        # 7. 종합 보고서 생성
        self.generate_summary_report_optimized(output_dir)
        
        # 최종 메모리 사용량
        final_memory = psutil.virtual_memory().percent
        print(f"최종 메모리 사용량: {final_memory:.1f}%")
        print(f"메모리 증가량: {final_memory - initial_memory:.1f}%")
        
        print("\n전체 분석이 완료되었습니다!")
        
        # 메모리 정리
        gc.collect()


# 사용 예시 함수
def example_large_data_usage():
    """대용량 데이터 사용 예시"""
    
    print("대용량 시계열 데이터 EDA 도구 사용 예시:")
    print("1. eda = LargeTimeseriesEDA(data_path='large_data.csv')")
    print("2. eda.preprocess_large_data(sample_ratio=0.1, resample_freq='1H')")
    print("3. eda.run_full_analysis_optimized(output_dir='large_eda_results')")
    print("4. 개별 분석: eda.analyze_first_timestep_optimized() 등")


if __name__ == "__main__":
    example_large_data_usage()
