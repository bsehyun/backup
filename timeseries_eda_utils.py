"""
시계열 데이터 EDA 유틸리티 함수들
대용량 고차원 시계열 데이터 분석을 위한 보조 함수들
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

def generate_sample_timeseries_data(n_timesteps=1000, n_features=200, noise_level=0.1):
    """
    샘플 시계열 데이터 생성 (테스트용)
    
    Args:
        n_timesteps (int): 시계열 길이
        n_features (int): 변수 수
        noise_level (float): 노이즈 수준
    
    Returns:
        pd.DataFrame: 샘플 시계열 데이터
    """
    print(f"샘플 시계열 데이터 생성: {n_timesteps} 시점 × {n_features} 변수")
    
    # 시간 인덱스 생성
    start_date = datetime(2020, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(n_timesteps)]
    
    # 다양한 패턴의 시계열 생성
    data = {}
    
    for i in range(n_features):
        if i < n_features * 0.3:  # 30% - 트렌드 패턴
            trend = np.linspace(0, 10, n_timesteps)
            seasonal = 2 * np.sin(2 * np.pi * np.arange(n_timesteps) / 24)  # 일일 계절성
            noise = np.random.normal(0, noise_level, n_timesteps)
            data[f'trend_var_{i}'] = trend + seasonal + noise
            
        elif i < n_features * 0.6:  # 30% - 계절성 패턴
            seasonal = 3 * np.sin(2 * np.pi * np.arange(n_timesteps) / 168)  # 주간 계절성
            noise = np.random.normal(0, noise_level, n_timesteps)
            data[f'seasonal_var_{i}'] = seasonal + noise
            
        elif i < n_features * 0.8:  # 20% - 랜덤 워크
            random_walk = np.cumsum(np.random.normal(0, 0.1, n_timesteps))
            data[f'random_walk_var_{i}'] = random_walk
            
        else:  # 20% - 스파이크 패턴
            base = np.random.normal(0, 0.5, n_timesteps)
            spikes = np.random.choice([0, 1], n_timesteps, p=[0.95, 0.05])
            spike_values = np.random.normal(5, 1, n_timesteps)
            data[f'spike_var_{i}'] = base + spikes * spike_values
    
    df = pd.DataFrame(data, index=dates)
    return df

def quick_data_overview(df):
    """
    데이터 빠른 개요 확인
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
    
    Returns:
        dict: 데이터 개요 정보
    """
    overview = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'dtypes': df.dtypes.value_counts().to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'missing_percentage': df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100,
        'duplicate_rows': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'datetime_columns': len(df.select_dtypes(include=['datetime']).columns)
    }
    
    print("=== 데이터 개요 ===")
    print(f"크기: {overview['shape'][0]} 행 × {overview['shape'][1]} 열")
    print(f"메모리 사용량: {overview['memory_usage_mb']:.2f} MB")
    print(f"결측값: {overview['missing_values']} ({overview['missing_percentage']:.2f}%)")
    print(f"중복 행: {overview['duplicate_rows']}")
    print(f"수치형 변수: {overview['numeric_columns']}")
    print(f"범주형 변수: {overview['categorical_columns']}")
    print(f"날짜/시간 변수: {overview['datetime_columns']}")
    
    return overview

def analyze_first_timestep_detailed(df, save_plots=False, output_dir='.'):
    """
    첫 번째 시점 상세 분석
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        save_plots (bool): 플롯 저장 여부
        output_dir (str): 저장 디렉토리
    
    Returns:
        dict: 분석 결과
    """
    first_row = df.iloc[0]
    
    # 기본 통계
    stats_info = {
        'mean': first_row.mean(),
        'std': first_row.std(),
        'min': first_row.min(),
        'max': first_row.max(),
        'median': first_row.median(),
        'skewness': skew(first_row.dropna()),
        'kurtosis': kurtosis(first_row.dropna()),
        'missing_count': first_row.isnull().sum(),
        'zero_count': (first_row == 0).sum(),
        'negative_count': (first_row < 0).sum(),
        'positive_count': (first_row > 0).sum()
    }
    
    # 분위수 정보
    quantiles = first_row.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    stats_info.update({f'q{int(q*100)}': val for q, val in quantiles.items()})
    
    print("=== 첫 번째 시점 상세 분석 ===")
    print(f"평균: {stats_info['mean']:.4f}")
    print(f"표준편차: {stats_info['std']:.4f}")
    print(f"왜도: {stats_info['skewness']:.4f}")
    print(f"첨도: {stats_info['kurtosis']:.4f}")
    print(f"결측값: {stats_info['missing_count']}")
    print(f"0값: {stats_info['zero_count']}")
    print(f"음수: {stats_info['negative_count']}")
    print(f"양수: {stats_info['positive_count']}")
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 히스토그램
    axes[0, 0].hist(first_row.dropna(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('첫 번째 시점 값 분포')
    axes[0, 0].set_xlabel('값')
    axes[0, 0].set_ylabel('빈도')
    
    # 2. 박스플롯
    axes[0, 1].boxplot(first_row.dropna())
    axes[0, 1].set_title('첫 번째 시점 박스플롯')
    axes[0, 1].set_ylabel('값')
    
    # 3. Q-Q 플롯
    stats.probplot(first_row.dropna(), dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('정규성 검정 (Q-Q Plot)')
    
    # 4. 상위 값들
    sorted_values = first_row.sort_values(ascending=False)
    axes[1, 0].bar(range(20), sorted_values.head(20))
    axes[1, 0].set_title('상위 20개 값')
    axes[1, 0].set_xlabel('순위')
    axes[1, 0].set_ylabel('값')
    
    # 5. 하위 값들
    axes[1, 1].bar(range(20), sorted_values.tail(20))
    axes[1, 1].set_title('하위 20개 값')
    axes[1, 1].set_xlabel('순위')
    axes[1, 1].set_ylabel('값')
    
    # 6. 값 범위별 분포
    value_ranges = [
        (first_row < 0, '음수'),
        (first_row == 0, '0'),
        ((first_row > 0) & (first_row <= first_row.quantile(0.5)), '0~중앙값'),
        (first_row > first_row.quantile(0.5), '중앙값 이상')
    ]
    
    range_counts = [first_row[condition].count() for condition, _ in value_ranges]
    range_labels = [label for _, label in value_ranges]
    
    axes[1, 2].pie(range_counts, labels=range_labels, autopct='%1.1f%%')
    axes[1, 2].set_title('값 범위별 분포')
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig(f'{output_dir}/first_timestep_detailed.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return stats_info

def analyze_variable_groups(df, n_groups=5):
    """
    변수들을 값 크기별로 그룹화하여 분석
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        n_groups (int): 그룹 수
    
    Returns:
        dict: 그룹별 분석 결과
    """
    first_row = df.iloc[0]
    
    # 값 크기별로 그룹화
    sorted_indices = first_row.sort_values().index
    group_size = len(sorted_indices) // n_groups
    
    groups = {}
    for i in range(n_groups):
        start_idx = i * group_size
        end_idx = start_idx + group_size if i < n_groups - 1 else len(sorted_indices)
        
        group_vars = sorted_indices[start_idx:end_idx]
        group_values = first_row[group_vars]
        
        groups[f'group_{i+1}'] = {
            'variables': list(group_vars),
            'values': group_values,
            'count': len(group_vars),
            'mean': group_values.mean(),
            'std': group_values.std(),
            'min': group_values.min(),
            'max': group_values.max(),
            'range': group_values.max() - group_values.min()
        }
    
    print("=== 변수 그룹별 분석 ===")
    for group_name, group_info in groups.items():
        print(f"\n{group_name} ({group_info['count']}개 변수):")
        print(f"  평균: {group_info['mean']:.4f}")
        print(f"  표준편차: {group_info['std']:.4f}")
        print(f"  범위: {group_info['min']:.4f} ~ {group_info['max']:.4f}")
        print(f"  변수 예시: {group_info['variables'][:3]}...")
    
    return groups

def detect_outliers_in_first_timestep(df, method='iqr', threshold=1.5):
    """
    첫 번째 시점에서 이상치 탐지
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        method (str): 이상치 탐지 방법 ('iqr', 'zscore', 'isolation')
        threshold (float): 임계값
    
    Returns:
        dict: 이상치 정보
    """
    first_row = df.iloc[0].dropna()
    
    outliers = {}
    
    if method == 'iqr':
        # IQR 방법
        Q1 = first_row.quantile(0.25)
        Q3 = first_row.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        outliers['lower'] = first_row[first_row < lower_bound]
        outliers['upper'] = first_row[first_row > upper_bound]
        outliers['method'] = 'IQR'
        outliers['bounds'] = (lower_bound, upper_bound)
        
    elif method == 'zscore':
        # Z-score 방법
        z_scores = np.abs(stats.zscore(first_row))
        outliers['all'] = first_row[z_scores > threshold]
        outliers['method'] = 'Z-score'
        outliers['threshold'] = threshold
        
    print(f"=== 이상치 탐지 ({method.upper()}) ===")
    if method == 'iqr':
        print(f"하한: {outliers['bounds'][0]:.4f}, 상한: {outliers['bounds'][1]:.4f}")
        print(f"하한 이상치: {len(outliers['lower'])}개")
        print(f"상한 이상치: {len(outliers['upper'])}개")
    else:
        print(f"임계값: {threshold}")
        print(f"이상치: {len(outliers['all'])}개")
    
    return outliers

def analyze_temporal_patterns(df, n_periods=100):
    """
    초기 구간의 시간적 패턴 분석
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        n_periods (int): 분석할 초기 구간 길이
    
    Returns:
        dict: 시간적 패턴 분석 결과
    """
    initial_data = df.head(n_periods)
    
    # 각 변수의 시간적 특성 계산
    temporal_features = {}
    
    for col in initial_data.columns:
        series = initial_data[col].dropna()
        if len(series) < 2:
            continue
            
        # 기본 통계
        temporal_features[col] = {
            'mean': series.mean(),
            'std': series.std(),
            'trend': np.polyfit(range(len(series)), series, 1)[0],  # 선형 트렌드
            'autocorr_lag1': series.autocorr(lag=1),
            'autocorr_lag5': series.autocorr(lag=5),
            'range': series.max() - series.min(),
            'cv': series.std() / abs(series.mean()) if series.mean() != 0 else np.inf
        }
    
    # 트렌드별 변수 분류
    trend_vars = {col: info for col, info in temporal_features.items() 
                  if abs(info['trend']) > 0.01}
    
    # 변동성별 변수 분류
    high_cv_vars = {col: info for col, info in temporal_features.items() 
                    if info['cv'] > 1.0}
    
    # 자기상관별 변수 분류
    high_autocorr_vars = {col: info for col, info in temporal_features.items() 
                          if abs(info['autocorr_lag1']) > 0.5}
    
    print("=== 시간적 패턴 분석 ===")
    print(f"트렌드가 있는 변수: {len(trend_vars)}개")
    print(f"높은 변동성 변수 (CV > 1): {len(high_cv_vars)}개")
    print(f"높은 자기상관 변수 (|r| > 0.5): {len(high_autocorr_vars)}개")
    
    return {
        'temporal_features': temporal_features,
        'trend_vars': trend_vars,
        'high_cv_vars': high_cv_vars,
        'high_autocorr_vars': high_autocorr_vars
    }

def create_summary_visualization(df, output_dir='.'):
    """
    종합 시각화 생성
    
    Args:
        df (pd.DataFrame): 분석할 데이터프레임
        output_dir (str): 저장 디렉토리
    """
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 첫 번째 시점 분포
    plt.subplot(3, 3, 1)
    first_row = df.iloc[0]
    plt.hist(first_row.dropna(), bins=50, alpha=0.7, color='skyblue')
    plt.title('첫 번째 시점 분포')
    plt.xlabel('값')
    plt.ylabel('빈도')
    
    # 2. 변수별 표준편차
    plt.subplot(3, 3, 2)
    std_by_var = df.std().sort_values(ascending=False)
    plt.bar(range(20), std_by_var.head(20))
    plt.title('상위 20개 변동성 변수')
    plt.xlabel('변수 순위')
    plt.ylabel('표준편차')
    
    # 3. 결측값 분포
    plt.subplot(3, 3, 3)
    missing_by_var = df.isnull().sum().sort_values(ascending=False)
    plt.bar(range(20), missing_by_var.head(20))
    plt.title('상위 20개 결측값 변수')
    plt.xlabel('변수 순위')
    plt.ylabel('결측값 수')
    
    # 4. 상관관계 히트맵 (상위 변수들)
    plt.subplot(3, 3, 4)
    corr_matrix = df.corr()
    top_vars = std_by_var.head(10).index
    corr_subset = corr_matrix.loc[top_vars, top_vars]
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('상위 변동성 변수들의 상관관계')
    
    # 5. 시계열 예시 (상위 5개 변동성 변수)
    plt.subplot(3, 3, 5)
    for i, var in enumerate(std_by_var.head(5).index):
        plt.plot(df.index[:100], df[var][:100], label=var, alpha=0.7)
    plt.title('상위 5개 변동성 변수 시계열')
    plt.xlabel('시간')
    plt.ylabel('값')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. 값 범위별 분포
    plt.subplot(3, 3, 6)
    value_ranges = [
        (first_row < first_row.quantile(0.25), 'Q1 이하'),
        ((first_row >= first_row.quantile(0.25)) & (first_row < first_row.quantile(0.5)), 'Q1-Q2'),
        ((first_row >= first_row.quantile(0.5)) & (first_row < first_row.quantile(0.75)), 'Q2-Q3'),
        (first_row >= first_row.quantile(0.75), 'Q3 이상')
    ]
    
    range_counts = [first_row[condition].count() for condition, _ in value_ranges]
    range_labels = [label for _, label in value_ranges]
    
    plt.pie(range_counts, labels=range_labels, autopct='%1.1f%%')
    plt.title('첫 번째 시점 사분위수 분포')
    
    # 7. 변수별 평균 vs 표준편차
    plt.subplot(3, 3, 7)
    mean_by_var = df.mean()
    plt.scatter(mean_by_var, std_by_var, alpha=0.6)
    plt.xlabel('평균')
    plt.ylabel('표준편차')
    plt.title('변수별 평균 vs 표준편차')
    
    # 8. 시간별 평균 변화
    plt.subplot(3, 3, 8)
    time_means = df.mean(axis=1)
    plt.plot(df.index[:100], time_means[:100])
    plt.title('시간별 전체 평균 변화')
    plt.xlabel('시간')
    plt.ylabel('평균값')
    
    # 9. 데이터 타입별 분포
    plt.subplot(3, 3, 9)
    dtype_counts = df.dtypes.value_counts()
    plt.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    plt.title('데이터 타입별 분포')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comprehensive_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def export_analysis_results(results, output_dir='.'):
    """
    분석 결과를 파일로 내보내기
    
    Args:
        results (dict): 분석 결과
        output_dir (str): 저장 디렉토리
    """
    import json
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 형태로 저장
    with open(f'{output_dir}/analysis_results.json', 'w', encoding='utf-8') as f:
        # JSON 직렬화 가능한 형태로 변환
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (pd.Series, pd.DataFrame)):
                        json_results[key][k] = v.to_dict()
                    elif isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    else:
                        json_results[key][k] = v
            else:
                json_results[key] = str(value)
        
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    # 텍스트 요약 저장
    with open(f'{output_dir}/analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("시계열 데이터 EDA 분석 결과 요약\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}:\n")
            if isinstance(value, dict):
                for k, v in value.items():
                    f.write(f"  {k}: {v}\n")
            else:
                f.write(f"  {value}\n")
            f.write("\n")
    
    print(f"분석 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    sample_df = generate_sample_timeseries_data(n_timesteps=500, n_features=100)
    
    # 빠른 개요 확인
    overview = quick_data_overview(sample_df)
    
    # 첫 번째 시점 상세 분석
    first_stats = analyze_first_timestep_detailed(sample_df)
    
    # 변수 그룹 분석
    groups = analyze_variable_groups(sample_df)
    
    # 이상치 탐지
    outliers = detect_outliers_in_first_timestep(sample_df)
    
    # 시간적 패턴 분석
    temporal_patterns = analyze_temporal_patterns(sample_df)
    
    # 종합 시각화
    create_summary_visualization(sample_df)
    
    # 결과 저장
    all_results = {
        'overview': overview,
        'first_timestep': first_stats,
        'variable_groups': groups,
        'outliers': outliers,
        'temporal_patterns': temporal_patterns
    }
    
    export_analysis_results(all_results)
