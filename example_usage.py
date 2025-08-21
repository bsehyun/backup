"""
시계열 데이터 EDA 사용 예시
대용량 고차원 시계열 데이터의 첫 번째 데이터 성질 분석 예시
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeseries_eda import TimeseriesEDA
from timeseries_eda_utils import *

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def example_1_basic_usage():
    """예시 1: 기본 사용법"""
    print("=== 예시 1: 기본 사용법 ===")
    
    # 1. 샘플 데이터 생성
    print("1. 샘플 데이터 생성 중...")
    sample_df = generate_sample_timeseries_data(n_timesteps=1000, n_features=200)
    
    # 2. TimeseriesEDA 클래스 사용
    print("2. TimeseriesEDA 클래스로 분석...")
    eda = TimeseriesEDA(df=sample_df)
    
    # 3. 전체 분석 실행
    print("3. 전체 분석 실행...")
    eda.run_full_analysis(output_dir='example_results_1', save_plots=True)
    
    print("기본 사용법 예시 완료!")

def example_2_step_by_step_analysis():
    """예시 2: 단계별 분석"""
    print("\n=== 예시 2: 단계별 분석 ===")
    
    # 1. 데이터 생성
    print("1. 데이터 생성...")
    sample_df = generate_sample_timeseries_data(n_timesteps=500, n_features=150)
    
    # 2. TimeseriesEDA 초기화
    print("2. EDA 도구 초기화...")
    eda = TimeseriesEDA(df=sample_df)
    
    # 3. 단계별 분석
    print("3. 첫 번째 시점 분석...")
    first_stats = eda.analyze_first_timestep(save_plots=True, output_dir='example_results_2')
    
    print("4. 차원 분석...")
    eda.analyze_dimensions(save_plots=True, output_dir='example_results_2')
    
    print("5. 변수 특성 분석...")
    eda.analyze_variable_characteristics()
    
    print("6. 초기 시계열 분석...")
    eda.analyze_initial_timeseries(n_periods=50, save_plots=True, output_dir='example_results_2')
    
    print("7. 상관관계 분석...")
    eda.analyze_correlations(save_plots=True, output_dir='example_results_2')
    
    print("8. 변수 클러스터링...")
    eda.cluster_variables(n_clusters=4, save_plots=True, output_dir='example_results_2')
    
    print("단계별 분석 예시 완료!")

def example_3_utility_functions():
    """예시 3: 유틸리티 함수들 사용"""
    print("\n=== 예시 3: 유틸리티 함수들 사용 ===")
    
    # 1. 데이터 생성
    print("1. 데이터 생성...")
    sample_df = generate_sample_timeseries_data(n_timesteps=800, n_features=100)
    
    # 2. 빠른 개요 확인
    print("2. 빠른 개요 확인...")
    overview = quick_data_overview(sample_df)
    
    # 3. 첫 번째 시점 상세 분석
    print("3. 첫 번째 시점 상세 분석...")
    first_stats = analyze_first_timestep_detailed(sample_df, save_plots=True, output_dir='example_results_3')
    
    # 4. 변수 그룹 분석
    print("4. 변수 그룹 분석...")
    groups = analyze_variable_groups(sample_df, n_groups=5)
    
    # 5. 이상치 탐지
    print("5. 이상치 탐지...")
    outliers_iqr = detect_outliers_in_first_timestep(sample_df, method='iqr')
    outliers_zscore = detect_outliers_in_first_timestep(sample_df, method='zscore', threshold=2.0)
    
    # 6. 시간적 패턴 분석
    print("6. 시간적 패턴 분석...")
    temporal_patterns = analyze_temporal_patterns(sample_df, n_periods=100)
    
    # 7. 종합 시각화
    print("7. 종합 시각화 생성...")
    create_summary_visualization(sample_df, output_dir='example_results_3')
    
    # 8. 결과 저장
    print("8. 결과 저장...")
    all_results = {
        'overview': overview,
        'first_timestep': first_stats,
        'variable_groups': groups,
        'outliers_iqr': outliers_iqr,
        'outliers_zscore': outliers_zscore,
        'temporal_patterns': temporal_patterns
    }
    export_analysis_results(all_results, output_dir='example_results_3')
    
    print("유틸리티 함수 사용 예시 완료!")

def example_4_real_data_simulation():
    """예시 4: 실제 데이터 시뮬레이션"""
    print("\n=== 예시 4: 실제 데이터 시뮬레이션 ===")
    
    # 더 현실적인 데이터 생성
    print("1. 현실적인 데이터 생성...")
    
    # 시간 인덱스 (1년간 시간별 데이터)
    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2023-12-31')
    dates = pd.date_range(start=start_date, end=end_date, freq='H')
    
    # 다양한 패턴의 변수들 생성
    data = {}
    
    # 센서 데이터 시뮬레이션
    for i in range(50):
        # 온도 센서 (계절성 + 일일 패턴)
        if i < 10:
            seasonal = 15 + 10 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 365))  # 연간 계절성
            daily = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)  # 일일 패턴
            noise = np.random.normal(0, 0.5, len(dates))
            data[f'temperature_sensor_{i}'] = seasonal + daily + noise
            
        # 습도 센서 (온도와 반대 패턴)
        elif i < 20:
            seasonal = 60 - 20 * np.sin(2 * np.pi * np.arange(len(dates)) / (24 * 365))
            daily = -5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
            noise = np.random.normal(0, 2, len(dates))
            data[f'humidity_sensor_{i}'] = seasonal + daily + noise
            
        # 압력 센서 (안정적 + 약간의 변동)
        elif i < 30:
            base = 1013.25  # 표준 대기압
            trend = 0.1 * np.arange(len(dates)) / len(dates)  # 약간의 상승 트렌드
            noise = np.random.normal(0, 1, len(dates))
            data[f'pressure_sensor_{i}'] = base + trend + noise
            
        # 전력 소비 (사용 패턴)
        elif i < 40:
            # 주간 패턴 (주말에 낮음)
            weekly = np.array([1, 1, 1, 1, 1, 0.7, 0.7] * (len(dates) // 7 + 1))[:len(dates)]
            # 일일 패턴 (새벽에 낮음)
            daily = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(len(dates)) / 24)
            base = 100 + 50 * weekly * daily
            noise = np.random.normal(0, 10, len(dates))
            data[f'power_consumption_{i}'] = base + noise
            
        # 네트워크 트래픽 (불규칙한 스파이크)
        else:
            base = np.random.poisson(50, len(dates))
            spikes = np.random.choice([0, 1], len(dates), p=[0.98, 0.02])
            spike_values = np.random.poisson(200, len(dates))
            data[f'network_traffic_{i}'] = base + spikes * spike_values
    
    # 추가 노이즈 변수들 (관련성 없는 변수들)
    for i in range(50, 100):
        if i < 75:
            # 랜덤 워크
            random_walk = np.cumsum(np.random.normal(0, 0.1, len(dates)))
            data[f'random_walk_{i}'] = random_walk
        else:
            # 순수 노이즈
            data[f'noise_{i}'] = np.random.normal(0, 1, len(dates))
    
    df = pd.DataFrame(data, index=dates)
    
    print(f"생성된 데이터: {df.shape[0]} 시점 × {df.shape[1]} 변수")
    
    # 2. EDA 분석
    print("2. EDA 분석 실행...")
    eda = TimeseriesEDA(df=df)
    eda.run_full_analysis(output_dir='example_results_4', save_plots=True)
    
    # 3. 추가 분석
    print("3. 추가 분석...")
    
    # 센서별 그룹 분석
    sensor_groups = {
        'temperature': [col for col in df.columns if 'temperature' in col],
        'humidity': [col for col in df.columns if 'humidity' in col],
        'pressure': [col for col in df.columns if 'pressure' in col],
        'power': [col for col in df.columns if 'power' in col],
        'network': [col for col in df.columns if 'network' in col]
    }
    
    print("\n센서별 첫 번째 시점 분석:")
    for group_name, group_cols in sensor_groups.items():
        if group_cols:
            group_data = df.iloc[0][group_cols]
            print(f"{group_name} 센서 ({len(group_cols)}개):")
            print(f"  평균: {group_data.mean():.4f}")
            print(f"  표준편차: {group_data.std():.4f}")
            print(f"  범위: {group_data.min():.4f} ~ {group_data.max():.4f}")
    
    print("실제 데이터 시뮬레이션 예시 완료!")

def example_5_custom_analysis():
    """예시 5: 커스텀 분석"""
    print("\n=== 예시 5: 커스텀 분석 ===")
    
    # 1. 데이터 생성
    print("1. 데이터 생성...")
    sample_df = generate_sample_timeseries_data(n_timesteps=600, n_features=80)
    
    # 2. 커스텀 분석 함수
    def custom_first_timestep_analysis(df):
        """첫 번째 시점 커스텀 분석"""
        first_row = df.iloc[0]
        
        # 값의 크기별 분류
        abs_values = first_row.abs()
        large_vars = abs_values[abs_values > abs_values.quantile(0.9)]
        small_vars = abs_values[abs_values < abs_values.quantile(0.1)]
        
        # 부호별 분류
        positive_vars = first_row[first_row > 0]
        negative_vars = first_row[first_row < 0]
        zero_vars = first_row[first_row == 0]
        
        print("=== 커스텀 첫 번째 시점 분석 ===")
        print(f"큰 값 변수 (상위 10%): {len(large_vars)}개")
        print(f"작은 값 변수 (하위 10%): {len(small_vars)}개")
        print(f"양수 변수: {len(positive_vars)}개")
        print(f"음수 변수: {len(negative_vars)}개")
        print(f"0값 변수: {len(zero_vars)}개")
        
        return {
            'large_vars': large_vars,
            'small_vars': small_vars,
            'positive_vars': positive_vars,
            'negative_vars': negative_vars,
            'zero_vars': zero_vars
        }
    
    # 3. 커스텀 시각화
    def custom_visualization(df, results):
        """커스텀 시각화"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 첫 번째 시점 값 분포
        axes[0, 0].hist(df.iloc[0].dropna(), bins=30, alpha=0.7, color='lightblue')
        axes[0, 0].set_title('첫 번째 시점 값 분포')
        axes[0, 0].set_xlabel('값')
        axes[0, 0].set_ylabel('빈도')
        
        # 부호별 분포
        sign_counts = [len(results['positive_vars']), len(results['negative_vars']), len(results['zero_vars'])]
        sign_labels = ['양수', '음수', '0']
        axes[0, 1].pie(sign_counts, labels=sign_labels, autopct='%1.1f%%')
        axes[0, 1].set_title('부호별 변수 분포')
        
        # 상위/하위 값 비교
        top_10 = df.iloc[0].sort_values(ascending=False).head(10)
        bottom_10 = df.iloc[0].sort_values(ascending=True).head(10)
        
        axes[1, 0].bar(range(10), top_10.values, color='red', alpha=0.7, label='상위 10개')
        axes[1, 0].bar(range(10), bottom_10.values, color='blue', alpha=0.7, label='하위 10개')
        axes[1, 0].set_title('상위/하위 10개 값 비교')
        axes[1, 0].set_xlabel('순위')
        axes[1, 0].set_ylabel('값')
        axes[1, 0].legend()
        
        # 시간별 변화 (처음 50개 시점)
        time_means = df.head(50).mean(axis=1)
        axes[1, 1].plot(df.index[:50], time_means, linewidth=2)
        axes[1, 1].set_title('시간별 전체 평균 변화')
        axes[1, 1].set_xlabel('시간')
        axes[1, 1].set_ylabel('평균값')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('example_results_5/custom_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. 분석 실행
    print("2. 커스텀 분석 실행...")
    custom_results = custom_first_timestep_analysis(sample_df)
    
    print("3. 커스텀 시각화 생성...")
    import os
    os.makedirs('example_results_5', exist_ok=True)
    custom_visualization(sample_df, custom_results)
    
    print("커스텀 분석 예시 완료!")

def main():
    """모든 예시 실행"""
    print("시계열 데이터 EDA 도구 사용 예시")
    print("=" * 50)
    
    try:
        # 예시 1: 기본 사용법
        example_1_basic_usage()
        
        # 예시 2: 단계별 분석
        example_2_step_by_step_analysis()
        
        # 예시 3: 유틸리티 함수들
        example_3_utility_functions()
        
        # 예시 4: 실제 데이터 시뮬레이션
        example_4_real_data_simulation()
        
        # 예시 5: 커스텀 분석
        example_5_custom_analysis()
        
        print("\n" + "=" * 50)
        print("모든 예시가 성공적으로 완료되었습니다!")
        print("결과 파일들은 각각의 example_results_* 디렉토리에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
