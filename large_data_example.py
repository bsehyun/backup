"""
대용량 시계열 데이터 (6,568,485 × 232) 처리 예시
메모리 효율적이고 성능 최적화된 분석 방법
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from timeseries_eda_large_data import LargeTimeseriesEDA
import psutil
import gc
import time
from datetime import datetime

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def analyze_large_timeseries_data(data_path, output_dir='large_data_results'):
    """
    대용량 시계열 데이터 분석
    
    Args:
        data_path (str): 데이터 파일 경로
        output_dir (str): 결과 저장 디렉토리
    """
    print("=== 대용량 시계열 데이터 분석 시작 ===")
    print(f"데이터 파일: {data_path}")
    print(f"결과 저장 디렉토리: {output_dir}")
    
    # 1. 시스템 정보 확인
    print("\n=== 시스템 정보 ===")
    memory_info = psutil.virtual_memory()
    print(f"총 메모리: {memory_info.total / (1024**3):.1f} GB")
    print(f"사용 가능한 메모리: {memory_info.available / (1024**3):.1f} GB")
    print(f"메모리 사용률: {memory_info.percent:.1f}%")
    
    # 2. 파일 크기 확인
    import os
    file_size_gb = os.path.getsize(data_path) / (1024**3)
    print(f"파일 크기: {file_size_gb:.2f} GB")
    
    # 3. 대용량 데이터 EDA 초기화
    print("\n=== EDA 도구 초기화 ===")
    start_time = time.time()
    
    # 메모리 제한 설정 (사용 가능한 메모리의 70%로 제한)
    max_memory_gb = memory_info.available / (1024**3) * 0.7
    
    eda = LargeTimeseriesEDA(
        data_path=data_path,
        chunk_size=50000,  # 청크 크기 증가
        max_memory_gb=max_memory_gb
    )
    
    init_time = time.time() - start_time
    print(f"초기화 완료 (소요시간: {init_time:.2f}초)")
    
    # 4. 데이터 전처리 (필수 단계)
    print("\n=== 데이터 전처리 ===")
    start_time = time.time()
    
    # 전처리 옵션 설정
    preprocess_options = {
        'sample_ratio': 0.05,  # 5% 샘플링 (약 328,424행)
        'resample_freq': '1H',  # 시간별 리샘플링
        'drop_columns': None  # 필요시 제거할 컬럼 지정
    }
    
    # 전처리 실행
    eda.preprocess_large_data(**preprocess_options)
    
    preprocess_time = time.time() - start_time
    print(f"전처리 완료 (소요시간: {preprocess_time:.2f}초)")
    
    # 5. 최적화된 분석 실행
    print("\n=== 최적화된 분석 실행 ===")
    start_time = time.time()
    
    eda.run_full_analysis_optimized(
        output_dir=output_dir,
        save_plots=True
    )
    
    analysis_time = time.time() - start_time
    print(f"분석 완료 (소요시간: {analysis_time:.2f}초)")
    
    # 6. 메모리 정리
    gc.collect()
    
    # 7. 최종 요약
    final_memory = psutil.virtual_memory().percent
    print(f"\n=== 분석 완료 ===")
    print(f"총 소요시간: {init_time + preprocess_time + analysis_time:.2f}초")
    print(f"최종 메모리 사용률: {final_memory:.1f}%")
    print(f"결과 저장 위치: {output_dir}")

def analyze_with_different_strategies(data_path):
    """
    다양한 전처리 전략으로 분석
    """
    print("=== 다양한 전처리 전략 분석 ===")
    
    strategies = [
        {
            'name': '높은 샘플링 (10%)',
            'sample_ratio': 0.1,
            'resample_freq': '30min',
            'description': '더 많은 데이터 사용, 더 정확한 분석'
        },
        {
            'name': '중간 샘플링 (5%)',
            'sample_ratio': 0.05,
            'resample_freq': '1H',
            'description': '균형잡힌 접근'
        },
        {
            'name': '낮은 샘플링 (1%)',
            'sample_ratio': 0.01,
            'resample_freq': '2H',
            'description': '빠른 분석, 메모리 효율적'
        }
    ]
    
    for i, strategy in enumerate(strategies):
        print(f"\n--- 전략 {i+1}: {strategy['name']} ---")
        print(f"설명: {strategy['description']}")
        
        # 메모리 사용량 확인
        memory_before = psutil.virtual_memory().percent
        
        # EDA 초기화
        eda = LargeTimeseriesEDA(
            data_path=data_path,
            chunk_size=50000,
            max_memory_gb=8
        )
        
        # 전처리
        eda.preprocess_large_data(
            sample_ratio=strategy['sample_ratio'],
            resample_freq=strategy['resample_freq']
        )
        
        # 첫 번째 시점 분석만 실행 (빠른 테스트)
        first_stats = eda.analyze_first_timestep_optimized(
            save_plots=False,
            output_dir=f'strategy_{i+1}_results'
        )
        
        # 메모리 사용량 확인
        memory_after = psutil.virtual_memory().percent
        memory_increase = memory_after - memory_before
        
        print(f"데이터 크기: {eda.df.shape}")
        print(f"메모리 증가량: {memory_increase:.1f}%")
        print(f"첫 번째 시점 평균: {first_stats['mean']:.4f}")
        
        # 메모리 정리
        del eda
        gc.collect()

def create_memory_efficient_sample(data_path, sample_size=100000):
    """
    메모리 효율적인 샘플 데이터 생성
    """
    print(f"=== 메모리 효율적인 샘플 생성 ({sample_size:,}행) ===")
    
    # 청크 단위로 읽어서 샘플링
    chunks = []
    total_rows = 0
    
    for chunk in pd.read_csv(data_path, chunksize=100000):
        # 각 청크에서 일정 비율 샘플링
        chunk_sample_size = int(sample_size * len(chunk) / 6568485)  # 전체 행 수 대비 비율
        if chunk_sample_size > 0:
            chunk_sample = chunk.sample(n=min(chunk_sample_size, len(chunk)), random_state=42)
            chunks.append(chunk_sample)
            total_rows += len(chunk_sample)
        
        if total_rows >= sample_size:
            break
    
    # 샘플 데이터 결합
    sample_df = pd.concat(chunks, ignore_index=True)
    sample_df = sample_df.head(sample_size)  # 정확한 크기로 조정
    
    print(f"샘플 데이터 생성 완료: {sample_df.shape}")
    print(f"메모리 사용량: {sample_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return sample_df

def analyze_column_importance(data_path, n_columns=50):
    """
    컬럼 중요도 분석을 통한 차원 축소
    """
    print(f"=== 컬럼 중요도 분석 (상위 {n_columns}개 컬럼 선택) ===")
    
    # 첫 번째 청크만 읽어서 컬럼 중요도 분석
    first_chunk = pd.read_csv(data_path, nrows=10000)
    
    # 각 컬럼의 변동성 계산
    column_variance = first_chunk.var().sort_values(ascending=False)
    
    # 상위 컬럼 선택
    important_columns = column_variance.head(n_columns).index.tolist()
    
    print(f"상위 {n_columns}개 중요 컬럼:")
    for i, col in enumerate(important_columns[:10]):  # 상위 10개만 출력
        print(f"  {i+1}. {col} (분산: {column_variance[col]:.4f})")
    
    return important_columns

def main():
    """메인 실행 함수"""
    print("대용량 시계열 데이터 (6,568,485 × 232) 분석 도구")
    print("=" * 60)
    
    # 데이터 파일 경로 (실제 파일 경로로 변경 필요)
    data_path = "your_large_timeseries_data.csv"  # 실제 파일 경로로 변경
    
    try:
        # 1. 기본 분석
        print("\n1. 기본 분석 실행")
        analyze_large_timeseries_data(data_path, 'basic_analysis')
        
        # 2. 다양한 전략 분석
        print("\n2. 다양한 전처리 전략 분석")
        analyze_with_different_strategies(data_path)
        
        # 3. 컬럼 중요도 분석
        print("\n3. 컬럼 중요도 분석")
        important_columns = analyze_column_importance(data_path, n_columns=50)
        
        # 4. 중요 컬럼만 사용한 분석
        print("\n4. 중요 컬럼만 사용한 분석")
        sample_df = create_memory_efficient_sample(data_path, sample_size=100000)
        
        # 중요 컬럼만 선택
        available_columns = [col for col in important_columns if col in sample_df.columns]
        sample_df_important = sample_df[available_columns]
        
        print(f"중요 컬럼만 선택된 데이터: {sample_df_important.shape}")
        
        # EDA 분석
        eda = LargeTimeseriesEDA(df=sample_df_important)
        eda.run_full_analysis_optimized(output_dir='important_columns_analysis')
        
        print("\n=== 모든 분석 완료 ===")
        print("결과 파일들을 확인하세요:")
        print("- basic_analysis/: 기본 분석 결과")
        print("- strategy_*_results/: 다양한 전략 분석 결과")
        print("- important_columns_analysis/: 중요 컬럼 분석 결과")
        
    except FileNotFoundError:
        print(f"오류: 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("실제 데이터 파일 경로로 변경해주세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc()

def example_with_synthetic_data():
    """합성 데이터로 예시 실행"""
    print("=== 합성 대용량 데이터로 예시 실행 ===")
    
    # 합성 데이터 생성 (6,568,485 × 232 크기 시뮬레이션)
    print("합성 데이터 생성 중...")
    
    # 실제로는 너무 크므로 작은 크기로 시뮬레이션
    n_rows = 100000  # 시뮬레이션용
    n_cols = 232
    
    # 시간 인덱스 생성
    dates = pd.date_range('2020-01-01', periods=n_rows, freq='1min')
    
    # 다양한 패턴의 데이터 생성
    data = {}
    for i in range(n_cols):
        if i < n_cols * 0.3:  # 30% - 트렌드 패턴
            trend = np.linspace(0, 10, n_rows)
            seasonal = 2 * np.sin(2 * np.pi * np.arange(n_rows) / 1440)  # 일일 계절성
            noise = np.random.normal(0, 0.1, n_rows)
            data[f'trend_var_{i}'] = trend + seasonal + noise
            
        elif i < n_cols * 0.6:  # 30% - 계절성 패턴
            seasonal = 3 * np.sin(2 * np.pi * np.arange(n_rows) / 10080)  # 주간 계절성
            noise = np.random.normal(0, 0.1, n_rows)
            data[f'seasonal_var_{i}'] = seasonal + noise
            
        elif i < n_cols * 0.8:  # 20% - 랜덤 워크
            random_walk = np.cumsum(np.random.normal(0, 0.01, n_rows))
            data[f'random_walk_var_{i}'] = random_walk
            
        else:  # 20% - 스파이크 패턴
            base = np.random.normal(0, 0.05, n_rows)
            spikes = np.random.choice([0, 1], n_rows, p=[0.99, 0.01])
            spike_values = np.random.normal(0.5, 0.1, n_rows)
            data[f'spike_var_{i}'] = base + spikes * spike_values
    
    # 데이터프레임 생성
    df = pd.DataFrame(data, index=dates)
    
    print(f"합성 데이터 생성 완료: {df.shape}")
    print(f"메모리 사용량: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # 파일로 저장
    output_file = 'synthetic_large_data.csv'
    df.to_csv(output_file)
    print(f"합성 데이터 저장: {output_file}")
    
    # 분석 실행
    print("\n합성 데이터로 분석 실행...")
    analyze_large_timeseries_data(output_file, 'synthetic_analysis')
    
    print("\n합성 데이터 분석 완료!")

if __name__ == "__main__":
    # 실제 데이터가 있는 경우
    # main()
    
    # 합성 데이터로 예시 실행
    example_with_synthetic_data()
