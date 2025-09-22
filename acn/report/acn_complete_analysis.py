"""
ACN 정제 공정 완전 분석 실행 파일
- 모든 분석을 통합하여 실행
- Input_source 증가 시 효율성 유지 방안 제공
- 품질값과 Output, Yield 간의 관계 분석
- 통합 HTML 리포트 생성
"""

import pandas as pd
import numpy as np
from acn_efficiency_optimization import main_efficiency_optimization
from acn_exp import main_advanced_experiments
from acn_integrated_report import create_integrated_report
import warnings
warnings.filterwarnings('ignore')

def run_complete_acn_analysis(data_path=None, df=None, output_file='acn_complete_analysis_report.html'):
    """
    ACN 정제 공정 완전 분석 실행
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    output_file: 출력 HTML 파일명
    
    Returns:
    results: 모든 분석 결과
    """
    print("=" * 100)
    print("ACN 정제 공정 완전 분석 시작")
    print("=" * 100)
    
    # 데이터 로드
    if df is not None:
        data = df.copy()
    elif data_path:
        data = pd.read_csv(data_path)
    else:
        raise ValueError("데이터 경로 또는 DataFrame을 제공해야 합니다.")
    
    print(f"데이터 로드 완료: {data.shape}")
    
    # 1. 효율성 최적화 분석 (새로운 접근법)
    print("\n" + "=" * 80)
    print("1. 효율성 최적화 분석 실행")
    print("=" * 80)
    
    efficiency_results = main_efficiency_optimization(df=data)
    
    # 2. 고급 실험 분석 (기존)
    print("\n" + "=" * 80)
    print("2. 고급 실험 분석 실행")
    print("=" * 80)
    
    advanced_exp_results = main_advanced_experiments(df=data)
    
    # 3. 통합 리포트 생성
    print("\n" + "=" * 80)
    print("3. 통합 HTML 리포트 생성")
    print("=" * 80)
    
    html_content = create_integrated_report(
        advanced_exp_results=advanced_exp_results,
        efficiency_optimization_results=efficiency_results,
        output_file=output_file
    )
    
    print(f"\n✅ 분석 완료!")
    print(f"📄 통합 리포트가 '{output_file}'로 저장되었습니다.")
    
    return {
        'efficiency_results': efficiency_results,
        'advanced_exp_results': advanced_exp_results,
        'html_content': html_content,
        'data': data
    }

def demonstrate_analysis_with_sample_data():
    """
    샘플 데이터로 분석 시연
    """
    print("=" * 100)
    print("샘플 데이터로 ACN 분석 시연")
    print("=" * 100)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 500
    
    # Input_source (주요 변수)
    input_source = np.random.normal(100, 15, n_samples)
    input_source = np.clip(input_source, 50, 150)
    
    # Control 변수들
    control_vars = {
        'Control_1': np.random.normal(25, 5, n_samples),
        'Control_2': np.random.normal(75, 10, n_samples),
        'Control_3': np.random.normal(50, 8, n_samples),
        'Control_4': np.random.normal(30, 6, n_samples),
        'Control_5': np.random.normal(60, 12, n_samples)
    }
    
    # Output (Input_source와 상관관계)
    output = 0.8 * input_source + 0.1 * control_vars['Control_1'] + 0.05 * control_vars['Control_2'] + np.random.normal(0, 5, n_samples)
    output = np.clip(output, 0, 200)
    
    # Yield (Output/Input_source)
    yield_val = output / input_source
    yield_val = np.clip(yield_val, 0, 2)
    
    # 품질값들 (Output과 음의 상관관계)
    quality_vars = {
        'AN-10_200nm': np.random.normal(0.5, 0.2, n_samples) - 0.1 * (output / 100),
        'AN-10_225nm': np.random.normal(0.3, 0.15, n_samples) - 0.05 * (output / 100),
        'AN-10_250nm': np.random.normal(0.2, 0.1, n_samples) - 0.03 * (output / 100),
        'AN-50_200nm': np.random.normal(0.4, 0.18, n_samples) - 0.08 * (output / 100),
        'AN-50_225nm': np.random.normal(0.25, 0.12, n_samples) - 0.04 * (output / 100),
        'AN-50_250nm': np.random.normal(0.15, 0.08, n_samples) - 0.02 * (output / 100)
    }
    
    # 품질값을 양수로 클리핑
    for key in quality_vars:
        quality_vars[key] = np.clip(quality_vars[key], 0, 2)
    
    # Final_FR Level (최종 분석용)
    final_fr = np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.2, 0.7])
    
    # DataFrame 생성
    sample_data = pd.DataFrame({
        'Input_source': input_source,
        'Output': output,
        'Yield': yield_val,
        'Final_FR': final_fr,
        **control_vars,
        **quality_vars
    })
    
    print(f"샘플 데이터 생성 완료: {sample_data.shape}")
    print("\n데이터 요약:")
    print(sample_data.describe().round(2))
    
    # 분석 실행
    results = run_complete_acn_analysis(
        df=sample_data, 
        output_file='acn_sample_analysis_report.html'
    )
    
    return results

def main():
    """
    메인 실행 함수
    """
    print("ACN 정제 공정 완전 분석 시스템")
    print("=" * 50)
    print("1. 샘플 데이터로 시연")
    print("2. 실제 데이터 파일로 분석")
    print("3. 종료")
    
    choice = input("\n선택하세요 (1-3): ").strip()
    
    if choice == '1':
        # 샘플 데이터로 시연
        results = demonstrate_analysis_with_sample_data()
        print("\n✅ 샘플 데이터 분석 완료!")
        print("📄 'acn_sample_analysis_report.html' 파일을 확인하세요.")
        
    elif choice == '2':
        # 실제 데이터 파일로 분석
        data_path = input("데이터 파일 경로를 입력하세요: ").strip()
        try:
            results = run_complete_acn_analysis(data_path=data_path)
            print("\n✅ 실제 데이터 분석 완료!")
            print("📄 'acn_complete_analysis_report.html' 파일을 확인하세요.")
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            
    elif choice == '3':
        print("프로그램을 종료합니다.")
        
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
