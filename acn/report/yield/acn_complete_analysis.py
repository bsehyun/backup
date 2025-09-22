"""
ACN 정제 공정 완전 분석 실행 파일
- Input_source와 Product 제외하고 Yield 변수만 분석
- 다중공선성 무시하고 모든 분석 종합
- 그래프가 포함된 HTML 리포트 생성
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def run_complete_acn_analysis(data_path=None, df=None):
    """
    ACN 정제 공정 Yield 종합 분석 실행
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    
    Returns:
    results: 모든 분석 결과
    """
    print("=" * 100)
    print("ACN 정제 공정 Yield 종합 분석 시작")
    print("=" * 100)
    
    # 데이터 로드
    if df is not None:
        data = df.copy()
    elif data_path:
        data = pd.read_csv(data_path)
    else:
        raise ValueError("데이터 경로 또는 DataFrame을 제공해야 합니다.")
    
    print(f"데이터 로드 완료: {data.shape}")
    
    # Yield 종합 분석 실행
    from acn_yield_comprehensive_analysis import main_yield_comprehensive_analysis
    
    results = main_yield_comprehensive_analysis(df=data)
    
    print(f"\n✅ 분석 완료!")
    print("📄 HTML 리포트가 생성되었습니다.")
    
    return results

def demonstrate_analysis_with_sample_data():
    """
    샘플 데이터로 분석 시연
    """
    print("=" * 100)
    print("샘플 데이터로 ACN Yield 종합 분석 시연")
    print("=" * 100)
    
    # 샘플 데이터 생성
    np.random.seed(42)
    n_samples = 500
    
    # Input_source (제외될 변수)
    input_source = np.random.normal(100, 15, n_samples)
    input_source = np.clip(input_source, 50, 150)
    
    # Product (제외될 변수)
    product = 0.8 * input_source + np.random.normal(0, 5, n_samples)
    product = np.clip(product, 0, 200)
    
    # Control 변수들 (분석에 사용될 변수들)
    control_vars = {
        'Control_1': np.random.normal(25, 5, n_samples),
        'Control_2': np.random.normal(75, 10, n_samples),
        'Control_3': np.random.normal(50, 8, n_samples),
        'Control_4': np.random.normal(30, 6, n_samples),
        'Control_5': np.random.normal(60, 12, n_samples),
        'Control_6': np.random.normal(40, 7, n_samples),
        'Control_7': np.random.normal(80, 9, n_samples),
        'Control_8': np.random.normal(35, 5, n_samples)
    }
    
    # Yield (분석 대상 변수)
    yield_val = (0.3 * control_vars['Control_1'] + 
                0.2 * control_vars['Control_2'] + 
                0.15 * control_vars['Control_3'] + 
                0.1 * control_vars['Control_4'] + 
                0.05 * control_vars['Control_5'] + 
                np.random.normal(0, 0.1, n_samples))
    yield_val = np.clip(yield_val, 0, 2)
    
    # 품질값들 (분석에 사용될 변수들)
    quality_vars = {
        'AN-10_200nm': np.random.normal(0.5, 0.2, n_samples),
        'AN-10_225nm': np.random.normal(0.3, 0.15, n_samples),
        'AN-10_250nm': np.random.normal(0.2, 0.1, n_samples),
        'AN-50_200nm': np.random.normal(0.4, 0.18, n_samples),
        'AN-50_225nm': np.random.normal(0.25, 0.12, n_samples),
        'AN-50_250nm': np.random.normal(0.15, 0.08, n_samples)
    }
    
    # 품질값을 양수로 클리핑
    for key in quality_vars:
        quality_vars[key] = np.clip(quality_vars[key], 0, 2)
    
    # Final_FR Level (최종 분석용)
    final_fr = np.random.choice([1, 2, 3], n_samples, p=[0.1, 0.2, 0.7])
    
    # DataFrame 생성
    sample_data = pd.DataFrame({
        'Input_source': input_source,  # 제외될 변수
        'Product': product,            # 제외될 변수
        'Yield': yield_val,            # 분석 대상 변수
        'Final_FR': final_fr,
        **control_vars,                # 분석에 사용될 변수들
        **quality_vars                 # 분석에 사용될 변수들
    })
    
    print(f"샘플 데이터 생성 완료: {sample_data.shape}")
    print("\n데이터 요약:")
    print(sample_data.describe().round(2))
    
    # 제외될 변수들 확인
    print("\n제외될 변수들 (다중공선성 무시):")
    exclude_vars = ['Input_source', 'Product']
    for var in exclude_vars:
        print(f"  - {var}")
    
    print(f"\n분석 대상 변수: Yield")
    print(f"분석에 사용될 특성 수: {len(control_vars) + len(quality_vars)}개")
    
    # 분석 실행
    results = run_complete_acn_analysis(df=sample_data)
    
    return results

def main():
    """
    메인 실행 함수
    """
    print("ACN 정제 공정 Yield 종합 분석 시스템")
    print("=" * 50)
    print("1. 샘플 데이터로 시연")
    print("2. 실제 데이터 파일로 분석")
    print("3. 종료")
    
    choice = input("\n선택하세요 (1-3): ").strip()
    
    if choice == '1':
        # 샘플 데이터로 시연
        results = demonstrate_analysis_with_sample_data()
        print("\n✅ 샘플 데이터 분석 완료!")
        print("📄 HTML 리포트를 확인하세요.")
        
    elif choice == '2':
        # 실제 데이터 파일로 분석
        data_path = input("데이터 파일 경로를 입력하세요: ").strip()
        try:
            results = run_complete_acn_analysis(data_path=data_path)
            print("\n✅ 실제 데이터 분석 완료!")
            print("📄 HTML 리포트를 확인하세요.")
        except Exception as e:
            print(f"❌ 오류 발생: {str(e)}")
            
    elif choice == '3':
        print("프로그램을 종료합니다.")
        
    else:
        print("잘못된 선택입니다.")

if __name__ == "__main__":
    main()
