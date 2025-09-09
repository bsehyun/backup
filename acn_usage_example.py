"""
ACN 수율 최적화 분석 - 사용 예시
이 파일은 제공된 분석 코드들의 실제 사용 방법을 보여줍니다.
"""

import pandas as pd
import numpy as np
from acn_yield_eda_analysis import ACNYieldAnalyzer
from acn_visualization_analysis import ACNVisualizationAnalyzer

def create_sample_data():
    """
    샘플 데이터 생성 함수
    실제 데이터가 없을 때 테스트용으로 사용
    """
    np.random.seed(42)
    n_samples = 200
    
    # 기본 데이터 생성
    data = {
        '폐기_촉매_재사용_횟수': np.random.randint(0, 10, n_samples),
        'batch_생산_날짜': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        '원료_투입_날짜': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        '원료_종류': np.random.choice(['A', 'B'], n_samples),
        '제품_품질값_a': np.random.normal(85, 5, n_samples),
        '제품_품질값_b': np.random.normal(90, 4, n_samples),
        '제품_품질값_c': np.random.normal(88, 6, n_samples),
        '제품_버블링_유무': np.random.choice(['Y', 'N'], n_samples),
        '냉동기_동시_가동_유무': np.random.choice(['Y', 'N'], n_samples),
        '원료_투입량': np.random.normal(1000, 50, n_samples),
        '스팀_압력': np.random.normal(15, 2, n_samples),
        '내온': np.random.normal(200, 10, n_samples),
        '내압': np.random.normal(5, 0.5, n_samples),
        '승온시간': np.random.normal(120, 15, n_samples),
        '안정화_시간': np.random.normal(60, 8, n_samples),
        '정제_시간': np.random.normal(180, 20, n_samples),
        'CWS': np.random.normal(25, 3, n_samples),
        'CWS_압': np.random.normal(3, 0.3, n_samples),
        'CWR_온도': np.random.normal(80, 5, n_samples),
        '현재_온도': np.random.normal(195, 8, n_samples),
        'product_level': np.random.normal(75, 5, n_samples),
        '폐기_잔량': np.random.normal(50, 10, n_samples),
        'condenser_온도': np.random.normal(45, 3, n_samples),
        '생산량': np.random.normal(800, 40, n_samples),
    }
    
    # 원료 품질값 생성 (제품 품질값과 상관관계 있도록)
    data['원료_품질값_a'] = data['제품_품질값_a'] + np.random.normal(0, 2, n_samples)
    data['원료_품질값_b'] = data['제품_품질값_b'] + np.random.normal(0, 2, n_samples)
    data['원료_품질값_c'] = data['제품_품질값_c'] + np.random.normal(0, 2, n_samples)
    
    # 수율 생성 (다양한 변수들의 조합으로)
    yield_base = (
        0.3 * (data['제품_품질값_a'] / 100) +
        0.2 * (data['제품_품질값_b'] / 100) +
        0.2 * (data['제품_품질값_c'] / 100) +
        0.1 * (1 - data['폐기_촉매_재사용_횟수'] / 10) +
        0.1 * (data['내온'] / 250) +
        0.1 * (data['내압'] / 10)
    )
    
    # 노이즈 추가
    data['수율'] = np.clip(yield_base + np.random.normal(0, 0.05, n_samples), 0, 1)
    
    return pd.DataFrame(data)

def main_analysis_example():
    """
    메인 분석 예시
    """
    print("=" * 80)
    print("ACN 수율 최적화 분석 - 사용 예시")
    print("=" * 80)
    
    # 1. 샘플 데이터 생성 (실제 데이터가 있다면 이 부분을 수정)
    print("1. 데이터 로드 중...")
    df = create_sample_data()
    print(f"데이터 크기: {df.shape}")
    print(f"컬럼 수: {len(df.columns)}")
    
    # 2. 기초 통계 분석
    print("\n2. 기초 통계 분석 시작...")
    analyzer = ACNYieldAnalyzer(df=df)
    results = analyzer.generate_summary_report()
    
    # 3. 시각화 분석
    print("\n3. 시각화 분석 시작...")
    visualizer = ACNVisualizationAnalyzer(df)
    visualizer.generate_comprehensive_visualization()
    
    # 4. 추가 분석 예시
    print("\n4. 추가 분석 예시...")
    additional_analysis_example(df)
    
    return results

def additional_analysis_example(df):
    """
    추가 분석 예시
    """
    print("\n" + "-" * 50)
    print("추가 분석 예시")
    print("-" * 50)
    
    # 1. 수율 구간별 분석
    print("1. 수율 구간별 분석")
    df['수율_구간'] = pd.cut(df['수율'], 
                           bins=[0, 0.7, 0.8, 0.9, 1.0], 
                           labels=['낮음', '보통', '높음', '매우높음'])
    
    yield_segment_analysis = df.groupby('수율_구간').agg({
        '수율': ['count', 'mean', 'std'],
        '내온': 'mean',
        '내압': 'mean',
        '승온시간': 'mean',
        '폐기_촉매_재사용_횟수': 'mean'
    }).round(3)
    
    print(yield_segment_analysis)
    
    # 2. 원료 종류별 상세 분석
    print("\n2. 원료 종류별 상세 분석")
    material_analysis = df.groupby('원료_종류').agg({
        '수율': ['count', 'mean', 'std'],
        '원료_품질값_a': 'mean',
        '원료_품질값_b': 'mean',
        '원료_품질값_c': 'mean',
        '생산량': 'mean'
    }).round(3)
    
    print(material_analysis)
    
    # 3. 촉매 재사용 횟수별 수율 분석
    print("\n3. 촉매 재사용 횟수별 수율 분석")
    catalyst_analysis = df.groupby('폐기_촉매_재사용_횟수')['수율'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(3)
    
    # 3개 이상 데이터가 있는 경우만 표시
    catalyst_analysis = catalyst_analysis[catalyst_analysis['count'] >= 3]
    print(catalyst_analysis)
    
    # 4. 공정 조건별 수율 분석
    print("\n4. 공정 조건별 수율 분석")
    
    # 온도 구간별 분석
    df['온도_구간'] = pd.cut(df['내온'], bins=5, labels=['매우낮음', '낮음', '보통', '높음', '매우높음'])
    temperature_analysis = df.groupby('온도_구간')['수율'].agg(['count', 'mean', 'std']).round(3)
    print("온도 구간별 수율:")
    print(temperature_analysis)
    
    # 압력 구간별 분석
    df['압력_구간'] = pd.cut(df['내압'], bins=5, labels=['매우낮음', '낮음', '보통', '높음', '매우높음'])
    pressure_analysis = df.groupby('압력_구간')['수율'].agg(['count', 'mean', 'std']).round(3)
    print("\n압력 구간별 수율:")
    print(pressure_analysis)

def real_data_usage_example():
    """
    실제 데이터 사용 예시
    """
    print("=" * 80)
    print("실제 데이터 사용 예시")
    print("=" * 80)
    
    print("실제 데이터를 사용하는 경우:")
    print("1. CSV 파일에서 로드:")
    print("   df = pd.read_csv('your_acn_data.csv')")
    print("   analyzer = ACNYieldAnalyzer(df=df)")
    print("   results = analyzer.generate_summary_report()")
    
    print("\n2. Excel 파일에서 로드:")
    print("   df = pd.read_excel('your_acn_data.xlsx')")
    print("   visualizer = ACNVisualizationAnalyzer(df)")
    print("   visualizer.generate_comprehensive_visualization()")
    
    print("\n3. 데이터베이스에서 로드:")
    print("   import sqlite3")
    print("   conn = sqlite3.connect('your_database.db')")
    print("   df = pd.read_sql_query('SELECT * FROM acn_batch_data', conn)")
    print("   conn.close()")
    
    print("\n4. 컬럼명이 다른 경우:")
    print("   # 컬럼명 매핑")
    print("   column_mapping = {")
    print("       'yield': '수율',")
    print("       'temperature': '내온',")
    print("       'pressure': '내압',")
    print("       # ... 기타 매핑")
    print("   }")
    print("   df = df.rename(columns=column_mapping)")

if __name__ == "__main__":
    # 샘플 데이터로 분석 실행
    results = main_analysis_example()
    
    # 실제 데이터 사용법 안내
    real_data_usage_example()
    
    print("\n" + "=" * 80)
    print("분석 완료!")
    print("=" * 80)
