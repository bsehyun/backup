# 추론 시스템 사용 예시
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# inference_notebook.py에서 함수들을 import
from inference_notebook import run_inference, load_models, predict

def create_sample_data():
    """
    테스트용 샘플 데이터를 생성합니다.
    """
    # 시간 인덱스 생성 (1분 간격, 2시간)
    start_time = datetime(2024, 1, 15, 12, 0, 0)
    time_index = pd.date_range(start=start_time, periods=120, freq='1min')
    
    # 샘플 피처 데이터 생성
    np.random.seed(42)  # 재현성을 위한 시드 설정
    
    data = {
        'feature1': np.random.normal(100, 10, 120) + np.sin(np.arange(120) * 0.1) * 20,
        'feature2': np.random.normal(50, 5, 120) + np.cos(np.arange(120) * 0.15) * 10,
        'feature3': np.random.normal(200, 15, 120) + np.random.walk(120) * 2,
        'feature4': np.random.normal(75, 8, 120) + np.sin(np.arange(120) * 0.2) * 15,
        'feature5': np.random.normal(150, 12, 120) + np.cos(np.arange(120) * 0.08) * 25
    }
    
    # 데이터프레임 생성
    df = pd.DataFrame(data, index=time_index)
    df.index.name = 'timestamp'
    
    return df

def create_sample_models():
    """
    테스트용 샘플 모델과 스케일러를 생성합니다.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import RobustScaler
    import joblib
    
    # models 디렉토리 생성
    os.makedirs('./models', exist_ok=True)
    
    # 샘플 모델 생성
    short_model = GradientBoostingRegressor(n_estimators=10, random_state=42)
    long_model = GradientBoostingRegressor(n_estimators=10, random_state=42)
    
    # 샘플 데이터로 모델 학습
    sample_data = create_sample_data()
    
    # 단기 모델용 피처 (feature1, feature2, feature3)
    short_features = sample_data[['feature1', 'feature2', 'feature3']].values[:-1]
    short_target = sample_data['feature1'].values[1:]  # 다음 시점의 feature1을 예측
    
    # 장기 모델용 피처 (feature1, feature2, feature4, feature5)
    long_features = sample_data[['feature1', 'feature2', 'feature4', 'feature5']].values[:-1]
    long_target = sample_data['feature1'].values[1:]  # 다음 시점의 feature1을 예측
    
    # 모델 학습
    short_model.fit(short_features, short_target)
    long_model.fit(long_features, long_target)
    
    # 스케일러 생성 및 학습
    short_scaler = RobustScaler()
    long_scaler = RobustScaler()
    
    short_scaler.fit(short_features)
    long_scaler.fit(long_features)
    
    # 모델과 스케일러 저장
    joblib.dump(short_model, './models/short_model.pkl')
    joblib.dump(long_model, './models/long_model.pkl')
    joblib.dump(short_scaler, './models/short_scaler.pkl')
    joblib.dump(long_scaler, './models/long_scaler.pkl')
    
    print("샘플 모델과 스케일러가 생성되었습니다.")

def test_inference_system():
    """
    추론 시스템을 테스트합니다.
    """
    # 샘플 데이터 생성 및 저장
    sample_data = create_sample_data()
    sample_data.to_csv('sample_data.csv')
    print("샘플 데이터가 생성되었습니다: sample_data.csv")
    
    # 샘플 모델 생성
    create_sample_models()
    
    # 추론 테스트
    print("\n=== 추론 시스템 테스트 ===")
    
    # 테스트 1: 현재 시간으로 추론
    print("\n1. 현재 시간으로 추론:")
    result1 = run_inference('sample_data.csv')
    print(f"결과: {result1}")
    
    # 테스트 2: 특정 시간으로 추론
    print("\n2. 특정 시간으로 추론:")
    target_time = "2024-01-15 14:30:00"
    result2 = run_inference('sample_data.csv', target_time)
    print(f"결과: {result2}")
    
    # 테스트 3: 결과 파일 확인
    print("\n3. 결과 파일 확인:")
    if os.path.exists('sample_data_results.csv'):
        results_df = pd.read_csv('sample_data_results.csv')
        print("생성된 결과:")
        print(results_df)
    else:
        print("결과 파일이 생성되지 않았습니다.")
    
    # 테스트 4: 두 번째 추론 (기존 결과에 추가)
    print("\n4. 두 번째 추론 (기존 결과에 추가):")
    result3 = run_inference('sample_data.csv', "2024-01-15 15:00:00")
    print(f"결과: {result3}")
    
    # 최종 결과 확인
    if os.path.exists('sample_data_results.csv'):
        final_results_df = pd.read_csv('sample_data_results.csv')
        print("\n최종 결과:")
        print(final_results_df)

def test_error_handling():
    """
    오류 처리 기능을 테스트합니다.
    """
    print("\n=== 오류 처리 테스트 ===")
    
    # 테스트 1: 존재하지 않는 파일
    print("\n1. 존재하지 않는 파일 테스트:")
    result = run_inference('nonexistent_file.csv')
    print(f"결과: {result}")
    
    # 테스트 2: 빈 데이터프레임
    print("\n2. 빈 데이터프레임 테스트:")
    empty_df = pd.DataFrame()
    empty_df.to_csv('empty_data.csv')
    result = run_inference('empty_data.csv')
    print(f"결과: {result}")
    
    # 테스트 3: 시간 인덱스가 없는 데이터
    print("\n3. 시간 인덱스가 없는 데이터 테스트:")
    no_time_df = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
    no_time_df.to_csv('no_time_data.csv')
    result = run_inference('no_time_data.csv')
    print(f"결과: {result}")
    
    # 테스트 4: 충분하지 않은 데이터 길이
    print("\n4. 충분하지 않은 데이터 길이 테스트:")
    short_df = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [6, 7, 8, 9, 10],
        'feature3': [11, 12, 13, 14, 15],
        'feature4': [16, 17, 18, 19, 20],
        'feature5': [21, 22, 23, 24, 25]
    }, index=pd.date_range('2024-01-15 12:00:00', periods=5, freq='1min'))
    short_df.index.name = 'timestamp'
    short_df.to_csv('short_data.csv')
    result = run_inference('short_data.csv')
    print(f"결과: {result}")

if __name__ == "__main__":
    # 전체 테스트 실행
    print("추론 시스템 테스트를 시작합니다...")
    
    # 기본 기능 테스트
    test_inference_system()
    
    # 오류 처리 테스트
    test_error_handling()
    
    print("\n테스트가 완료되었습니다.")
