import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def generate_simulation_data():
    """시뮬레이션 데이터 생성"""
    np.random.seed(42)
    
    # 날짜 범위 설정 (30일)
    dates = pd.date_range('2024-01-01', periods=30, freq='D')
    minutes = pd.date_range('2024-01-01', periods=30*24*60, freq='1min')
    
    # 실측 데이터 생성 (1일 1회)
    measured_data = pd.DataFrame({
        'date': dates,
        'inflow_rate': np.random.normal(1000, 100, 30),  # m³/day
        'do_measured': np.random.normal(2.5, 0.3, 30),   # mg/L
        'mlss': np.random.normal(3000, 200, 30),         # mg/L
        'cod_inflow': np.random.normal(400, 50, 30)      # mg/L
    })
    
    # 센서 데이터 생성 (1분 간격, 30일)
    sensor_data = pd.DataFrame({
        'timestamp': minutes,
        'do_sensor': np.random.normal(2.0, 0.5, len(minutes)),
        'inflow_rate_sensor': np.random.normal(1000, 150, len(minutes)),
        'blower_suction': np.random.normal(50, 5, len(minutes)),
        'temperature': np.random.normal(25, 3, len(minutes)),
        'filter_pressure': np.random.normal(0.5, 0.1, len(minutes)),
        'air_flow_rate': np.random.normal(200, 20, len(minutes))
    })
    
    return measured_data, sensor_data

def preprocess_data(measured_data, sensor_data):
    """데이터 전처리 및 융합"""
    # 실측 데이터에 날짜 정보 추가
    measured_data['date_only'] = measured_data['date'].dt.date
    sensor_data['date_only'] = sensor_data['timestamp'].dt.date
    
    # 실측 데이터를 센서 데이터에 병합
    merged_data = sensor_data.merge(measured_data, on='date_only', how='left')
    
    # 결측값 처리 (전날 데이터로 채우기)
    merged_data = merged_data.fillna(method='ffill')
    
    # 추가 특성 생성
    merged_data['hour'] = merged_data['timestamp'].dt.hour
    merged_data['day_of_week'] = merged_data['timestamp'].dt.dayofweek
    merged_data['do_diff'] = merged_data['do_measured'] - merged_data['do_sensor']
    merged_data['inflow_diff'] = merged_data['inflow_rate'] - merged_data['inflow_rate_sensor']
    
    return merged_data

def calculate_target_do(inflow_rate, cod_inflow, mlss, temperature=25):
    """목표 DO 계산"""
    # 기본 목표 DO (mg/L)
    base_target = 2.0
    
    # 유입량 영향 (정규화된 값)
    inflow_factor = (inflow_rate - 800) / 400  # 800-1200 범위를 0-1로 정규화
    
    # COD 영향
    cod_factor = (cod_inflow - 300) / 200  # 300-500 범위를 0-1로 정규화
    
    # MLSS 영향
    mlss_factor = (mlss - 2500) / 1000  # 2500-3500 범위를 0-1로 정규화
    
    # 온도 영향 (온도가 높을수록 DO 용해도 감소)
    temp_factor = max(0, (30 - temperature) / 10)  # 20-30도 범위
    
    # 목표 DO 계산
    target_do = base_target + \
                0.5 * inflow_factor + \
                0.3 * cod_factor + \
                0.2 * mlss_factor + \
                0.1 * temp_factor
    
    return max(1.5, min(4.0, target_do))  # 1.5-4.0 범위로 제한

def optimize_air_flow_rate(current_do, target_do, current_air_flow, inflow_rate, cod_inflow):
    """송풍량 최적화"""
    # DO 차이
    do_diff = target_do - current_do
    
    # 기본 조정 비율
    base_adjustment = do_diff * 0.1  # DO 차이 1mg/L당 10% 조정
    
    # 유입량 영향
    inflow_factor = (inflow_rate - 800) / 400  # 정규화
    
    # COD 영향
    cod_factor = (cod_inflow - 300) / 200  # 정규화
    
    # 총 조정 비율
    total_adjustment = base_adjustment + 0.05 * inflow_factor + 0.03 * cod_factor
    
    # 현재 송풍량 백분율 추정 (실제로는 센서에서 측정)
    current_percentage = (current_air_flow / 250) * 100  # 250 m³/min을 100%로 가정
    
    # 새로운 백분율 계산
    new_percentage = current_percentage + total_adjustment * 100
    
    # 송풍량 백분율 옵션으로 제한 (94%, 96%, 98%)
    options = [94, 96, 98]
    optimal_percentage = min(options, key=lambda x: abs(x - new_percentage))
    
    return optimal_percentage

def simulate_control_system(data, control_interval=60):
    """실시간 제어 시스템 시뮬레이션"""
    control_results = []
    current_air_flow_percentage = 96  # 초기값
    
    for i in range(0, len(data), control_interval):
        current_data = data.iloc[i]
        
        # 현재 상태
        current_do = current_data['do_sensor']
        target_do = current_data['target_do']
        
        # 송풍량 최적화
        optimal_percentage = optimize_air_flow_rate(
            current_do, target_do, current_data['air_flow_rate'],
            current_data['inflow_rate'], current_data['cod_inflow']
        )
        
        # 제어 결과 기록
        control_results.append({
            'timestamp': current_data['timestamp'],
            'current_do': current_do,
            'target_do': target_do,
            'do_error': target_do - current_do,
            'current_air_flow_percentage': current_air_flow_percentage,
            'optimal_air_flow_percentage': optimal_percentage,
            'control_action': 'increase' if optimal_percentage > current_air_flow_percentage else 'decrease' if optimal_percentage < current_air_flow_percentage else 'maintain'
        })
        
        # 다음 제어를 위해 현재 값 업데이트
        current_air_flow_percentage = optimal_percentage
    
    return pd.DataFrame(control_results)

def evaluate_control_performance(control_results):
    """제어 성능 평가"""
    # DO 오차 통계
    do_error = control_results['do_error']
    
    # 성능 지표
    performance = {
        'mean_do_error': do_error.mean(),
        'std_do_error': do_error.std(),
        'max_do_error': do_error.abs().max(),
        'do_error_within_0.5': (do_error.abs() <= 0.5).mean() * 100,
        'do_error_within_1.0': (do_error.abs() <= 1.0).mean() * 100,
        'control_stability': (control_results['control_action'] == 'maintain').mean() * 100
    }
    
    return performance

def main():
    """메인 실행 함수"""
    print("=== 폐수 정제 AI 솔루션 ===\n")
    
    # 1. 데이터 생성
    print("1. 시뮬레이션 데이터 생성 중...")
    measured_data, sensor_data = generate_simulation_data()
    print(f"   실측 데이터: {measured_data.shape}")
    print(f"   센서 데이터: {sensor_data.shape}")
    
    # 2. 데이터 전처리
    print("\n2. 데이터 전처리 중...")
    merged_data = preprocess_data(measured_data, sensor_data)
    print(f"   병합된 데이터: {merged_data.shape}")
    
    # 3. 목표 DO 계산
    print("\n3. 목표 DO 계산 중...")
    merged_data['target_do'] = calculate_target_do(
        merged_data['inflow_rate'],
        merged_data['cod_inflow'],
        merged_data['mlss'],
        merged_data['temperature']
    )
    print(f"   목표 DO 평균: {merged_data['target_do'].mean():.2f} mg/L")
    
    # 4. 제어 시스템 시뮬레이션
    print("\n4. 제어 시스템 시뮬레이션 중...")
    control_results = simulate_control_system(merged_data)
    print(f"   총 제어 횟수: {len(control_results)}")
    
    # 5. 성능 평가
    print("\n5. 성능 평가 중...")
    performance = evaluate_control_performance(control_results)
    
    print("\n=== 성능 결과 ===")
    print(f"평균 DO 오차: {performance['mean_do_error']:.3f} mg/L")
    print(f"DO 오차 표준편차: {performance['std_do_error']:.3f} mg/L")
    print(f"최대 DO 오차: {performance['max_do_error']:.3f} mg/L")
    print(f"DO 오차 ±0.5mg/L 이내: {performance['do_error_within_0.5']:.1f}%")
    print(f"DO 오차 ±1.0mg/L 이내: {performance['do_error_within_1.0']:.1f}%")
    print(f"제어 안정성: {performance['control_stability']:.1f}%")
    
    # 6. 송풍량 백분율 분포
    print("\n=== 송풍량 백분율 분포 ===")
    percentage_counts = control_results['optimal_air_flow_percentage'].value_counts().sort_index()
    for percentage, count in percentage_counts.items():
        print(f"  {percentage}%: {count}회 ({count/len(control_results)*100:.1f}%)")
    
    # 7. 제어 액션 분포
    print("\n=== 제어 액션 분포 ===")
    action_counts = control_results['control_action'].value_counts()
    for action, count in action_counts.items():
        print(f"  {action}: {count}회 ({count/len(control_results)*100:.1f}%)")
    
    print("\n=== 시스템 요약 ===")
    print("✓ 실시간 DO 모니터링 및 제어")
    print("✓ 유입량 및 COD 기반 적응형 제어")
    print("✓ 3단계 송풍량 제어 (94%, 96%, 98%)")
    print("✓ 60분 간격 제어 업데이트")
    print("✓ 성능 모니터링 및 최적화")
    
    return merged_data, control_results, performance

if __name__ == "__main__":
    merged_data, control_results, performance = main()
