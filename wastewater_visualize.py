import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wastewater_ai_solution import main

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def create_performance_plots(merged_data, control_results, performance):
    """성능 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. DO 추이 (첫 24시간)
    first_day = merged_data[merged_data['timestamp'] < merged_data['timestamp'].iloc[0] + pd.Timedelta(days=1)]
    axes[0,0].plot(first_day['timestamp'], first_day['do_sensor'], 'b-', label='실제 DO', alpha=0.7)
    axes[0,0].plot(first_day['timestamp'], first_day['target_do'], 'r--', label='목표 DO', linewidth=2)
    axes[0,0].set_title('DO 모니터링 (첫 24시간)')
    axes[0,0].set_ylabel('DO (mg/L)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. DO 오차 분포
    axes[0,1].hist(control_results['do_error'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0,1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='목표선')
    axes[0,1].set_title('DO 오차 분포')
    axes[0,1].set_xlabel('DO 오차 (mg/L)')
    axes[0,1].set_ylabel('빈도')
    axes[0,1].legend()
    
    # 3. 송풍량 백분율 분포
    percentage_counts = control_results['optimal_air_flow_percentage'].value_counts().sort_index()
    axes[1,0].bar(percentage_counts.index, percentage_counts.values, color='lightgreen', edgecolor='black')
    axes[1,0].set_title('송풍량 백분율 분포')
    axes[1,0].set_xlabel('송풍량 백분율 (%)')
    axes[1,0].set_ylabel('빈도')
    
    # 4. 제어 액션 분포
    action_counts = control_results['control_action'].value_counts()
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    axes[1,1].pie(action_counts.values, labels=action_counts.index, autopct='%1.1f%%', colors=colors)
    axes[1,1].set_title('제어 액션 분포')
    
    plt.tight_layout()
    plt.show()

def create_time_series_plots(merged_data, control_results):
    """시계열 시각화"""
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 1. DO 시계열
    axes[0].plot(control_results['timestamp'], control_results['current_do'], 'b-', label='현재 DO', alpha=0.7)
    axes[0].plot(control_results['timestamp'], control_results['target_do'], 'r--', label='목표 DO', linewidth=2)
    axes[0].set_title('DO 시계열')
    axes[0].set_ylabel('DO (mg/L)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. DO 오차 시계열
    axes[1].plot(control_results['timestamp'], control_results['do_error'], 'g-', alpha=0.7)
    axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title('DO 오차 시계열')
    axes[1].set_ylabel('DO 오차 (mg/L)')
    axes[1].grid(True, alpha=0.3)
    
    # 3. 송풍량 백분율 시계열
    axes[2].plot(control_results['timestamp'], control_results['optimal_air_flow_percentage'], 'o-', color='orange')
    axes[2].set_title('송풍량 백분율 시계열')
    axes[2].set_ylabel('송풍량 백분율 (%)')
    axes[2].set_ylim(90, 100)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_correlation_analysis(merged_data):
    """상관관계 분석"""
    # 주요 변수 선택
    correlation_vars = ['do_sensor', 'inflow_rate_sensor', 'temperature', 
                       'air_flow_rate', 'blower_suction', 'filter_pressure']
    
    correlation_data = merged_data[correlation_vars].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_data, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f')
    plt.title('변수 간 상관관계 분석')
    plt.tight_layout()
    plt.show()

def create_hourly_analysis(merged_data):
    """시간대별 분석"""
    # 시간대별 평균 DO
    hourly_do = merged_data.groupby('hour')['do_sensor'].agg(['mean', 'std']).reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.errorbar(hourly_do['hour'], hourly_do['mean'], yerr=hourly_do['std'], 
                fmt='o-', capsize=5, capthick=2)
    plt.title('시간대별 평균 DO')
    plt.xlabel('시간')
    plt.ylabel('평균 DO (mg/L)')
    plt.grid(True, alpha=0.3)
    plt.show()

def create_control_dashboard(merged_data, control_results, current_time_idx=1000):
    """실시간 제어 대시보드"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 현재 상태
    current_data = merged_data.iloc[current_time_idx]
    current_control = control_results.iloc[current_time_idx // 60] if current_time_idx // 60 < len(control_results) else control_results.iloc[-1]
    
    # 1. DO 상태
    ax1.plot(merged_data['timestamp'][:current_time_idx], merged_data['do_sensor'][:current_time_idx], 'b-', label='실제 DO', alpha=0.7)
    ax1.plot(merged_data['timestamp'][:current_time_idx], merged_data['target_do'][:current_time_idx], 'r--', label='목표 DO', linewidth=2)
    ax1.axhline(y=current_data['do_sensor'], color='green', linewidth=3, label=f'현재 DO: {current_data["do_sensor"]:.2f}')
    ax1.set_title('DO 모니터링')
    ax1.set_ylabel('DO (mg/L)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 유입량
    ax2.plot(merged_data['timestamp'][:current_time_idx], merged_data['inflow_rate_sensor'][:current_time_idx], 'g-', alpha=0.7)
    ax2.axhline(y=current_data['inflow_rate_sensor'], color='red', linewidth=3, label=f'현재: {current_data["inflow_rate_sensor"]:.0f}')
    ax2.set_title('유입량 모니터링')
    ax2.set_ylabel('유입량 (m³/day)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 송풍량 백분율
    control_times = control_results['timestamp'][:current_time_idx//60]
    control_percentages = control_results['optimal_air_flow_percentage'][:current_time_idx//60]
    ax3.plot(control_times, control_percentages, 'o-', color='orange')
    ax3.axhline(y=current_control['optimal_air_flow_percentage'], color='red', linewidth=3, 
                label=f'현재: {current_control["optimal_air_flow_percentage"]}%')
    ax3.set_title('송풍량 백분율 제어')
    ax3.set_ylabel('송풍량 백분율 (%)')
    ax3.set_ylim(90, 100)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 온도 및 필터 차압
    ax4_twin = ax4.twinx()
    ax4.plot(merged_data['timestamp'][:current_time_idx], merged_data['temperature'][:current_time_idx], 'r-', label='온도', alpha=0.7)
    ax4_twin.plot(merged_data['timestamp'][:current_time_idx], merged_data['filter_pressure'][:current_time_idx], 'b-', label='필터 차압', alpha=0.7)
    ax4.set_title('온도 및 필터 차압')
    ax4.set_ylabel('온도 (°C)', color='red')
    ax4_twin.set_ylabel('필터 차압 (bar)', color='blue')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 상태 요약 출력
    print("=== 현재 시스템 상태 ===")
    print(f"시간: {current_data['timestamp']}")
    print(f"현재 DO: {current_data['do_sensor']:.2f} mg/L")
    print(f"목표 DO: {current_data['target_do']:.2f} mg/L")
    print(f"DO 오차: {current_data['target_do'] - current_data['do_sensor']:.2f} mg/L")
    print(f"현재 송풍량 백분율: {current_control['optimal_air_flow_percentage']}%")
    print(f"제어 액션: {current_control['control_action']}")
    print(f"유입량: {current_data['inflow_rate_sensor']:.0f} m³/day")
    print(f"온도: {current_data['temperature']:.1f}°C")

def main_visualization():
    """메인 시각화 함수"""
    print("=== 폐수 정제 AI 솔루션 시각화 ===\n")
    
    # 메인 시스템 실행
    merged_data, control_results, performance = main()
    
    print("\n=== 시각화 생성 중... ===")
    
    # 1. 성능 시각화
    print("1. 성능 시각화 생성...")
    create_performance_plots(merged_data, control_results, performance)
    
    # 2. 시계열 시각화
    print("2. 시계열 시각화 생성...")
    create_time_series_plots(merged_data, control_results)
    
    # 3. 상관관계 분석
    print("3. 상관관계 분석 생성...")
    create_correlation_analysis(merged_data)
    
    # 4. 시간대별 분석
    print("4. 시간대별 분석 생성...")
    create_hourly_analysis(merged_data)
    
    # 5. 실시간 제어 대시보드
    print("5. 실시간 제어 대시보드 생성...")
    create_control_dashboard(merged_data, control_results, current_time_idx=2000)
    
    print("\n=== 시각화 완료 ===")
    print("모든 시각화가 성공적으로 생성되었습니다.")

if __name__ == "__main__":
    main_visualization()
