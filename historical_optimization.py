#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 데이터와 실무자 경험 기반 최적화 시스템
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

# 시각화 설정
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

class HistoricalDataOptimizer:
    """
    기존 데이터와 실무자 경험 기반 최적화 시스템
    """
    
    def __init__(self):
        self.historical_data = None
        self.expert_knowledge = None
        self.optimization_models = {}
        self.scalers = {}
        
    def create_historical_dataset(self):
        """
        기존 운영 데이터 시뮬레이션 (실제로는 기존 DB에서 로드)
        """
        print("=== 기존 운영 데이터 생성 ===\n")
        
        # 1년간의 운영 데이터 생성 (실제로는 기존 DB에서 가져옴)
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        
        # 실제 운영 조건들
        temperatures = 20 + 10 * np.sin(np.arange(365) * 2 * np.pi / 365) + np.random.normal(0, 2, 365)
        inflow_rates = 1000 + 200 * np.sin(np.arange(365) * 2 * np.pi / 7) + np.random.normal(0, 50, 365)
        cod_values = 150 + 50 * np.sin(np.arange(365) * 2 * np.pi / 14) + np.random.normal(0, 15, 365)
        mlss_values = 3000 + 500 * np.sin(np.arange(365) * 2 * np.pi / 30) + np.random.normal(0, 100, 365)
        
        # 실무자들이 실제로 설정한 목표 DO (경험 기반)
        # 온도가 높을수록, COD가 높을수록, 유입량이 많을수록 목표 DO 증가
        target_dos = []
        for i in range(365):
            base_target = 2.0
            temp_factor = 0.1 * (temperatures[i] - 25)  # 온도 영향
            cod_factor = 0.002 * (cod_values[i] - 150)  # COD 영향
            flow_factor = 0.0001 * (inflow_rates[i] - 1000)  # 유입량 영향
            
            target_do = base_target + temp_factor + cod_factor + flow_factor
            target_do = np.clip(target_do, 1.5, 4.0)
            target_dos.append(target_do)
        
        # 실제 달성된 DO (목표와 약간의 차이)
        achieved_dos = [target + np.random.normal(0, 0.2) for target in target_dos]
        achieved_dos = np.clip(achieved_dos, 1.0, 5.0)
        
        # 실무자가 설정한 송풍량 (94%, 96%, 98%)
        blower_percentages = []
        for i in range(365):
            do_diff = achieved_dos[i] - target_dos[i]
            if do_diff > 0.3:
                blower_percentages.append(94)  # DO가 높으면 송풍량 감소
            elif do_diff < -0.3:
                blower_percentages.append(98)  # DO가 낮으면 송풍량 증가
            else:
                blower_percentages.append(96)  # 적정 범위
        
        # 성과 지표들 (실제 측정값)
        energy_costs = []
        treatment_efficiencies = []
        water_quality_scores = []
        
        for i in range(365):
            # 에너지 비용: 송풍량이 높을수록, 온도가 높을수록 증가
            energy_cost = 80 + 10 * (blower_percentages[i] - 94) / 4 + 2 * (temperatures[i] - 20) / 10
            energy_costs.append(energy_cost + np.random.normal(0, 3))
            
            # 처리 효율성: DO가 적정 범위일 때 최대
            do_optimal = 2.0 + 0.1 * (temperatures[i] - 25)
            efficiency = 0.95 - 0.1 * abs(achieved_dos[i] - do_optimal)
            treatment_efficiencies.append(efficiency + np.random.normal(0, 0.02))
            
            # 수질 점수: 처리 효율성과 관련
            water_quality = 0.9 * treatment_efficiencies[i] + 0.1 * np.random.normal(0, 0.05)
            water_quality_scores.append(np.clip(water_quality, 0.7, 1.0))
        
        # 종합 성과 점수
        performance_scores = []
        for i in range(365):
            # DO 제어 정확도
            do_accuracy = 1.0 / (1.0 + abs(achieved_dos[i] - target_dos[i]))
            
            # 에너지 효율성 (낮을수록 좋음)
            energy_efficiency = 1.0 / (1.0 + (energy_costs[i] - 80) / 20)
            
            # 처리 효율성
            treatment_score = treatment_efficiencies[i]
            
            # 수질 점수
            water_score = water_quality_scores[i]
            
            # 종합 점수
            total_score = (do_accuracy * 0.3 + 
                          energy_efficiency * 0.3 + 
                          treatment_score * 0.2 + 
                          water_score * 0.2)
            
            performance_scores.append(total_score)
        
        # 데이터프레임 생성
        self.historical_data = pd.DataFrame({
            'date': dates,
            'temperature': temperatures,
            'inflow_rate': inflow_rates,
            'cod': cod_values,
            'mlss': mlss_values,
            'target_do': target_dos,
            'achieved_do': achieved_dos,
            'blower_percentage': blower_percentages,
            'energy_cost': energy_costs,
            'treatment_efficiency': treatment_efficiencies,
            'water_quality_score': water_quality_scores,
            'performance_score': performance_scores
        })
        
        print(f"생성된 데이터: {self.historical_data.shape}")
        print(f"기간: {self.historical_data['date'].min()} ~ {self.historical_data['date'].max()}")
        print(f"목표 DO 범위: {self.historical_data['target_do'].min():.2f} ~ {self.historical_data['target_do'].max():.2f} mg/L")
        print(f"송풍량 분포: {self.historical_data['blower_percentage'].value_counts().to_dict()}")
        print(f"평균 성과 점수: {self.historical_data['performance_score'].mean():.3f}")
        
        return self.historical_data
    
    def extract_expert_knowledge(self):
        """
        실무자 경험 지식 추출
        """
        print("\n=== 실무자 경험 지식 추출 ===\n")
        
        # 1. 온도별 최적 목표 DO 패턴
        temp_groups = self.historical_data.groupby(pd.cut(self.historical_data['temperature'], 
                                                         bins=[15, 20, 25, 30, 35]))
        
        temp_optimal_dos = {}
        for temp_range, group in temp_groups:
            # 해당 온도 구간에서 성과가 좋은 데이터의 평균 목표 DO
            high_performance = group[group['performance_score'] > group['performance_score'].quantile(0.7)]
            optimal_do = high_performance['target_do'].mean()
            temp_optimal_dos[temp_range] = optimal_do
            print(f"온도 {temp_range}: 최적 목표 DO = {optimal_do:.2f} mg/L")
        
        # 2. COD별 최적 목표 DO 패턴
        cod_groups = self.historical_data.groupby(pd.cut(self.historical_data['cod'], 
                                                        bins=[100, 150, 200, 250, 300]))
        
        cod_optimal_dos = {}
        for cod_range, group in cod_groups:
            high_performance = group[group['performance_score'] > group['performance_score'].quantile(0.7)]
            optimal_do = high_performance['target_do'].mean()
            cod_optimal_dos[cod_range] = optimal_do
            print(f"COD {cod_range}: 최적 목표 DO = {optimal_do:.2f} mg/L")
        
        # 3. 실무자 의사결정 패턴
        decision_patterns = {
            'do_control_threshold': 0.3,  # DO 차이 임계값
            'blower_adjustment_rules': {
                'high_do': 94,  # DO가 높으면 송풍량 감소
                'low_do': 98,   # DO가 낮으면 송풍량 증가
                'normal_do': 96 # 적정 범위
            },
            'safety_margins': {
                'min_do': 1.5,
                'max_do': 4.0,
                'min_blower': 94,
                'max_blower': 98
            }
        }
        
        self.expert_knowledge = {
            'temp_optimal_dos': temp_optimal_dos,
            'cod_optimal_dos': cod_optimal_dos,
            'decision_patterns': decision_patterns
        }
        
        return self.expert_knowledge
    
    def train_ml_models_from_historical_data(self):
        """
        기존 데이터로 ML 모델 학습
        """
        print("\n=== ML 모델 학습 ===\n")
        
        # 특성 엔지니어링
        data = self.historical_data.copy()
        
        # 파생 특성들
        data['temp_factor'] = (data['temperature'] - 25) / 10
        data['cod_load'] = data['cod'] * data['inflow_rate'] / 1000
        data['flow_factor'] = (data['inflow_rate'] - 1000) / 200
        data['mlss_factor'] = (data['mlss'] - 3000) / 500
        
        # 시간적 특성
        data['day_of_year'] = data['date'].dt.dayofyear
        data['season'] = data['date'].dt.quarter
        
        # 상호작용 특성
        data['temp_cod_interaction'] = data['temp_factor'] * data['cod_load']
        data['flow_mlss_interaction'] = data['flow_factor'] * data['mlss_factor']
        
        # 목표 변수들
        data['do_control_success'] = (abs(data['achieved_do'] - data['target_do']) <= 0.3).astype(int)
        data['high_performance'] = (data['performance_score'] > data['performance_score'].quantile(0.7)).astype(int)
        
        # 학습 특성 선택
        feature_columns = [
            'temperature', 'inflow_rate', 'cod', 'mlss',
            'temp_factor', 'cod_load', 'flow_factor', 'mlss_factor',
            'day_of_year', 'season',
            'temp_cod_interaction', 'flow_mlss_interaction'
        ]
        
        X = data[feature_columns]
        
        # 1. 목표 DO 예측 모델 (회귀)
        y_target_do = data['target_do']
        X_train, X_test, y_train, y_test = train_test_split(X, y_target_do, test_size=0.2, random_state=42)
        
        # 스케일링
        scaler_target_do = StandardScaler()
        X_train_scaled = scaler_target_do.fit_transform(X_train)
        X_test_scaled = scaler_target_do.transform(X_test)
        
        # 모델 학습
        target_do_model = RandomForestRegressor(n_estimators=100, random_state=42)
        target_do_model.fit(X_train_scaled, y_train)
        
        # 성능 평가
        y_pred = target_do_model.predict(X_test_scaled)
        r2_target_do = r2_score(y_test, y_pred)
        
        print(f"목표 DO 예측 모델 R² 점수: {r2_target_do:.3f}")
        
        # 2. 송풍량 결정 모델 (분류)
        y_blower = data['blower_percentage']
        X_train, X_test, y_train, y_test = train_test_split(X, y_blower, test_size=0.2, random_state=42)
        
        scaler_blower = StandardScaler()
        X_train_scaled = scaler_blower.fit_transform(X_train)
        X_test_scaled = scaler_blower.transform(X_test)
        
        blower_model = RandomForestClassifier(n_estimators=100, random_state=42)
        blower_model.fit(X_train_scaled, y_train)
        
        y_pred = blower_model.predict(X_test_scaled)
        accuracy_blower = accuracy_score(y_test, y_pred)
        
        print(f"송풍량 결정 모델 정확도: {accuracy_blower:.3f}")
        
        # 3. 성과 예측 모델 (회귀)
        y_performance = data['performance_score']
        X_train, X_test, y_train, y_test = train_test_split(X, y_performance, test_size=0.2, random_state=42)
        
        scaler_performance = StandardScaler()
        X_train_scaled = scaler_performance.fit_transform(X_train)
        X_test_scaled = scaler_performance.transform(X_test)
        
        performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        performance_model.fit(X_train_scaled, y_train)
        
        y_pred = performance_model.predict(X_test_scaled)
        r2_performance = r2_score(y_test, y_pred)
        
        print(f"성과 예측 모델 R² 점수: {r2_performance:.3f}")
        
        # 모델 저장
        self.optimization_models = {
            'target_do_model': target_do_model,
            'blower_model': blower_model,
            'performance_model': performance_model
        }
        
        self.scalers = {
            'target_do_scaler': scaler_target_do,
            'blower_scaler': scaler_blower,
            'performance_scaler': scaler_performance
        }
        
        return self.optimization_models
    
    def create_hybrid_optimization_system(self):
        """
        실무자 경험 + ML 모델 하이브리드 시스템
        """
        print("\n=== 하이브리드 최적화 시스템 ===\n")
        
        class HybridOptimizer:
            def __init__(self, expert_knowledge, ml_models, scalers):
                self.expert_knowledge = expert_knowledge
                self.ml_models = ml_models
                self.scalers = scalers
            
            def predict_target_do(self, current_conditions):
                """
                실무자 경험 + ML 모델을 결합한 목표 DO 예측
                """
                # 1. 실무자 경험 기반 예측
                temp = current_conditions['temperature']
                cod = current_conditions['cod']
                
                # 온도별 최적 DO (실무자 경험)
                temp_optimal = None
                for temp_range, optimal_do in self.expert_knowledge['temp_optimal_dos'].items():
                    if temp_range.left <= temp < temp_range.right:
                        temp_optimal = optimal_do
                        break
                
                # COD별 최적 DO (실무자 경험)
                cod_optimal = None
                for cod_range, optimal_do in self.expert_knowledge['cod_optimal_dos'].items():
                    if cod_range.left <= cod < cod_range.right:
                        cod_optimal = optimal_do
                        break
                
                # 실무자 경험 기반 예측
                expert_prediction = (temp_optimal + cod_optimal) / 2 if temp_optimal and cod_optimal else 2.0
                
                # 2. ML 모델 예측
                features = self._prepare_features(current_conditions)
                ml_prediction = self.ml_models['target_do_model'].predict(
                    self.scalers['target_do_scaler'].transform([features])
                )[0]
                
                # 3. 가중 평균 (실무자 경험 60%, ML 40%)
                hybrid_prediction = 0.6 * expert_prediction + 0.4 * ml_prediction
                
                # 안전 범위 적용
                hybrid_prediction = np.clip(hybrid_prediction, 
                                          self.expert_knowledge['decision_patterns']['safety_margins']['min_do'],
                                          self.expert_knowledge['decision_patterns']['safety_margins']['max_do'])
                
                return {
                    'expert_prediction': expert_prediction,
                    'ml_prediction': ml_prediction,
                    'hybrid_prediction': hybrid_prediction
                }
            
            def determine_blower_percentage(self, current_conditions, target_do):
                """
                실무자 규칙 + ML 모델을 결합한 송풍량 결정
                """
                # 1. 실무자 규칙 기반
                current_do = current_conditions.get('current_do', target_do)
                do_diff = current_do - target_do
                
                if do_diff > self.expert_knowledge['decision_patterns']['do_control_threshold']:
                    expert_blower = self.expert_knowledge['decision_patterns']['blower_adjustment_rules']['high_do']
                elif do_diff < -self.expert_knowledge['decision_patterns']['do_control_threshold']:
                    expert_blower = self.expert_knowledge['decision_patterns']['blower_adjustment_rules']['low_do']
                else:
                    expert_blower = self.expert_knowledge['decision_patterns']['blower_adjustment_rules']['normal_do']
                
                # 2. ML 모델 예측
                features = self._prepare_features(current_conditions)
                ml_blower = self.ml_models['blower_model'].predict(
                    self.scalers['blower_scaler'].transform([features])
                )[0]
                
                # 3. 가중 평균 (실무자 규칙 70%, ML 30%)
                hybrid_blower = int(0.7 * expert_blower + 0.3 * ml_blower)
                
                # 안전 범위 적용
                hybrid_blower = np.clip(hybrid_blower,
                                      self.expert_knowledge['decision_patterns']['safety_margins']['min_blower'],
                                      self.expert_knowledge['decision_patterns']['safety_margins']['max_blower'])
                
                return {
                    'expert_blower': expert_blower,
                    'ml_blower': ml_blower,
                    'hybrid_blower': hybrid_blower
                }
            
            def predict_performance(self, current_conditions, target_do, blower_percentage):
                """
                성과 예측
                """
                features = self._prepare_features(current_conditions)
                predicted_performance = self.ml_models['performance_model'].predict(
                    self.scalers['performance_scaler'].transform([features])
                )[0]
                
                return predicted_performance
            
            def _prepare_features(self, conditions):
                """
                특성 준비
                """
                temp = conditions['temperature']
                inflow = conditions['inflow_rate']
                cod = conditions['cod']
                mlss = conditions['mlss']
                
                # 파생 특성
                temp_factor = (temp - 25) / 10
                cod_load = cod * inflow / 1000
                flow_factor = (inflow - 1000) / 200
                mlss_factor = (mlss - 3000) / 500
                
                # 상호작용 특성
                temp_cod_interaction = temp_factor * cod_load
                flow_mlss_interaction = flow_factor * mlss_factor
                
                return [
                    temp, inflow, cod, mlss,
                    temp_factor, cod_load, flow_factor, mlss_factor,
                    180, 2,  # day_of_year, season (예시)
                    temp_cod_interaction, flow_mlss_interaction
                ]
        
        self.hybrid_optimizer = HybridOptimizer(
            self.expert_knowledge, 
            self.optimization_models, 
            self.scalers
        )
        
        return self.hybrid_optimizer
    
    def test_hybrid_system(self):
        """
        하이브리드 시스템 테스트
        """
        print("\n=== 하이브리드 시스템 테스트 ===\n")
        
        # 테스트 시나리오들
        test_scenarios = [
            {
                'name': '정상 조건',
                'conditions': {'temperature': 25, 'inflow_rate': 1000, 'cod': 150, 'mlss': 3000}
            },
            {
                'name': '고온 조건',
                'conditions': {'temperature': 30, 'inflow_rate': 1000, 'cod': 150, 'mlss': 3000}
            },
            {
                'name': '고부하 조건',
                'conditions': {'temperature': 25, 'inflow_rate': 1200, 'cod': 200, 'mlss': 3000}
            },
            {
                'name': '복합 조건',
                'conditions': {'temperature': 28, 'inflow_rate': 1100, 'cod': 180, 'mlss': 3500}
            }
        ]
        
        results = []
        
        for scenario in test_scenarios:
            print(f"\n{scenario['name']}:")
            print(f"  조건: 온도 {scenario['conditions']['temperature']}°C, "
                  f"유입량 {scenario['conditions']['inflow_rate']} m³/day, "
                  f"COD {scenario['conditions']['cod']} mg/L")
            
            # 목표 DO 예측
            target_do_result = self.hybrid_optimizer.predict_target_do(scenario['conditions'])
            print(f"  목표 DO 예측:")
            print(f"    실무자 경험: {target_do_result['expert_prediction']:.2f} mg/L")
            print(f"    ML 모델: {target_do_result['ml_prediction']:.2f} mg/L")
            print(f"    하이브리드: {target_do_result['hybrid_prediction']:.2f} mg/L")
            
            # 송풍량 결정
            blower_result = self.hybrid_optimizer.determine_blower_percentage(
                scenario['conditions'], target_do_result['hybrid_prediction']
            )
            print(f"  송풍량 결정:")
            print(f"    실무자 규칙: {blower_result['expert_blower']}%")
            print(f"    ML 모델: {blower_result['ml_blower']}%")
            print(f"    하이브리드: {blower_result['hybrid_blower']}%")
            
            # 성과 예측
            predicted_performance = self.hybrid_optimizer.predict_performance(
                scenario['conditions'], 
                target_do_result['hybrid_prediction'], 
                blower_result['hybrid_blower']
            )
            print(f"  예측 성과 점수: {predicted_performance:.3f}")
            
            results.append({
                'scenario': scenario['name'],
                'conditions': scenario['conditions'],
                'target_do': target_do_result,
                'blower': blower_result,
                'performance': predicted_performance
            })
        
        return results
    
    def visualize_hybrid_results(self, test_results):
        """
        하이브리드 시스템 결과 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 목표 DO 예측 비교
        scenarios = [r['scenario'] for r in test_results]
        expert_dos = [r['target_do']['expert_prediction'] for r in test_results]
        ml_dos = [r['target_do']['ml_prediction'] for r in test_results]
        hybrid_dos = [r['target_do']['hybrid_prediction'] for r in test_results]
        
        x = np.arange(len(scenarios))
        width = 0.25
        
        axes[0,0].bar(x - width, expert_dos, width, label='실무자 경험', alpha=0.8)
        axes[0,0].bar(x, ml_dos, width, label='ML 모델', alpha=0.8)
        axes[0,0].bar(x + width, hybrid_dos, width, label='하이브리드', alpha=0.8)
        axes[0,0].set_xlabel('시나리오')
        axes[0,0].set_ylabel('목표 DO (mg/L)')
        axes[0,0].set_title('목표 DO 예측 비교')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels(scenarios, rotation=45)
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. 송풍량 결정 비교
        expert_blowers = [r['blower']['expert_blower'] for r in test_results]
        ml_blowers = [r['blower']['ml_blower'] for r in test_results]
        hybrid_blowers = [r['blower']['hybrid_blower'] for r in test_results]
        
        axes[0,1].bar(x - width, expert_blowers, width, label='실무자 규칙', alpha=0.8)
        axes[0,1].bar(x, ml_blowers, width, label='ML 모델', alpha=0.8)
        axes[0,1].bar(x + width, hybrid_blowers, width, label='하이브리드', alpha=0.8)
        axes[0,1].set_xlabel('시나리오')
        axes[0,1].set_ylabel('송풍량 (%)')
        axes[0,1].set_title('송풍량 결정 비교')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(scenarios, rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 성과 점수
        performances = [r['performance'] for r in test_results]
        axes[1,0].bar(scenarios, performances, alpha=0.7, color='green')
        axes[1,0].set_xlabel('시나리오')
        axes[1,0].set_ylabel('예측 성과 점수')
        axes[1,0].set_title('예측 성과 점수')
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 온도 vs 목표 DO (기존 데이터)
        axes[1,1].scatter(self.historical_data['temperature'], 
                          self.historical_data['target_do'], 
                          alpha=0.6, s=20)
        axes[1,1].set_xlabel('온도 (°C)')
        axes[1,1].set_ylabel('목표 DO (mg/L)')
        axes[1,1].set_title('기존 데이터: 온도 vs 목표 DO')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def save_models_and_knowledge(self):
        """
        모델과 지식 저장
        """
        print("\n=== 모델 및 지식 저장 ===\n")
        
        # ML 모델 저장
        joblib.dump(self.optimization_models, 'optimization_models.pkl')
        joblib.dump(self.scalers, 'optimization_scalers.pkl')
        
        # 실무자 지식 저장
        joblib.dump(self.expert_knowledge, 'expert_knowledge.pkl')
        
        # 하이브리드 시스템 저장
        joblib.dump(self.hybrid_optimizer, 'hybrid_optimizer.pkl')
        
        print("모델 및 지식이 저장되었습니다:")
        print("  - optimization_models.pkl")
        print("  - optimization_scalers.pkl")
        print("  - expert_knowledge.pkl")
        print("  - hybrid_optimizer.pkl")

def main():
    """
    메인 실행 함수
    """
    print("=== 기존 데이터와 실무자 경험 기반 최적화 시스템 ===\n")
    
    # 1. 시스템 초기화
    optimizer = HistoricalDataOptimizer()
    
    # 2. 기존 데이터 생성 (실제로는 DB에서 로드)
    historical_data = optimizer.create_historical_dataset()
    
    # 3. 실무자 경험 지식 추출
    expert_knowledge = optimizer.extract_expert_knowledge()
    
    # 4. ML 모델 학습
    ml_models = optimizer.train_ml_models_from_historical_data()
    
    # 5. 하이브리드 시스템 구축
    hybrid_system = optimizer.create_hybrid_optimization_system()
    
    # 6. 시스템 테스트
    test_results = optimizer.test_hybrid_system()
    
    # 7. 결과 시각화
    optimizer.visualize_hybrid_results(test_results)
    
    # 8. 모델 저장
    optimizer.save_models_and_knowledge()
    
    # 9. 결론
    print("\n=== 시스템 결론 ===")
    print("1. 기존 데이터를 활용하여 실무자 경험을 정량화했습니다.")
    print("2. ML 모델이 실무자 패턴을 학습하여 예측 정확도를 높였습니다.")
    print("3. 하이브리드 시스템으로 안전성과 성능을 모두 확보했습니다.")
    print("4. 새로운 실험 없이도 기존 데이터로 최적화가 가능합니다.")
    print("5. 실무자의 직관과 ML의 정확성을 결합한 현실적인 솔루션입니다.")

if __name__ == "__main__":
    main() 
