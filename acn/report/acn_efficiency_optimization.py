import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ACNEfficiencyOptimizer:
    """
    ACN 정제 공정 효율성 최적화 분석기
    - Input_source 증가 시에도 높은 수율을 유지하는 방안 탐색
    - 품질값과 Output, Yield 간의 관계 분석
    - 미래 Input 증가에 대한 명시적 Control 가이드 제공
    """
    
    def __init__(self, data_path=None, df=None):
        """
        초기화
        
        Parameters:
        data_path: 데이터 파일 경로
        df: 이미 로드된 DataFrame
        """
        if df is not None:
            self.df = df.copy()
        elif data_path:
            self.df = pd.read_csv(data_path)
        else:
            raise ValueError("데이터 경로 또는 DataFrame을 제공해야 합니다.")
        
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.feature_names = None
        self.quality_columns = None
        self.efficiency_target = None
        self.results = {}
        
        print("ACN 효율성 최적화 분석기 초기화 완료")
    
    def preprocess_data(self):
        """
        데이터 전처리 및 효율성 지표 생성
        """
        print("=" * 80)
        print("데이터 전처리 및 효율성 지표 생성")
        print("=" * 80)
        
        # 1. 최종 F/R Level에서 분석한 데이터만 필터링
        if 'Final_FR' in self.df.columns:
            max_fr_level = self.df['Final_FR'].max()
            self.df = self.df[self.df['Final_FR'] == max_fr_level].copy()
            print(f"최종 F/R Level 필터링 후 데이터 크기: {self.df.shape}")
        
        # 2. 품질값 컬럼 정의
        self.quality_columns = ['AN-10_200nm', 'AN-10_225nm', 'AN-10_250nm', 
                               'AN-50_200nm', 'AN-50_225nm', 'AN-50_250nm']
        
        # 3. 효율성 지표 생성 (Yield 대신 사용)
        self._create_efficiency_metrics()
        
        # 4. Input_source 관련 컬럼 찾기
        input_columns = [col for col in self.df.columns if 'input' in col.lower() or 'source' in col.lower()]
        print(f"Input 관련 컬럼: {input_columns}")
        
        # 5. 수치형 변수만 선택 (Yield, Output, 품질값 제외)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Yield', 'Output'] + self.quality_columns + ['Efficiency_Score', 'Quality_Score']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 결측치가 있는 컬럼 제외
        numeric_cols = [col for col in numeric_cols if not self.df[col].isnull().all()]
        
        # 6. 결측치 처리
        self.X = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        self.y = self.df['Efficiency_Score'].fillna(self.df['Efficiency_Score'].median())
        
        # 7. 특성 스케일링
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.feature_names = numeric_cols
        
        print(f"분석 대상 특성 수: {len(self.feature_names)}")
        print(f"분석 대상 샘플 수: {len(self.X)}")
        print(f"효율성 지표 범위: {self.y.min():.4f} ~ {self.y.max():.4f}")
        
        return self.X, self.y
    
    def _create_efficiency_metrics(self):
        """
        효율성 지표 생성
        - Yield 대신 Input 대비 Output의 효율성을 측정
        - 품질값을 고려한 종합 효율성 점수
        """
        # 1. 기본 효율성 (Output / Input_source)
        if 'Input_source' in self.df.columns and 'Output' in self.df.columns:
            # 0으로 나누기 방지
            input_safe = self.df['Input_source'].replace(0, np.nan)
            self.df['Basic_Efficiency'] = self.df['Output'] / input_safe
        else:
            # Input_source가 없는 경우 Yield를 기본 효율성으로 사용
            self.df['Basic_Efficiency'] = self.df['Yield']
        
        # 2. 품질 점수 계산 (품질값이 낮을수록 좋음)
        quality_score = 0
        quality_count = 0
        
        for col in self.quality_columns:
            if col in self.df.columns:
                # 품질값이 낮을수록 좋은 것으로 가정 (0에 가까울수록 좋음)
                normalized_quality = 1 / (1 + np.abs(self.df[col]))
                quality_score += normalized_quality
                quality_count += 1
        
        if quality_count > 0:
            self.df['Quality_Score'] = quality_score / quality_count
        else:
            self.df['Quality_Score'] = 1.0  # 품질 데이터가 없으면 중립값
        
        # 3. 종합 효율성 점수 (기본 효율성 × 품질 점수)
        self.df['Efficiency_Score'] = self.df['Basic_Efficiency'] * self.df['Quality_Score']
        
        # 4. 정규화 (0-1 범위)
        self.df['Efficiency_Score'] = (self.df['Efficiency_Score'] - self.df['Efficiency_Score'].min()) / \
                                     (self.df['Efficiency_Score'].max() - self.df['Efficiency_Score'].min())
        
        print("효율성 지표 생성 완료:")
        print(f"  - 기본 효율성 범위: {self.df['Basic_Efficiency'].min():.4f} ~ {self.df['Basic_Efficiency'].max():.4f}")
        print(f"  - 품질 점수 범위: {self.df['Quality_Score'].min():.4f} ~ {self.df['Quality_Score'].max():.4f}")
        print(f"  - 종합 효율성 점수 범위: {self.df['Efficiency_Score'].min():.4f} ~ {self.df['Efficiency_Score'].max():.4f}")
    
    def analyze_input_output_relationship(self):
        """
        Input-Output 관계 분석
        """
        print("\n" + "=" * 80)
        print("Input-Output 관계 분석")
        print("=" * 80)
        
        # Input_source와 Output, Yield, Efficiency_Score 관계 분석
        input_cols = [col for col in self.df.columns if 'input' in col.lower() or 'source' in col.lower()]
        
        if not input_cols:
            print("Input 관련 컬럼을 찾을 수 없습니다.")
            return None
        
        input_col = input_cols[0]  # 첫 번째 Input 컬럼 사용
        print(f"분석 대상 Input 컬럼: {input_col}")
        
        # 상관관계 분석
        target_cols = ['Output', 'Yield', 'Efficiency_Score']
        correlations = {}
        
        for target in target_cols:
            if target in self.df.columns:
                corr, p_value = stats.pearsonr(self.df[input_col], self.df[target])
                correlations[target] = {
                    'correlation': corr,
                    'p_value': p_value
                }
                print(f"{input_col} vs {target}: r={corr:.4f}, p={p_value:.4f}")
        
        # 시각화
        self._plot_input_output_relationship(input_col, target_cols)
        
        return correlations
    
    def _plot_input_output_relationship(self, input_col, target_cols):
        """
        Input-Output 관계 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Input vs Output
        if 'Output' in self.df.columns:
            axes[0, 0].scatter(self.df[input_col], self.df['Output'], alpha=0.6)
            # 회귀선 추가
            z = np.polyfit(self.df[input_col], self.df['Output'], 1)
            p = np.poly1d(z)
            axes[0, 0].plot(self.df[input_col], p(self.df[input_col]), "r--", alpha=0.8)
            axes[0, 0].set_xlabel(input_col)
            axes[0, 0].set_ylabel('Output')
            axes[0, 0].set_title(f'{input_col} vs Output')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Input vs Yield
        if 'Yield' in self.df.columns:
            axes[0, 1].scatter(self.df[input_col], self.df['Yield'], alpha=0.6, color='green')
            # 회귀선 추가
            z = np.polyfit(self.df[input_col], self.df['Yield'], 1)
            p = np.poly1d(z)
            axes[0, 1].plot(self.df[input_col], p(self.df[input_col]), "r--", alpha=0.8)
            axes[0, 1].set_xlabel(input_col)
            axes[0, 1].set_ylabel('Yield')
            axes[0, 1].set_title(f'{input_col} vs Yield')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Input vs Efficiency_Score
        axes[1, 0].scatter(self.df[input_col], self.df['Efficiency_Score'], alpha=0.6, color='orange')
        # 회귀선 추가
        z = np.polyfit(self.df[input_col], self.df['Efficiency_Score'], 1)
        p = np.poly1d(z)
        axes[1, 0].plot(self.df[input_col], p(self.df[input_col]), "r--", alpha=0.8)
        axes[1, 0].set_xlabel(input_col)
        axes[1, 0].set_ylabel('Efficiency Score')
        axes[1, 0].set_title(f'{input_col} vs Efficiency Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Output vs Yield vs Efficiency_Score (3D scatter)
        if 'Output' in self.df.columns and 'Yield' in self.df.columns:
            scatter = axes[1, 1].scatter(self.df['Output'], self.df['Yield'], 
                                       c=self.df['Efficiency_Score'], 
                                       cmap='viridis', alpha=0.6)
            axes[1, 1].set_xlabel('Output')
            axes[1, 1].set_ylabel('Yield')
            axes[1, 1].set_title('Output vs Yield (색상: Efficiency Score)')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Efficiency Score')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_quality_relationships(self):
        """
        품질값과 Output, Yield 간의 관계 분석
        """
        print("\n" + "=" * 80)
        print("품질값과 Output, Yield 간의 관계 분석")
        print("=" * 80)
        
        # 품질값과 각 지표 간의 상관관계
        target_cols = ['Output', 'Yield', 'Efficiency_Score']
        quality_relationships = {}
        
        for quality_col in self.quality_columns:
            if quality_col in self.df.columns:
                quality_relationships[quality_col] = {}
                
                for target in target_cols:
                    if target in self.df.columns:
                        corr, p_value = stats.pearsonr(self.df[quality_col], self.df[target])
                        quality_relationships[quality_col][target] = {
                            'correlation': corr,
                            'p_value': p_value
                        }
        
        # 결과 출력
        print("품질값과 각 지표 간의 상관관계:")
        for quality_col, targets in quality_relationships.items():
            print(f"\n{quality_col}:")
            for target, stats in targets.items():
                print(f"  vs {target}: r={stats['correlation']:.4f}, p={stats['p_value']:.4f}")
        
        # 시각화
        self._plot_quality_relationships(quality_relationships)
        
        return quality_relationships
    
    def _plot_quality_relationships(self, quality_relationships):
        """
        품질값 관계 시각화
        """
        # 상위 3개 품질값만 시각화
        top_quality_cols = list(quality_relationships.keys())[:3]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        for i, quality_col in enumerate(top_quality_cols):
            # Output vs 품질값
            if 'Output' in self.df.columns:
                axes[0, i].scatter(self.df[quality_col], self.df['Output'], alpha=0.6)
                axes[0, i].set_xlabel(quality_col)
                axes[0, i].set_ylabel('Output')
                axes[0, i].set_title(f'{quality_col} vs Output')
                axes[0, i].grid(True, alpha=0.3)
            
            # Yield vs 품질값
            if 'Yield' in self.df.columns:
                axes[1, i].scatter(self.df[quality_col], self.df['Yield'], alpha=0.6, color='green')
                axes[1, i].set_xlabel(quality_col)
                axes[1, i].set_ylabel('Yield')
                axes[1, i].set_title(f'{quality_col} vs Yield')
                axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def build_efficiency_prediction_model(self):
        """
        효율성 예측 모델 구축
        """
        print("\n" + "=" * 80)
        print("효율성 예측 모델 구축")
        print("=" * 80)
        
        # 모델 정의
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Lasso Regression': Lasso(alpha=0.1),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale')
        }
        
        # 훈련/테스트 분할
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_scaled, self.y, test_size=0.2, random_state=42
        )
        
        model_results = {}
        
        for name, model in models.items():
            # 모델 훈련
            model.fit(X_train, y_train)
            
            # 예측
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # 성능 평가
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            
            # 교차 검증
            cv_scores = cross_val_score(model, self.X_scaled, self.y, cv=5, scoring='r2')
            
            model_results[name] = {
                'model': model,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_test_pred
            }
            
            print(f"\n{name}:")
            print(f"  Train R²: {train_r2:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Train RMSE: {train_rmse:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        # 최고 성능 모델 선택
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        best_model = model_results[best_model_name]['model']
        
        print(f"\n최고 성능 모델: {best_model_name}")
        
        # 특성 중요도 분석 (Random Forest인 경우)
        if best_model_name == 'Random Forest':
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\n특성 중요도 (상위 10개):")
            print(feature_importance.head(10))
        
        # 시각화
        self._plot_model_performance(model_results, y_test)
        
        self.results['model_results'] = model_results
        self.results['best_model'] = best_model
        self.results['best_model_name'] = best_model_name
        
        return model_results, best_model
    
    def _plot_model_performance(self, model_results, y_test):
        """
        모델 성능 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 모델별 R² 점수 비교
        model_names = list(model_results.keys())
        test_r2_scores = [model_results[name]['test_r2'] for name in model_names]
        cv_scores = [model_results[name]['cv_mean'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
        axes[0, 0].bar(x + width/2, cv_scores, width, label='CV R²', alpha=0.8)
        axes[0, 0].set_xlabel('Models')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].set_title('Model Performance Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(model_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 모델별 RMSE 비교
        test_rmse_scores = [model_results[name]['test_rmse'] for name in model_names]
        
        axes[0, 1].bar(model_names, test_rmse_scores, alpha=0.8, color='orange')
        axes[0, 1].set_xlabel('Models')
        axes[0, 1].set_ylabel('Test RMSE')
        axes[0, 1].set_title('Model RMSE Comparison')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 최고 성능 모델의 예측 vs 실제값
        best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['test_r2'])
        y_pred = model_results[best_model_name]['predictions']
        
        axes[1, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[1, 0].set_xlabel('Actual Efficiency Score')
        axes[1, 0].set_ylabel('Predicted Efficiency Score')
        axes[1, 0].set_title(f'Prediction vs Actual ({best_model_name})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 잔차 플롯
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Efficiency Score')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_input_increase_guidance(self):
        """
        Input 증가 시 Control 가이드 생성
        """
        print("\n" + "=" * 80)
        print("Input 증가 시 Control 가이드 생성")
        print("=" * 80)
        
        if 'best_model' not in self.results:
            print("모델이 구축되지 않았습니다. build_efficiency_prediction_model()을 먼저 실행하세요.")
            return None
        
        best_model = self.results['best_model']
        
        # 현재 평균 Input_source 값
        input_cols = [col for col in self.df.columns if 'input' in col.lower() or 'source' in col.lower()]
        if not input_cols:
            print("Input 관련 컬럼을 찾을 수 없습니다.")
            return None
        
        input_col = input_cols[0]
        current_input = self.df[input_col].mean()
        
        # Input 증가 시나리오 (10%, 20%, 30% 증가)
        increase_scenarios = [0.1, 0.2, 0.3]
        
        guidance_results = {}
        
        for increase in increase_scenarios:
            new_input = current_input * (1 + increase)
            print(f"\nInput {increase*100:.0f}% 증가 시나리오 (현재: {current_input:.2f} → 새로운: {new_input:.2f}):")
            
            # 현재 조건에서의 예측 효율성
            current_conditions = self.X.mean().values.reshape(1, -1)
            current_conditions_scaled = self.scaler.transform(current_conditions)
            current_efficiency = best_model.predict(current_conditions_scaled)[0]
            
            # Input 증가 시 효율성 유지를 위한 최적 Control 값 찾기
            optimal_controls = self._find_optimal_controls_for_input_increase(
                best_model, current_conditions, new_input, current_efficiency
            )
            
            guidance_results[f'increase_{increase*100:.0f}%'] = {
                'new_input': new_input,
                'current_efficiency': current_efficiency,
                'optimal_controls': optimal_controls
            }
            
            print(f"  현재 효율성: {current_efficiency:.4f}")
            print(f"  효율성 유지를 위한 최적 Control 값:")
            for control_name, control_value in optimal_controls.items():
                print(f"    {control_name}: {control_value:.4f}")
        
        self.results['input_increase_guidance'] = guidance_results
        return guidance_results
    
    def _find_optimal_controls_for_input_increase(self, model, current_conditions, new_input, target_efficiency):
        """
        Input 증가 시 효율성 유지를 위한 최적 Control 값 찾기
        """
        # Input 컬럼 인덱스 찾기
        input_cols = [col for col in self.df.columns if 'input' in col.lower() or 'source' in col.lower()]
        if not input_cols:
            return {}
        
        input_col = input_cols[0]
        input_idx = self.feature_names.index(input_col) if input_col in self.feature_names else None
        
        if input_idx is None:
            return {}
        
        # 최적화 함수 정의
        def objective(controls):
            # 현재 조건을 복사하고 Control 값들을 업데이트
            new_conditions = current_conditions.copy()
            new_conditions[0, input_idx] = new_input
            
            # 다른 Control 값들 업데이트
            for i, (feature_name, control_value) in enumerate(controls.items()):
                if feature_name in self.feature_names:
                    feature_idx = self.feature_names.index(feature_name)
                    new_conditions[0, feature_idx] = control_value
            
            # 스케일링
            new_conditions_scaled = self.scaler.transform(new_conditions)
            
            # 예측
            predicted_efficiency = model.predict(new_conditions_scaled)[0]
            
            # 목표 효율성과의 차이를 최소화
            return (predicted_efficiency - target_efficiency) ** 2
        
        # Control 변수들 (Input_source 제외한 상위 5개 특성)
        control_features = [feat for feat in self.feature_names[:5] if feat != input_col]
        
        # 초기값 설정 (현재 평균값)
        initial_controls = {}
        for feature in control_features:
            feature_idx = self.feature_names.index(feature)
            initial_controls[feature] = current_conditions[0, feature_idx]
        
        # 제약 조건 (각 특성의 범위 내에서)
        bounds = []
        for feature in control_features:
            feature_idx = self.feature_names.index(feature)
            min_val = self.X.iloc[:, feature_idx].min()
            max_val = self.X.iloc[:, feature_idx].max()
            bounds.append((min_val, max_val))
        
        # 최적화 실행
        try:
            result = minimize(
                objective,
                list(initial_controls.values()),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_controls = dict(zip(control_features, result.x))
                return optimal_controls
            else:
                print(f"최적화 실패: {result.message}")
                return initial_controls
                
        except Exception as e:
            print(f"최적화 중 오류 발생: {str(e)}")
            return initial_controls
    
    def generate_comprehensive_report(self):
        """
        종합 분석 리포트 생성
        """
        print("\n" + "=" * 80)
        print("종합 분석 리포트 생성")
        print("=" * 80)
        
        report = {
            'data_info': {
                'n_samples': len(self.X) if self.X is not None else 0,
                'n_features': len(self.feature_names) if self.feature_names is not None else 0,
                'efficiency_range': (self.y.min(), self.y.max()) if self.y is not None else (0, 0)
            },
            'key_findings': [],
            'recommendations': [],
            'input_guidance': {}
        }
        
        # 주요 발견사항
        if 'model_results' in self.results:
            best_model_name = self.results['best_model_name']
            best_r2 = self.results['model_results'][best_model_name]['test_r2']
            report['key_findings'].append(f"최고 성능 모델: {best_model_name} (R² = {best_r2:.4f})")
        
        # Input 증가 가이드
        if 'input_increase_guidance' in self.results:
            guidance = self.results['input_increase_guidance']
            report['input_guidance'] = guidance
            
            for scenario, data in guidance.items():
                report['recommendations'].append(
                    f"{scenario}: Input {data['new_input']:.2f}에서 효율성 유지를 위한 Control 조정 필요"
                )
        
        # 리포트 출력
        print("\n📊 ACN 효율성 최적화 분석 결과")
        print("-" * 50)
        
        print(f"\n데이터 정보:")
        print(f"  • 샘플 수: {report['data_info']['n_samples']}개")
        print(f"  • 특성 수: {report['data_info']['n_features']}개")
        print(f"  • 효율성 점수 범위: {report['data_info']['efficiency_range'][0]:.4f} ~ {report['data_info']['efficiency_range'][1]:.4f}")
        
        print(f"\n주요 발견사항:")
        for finding in report['key_findings']:
            print(f"  • {finding}")
        
        print(f"\n권장사항:")
        for recommendation in report['recommendations']:
            print(f"  • {recommendation}")
        
        return report

def main_efficiency_optimization(data_path=None, df=None):
    """
    ACN 효율성 최적화 분석 메인 함수
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    
    Returns:
    results: 분석 결과 딕셔너리
    """
    # 1. 분석기 초기화
    optimizer = ACNEfficiencyOptimizer(data_path, df)
    
    # 2. 데이터 전처리
    optimizer.preprocess_data()
    
    # 3. Input-Output 관계 분석
    input_output_analysis = optimizer.analyze_input_output_relationship()
    
    # 4. 품질값 관계 분석
    quality_analysis = optimizer.analyze_quality_relationships()
    
    # 5. 효율성 예측 모델 구축
    model_results, best_model = optimizer.build_efficiency_prediction_model()
    
    # 6. Input 증가 가이드 생성
    input_guidance = optimizer.generate_input_increase_guidance()
    
    # 7. 종합 리포트 생성
    report = optimizer.generate_comprehensive_report()
    
    return {
        'optimizer': optimizer,
        'input_output_analysis': input_output_analysis,
        'quality_analysis': quality_analysis,
        'model_results': model_results,
        'best_model': best_model,
        'input_guidance': input_guidance,
        'report': report
    }

# 사용 예시
if __name__ == "__main__":
    print("ACN 효율성 최적화 분석을 시작합니다.")
    print("\n사용법:")
    print("1. CSV 파일에서 분석:")
    print("   results = main_efficiency_optimization(data_path='your_data.csv')")
    print("\n2. DataFrame에서 분석:")
    print("   results = main_efficiency_optimization(df=your_dataframe)")
    print("\n3. 결과 확인:")
    print("   print(results['report'])")
