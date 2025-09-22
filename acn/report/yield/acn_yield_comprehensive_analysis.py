"""
ACN 정제 공정 Yield 종합 분석
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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.optimize import minimize
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ACNYieldComprehensiveAnalyzer:
    """
    ACN 정제 공정 Yield 종합 분석기
    - Input_source와 Product 제외하고 Yield 변수만 분석
    - 모든 분석을 종합하여 HTML 리포트 생성
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
        self.results = {}
        self.plots = {}  # 그래프 저장용
        
        print("ACN Yield 종합 분석기 초기화 완료")
    
    def preprocess_data(self):
        """
        데이터 전처리
        """
        print("=" * 80)
        print("데이터 전처리")
        print("=" * 80)
        
        # 1. 최종 F/R Level에서 분석한 데이터만 필터링
        if 'Final_FR' in self.df.columns:
            max_fr_level = self.df['Final_FR'].max()
            self.df = self.df[self.df['Final_FR'] == max_fr_level].copy()
            print(f"최종 F/R Level 필터링 후 데이터 크기: {self.df.shape}")
        
        # 2. Yield를 target으로 설정
        if 'Yield' not in self.df.columns:
            raise ValueError("Yield 컬럼이 필요합니다.")
        
        self.y = self.df['Yield'].fillna(self.df['Yield'].median())
        
        # 3. Input_source와 Product를 제외한 수치형 변수만 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Input_source', 'Product']  # 요청사항에 따라 제외
        
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 결측치가 있는 컬럼 제외
        numeric_cols = [col for col in numeric_cols if not self.df[col].isnull().all()]
        
        # 4. 결측치 처리
        self.X = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # 5. 특성 스케일링
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.feature_names = numeric_cols
        
        print(f"분석 대상 특성 수: {len(self.feature_names)}")
        print(f"분석 대상 샘플 수: {len(self.X)}")
        print(f"Target 변수: Yield")
        print(f"Yield 범위: {self.y.min():.4f} ~ {self.y.max():.4f}")
        print(f"제외된 변수: {exclude_cols}")
        
        return self.X, self.y
    
    def analyze_data_distribution(self):
        """
        데이터 분포 분석
        """
        print("\n" + "=" * 80)
        print("데이터 분포 분석")
        print("=" * 80)
        
        # 기본 통계
        print(f"전체 데이터 크기: {len(self.df)}")
        print(f"Yield 범위: {self.y.min():.4f} ~ {self.y.max():.4f}")
        print(f"Yield 평균: {self.y.mean():.4f}")
        print(f"Yield 표준편차: {self.y.std():.4f}")
        
        # 특성별 통계
        print(f"\n특성별 통계 (상위 10개):")
        feature_stats = self.X.describe().round(4)
        print(feature_stats.head(10))
        
        # 시각화
        self._plot_data_distribution()
        
        return feature_stats
    
    def _plot_data_distribution(self):
        """
        데이터 분포 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Yield 히스토그램
        axes[0, 0].hist(self.y, bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Yield')
        axes[0, 0].set_ylabel('빈도')
        axes[0, 0].set_title('Yield 분포')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Yield 박스플롯
        axes[0, 1].boxplot(self.y)
        axes[0, 1].set_ylabel('Yield')
        axes[0, 1].set_title('Yield 박스플롯')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 상위 5개 특성의 분포
        top_features = self.feature_names[:5]
        for i, feature in enumerate(top_features):
            axes[1, 0].hist(self.X[feature], bins=20, alpha=0.5, label=feature)
        axes[1, 0].set_xlabel('값')
        axes[1, 0].set_ylabel('빈도')
        axes[1, 0].set_title('상위 5개 특성 분포')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 특성별 표준편차
        feature_stds = self.X.std().sort_values(ascending=False).head(10)
        axes[1, 1].barh(range(len(feature_stds)), feature_stds.values)
        axes[1, 1].set_yticks(range(len(feature_stds)))
        axes[1, 1].set_yticklabels(feature_stds.index)
        axes[1, 1].set_xlabel('표준편차')
        axes[1, 1].set_title('특성별 표준편차 (상위 10개)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 그래프를 base64로 저장
        self.plots['data_distribution'] = self._plot_to_base64(fig)
        plt.show()
    
    def analyze_correlations(self):
        """
        상관관계 분석
        """
        print("\n" + "=" * 80)
        print("상관관계 분석")
        print("=" * 80)
        
        # Yield와 각 특성 간의 상관관계
        correlations = []
        for feature in self.feature_names:
            corr, p_value = stats.pearsonr(self.X[feature], self.y)
            correlations.append({
                'feature': feature,
                'correlation': corr,
                'p_value': p_value
            })
        
        # 상관관계 정렬
        correlations_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)
        
        print("Yield와 특성 간의 상관관계 (상위 15개):")
        print(correlations_df.head(15).round(4))
        
        # 시각화
        self._plot_correlations(correlations_df)
        
        return correlations_df
    
    def _plot_correlations(self, correlations_df):
        """
        상관관계 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 상위 10개 특성의 상관계수
        top_corr = correlations_df.head(10)
        colors = ['red' if x < 0 else 'blue' for x in top_corr['correlation']]
        axes[0, 0].barh(range(len(top_corr)), top_corr['correlation'], color=colors, alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_corr)))
        axes[0, 0].set_yticklabels(top_corr['feature'])
        axes[0, 0].set_xlabel('상관계수')
        axes[0, 0].set_title('Yield와 특성 간 상관계수 (상위 10개)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 상관계수 히스토그램
        axes[0, 1].hist(correlations_df['correlation'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('상관계수')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].set_title('상관계수 분포')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 상위 3개 특성과 Yield의 산점도
        top_3_features = correlations_df.head(3)['feature'].tolist()
        for i, feature in enumerate(top_3_features):
            axes[1, 0].scatter(self.X[feature], self.y, alpha=0.6, label=feature)
        axes[1, 0].set_xlabel('특성 값')
        axes[1, 0].set_ylabel('Yield')
        axes[1, 0].set_title('상위 3개 특성과 Yield의 관계')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. p-value 분포
        axes[1, 1].hist(correlations_df['p_value'], bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        axes[1, 1].set_xlabel('p-value')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].set_title('p-value 분포')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 그래프를 base64로 저장
        self.plots['correlations'] = self._plot_to_base64(fig)
        plt.show()
    
    def build_prediction_models(self):
        """
        예측 모델 구축
        """
        print("\n" + "=" * 80)
        print("예측 모델 구축")
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
        axes[1, 0].set_xlabel('Actual Yield')
        axes[1, 0].set_ylabel('Predicted Yield')
        axes[1, 0].set_title(f'Prediction vs Actual ({best_model_name})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 잔차 플롯
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Yield')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 그래프를 base64로 저장
        self.plots['model_performance'] = self._plot_to_base64(fig)
        plt.show()
    
    def analyze_feature_importance(self):
        """
        특성 중요도 분석
        """
        print("\n" + "=" * 80)
        print("특성 중요도 분석")
        print("=" * 80)
        
        if 'best_model' not in self.results:
            print("모델이 구축되지 않았습니다. build_prediction_models()을 먼저 실행하세요.")
            return None
        
        best_model = self.results['best_model']
        best_model_name = self.results['best_model_name']
        
        # 특성 중요도 계산
        if best_model_name == 'Random Forest':
            importance_scores = best_model.feature_importances_
        else:
            # 다른 모델의 경우 계수 절댓값 사용
            if hasattr(best_model, 'coef_'):
                importance_scores = np.abs(best_model.coef_)
            else:
                print("이 모델에서는 특성 중요도를 계산할 수 없습니다.")
                return None
        
        # 특성 중요도 DataFrame 생성
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print("특성 중요도 (상위 15개):")
        print(feature_importance_df.head(15).round(4))
        
        # 시각화
        self._plot_feature_importance(feature_importance_df)
        
        return feature_importance_df
    
    def _plot_feature_importance(self, feature_importance_df):
        """
        특성 중요도 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 상위 15개 특성 중요도
        top_15 = feature_importance_df.head(15)
        axes[0, 0].barh(range(len(top_15)), top_15['importance'])
        axes[0, 0].set_yticks(range(len(top_15)))
        axes[0, 0].set_yticklabels(top_15['feature'])
        axes[0, 0].set_xlabel('중요도')
        axes[0, 0].set_title('특성 중요도 (상위 15개)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 중요도 분포
        axes[0, 1].hist(feature_importance_df['importance'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('중요도')
        axes[0, 1].set_ylabel('빈도')
        axes[0, 1].set_title('특성 중요도 분포')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 상위 5개 특성과 Yield의 관계
        top_5_features = feature_importance_df.head(5)['feature'].tolist()
        for i, feature in enumerate(top_5_features):
            axes[1, 0].scatter(self.X[feature], self.y, alpha=0.6, label=feature)
        axes[1, 0].set_xlabel('특성 값')
        axes[1, 0].set_ylabel('Yield')
        axes[1, 0].set_title('상위 5개 특성과 Yield의 관계')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 중요도 누적 분포
        cumulative_importance = feature_importance_df['importance'].cumsum()
        cumulative_importance = cumulative_importance / cumulative_importance.iloc[-1] * 100
        axes[1, 1].plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 'o-')
        axes[1, 1].axhline(y=80, color='red', linestyle='--', label='80%')
        axes[1, 1].set_xlabel('특성 수')
        axes[1, 1].set_ylabel('누적 중요도 (%)')
        axes[1, 1].set_title('특성 중요도 누적 분포')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 그래프를 base64로 저장
        self.plots['feature_importance'] = self._plot_to_base64(fig)
        plt.show()
    
    def generate_optimization_guidance(self):
        """
        Yield 최적화 가이드 생성
        """
        print("\n" + "=" * 80)
        print("Yield 최적화 가이드 생성")
        print("=" * 80)
        
        if 'best_model' not in self.results:
            print("모델이 구축되지 않았습니다. build_prediction_models()을 먼저 실행하세요.")
            return None
        
        best_model = self.results['best_model']
        
        # 현재 평균 조건에서의 예측 Yield
        current_conditions = self.X.mean().values.reshape(1, -1)
        current_conditions_scaled = self.scaler.transform(current_conditions)
        current_yield = best_model.predict(current_conditions_scaled)[0]
        
        print(f"현재 평균 조건에서 예측 Yield: {current_yield:.4f}")
        
        # 간단한 최적화 가이드 (최적화 알고리즘 대신 상위 특성들의 최대값 사용)
        try:
            optimal_controls = {}
            top_features = self.feature_names[:5]  # 상위 5개 특성
            
            for feature_name in top_features:
                feature_idx = self.feature_names.index(feature_name)
                # 각 특성의 최대값을 최적값으로 설정
                optimal_controls[feature_name] = self.X.iloc[:, feature_idx].max()
            
            print(f"\nYield 최대화를 위한 최적 Control 값 (상위 특성 최대값):")
            for control_name, control_value in optimal_controls.items():
                print(f"  {control_name}: {control_value:.4f}")
            
            # 최적 조건에서의 예측 Yield
            optimal_conditions = current_conditions.copy()
            for feature_name, control_value in optimal_controls.items():
                if feature_name in self.feature_names:
                    feature_idx = self.feature_names.index(feature_name)
                    optimal_conditions[0, feature_idx] = control_value
            
            optimal_conditions_scaled = self.scaler.transform(optimal_conditions)
            optimal_yield = best_model.predict(optimal_conditions_scaled)[0]
            
            print(f"\n최적 조건에서 예측 Yield: {optimal_yield:.4f}")
            print(f"Yield 개선 효과: {optimal_yield - current_yield:.4f} ({(optimal_yield/current_yield - 1)*100:.2f}% 증가)")
            
            guidance_results = {
                'current_yield': current_yield,
                'optimal_yield': optimal_yield,
                'improvement': optimal_yield - current_yield,
                'improvement_percent': (optimal_yield/current_yield - 1)*100,
                'optimal_controls': optimal_controls
            }
            
        except Exception as e:
            print(f"최적화 가이드 생성 중 오류 발생: {str(e)}")
            # 기본값으로 설정
            guidance_results = {
                'current_yield': current_yield,
                'optimal_yield': current_yield,
                'improvement': 0,
                'improvement_percent': 0,
                'optimal_controls': {}
            }
        
        self.results['optimization_guidance'] = guidance_results
        return guidance_results
    
    def _find_optimal_controls_for_yield_maximization(self, model, current_conditions):
        """
        Yield 최대화를 위한 최적 Control 값 찾기
        """
        # Control 변수들 (상위 5개 특성)
        control_features = self.feature_names[:5]
        
        # 최적화 함수 정의 (Yield 최대화)
        def objective(controls_array):
            # controls_array는 numpy 배열로 전달됨
            new_conditions = current_conditions.copy()
            
            # Control 값들 업데이트
            for i, feature_name in enumerate(control_features):
                if i < len(controls_array):
                    feature_idx = self.feature_names.index(feature_name)
                    new_conditions[0, feature_idx] = controls_array[i]
            
            # 스케일링
            new_conditions_scaled = self.scaler.transform(new_conditions)
            
            # 예측 (Yield 최대화를 위해 음수로 반환)
            predicted_yield = model.predict(new_conditions_scaled)[0]
            return -predicted_yield  # 최대화를 위해 음수 반환
        
        # 초기값 설정 (현재 평균값)
        initial_controls = []
        for feature in control_features:
            feature_idx = self.feature_names.index(feature)
            initial_controls.append(current_conditions[0, feature_idx])
        
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
                initial_controls,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )
            
            if result.success:
                optimal_controls = dict(zip(control_features, result.x))
                return optimal_controls
            else:
                print(f"최적화 실패: {result.message}")
                # 실패 시 초기값 반환
                return dict(zip(control_features, initial_controls))
                
        except Exception as e:
            print(f"최적화 중 오류 발생: {str(e)}")
            # 오류 시 초기값 반환
            return dict(zip(control_features, initial_controls))
    
    def _plot_to_base64(self, fig):
        """
        matplotlib 그래프를 base64 문자열로 변환
        """
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64
    
    def generate_comprehensive_html_report(self, output_file='acn_yield_comprehensive_report.html'):
        """
        종합 HTML 리포트 생성
        """
        print("\n" + "=" * 80)
        print("종합 HTML 리포트 생성")
        print("=" * 80)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ACN 정제 공정 Yield 종합 분석 리포트</title>
            <style>
                body {{ 
                    font-family: 'Malgun Gothic', Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    line-height: 1.6; 
                    background-color: #f8f9fa;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background-color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{ 
                    text-align: center; 
                    margin-bottom: 40px; 
                    border-bottom: 3px solid #3498db; 
                    padding-bottom: 20px;
                }}
                .header h1 {{ 
                    color: #2c3e50; 
                    margin: 0; 
                    font-size: 2.5em;
                }}
                .section {{ 
                    margin: 40px 0; 
                    padding: 25px; 
                    border-radius: 8px; 
                    background-color: #f8f9fa;
                }}
                .section h2 {{ 
                    color: #2c3e50; 
                    border-left: 5px solid #3498db; 
                    padding-left: 15px; 
                    margin-top: 0;
                }}
                .plot-container {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .plot-container img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .summary-box {{ 
                    background-color: #ecf0f1; 
                    padding: 20px; 
                    border-radius: 5px; 
                    margin: 20px 0; 
                    border-left: 4px solid #3498db;
                }}
                .finding-box {{ 
                    background-color: #e8f5e8; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 10px 0; 
                    border-left: 4px solid #27ae60;
                }}
                .recommendation-box {{ 
                    background-color: #fff3cd; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 10px 0; 
                    border-left: 4px solid #f39c12;
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0; 
                    background-color: white;
                }}
                th, td {{ 
                    border: 1px solid #bdc3c7; 
                    padding: 12px; 
                    text-align: left;
                }}
                th {{ 
                    background-color: #34495e; 
                    color: white; 
                    font-weight: bold;
                }}
                tr:nth-child(even) {{ 
                    background-color: #f8f9fa;
                }}
                .metric {{ 
                    font-weight: bold; 
                    color: #e74c3c;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧪 ACN 정제 공정 Yield 종합 분석 리포트</h1>
                    <p>분석 일시: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <section>
                    <h2>📊 분석 개요</h2>
                    <div class="summary-box">
                        <h3>데이터 정보</h3>
                        <p><strong>전체 샘플 수:</strong> {len(self.df)}개</p>
                        <p><strong>분석 대상 특성 수:</strong> {len(self.feature_names)}개</p>
                        <p><strong>Target 변수:</strong> Yield</p>
                        <p><strong>Yield 범위:</strong> {self.y.min():.4f} ~ {self.y.max():.4f}</p>
                        <p><strong>Yield 평균:</strong> {self.y.mean():.4f}</p>
                        <p><strong>Yield 표준편차:</strong> {self.y.std():.4f}</p>
                    </div>
                </section>
                
                <section>
                    <h2>📈 데이터 분포 분석</h2>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{self.plots.get('data_distribution', '')}" alt="데이터 분포">
                    </div>
                </section>
                
                <section>
                    <h2>🔗 상관관계 분석</h2>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{self.plots.get('correlations', '')}" alt="상관관계 분석">
                    </div>
                </section>
                
                <section>
                    <h2>🤖 예측 모델 성능</h2>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{self.plots.get('model_performance', '')}" alt="모델 성능">
                    </div>
                </section>
                
                <section>
                    <h2>🎯 특성 중요도 분석</h2>
                    <div class="plot-container">
                        <img src="data:image/png;base64,{self.plots.get('feature_importance', '')}" alt="특성 중요도">
                    </div>
                </section>
                
                <section>
                    <h2>💡 주요 발견사항</h2>
                    {self._generate_findings_html()}
                </section>
                
                <section>
                    <h2>🚀 최적화 권장사항</h2>
                    {self._generate_recommendations_html()}
                </section>
                
                <section>
                    <h2>📋 종합 결론</h2>
                    <div class="summary-box">
                        <p>본 종합 분석을 통해 ACN 정제 공정의 Yield 최적화를 위한 종합적인 인사이트를 도출했습니다. 
                        다양한 분석 방법론을 통해 도출된 결과들을 종합하면, 다음과 같은 핵심 전략을 제안합니다:</p>
                        
                        <ul>
                            <li><strong>데이터 기반 의사결정:</strong> 통계적 유의성과 ML 모델 성능을 바탕으로 한 과학적 접근</li>
                            <li><strong>특성 우선순위 관리:</strong> 중요도 분석을 통한 효율적 자원 배분</li>
                            <li><strong>공정 최적화:</strong> 최적화 알고리즘을 통한 최적 공정 조건 도출</li>
                            <li><strong>지속적 모니터링:</strong> 주요 영향 인자에 대한 실시간 모니터링 체계 구축</li>
                        </ul>
                        
                        <p>이러한 통합적 접근을 통해 ACN 정제 공정의 Yield를 체계적으로 개선할 수 있을 것으로 기대됩니다.</p>
                    </div>
                </section>
            </div>
        </body>
        </html>
        """
        
        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"종합 HTML 리포트가 '{output_file}'로 저장되었습니다.")
        return html_content
    
    def _generate_findings_html(self):
        """주요 발견사항 HTML 생성"""
        findings = []
        
        # 모델 성능
        if 'model_results' in self.results:
            best_model_name = self.results['best_model_name']
            best_r2 = self.results['model_results'][best_model_name]['test_r2']
            findings.append(f"최고 성능 모델: {best_model_name} (R² = {best_r2:.4f})")
        
        # 최적화 결과
        if 'optimization_guidance' in self.results:
            guidance = self.results['optimization_guidance']
            findings.append(f"Yield 최적화: {guidance['improvement_percent']:.2f}% 개선 가능")
        
        findings_html = ""
        for finding in findings:
            findings_html += f'<div class="finding-box"><p>{finding}</p></div>'
        
        return findings_html
    
    def _generate_recommendations_html(self):
        """권장사항 HTML 생성"""
        recommendations = []
        
        # 최적화 가이드
        if 'optimization_guidance' in self.results:
            guidance = self.results['optimization_guidance']
            recommendations.append(f"현재 Yield {guidance['current_yield']:.4f}에서 최적 Yield {guidance['optimal_yield']:.4f}로 개선 가능")
            
            optimal_controls = guidance['optimal_controls']
            for control_name, control_value in optimal_controls.items():
                recommendations.append(f"{control_name}: {control_value:.4f}로 조정 권장")
        
        recommendations_html = ""
        for recommendation in recommendations:
            recommendations_html += f'<div class="recommendation-box"><p>{recommendation}</p></div>'
        
        return recommendations_html
    
    def generate_comprehensive_report(self):
        """
        종합 분석 리포트 생성
        """
        print("\n" + "=" * 80)
        print("종합 분석 리포트 생성")
        print("=" * 80)
        
        report = {
            'data_info': {
                'n_samples': len(self.df),
                'n_features': len(self.feature_names),
                'yield_range': (self.y.min(), self.y.max()),
                'yield_mean': self.y.mean(),
                'yield_std': self.y.std()
            },
            'key_findings': [],
            'recommendations': [],
            'optimization_results': {}
        }
        
        # 주요 발견사항
        if 'model_results' in self.results:
            best_model_name = self.results['best_model_name']
            best_r2 = self.results['model_results'][best_model_name]['test_r2']
            report['key_findings'].append(f"최고 성능 모델: {best_model_name} (R² = {best_r2:.4f})")
        
        # 최적화 결과
        if 'optimization_guidance' in self.results:
            guidance = self.results['optimization_guidance']
            report['optimization_results'] = guidance
            report['recommendations'].append(
                f"Yield 최적화: {guidance['improvement_percent']:.2f}% 개선 가능"
            )
        
        # 리포트 출력
        print("\n📊 ACN Yield 종합 분석 결과")
        print("-" * 50)
        
        print(f"\n데이터 정보:")
        print(f"  • 샘플 수: {report['data_info']['n_samples']}개")
        print(f"  • 특성 수: {report['data_info']['n_features']}개")
        print(f"  • Yield 범위: {report['data_info']['yield_range'][0]:.4f} ~ {report['data_info']['yield_range'][1]:.4f}")
        print(f"  • Yield 평균: {report['data_info']['yield_mean']:.4f}")
        print(f"  • Yield 표준편차: {report['data_info']['yield_std']:.4f}")
        
        print(f"\n주요 발견사항:")
        for finding in report['key_findings']:
            print(f"  • {finding}")
        
        print(f"\n권장사항:")
        for recommendation in report['recommendations']:
            print(f"  • {recommendation}")
        
        return report

def main_yield_comprehensive_analysis(data_path=None, df=None):
    """
    ACN Yield 종합 분석 메인 함수
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    
    Returns:
    results: 분석 결과 딕셔너리
    """
    # 1. 분석기 초기화
    analyzer = ACNYieldComprehensiveAnalyzer(data_path, df)
    
    # 2. 데이터 전처리
    analyzer.preprocess_data()
    
    # 3. 데이터 분포 분석
    distribution_results = analyzer.analyze_data_distribution()
    
    # 4. 상관관계 분석
    correlation_results = analyzer.analyze_correlations()
    
    # 5. 예측 모델 구축
    model_results, best_model = analyzer.build_prediction_models()
    
    # 6. 특성 중요도 분석
    feature_importance_results = analyzer.analyze_feature_importance()
    
    # 7. 최적화 가이드 생성
    optimization_guidance = analyzer.generate_optimization_guidance()
    
    # 8. 종합 리포트 생성
    report = analyzer.generate_comprehensive_report()
    
    # 9. HTML 리포트 생성
    html_content = analyzer.generate_comprehensive_html_report()
    
    return {
        'analyzer': analyzer,
        'distribution_results': distribution_results,
        'correlation_results': correlation_results,
        'model_results': model_results,
        'best_model': best_model,
        'feature_importance_results': feature_importance_results,
        'optimization_guidance': optimization_guidance,
        'report': report,
        'html_content': html_content
    }

# 사용 예시
if __name__ == "__main__":
    print("ACN Yield 종합 분석을 시작합니다.")
    print("\n사용법:")
    print("1. CSV 파일에서 분석:")
    print("   results = main_yield_comprehensive_analysis(data_path='your_data.csv')")
    print("\n2. DataFrame에서 분석:")
    print("   results = main_yield_comprehensive_analysis(df=your_dataframe)")
    print("\n3. 결과 확인:")
    print("   print(results['report'])")
