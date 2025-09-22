"""
ACN 정제 공정 Product 최적화 분석
- Input_source 조건에서 높은 Product 생산을 위한 분석
- 다중공선성 문제 해결 (Input_source, Product, Yield 분리)
- Product 최적화에 집중한 분석
- 시각화 중심의 분석 결과 제공
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

class ACNProductOptimizer:
    """
    ACN 정제 공정 Product 최적화 분석기
    - Input_source 조건에서 높은 Product 생산을 위한 분석
    - 다중공선성 문제 해결 (Input_source, Product, Yield 분리)
    - Product 최적화에 집중한 분석
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
        
        print("ACN Product 최적화 분석기 초기화 완료")
    
    def preprocess_data(self):
        """
        데이터 전처리 - 다중공선성 문제 해결
        """
        print("=" * 80)
        print("데이터 전처리 - 다중공선성 문제 해결")
        print("=" * 80)
        
        # 1. 최종 F/R Level에서 분석한 데이터만 필터링
        if 'Final_FR' in self.df.columns:
            max_fr_level = self.df['Final_FR'].max()
            self.df = self.df[self.df['Final_FR'] == max_fr_level].copy()
            print(f"최종 F/R Level 필터링 후 데이터 크기: {self.df.shape}")
        
        # 2. 다중공선성 문제 해결을 위한 변수 분리
        # Input_source, Product, Yield는 함께 사용하지 않음
        
        # 3. Product를 target으로 설정 (Yield 제외)
        if 'Product' in self.df.columns:
            self.y = self.df['Product'].fillna(self.df['Product'].median())
        elif 'Output' in self.df.columns:
            self.y = self.df['Output'].fillna(self.df['Output'].median())
        else:
            raise ValueError("Product 또는 Output 컬럼이 필요합니다.")
        
        # 4. Input_source와 Yield를 제외한 수치형 변수만 선택
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['Input_source', 'Yield', 'Product', 'Output']  # 다중공선성 방지
        
        # 품질값 컬럼들도 제외 (선택사항)
        quality_cols = ['AN-10_200nm', 'AN-10_225nm', 'AN-10_250nm', 
                       'AN-50_200nm', 'AN-50_225nm', 'AN-50_250nm']
        exclude_cols.extend(quality_cols)
        
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # 결측치가 있는 컬럼 제외
        numeric_cols = [col for col in numeric_cols if not self.df[col].isnull().all()]
        
        # 5. 결측치 처리
        self.X = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        
        # 6. 특성 스케일링
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.feature_names = numeric_cols
        
        print(f"분석 대상 특성 수: {len(self.feature_names)}")
        print(f"분석 대상 샘플 수: {len(self.X)}")
        print(f"Target 변수: {'Product' if 'Product' in self.df.columns else 'Output'}")
        print(f"Target 범위: {self.y.min():.4f} ~ {self.y.max():.4f}")
        print(f"제외된 변수 (다중공선성 방지): {exclude_cols}")
        
        return self.X, self.y
    
    def analyze_multicollinearity(self):
        """
        다중공선성 분석
        """
        print("\n" + "=" * 80)
        print("다중공선성 분석")
        print("=" * 80)
        
        # Input_source, Product, Yield 간의 상관관계 분석
        multicollinearity_vars = []
        
        if 'Input_source' in self.df.columns:
            multicollinearity_vars.append('Input_source')
        if 'Product' in self.df.columns:
            multicollinearity_vars.append('Product')
        elif 'Output' in self.df.columns:
            multicollinearity_vars.append('Output')
        if 'Yield' in self.df.columns:
            multicollinearity_vars.append('Yield')
        
        if len(multicollinearity_vars) >= 2:
            corr_matrix = self.df[multicollinearity_vars].corr()
            print("다중공선성 변수들 간의 상관관계:")
            print(corr_matrix.round(4))
            
            # 높은 상관관계 확인 (|r| > 0.8)
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.8:
                        high_corr_pairs.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_val
                        })
            
            if high_corr_pairs:
                print(f"\n⚠️ 높은 상관관계 (|r| > 0.8) 발견: {len(high_corr_pairs)}개")
                for pair in high_corr_pairs:
                    print(f"  {pair['var1']} vs {pair['var2']}: r = {pair['correlation']:.4f}")
            else:
                print("\n✅ 높은 상관관계 없음 (|r| ≤ 0.8)")
        
        return corr_matrix if len(multicollinearity_vars) >= 2 else None
    
    def build_product_prediction_model(self):
        """
        Product 예측 모델 구축
        """
        print("\n" + "=" * 80)
        print("Product 예측 모델 구축")
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
        axes[1, 0].set_xlabel('Actual Product')
        axes[1, 0].set_ylabel('Predicted Product')
        axes[1, 0].set_title(f'Prediction vs Actual ({best_model_name})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 잔차 플롯
        residuals = y_test - y_pred
        axes[1, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Product')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residual Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def analyze_input_source_impact(self):
        """
        Input_source가 Product에 미치는 영향 분석
        """
        print("\n" + "=" * 80)
        print("Input_source가 Product에 미치는 영향 분석")
        print("=" * 80)
        
        if 'Input_source' not in self.df.columns:
            print("Input_source 컬럼이 없습니다.")
            return None
        
        # Input_source와 Product 간의 상관관계
        corr, p_value = stats.pearsonr(self.df['Input_source'], self.y)
        print(f"Input_source vs Product 상관계수: r = {corr:.4f}, p = {p_value:.4f}")
        
        # Input_source 구간별 Product 분석
        input_source = self.df['Input_source']
        product = self.y
        
        # Input_source를 5개 구간으로 나누기
        input_quantiles = input_source.quantile([0.2, 0.4, 0.6, 0.8])
        input_bins = pd.cut(input_source, 
                           bins=[input_source.min(), input_quantiles[0.2], input_quantiles[0.4], 
                                input_quantiles[0.6], input_quantiles[0.8], input_source.max()],
                           labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        
        # 구간별 Product 통계
        product_by_input = pd.DataFrame({
            'Input_Source_Bin': input_bins,
            'Product': product
        }).groupby('Input_Source_Bin')['Product'].agg([
            'count', 'mean', 'std', 'min', 'max'
        ]).round(2)
        
        print("\nInput_source 구간별 Product 통계:")
        print(product_by_input)
        
        # 시각화
        self._plot_input_source_impact(input_source, product, input_bins)
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'product_by_input': product_by_input
        }
    
    def _plot_input_source_impact(self, input_source, product, input_bins):
        """
        Input_source 영향 시각화
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Input_source vs Product 산점도
        axes[0, 0].scatter(input_source, product, alpha=0.6)
        # 회귀선 추가
        z = np.polyfit(input_source, product, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(input_source, p(input_source), "r--", alpha=0.8)
        axes[0, 0].set_xlabel('Input_source')
        axes[0, 0].set_ylabel('Product')
        axes[0, 0].set_title('Input_source vs Product')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Input_source 구간별 Product 분포 (Box plot)
        product_by_bin = [product[input_bins == bin_name] for bin_name in input_bins.cat.categories]
        axes[0, 1].boxplot(product_by_bin, labels=input_bins.cat.categories)
        axes[0, 1].set_xlabel('Input_source 구간')
        axes[0, 1].set_ylabel('Product')
        axes[0, 1].set_title('Input_source 구간별 Product 분포')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Input_source 구간별 평균 Product
        mean_product_by_bin = [product[input_bins == bin_name].mean() for bin_name in input_bins.cat.categories]
        axes[1, 0].bar(input_bins.cat.categories, mean_product_by_bin, alpha=0.8, color='green')
        axes[1, 0].set_xlabel('Input_source 구간')
        axes[1, 0].set_ylabel('평균 Product')
        axes[1, 0].set_title('Input_source 구간별 평균 Product')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Input_source 히스토그램
        axes[1, 1].hist(input_source, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Input_source')
        axes[1, 1].set_ylabel('빈도')
        axes[1, 1].set_title('Input_source 분포')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_optimization_guidance(self):
        """
        Product 최적화 가이드 생성
        """
        print("\n" + "=" * 80)
        print("Product 최적화 가이드 생성")
        print("=" * 80)
        
        if 'best_model' not in self.results:
            print("모델이 구축되지 않았습니다. build_product_prediction_model()을 먼저 실행하세요.")
            return None
        
        best_model = self.results['best_model']
        
        # 현재 평균 조건에서의 예측 Product
        current_conditions = self.X.mean().values.reshape(1, -1)
        current_conditions_scaled = self.scaler.transform(current_conditions)
        current_product = best_model.predict(current_conditions_scaled)[0]
        
        print(f"현재 평균 조건에서 예측 Product: {current_product:.4f}")
        
        # Product 최대화를 위한 최적 Control 값 찾기
        optimal_controls = self._find_optimal_controls_for_product_maximization(best_model, current_conditions)
        
        print(f"\nProduct 최대화를 위한 최적 Control 값:")
        for control_name, control_value in optimal_controls.items():
            print(f"  {control_name}: {control_value:.4f}")
        
        # 최적 조건에서의 예측 Product
        optimal_conditions = current_conditions.copy()
        for i, (feature_name, control_value) in enumerate(optimal_controls.items()):
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                optimal_conditions[0, feature_idx] = control_value
        
        optimal_conditions_scaled = self.scaler.transform(optimal_conditions)
        optimal_product = best_model.predict(optimal_conditions_scaled)[0]
        
        print(f"\n최적 조건에서 예측 Product: {optimal_product:.4f}")
        print(f"Product 개선 효과: {optimal_product - current_product:.4f} ({(optimal_product/current_product - 1)*100:.2f}% 증가)")
        
        guidance_results = {
            'current_product': current_product,
            'optimal_product': optimal_product,
            'improvement': optimal_product - current_product,
            'improvement_percent': (optimal_product/current_product - 1)*100,
            'optimal_controls': optimal_controls
        }
        
        self.results['optimization_guidance'] = guidance_results
        return guidance_results
    
    def _find_optimal_controls_for_product_maximization(self, model, current_conditions):
        """
        Product 최대화를 위한 최적 Control 값 찾기
        """
        # 최적화 함수 정의 (Product 최대화)
        def objective(controls):
            # 현재 조건을 복사하고 Control 값들을 업데이트
            new_conditions = current_conditions.copy()
            
            # Control 값들 업데이트
            for i, (feature_name, control_value) in enumerate(controls.items()):
                if feature_name in self.feature_names:
                    feature_idx = self.feature_names.index(feature_name)
                    new_conditions[0, feature_idx] = control_value
            
            # 스케일링
            new_conditions_scaled = self.scaler.transform(new_conditions)
            
            # 예측 (Product 최대화를 위해 음수로 반환)
            predicted_product = model.predict(new_conditions_scaled)[0]
            return -predicted_product  # 최대화를 위해 음수 반환
        
        # Control 변수들 (상위 5개 특성)
        control_features = self.feature_names[:5]
        
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
                'product_range': (self.y.min(), self.y.max()) if self.y is not None else (0, 0)
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
                f"Product 최적화: {guidance['improvement_percent']:.2f}% 개선 가능"
            )
        
        # 리포트 출력
        print("\n📊 ACN Product 최적화 분석 결과")
        print("-" * 50)
        
        print(f"\n데이터 정보:")
        print(f"  • 샘플 수: {report['data_info']['n_samples']}개")
        print(f"  • 특성 수: {report['data_info']['n_features']}개")
        print(f"  • Product 범위: {report['data_info']['product_range'][0]:.4f} ~ {report['data_info']['product_range'][1]:.4f}")
        
        print(f"\n주요 발견사항:")
        for finding in report['key_findings']:
            print(f"  • {finding}")
        
        print(f"\n권장사항:")
        for recommendation in report['recommendations']:
            print(f"  • {recommendation}")
        
        return report

def main_product_optimization(data_path=None, df=None):
    """
    ACN Product 최적화 분석 메인 함수
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    
    Returns:
    results: 분석 결과 딕셔너리
    """
    # 1. 분석기 초기화
    optimizer = ACNProductOptimizer(data_path, df)
    
    # 2. 데이터 전처리
    optimizer.preprocess_data()
    
    # 3. 다중공선성 분석
    multicollinearity_results = optimizer.analyze_multicollinearity()
    
    # 4. Product 예측 모델 구축
    model_results, best_model = optimizer.build_product_prediction_model()
    
    # 5. Input_source 영향 분석
    input_impact_results = optimizer.analyze_input_source_impact()
    
    # 6. 최적화 가이드 생성
    optimization_guidance = optimizer.generate_optimization_guidance()
    
    # 7. 종합 리포트 생성
    report = optimizer.generate_comprehensive_report()
    
    return {
        'optimizer': optimizer,
        'multicollinearity_results': multicollinearity_results,
        'model_results': model_results,
        'best_model': best_model,
        'input_impact_results': input_impact_results,
        'optimization_guidance': optimization_guidance,
        'report': report
    }

# 사용 예시
if __name__ == "__main__":
    print("ACN Product 최적화 분석을 시작합니다.")
    print("\n사용법:")
    print("1. CSV 파일에서 분석:")
    print("   results = main_product_optimization(data_path='your_data.csv')")
    print("\n2. DataFrame에서 분석:")
    print("   results = main_product_optimization(df=your_dataframe)")
    print("\n3. 결과 확인:")
    print("   print(results['report'])")
