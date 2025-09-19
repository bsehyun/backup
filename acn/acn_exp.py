import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f, t
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.inspection import partial_dependence
import warnings
warnings.filterwarnings('ignore')

# 고급 분석 라이브러리
try:
    import SALib
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    SOBOL_AVAILABLE = True
except ImportError:
    SOBOL_AVAILABLE = False
    print("SALib가 설치되지 않았습니다. pip install SALib로 설치하세요.")

try:
    from statsmodels.stats.anova import anova_lm
    from statsmodels.formula.api import ols
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("statsmodels가 설치되지 않았습니다. pip install statsmodels로 설치하세요.")

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ACNAdvancedExperiments:
    """
    ACN 정제 공정 고급 통계 분석 클래스
    - Partial Regression Plot
    - ANOVA 분석
    - Sobol 민감도 분석
    - RSM (Response Surface Methodology)
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
        self.model = None
        self.results = {}
        
        print("ACN 고급 실험 분석기 초기화 완료")
    
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
        
        # 2. 품질값 정규화 (spec 기준)
        quality_columns = ['AN-10_200nm', 'AN-10_225nm', 'AN-10_250nm', 
                          'AN-50_200nm', 'AN-50_225nm', 'AN-50_250nm']
        
        for col in quality_columns:
            if col in self.df.columns:
                # 0을 기준으로 정규화 (실제 spec 값에 따라 조정 필요)
                self.df[f'{col}_normalized'] = self.df[col] - 0
        
        # 3. 수치형 변수만 선택 (Yield 제외)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Yield' in numeric_cols:
            numeric_cols.remove('Yield')
        
        # 결측치가 있는 컬럼 제외
        numeric_cols = [col for col in numeric_cols if not self.df[col].isnull().all()]
        
        # 4. 결측치 처리
        self.X = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        self.y = self.df['Yield'].fillna(self.df['Yield'].median())
        
        # 5. 특성 스케일링
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.feature_names = numeric_cols
        
        print(f"분석 대상 특성 수: {len(self.feature_names)}")
        print(f"분석 대상 샘플 수: {len(self.X)}")
        
        return self.X, self.y
    
    def partial_regression_analysis(self):
        """
        Partial Regression Plot 분석
        """
        print("\n" + "=" * 80)
        print("Partial Regression Plot 분석")
        print("=" * 80)
        
        if self.X is None or self.y is None:
            print("데이터가 전처리되지 않았습니다. preprocess_data()를 먼저 실행하세요.")
            return None
        
        # 선형 회귀 모델 훈련
        self.model = LinearRegression()
        self.model.fit(self.X_scaled, self.y)
        
        # Partial Regression Plot 생성
        n_features = min(12, len(self.feature_names))  # 상위 12개 특성만 분석
        top_features = self._get_top_features(n_features)
        
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.ravel()
        
        partial_results = {}
        
        for i, feature in enumerate(top_features):
            if i >= 12:
                break
                
            # Partial regression 계산
            partial_x, partial_y = self._calculate_partial_regression(feature)
            
            # 산점도 그리기
            axes[i].scatter(partial_x, partial_y, alpha=0.6)
            
            # 회귀선 추가
            z = np.polyfit(partial_x, partial_y, 1)
            p = np.poly1d(z)
            axes[i].plot(partial_x, p(partial_x), "r--", alpha=0.8)
            
            # 상관계수 계산
            corr, p_value = stats.pearsonr(partial_x, partial_y)
            
            axes[i].set_xlabel(f'Partial {feature}')
            axes[i].set_ylabel('Partial Yield')
            axes[i].set_title(f'{feature}\nCorr: {corr:.3f}, p: {p_value:.3f}')
            axes[i].grid(True, alpha=0.3)
            
            partial_results[feature] = {
                'correlation': corr,
                'p_value': p_value,
                'partial_x': partial_x,
                'partial_y': partial_y
            }
        
        # 빈 subplot 제거
        for i in range(len(top_features), 12):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        self.results['partial_regression'] = partial_results
        
        # 결과 요약
        print("\nPartial Regression 분석 결과 (상위 10개):")
        partial_summary = pd.DataFrame([
            {'feature': feat, 'correlation': result['correlation'], 'p_value': result['p_value']}
            for feat, result in partial_results.items()
        ]).sort_values('correlation', key=abs, ascending=False)
        
        print(partial_summary.head(10).round(4))
        
        return partial_results
    
    def _get_top_features(self, n_features):
        """상위 n개 특성 선택"""
        # 모델 계수 기준으로 상위 특성 선택
        if self.model is not None:
            feature_importance = np.abs(self.model.coef_)
            top_indices = np.argsort(feature_importance)[-n_features:][::-1]
            return [self.feature_names[i] for i in top_indices]
        else:
            # 상관관계 기준으로 선택
            correlations = []
            for i, feature in enumerate(self.feature_names):
                corr, _ = stats.pearsonr(self.X.iloc[:, i], self.y)
                correlations.append((feature, abs(corr)))
            
            correlations.sort(key=lambda x: x[1], reverse=True)
            return [feat for feat, _ in correlations[:n_features]]
    
    def _calculate_partial_regression(self, target_feature):
        """Partial regression 계산"""
        target_idx = self.feature_names.index(target_feature)
        
        # 다른 특성들로 모델 훈련
        other_features = [i for i in range(len(self.feature_names)) if i != target_idx]
        X_other = self.X_scaled[:, other_features]
        
        # Yield를 다른 특성들로 예측
        model_other = LinearRegression()
        model_other.fit(X_other, self.y)
        y_pred_other = model_other.predict(X_other)
        
        # Target 특성을 다른 특성들로 예측
        model_target = LinearRegression()
        model_target.fit(X_other, self.X_scaled[:, target_idx])
        x_pred_other = model_target.predict(X_other)
        
        # Partial residuals 계산
        partial_y = self.y - y_pred_other
        partial_x = self.X_scaled[:, target_idx] - x_pred_other
        
        return partial_x, partial_y
    
    def anova_analysis(self):
        """
        ANOVA 분석
        """
        print("\n" + "=" * 80)
        print("ANOVA 분석")
        print("=" * 80)
        
        if not STATSMODELS_AVAILABLE:
            print("statsmodels가 설치되지 않아 ANOVA 분석을 수행할 수 없습니다.")
            return None
        
        if self.X is None or self.y is None:
            print("데이터가 전처리되지 않았습니다. preprocess_data()를 먼저 실행하세요.")
            return None
        
        # 상위 10개 특성만 선택
        top_features = self._get_top_features(10)
        
        # OLS 모델 생성
        formula = 'Yield ~ ' + ' + '.join(top_features)
        
        # 데이터 준비
        anova_data = self.df[top_features + ['Yield']].copy()
        
        try:
            # OLS 모델 피팅
            model = ols(formula, data=anova_data).fit()
            
            # ANOVA 테이블 생성
            anova_table = anova_lm(model, typ=2)
            
            print("ANOVA 분석 결과:")
            print(anova_table.round(4))
            
            # F-통계량과 p-value 분석
            significant_features = anova_table[anova_table['PR(>F)'] < 0.05].sort_values('F', ascending=False)
            
            print(f"\n유의한 특성 (p < 0.05): {len(significant_features)}개")
            if len(significant_features) > 0:
                print(significant_features[['F', 'PR(>F)']].round(4))
            
            # 모델 요약
            print(f"\n모델 요약:")
            print(f"R-squared: {model.rsquared:.4f}")
            print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
            print(f"F-statistic: {model.fvalue:.4f}")
            print(f"F p-value: {model.f_pvalue:.4f}")
            
            # VIF (Variance Inflation Factor) 계산
            vif_data = pd.DataFrame()
            vif_data["Feature"] = top_features
            vif_data["VIF"] = [variance_inflation_factor(anova_data[top_features].values, i) 
                              for i in range(len(top_features))]
            
            print(f"\nVIF (Variance Inflation Factor):")
            print(vif_data.sort_values('VIF', ascending=False).round(4))
            
            # 다중공선성 경고
            high_vif = vif_data[vif_data['VIF'] > 10]
            if len(high_vif) > 0:
                print(f"\n⚠️ 다중공선성 경고: VIF > 10인 특성 {len(high_vif)}개")
                print(high_vif[['Feature', 'VIF']].round(2))
            
            anova_results = {
                'anova_table': anova_table,
                'significant_features': significant_features,
                'model_summary': {
                    'r_squared': model.rsquared,
                    'adj_r_squared': model.rsquared_adj,
                    'f_statistic': model.fvalue,
                    'f_pvalue': model.f_pvalue
                },
                'vif': vif_data,
                'high_vif_features': high_vif
            }
            
            self.results['anova'] = anova_results
            return anova_results
            
        except Exception as e:
            print(f"ANOVA 분석 중 오류 발생: {str(e)}")
            return None
    
    def sobol_sensitivity_analysis(self):
        """
        Sobol 민감도 분석
        """
        print("\n" + "=" * 80)
        print("Sobol 민감도 분석")
        print("=" * 80)
        
        if not SOBOL_AVAILABLE:
            print("SALib가 설치되지 않아 Sobol 민감도 분석을 수행할 수 없습니다.")
            return None
        
        if self.X is None or self.y is None:
            print("데이터가 전처리되지 않았습니다. preprocess_data()를 먼저 실행하세요.")
            return None
        
        # 상위 8개 특성만 선택 (Sobol 분석은 계산량이 많음)
        top_features = self._get_top_features(8)
        X_selected = self.X[top_features]
        
        # 문제 정의
        problem = {
            'num_vars': len(top_features),
            'names': top_features,
            'bounds': [[X_selected[feat].min(), X_selected[feat].max()] for feat in top_features]
        }
        
        # Saltelli 샘플링
        param_values = saltelli.sample(problem, 1000)  # N=1000
        
        print(f"Sobol 샘플링 완료: {param_values.shape[0]}개 샘플")
        
        # 모델 예측
        try:
            # 선형 회귀 모델로 예측
            model = LinearRegression()
            model.fit(X_selected, self.y)
            Y = model.predict(param_values)
            
            # Sobol 분석
            Si = sobol.analyze(problem, Y)
            
            # 결과 정리
            sobol_results = {
                'first_order': pd.DataFrame({
                    'feature': top_features,
                    'S1': Si['S1'],
                    'S1_conf': Si['S1_conf']
                }).sort_values('S1', ascending=False),
                
                'total_order': pd.DataFrame({
                    'feature': top_features,
                    'ST': Si['ST'],
                    'ST_conf': Si['ST_conf']
                }).sort_values('ST', ascending=False),
                
                'second_order': Si['S2'] if 'S2' in Si else None
            }
            
            print("Sobol 민감도 분석 결과:")
            print("\n1차 민감도 지수 (S1):")
            print(sobol_results['first_order'].round(4))
            
            print("\n총 민감도 지수 (ST):")
            print(sobol_results['total_order'].round(4))
            
            # 시각화
            self._plot_sobol_results(sobol_results)
            
            self.results['sobol'] = sobol_results
            return sobol_results
            
        except Exception as e:
            print(f"Sobol 분석 중 오류 발생: {str(e)}")
            return None
    
    def _plot_sobol_results(self, sobol_results):
        """Sobol 결과 시각화"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1차 민감도 지수
        first_order = sobol_results['first_order']
        ax1.barh(range(len(first_order)), first_order['S1'])
        ax1.set_yticks(range(len(first_order)))
        ax1.set_yticklabels(first_order['feature'])
        ax1.set_xlabel('1차 민감도 지수 (S1)')
        ax1.set_title('Sobol 1차 민감도 지수')
        ax1.grid(True, alpha=0.3)
        
        # 총 민감도 지수
        total_order = sobol_results['total_order']
        ax2.barh(range(len(total_order)), total_order['ST'])
        ax2.set_yticks(range(len(total_order)))
        ax2.set_yticklabels(total_order['feature'])
        ax2.set_xlabel('총 민감도 지수 (ST)')
        ax2.set_title('Sobol 총 민감도 지수')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def response_surface_methodology(self):
        """
        RSM (Response Surface Methodology) 분석
        """
        print("\n" + "=" * 80)
        print("RSM (Response Surface Methodology) 분석")
        print("=" * 80)
        
        if self.X is None or self.y is None:
            print("데이터가 전처리되지 않았습니다. preprocess_data()를 먼저 실행하세요.")
            return None
        
        # 상위 4개 특성만 선택 (RSM은 차원이 높아지면 복잡해짐)
        top_features = self._get_top_features(4)
        X_selected = self.X[top_features]
        
        print(f"RSM 분석 대상 특성: {top_features}")
        
        # 1차 모델 (선형)
        linear_model = LinearRegression()
        linear_model.fit(X_selected, self.y)
        linear_r2 = r2_score(self.y, linear_model.predict(X_selected))
        
        # 2차 모델 (이차항 포함)
        poly_features = PolynomialFeatures(degree=2, include_bias=False)
        X_poly = poly_features.fit_transform(X_selected)
        
        poly_model = LinearRegression()
        poly_model.fit(X_poly, self.y)
        poly_r2 = r2_score(self.y, poly_model.predict(X_poly))
        
        print(f"1차 모델 R²: {linear_r2:.4f}")
        print(f"2차 모델 R²: {poly_r2:.4f}")
        
        # 모델 비교
        if poly_r2 > linear_r2 + 0.05:  # 2차 모델이 5% 이상 개선
            print("2차 모델이 더 적합합니다.")
            best_model = poly_model
            best_features = X_poly
            model_type = "2차"
        else:
            print("1차 모델이 더 적합합니다.")
            best_model = linear_model
            best_features = X_selected
            model_type = "1차"
        
        # 교차 검증
        cv_scores = cross_val_score(best_model, best_features, self.y, cv=5, scoring='r2')
        print(f"교차 검증 R² (평균 ± 표준편차): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # 잔차 분석
        y_pred = best_model.predict(best_features)
        residuals = self.y - y_pred
        
        # RSM 시각화
        self._plot_rsm_results(top_features, X_selected, self.y, y_pred, residuals, model_type)
        
        # 최적점 찾기 (간단한 그리드 서치)
        optimal_point = self._find_optimal_point(top_features, best_model, poly_features if model_type == "2차" else None)
        
        rsm_results = {
            'linear_model': linear_model,
            'poly_model': poly_model,
            'best_model': best_model,
            'model_type': model_type,
            'linear_r2': linear_r2,
            'poly_r2': poly_r2,
            'cv_scores': cv_scores,
            'optimal_point': optimal_point,
            'features': top_features
        }
        
        self.results['rsm'] = rsm_results
        return rsm_results
    
    def _plot_rsm_results(self, features, X, y, y_pred, residuals, model_type):
        """RSM 결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 예측 vs 실제
        axes[0, 0].scatter(y, y_pred, alpha=0.6)
        axes[0, 0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('실제 Yield')
        axes[0, 0].set_ylabel('예측 Yield')
        axes[0, 0].set_title(f'예측 vs 실제 ({model_type} 모델)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 잔차 플롯
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('예측 Yield')
        axes[0, 1].set_ylabel('잔차')
        axes[0, 1].set_title('잔차 플롯')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 잔차 히스토그램
        axes[0, 2].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 2].set_xlabel('잔차')
        axes[0, 2].set_ylabel('빈도')
        axes[0, 2].set_title('잔차 분포')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4-6. 각 특성별 Yield 관계 (상위 3개)
        for i in range(min(3, len(features))):
            feature = features[i]
            axes[1, i].scatter(X[feature], y, alpha=0.6, label='실제')
            axes[1, i].scatter(X[feature], y_pred, alpha=0.6, label='예측')
            axes[1, i].set_xlabel(feature)
            axes[1, i].set_ylabel('Yield')
            axes[1, i].set_title(f'{feature} vs Yield')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _find_optimal_point(self, features, model, poly_features=None):
        """최적점 찾기 (간단한 그리드 서치)"""
        print("\n최적점 탐색 중...")
        
        # 각 특성의 범위 정의
        bounds = []
        for feature in features:
            min_val = self.X[feature].min()
            max_val = self.X[feature].max()
            bounds.append((min_val, max_val))
        
        # 간단한 그리드 서치 (각 특성을 10개 구간으로 나눔)
        best_yield = -np.inf
        best_point = None
        
        # 2차 모델인 경우 다항식 특성 변환 필요
        if poly_features is not None:
            # 2차 모델의 경우 복잡하므로 간단한 탐색
            n_samples = 1000
            random_points = []
            for i, (min_val, max_val) in enumerate(bounds):
                random_points.append(np.random.uniform(min_val, max_val, n_samples))
            
            X_test = np.column_stack(random_points)
            X_test_poly = poly_features.transform(X_test)
            y_test = model.predict(X_test_poly)
            
            best_idx = np.argmax(y_test)
            best_point = X_test[best_idx]
            best_yield = y_test[best_idx]
        else:
            # 1차 모델인 경우
            n_samples = 1000
            random_points = []
            for i, (min_val, max_val) in enumerate(bounds):
                random_points.append(np.random.uniform(min_val, max_val, n_samples))
            
            X_test = np.column_stack(random_points)
            y_test = model.predict(X_test)
            
            best_idx = np.argmax(y_test)
            best_point = X_test[best_idx]
            best_yield = y_test[best_idx]
        
        optimal_result = {
            'optimal_point': dict(zip(features, best_point)),
            'predicted_yield': best_yield,
            'features': features
        }
        
        print("최적점 탐색 결과:")
        for feature, value in optimal_result['optimal_point'].items():
            print(f"  {feature}: {value:.4f}")
        print(f"  예측 Yield: {best_yield:.4f}")
        
        return optimal_result
    
    def generate_comprehensive_report(self):
        """
        종합 실험 리포트 생성
        """
        print("\n" + "=" * 80)
        print("종합 실험 리포트 생성")
        print("=" * 80)
        
        report = {
            'data_info': {
                'n_samples': len(self.X) if self.X is not None else 0,
                'n_features': len(self.feature_names) if self.feature_names is not None else 0,
                'features': self.feature_names[:10] if self.feature_names else []  # 상위 10개만
            },
            'experiments': {}
        }
        
        # Partial Regression 결과
        if 'partial_regression' in self.results:
            partial_data = self.results['partial_regression']
            top_partial = sorted(partial_data.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)[:5]
            
            report['experiments']['partial_regression'] = {
                'top_features': [{'feature': feat, 'correlation': result['correlation'], 'p_value': result['p_value']} 
                               for feat, result in top_partial],
                'summary': f"상위 5개 특성의 평균 절댓값 상관계수: {np.mean([abs(result['correlation']) for _, result in top_partial]):.4f}"
            }
        
        # ANOVA 결과
        if 'anova' in self.results:
            anova_data = self.results['anova']
            report['experiments']['anova'] = {
                'significant_features': len(anova_data['significant_features']),
                'r_squared': anova_data['model_summary']['r_squared'],
                'high_vif_count': len(anova_data['high_vif_features']),
                'summary': f"유의한 특성 {len(anova_data['significant_features'])}개, R² = {anova_data['model_summary']['r_squared']:.4f}"
            }
        
        # Sobol 결과
        if 'sobol' in self.results:
            sobol_data = self.results['sobol']
            top_sobol = sobol_data['first_order'].head(3)
            
            report['experiments']['sobol'] = {
                'top_sensitive_features': [{'feature': row['feature'], 's1': row['S1'], 'st': row['ST']} 
                                         for _, row in top_sobol.iterrows()],
                'summary': f"상위 3개 민감 특성의 평균 1차 민감도: {top_sobol['S1'].mean():.4f}"
            }
        
        # RSM 결과
        if 'rsm' in self.results:
            rsm_data = self.results['rsm']
            report['experiments']['rsm'] = {
                'model_type': rsm_data['model_type'],
                'r_squared': rsm_data['poly_r2'] if rsm_data['model_type'] == '2차' else rsm_data['linear_r2'],
                'cv_score': rsm_data['cv_scores'].mean(),
                'optimal_yield': rsm_data['optimal_point']['predicted_yield'],
                'summary': f"{rsm_data['model_type']} 모델, R² = {rsm_data['poly_r2'] if rsm_data['model_type'] == '2차' else rsm_data['linear_r2']:.4f}"
            }
        
        # 리포트 출력
        print("\n📊 종합 실험 결과 요약")
        print("-" * 50)
        
        for exp_name, exp_data in report['experiments'].items():
            print(f"\n{exp_name.upper()}:")
            print(f"  {exp_data['summary']}")
        
        # 주요 권장사항
        print("\n💡 주요 권장사항:")
        
        if 'partial_regression' in report['experiments']:
            top_feature = report['experiments']['partial_regression']['top_features'][0]['feature']
            print(f"  1. Partial Regression에서 가장 중요한 특성: {top_feature}")
        
        if 'sobol' in report['experiments']:
            top_sensitive = report['experiments']['sobol']['top_sensitive_features'][0]['feature']
            print(f"  2. Sobol 민감도 분석에서 가장 민감한 특성: {top_sensitive}")
        
        if 'rsm' in report['experiments']:
            optimal_yield = report['experiments']['rsm']['optimal_yield']
            print(f"  3. RSM 최적화로 예측 가능한 최대 Yield: {optimal_yield:.4f}")
        
        return report

def main_advanced_experiments(data_path=None, df=None):
    """
    ACN 고급 실험 분석 메인 함수
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    
    Returns:
    results: 분석 결과 딕셔너리
    """
    # 1. 분석기 초기화
    analyzer = ACNAdvancedExperiments(data_path, df)
    
    # 2. 데이터 전처리
    analyzer.preprocess_data()
    
    # 3. Partial Regression 분석
    partial_results = analyzer.partial_regression_analysis()
    
    # 4. ANOVA 분석
    anova_results = analyzer.anova_analysis()
    
    # 5. Sobol 민감도 분석
    sobol_results = analyzer.sobol_sensitivity_analysis()
    
    # 6. RSM 분석
    rsm_results = analyzer.response_surface_methodology()
    
    # 7. 종합 리포트 생성
    report = analyzer.generate_comprehensive_report()
    
    return {
        'analyzer': analyzer,
        'partial_regression': partial_results,
        'anova': anova_results,
        'sobol': sobol_results,
        'rsm': rsm_results,
        'report': report
    }

# 사용 예시
if __name__ == "__main__":
    print("ACN 정제 공정 고급 실험 분석을 시작합니다.")
    print("\n사용법:")
    print("1. CSV 파일에서 분석:")
    print("   results = main_advanced_experiments(data_path='your_data.csv')")
    print("\n2. DataFrame에서 분석:")
    print("   results = main_advanced_experiments(df=your_dataframe)")
    print("\n3. 결과 확인:")
    print("   print(results['report']['experiments'])")
    print("\n필요한 라이브러리:")
    print("   pip install SALib statsmodels")
