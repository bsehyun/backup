import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# 머신러닝 라이브러리
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

# SHAP 라이브러리
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP이 설치되지 않았습니다. pip install shap로 설치하세요.")

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ACNMLFeatureImportance:
    """
    ACN 정제 공정 Yield 예측을 위한 머신러닝 기반 Feature Importance 분석 클래스
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
        
        self.models = {}
        self.feature_importance_results = {}
        self.shap_values = {}
        self.scaler = StandardScaler()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
        print("ACN ML Feature Importance 분석기 초기화 완료")
    
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
        X = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        y = self.df['Yield'].fillna(self.df['Yield'].median())
        
        # 5. 데이터 분할
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 6. 특성 스케일링
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.feature_names = numeric_cols
        
        print(f"훈련 데이터 크기: {self.X_train.shape}")
        print(f"테스트 데이터 크기: {self.X_test.shape}")
        print(f"특성 수: {len(self.feature_names)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """
        다양한 머신러닝 모델 훈련
        """
        print("\n" + "=" * 80)
        print("머신러닝 모델 훈련")
        print("=" * 80)
        
        # 모델 정의
        models_config = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.1),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
            'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'MLP': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        }
        
        # 모델 훈련 및 평가
        model_results = {}
        
        for name, model in models_config.items():
            print(f"\n{name} 모델 훈련 중...")
            
            try:
                # 모델 훈련
                if name in ['SVR', 'Ridge', 'Lasso', 'ElasticNet', 'MLP']:
                    model.fit(self.X_train_scaled, self.y_train)
                    y_pred = model.predict(self.X_test_scaled)
                else:
                    model.fit(self.X_train, self.y_train)
                    y_pred = model.predict(self.X_test)
                
                # 성능 평가
                mse = mean_squared_error(self.y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(self.y_test, y_pred)
                mae = mean_absolute_error(self.y_test, y_pred)
                
                # 교차 검증
                if name in ['SVR', 'Ridge', 'Lasso', 'ElasticNet', 'MLP']:
                    cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                              cv=5, scoring='r2')
                else:
                    cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                              cv=5, scoring='r2')
                
                model_results[name] = {
                    'model': model,
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,
                    'mae': mae,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                print(f"  R² Score: {r2:.4f}")
                print(f"  RMSE: {rmse:.4f}")
                print(f"  CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
                
            except Exception as e:
                print(f"  {name} 모델 훈련 실패: {str(e)}")
        
        self.models = model_results
        
        # 최고 성능 모델 선택
        best_model_name = max(model_results.keys(), 
                            key=lambda x: model_results[x]['r2'])
        print(f"\n최고 성능 모델: {best_model_name} (R² = {model_results[best_model_name]['r2']:.4f})")
        
        return model_results
    
    def calculate_feature_importance(self):
        """
        다양한 방법으로 Feature Importance 계산
        """
        print("\n" + "=" * 80)
        print("Feature Importance 계산")
        print("=" * 80)
        
        importance_results = {}
        
        # 1. Tree-based 모델들의 Feature Importance
        print("\n1. Tree-based 모델 Feature Importance")
        print("-" * 50)
        
        tree_models = ['RandomForest', 'GradientBoosting', 'ExtraTrees']
        for model_name in tree_models:
            if model_name in self.models:
                model = self.models[model_name]['model']
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    importance_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    importance_results[f'{model_name}_importance'] = importance_df
                    print(f"{model_name} 상위 5개 특성:")
                    print(importance_df.head().round(4))
        
        # 2. Linear 모델들의 계수
        print("\n2. Linear 모델 계수")
        print("-" * 50)
        
        linear_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']
        for model_name in linear_models:
            if model_name in self.models:
                model = self.models[model_name]['model']
                if hasattr(model, 'coef_'):
                    coef = model.coef_
                    coef_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'coefficient': coef,
                        'abs_coefficient': np.abs(coef)
                    }).sort_values('abs_coefficient', ascending=False)
                    
                    importance_results[f'{model_name}_coefficient'] = coef_df
                    print(f"{model_name} 상위 5개 특성:")
                    print(coef_df.head().round(4))
        
        # 3. Permutation Importance
        print("\n3. Permutation Importance")
        print("-" * 50)
        
        from sklearn.inspection import permutation_importance
        
        # 최고 성능 모델로 permutation importance 계산
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['r2'])
        best_model = self.models[best_model_name]['model']
        
        if best_model_name in ['SVR', 'Ridge', 'Lasso', 'ElasticNet', 'MLP']:
            perm_importance = permutation_importance(
                best_model, self.X_test_scaled, self.y_test, 
                n_repeats=10, random_state=42
            )
        else:
            perm_importance = permutation_importance(
                best_model, self.X_test, self.y_test, 
                n_repeats=10, random_state=42
            )
        
        perm_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        importance_results['permutation_importance'] = perm_df
        print(f"Permutation Importance (기준 모델: {best_model_name}) 상위 5개:")
        print(perm_df.head().round(4))
        
        # 4. SHAP Values (가능한 경우)
        if SHAP_AVAILABLE:
            print("\n4. SHAP Values")
            print("-" * 50)
            
            try:
                # Tree-based 모델에 대해서만 SHAP 계산
                tree_model_name = None
                for name in tree_models:
                    if name in self.models:
                        tree_model_name = name
                        break
                
                if tree_model_name:
                    model = self.models[tree_model_name]['model']
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(self.X_test)
                    
                    # SHAP 중요도 계산
                    shap_importance = np.abs(shap_values).mean(axis=0)
                    shap_df = pd.DataFrame({
                        'feature': self.feature_names,
                        'shap_importance': shap_importance
                    }).sort_values('shap_importance', ascending=False)
                    
                    importance_results['shap_importance'] = shap_df
                    self.shap_values[tree_model_name] = shap_values
                    
                    print(f"SHAP Importance (기준 모델: {tree_model_name}) 상위 5개:")
                    print(shap_df.head().round(4))
                
            except Exception as e:
                print(f"SHAP 계산 실패: {str(e)}")
        
        self.feature_importance_results = importance_results
        return importance_results
    
    def calculate_combined_importance(self):
        """
        모든 방법의 Feature Importance를 종합하여 최종 점수 계산
        """
        print("\n" + "=" * 80)
        print("종합 Feature Importance 계산")
        print("=" * 80)
        
        # 모든 중요도 점수를 정규화하여 종합
        combined_scores = pd.DataFrame({'feature': self.feature_names})
        
        # 각 방법별 점수 추가
        for method, df in self.feature_importance_results.items():
            if 'importance' in method or 'coefficient' in method:
                # 중요도 점수 정규화
                if 'abs_coefficient' in df.columns:
                    score_col = 'abs_coefficient'
                else:
                    score_col = 'importance'
                
                normalized_score = (df[score_col] - df[score_col].min()) / (df[score_col].max() - df[score_col].min())
                
                # feature 이름으로 매칭
                method_scores = pd.DataFrame({
                    'feature': df['feature'],
                    f'{method}_score': normalized_score
                })
                combined_scores = combined_scores.merge(method_scores, on='feature', how='left')
        
        # 결측치를 0으로 채우기
        score_columns = [col for col in combined_scores.columns if col.endswith('_score')]
        combined_scores[score_columns] = combined_scores[score_columns].fillna(0)
        
        # 가중 평균으로 최종 점수 계산
        weights = {
            'RandomForest_importance_score': 0.2,
            'GradientBoosting_importance_score': 0.2,
            'ExtraTrees_importance_score': 0.15,
            'permutation_importance_score': 0.2,
            'shap_importance_score': 0.15,
            'LinearRegression_coefficient_score': 0.05,
            'Ridge_coefficient_score': 0.05
        }
        
        final_score = 0
        for col, weight in weights.items():
            if col in combined_scores.columns:
                final_score += combined_scores[col] * weight
        
        combined_scores['final_importance_score'] = final_score
        combined_scores = combined_scores.sort_values('final_importance_score', ascending=False)
        
        print("종합 Feature Importance (상위 20개):")
        print(combined_scores[['feature', 'final_importance_score']].head(20).round(4))
        
        return combined_scores
    
    def create_visualizations(self):
        """
        Feature Importance 시각화
        """
        print("\n" + "=" * 80)
        print("Feature Importance 시각화")
        print("=" * 80)
        
        # 1. 모델 성능 비교
        plt.figure(figsize=(20, 15))
        
        plt.subplot(3, 3, 1)
        model_names = list(self.models.keys())
        r2_scores = [self.models[name]['r2'] for name in model_names]
        
        plt.bar(model_names, r2_scores)
        plt.title('모델별 R² Score')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 2. Tree-based 모델 Feature Importance
        plt.subplot(3, 3, 2)
        if 'RandomForest_importance' in self.feature_importance_results:
            rf_importance = self.feature_importance_results['RandomForest_importance'].head(10)
            plt.barh(range(len(rf_importance)), rf_importance['importance'])
            plt.yticks(range(len(rf_importance)), rf_importance['feature'])
            plt.title('Random Forest Feature Importance')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
        
        # 3. Permutation Importance
        plt.subplot(3, 3, 3)
        if 'permutation_importance' in self.feature_importance_results:
            perm_importance = self.feature_importance_results['permutation_importance'].head(10)
            plt.barh(range(len(perm_importance)), perm_importance['importance'])
            plt.yticks(range(len(perm_importance)), perm_importance['feature'])
            plt.title('Permutation Importance')
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
        
        # 4. SHAP Summary Plot (가능한 경우)
        plt.subplot(3, 3, 4)
        if SHAP_AVAILABLE and self.shap_values:
            try:
                model_name = list(self.shap_values.keys())[0]
                shap_values = self.shap_values[model_name]
                
                # SHAP 중요도 계산
                shap_importance = np.abs(shap_values).mean(axis=0)
                top_indices = np.argsort(shap_importance)[-10:]
                
                plt.barh(range(len(top_indices)), shap_importance[top_indices])
                plt.yticks(range(len(top_indices)), [self.feature_names[i] for i in top_indices])
                plt.title('SHAP Feature Importance')
                plt.xlabel('Mean |SHAP Value|')
                plt.gca().invert_yaxis()
            except Exception as e:
                plt.text(0.5, 0.5, f'SHAP Plot Error:\n{str(e)}', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title('SHAP Plot (Error)')
        
        # 5. 종합 중요도
        plt.subplot(3, 3, 5)
        combined_scores = self.calculate_combined_importance()
        top_combined = combined_scores.head(10)
        
        plt.barh(range(len(top_combined)), top_combined['final_importance_score'])
        plt.yticks(range(len(top_combined)), top_combined['feature'])
        plt.title('종합 Feature Importance')
        plt.xlabel('Final Score')
        plt.gca().invert_yaxis()
        
        # 6. 모델 예측 vs 실제값
        plt.subplot(3, 3, 6)
        best_model_name = max(self.models.keys(), key=lambda x: self.models[x]['r2'])
        best_model = self.models[best_model_name]['model']
        
        if best_model_name in ['SVR', 'Ridge', 'Lasso', 'ElasticNet', 'MLP']:
            y_pred = best_model.predict(self.X_test_scaled)
        else:
            y_pred = best_model.predict(self.X_test)
        
        plt.scatter(self.y_test, y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Yield')
        plt.ylabel('Predicted Yield')
        plt.title(f'{best_model_name} 예측 vs 실제')
        plt.grid(True, alpha=0.3)
        
        # 7. 잔차 플롯
        plt.subplot(3, 3, 7)
        residuals = self.y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Yield')
        plt.ylabel('Residuals')
        plt.title('잔차 플롯')
        plt.grid(True, alpha=0.3)
        
        # 8. 특성별 상관관계 히트맵 (상위 10개)
        plt.subplot(3, 3, 8)
        top_features = combined_scores.head(10)['feature'].tolist()
        top_features.append('Yield')
        
        corr_data = self.df[top_features].corr()
        sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, fmt='.3f')
        plt.title('상위 특성 간 상관관계')
        
        # 9. Yield 분포
        plt.subplot(3, 3, 9)
        plt.hist(self.df['Yield'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(self.df['Yield'].mean(), color='red', linestyle='--', 
                   label=f'평균: {self.df["Yield"].mean():.3f}')
        plt.xlabel('Yield')
        plt.ylabel('빈도')
        plt.title('Yield 분포')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_shap_plots(self):
        """
        SHAP 시각화 (가능한 경우)
        """
        if not SHAP_AVAILABLE or not self.shap_values:
            print("SHAP이 사용 불가능하거나 SHAP 값이 없습니다.")
            return
        
        print("\n" + "=" * 80)
        print("SHAP 시각화")
        print("=" * 80)
        
        try:
            model_name = list(self.shap_values.keys())[0]
            shap_values = self.shap_values[model_name]
            
            # SHAP Summary Plot
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, 
                            show=False, max_display=10)
            plt.title(f'SHAP Summary Plot ({model_name})')
            
            plt.subplot(2, 2, 2)
            shap.summary_plot(shap_values, self.X_test, feature_names=self.feature_names, 
                            plot_type="bar", show=False, max_display=10)
            plt.title(f'SHAP Bar Plot ({model_name})')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"SHAP 시각화 실패: {str(e)}")
    
    def generate_report(self):
        """
        종합 분석 리포트 생성
        """
        print("\n" + "=" * 80)
        print("ACN 정제 공정 ML Feature Importance 분석 리포트")
        print("=" * 80)
        
        # 1. 모델 성능 요약
        print("\n1. 모델 성능 요약")
        print("-" * 50)
        model_performance = []
        for name, results in self.models.items():
            model_performance.append({
                'Model': name,
                'R²': results['r2'],
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'CV_Score': results['cv_mean']
            })
        
        performance_df = pd.DataFrame(model_performance).sort_values('R²', ascending=False)
        print(performance_df.round(4))
        
        # 2. 최고 성능 모델
        best_model_name = performance_df.iloc[0]['Model']
        print(f"\n최고 성능 모델: {best_model_name}")
        print(f"R² Score: {performance_df.iloc[0]['R²']:.4f}")
        print(f"RMSE: {performance_df.iloc[0]['RMSE']:.4f}")
        
        # 3. Feature Importance 요약
        print("\n2. Feature Importance 요약")
        print("-" * 50)
        
        combined_scores = self.calculate_combined_importance()
        print("상위 10개 중요 특성:")
        for i, (_, row) in enumerate(combined_scores.head(10).iterrows(), 1):
            print(f"{i:2d}. {row['feature']:<25} {row['final_importance_score']:.4f}")
        
        # 4. 권장사항
        print("\n3. Yield 최적화 권장사항")
        print("-" * 50)
        
        top_3_features = combined_scores.head(3)
        print("주요 영향 인자 모니터링 강화:")
        for _, row in top_3_features.iterrows():
            print(f"  - {row['feature']}: 중요도 {row['final_importance_score']:.4f}")
        
        # 5. 모델 해석성
        print("\n4. 모델 해석성")
        print("-" * 50)
        print(f"최고 성능 모델 ({best_model_name})의 해석성:")
        
        if best_model_name in ['RandomForest', 'GradientBoosting', 'ExtraTrees']:
            print("  - Tree-based 모델로 높은 해석성")
            print("  - Feature Importance 직접 제공")
            print("  - SHAP Values로 개별 예측 해석 가능")
        elif best_model_name in ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']:
            print("  - Linear 모델로 높은 해석성")
            print("  - 계수를 통한 직접적 영향도 해석")
        else:
            print("  - 복잡한 모델로 상대적으로 낮은 해석성")
            print("  - Permutation Importance로 간접적 해석")
        
        return {
            'model_performance': performance_df,
            'feature_importance': combined_scores,
            'best_model': best_model_name,
            'recommendations': top_3_features
        }

def main_ml_analysis(data_path=None, df=None):
    """
    ACN 정제 공정 ML Feature Importance 분석 메인 함수
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    
    Returns:
    results: 분석 결과 딕셔너리
    """
    # 1. 분석기 초기화
    analyzer = ACNMLFeatureImportance(data_path, df)
    
    # 2. 데이터 전처리
    analyzer.preprocess_data()
    
    # 3. 모델 훈련
    model_results = analyzer.train_models()
    
    # 4. Feature Importance 계산
    importance_results = analyzer.calculate_feature_importance()
    
    # 5. 종합 중요도 계산
    combined_importance = analyzer.calculate_combined_importance()
    
    # 6. 시각화
    analyzer.create_visualizations()
    
    # 7. SHAP 시각화 (가능한 경우)
    analyzer.generate_shap_plots()
    
    # 8. 리포트 생성
    report = analyzer.generate_report()
    
    return {
        'analyzer': analyzer,
        'model_results': model_results,
        'importance_results': importance_results,
        'combined_importance': combined_importance,
        'report': report
    }

# 사용 예시
if __name__ == "__main__":
    print("ACN 정제 공정 ML Feature Importance 분석을 시작합니다.")
    print("\n사용법:")
    print("1. CSV 파일에서 분석:")
    print("   results = main_ml_analysis(data_path='your_data.csv')")
    print("\n2. DataFrame에서 분석:")
    print("   results = main_ml_analysis(df=your_dataframe)")
    print("\n3. 결과 확인:")
    print("   print(results['report']['feature_importance'].head(10))")
    print("   print(results['report']['model_performance'])")