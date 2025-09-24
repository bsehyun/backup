# 필수 라이브러리 import
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import PC as pgmpy_PC, HillClimbSearch, BicScore
import dowhy
from dowhy import CausalModel
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import KNNImputer
import networkx as nx

# ACN 공정 데이터 전처리 함수
def preprocess_acn_data(raw_data):
    """
    ACN 공정 데이터 전처리 - 이상치 제거, 정규화, 결측치 처리
    """
    # 1. 이상치 제거 (화학공정 특화)
    def detect_process_outliers(data, columns, method='IQR'):
        outlier_mask = pd.Series([False] * len(data))
        
        for col in columns:
            if method == 'IQR':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_mask |= (data[col] < lower) | (data[col] > upper)
        
        return ~outlier_mask
    
    # 2. 공정 변수별 정규화
    process_vars = ['temperature', 'pressure', 'nh3_c3h6_ratio', 
                   'o2_c3h6_ratio', 'ghsv', 'catalyst_age']
    
    clean_mask = detect_process_outliers(raw_data, process_vars)
    clean_data = raw_data[clean_mask].copy()
    
    # 3. Robust Scaling (공정 데이터 특성 고려)
    scaler = RobustScaler()
    clean_data[process_vars] = scaler.fit_transform(clean_data[process_vars])
    
    # 4. KNN 결측치 보간
    imputer = KNNImputer(n_neighbors=5)
    clean_data[process_vars] = imputer.fit_transform(clean_data[process_vars])
    
    return clean_data, scaler

# VIF 기반 다중공선성 처리
def handle_multicollinearity(df, threshold=5):
    """
    VIF 임계값 기반 다중공선성 변수 제거
    """
    def calculate_vif(df):
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                          for i in range(len(df.columns))]
        return vif_data
    
    features_to_keep = df.columns.tolist()
    
    while True:
        vif_df = calculate_vif(df[features_to_keep])
        max_vif = vif_df['VIF'].max()
        
        if max_vif <= threshold:
            break
            
        feature_to_remove = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
        features_to_keep.remove(feature_to_remove)
        print(f"Removed {feature_to_remove} (VIF: {max_vif:.2f})")
    
    return features_to_keep



# PC 알고리즘 기반 인과구조 발견
def discover_causal_structure_pc(data, alpha=0.05):
    """
    PC 알고리즘으로 ACN 공정의 인과구조 발견
    """
    # PC 알고리즘 실행
    cg = pc(data.values, alpha=alpha, indep_test='fisherz', 
            stable=True, uc_rule=0)
    
    # NetworkX 그래프로 변환
    G = nx.DiGraph()
    node_names = data.columns.tolist()
    
    # 노드 추가
    G.add_nodes_from(node_names)
    
    # 엣지 추가 (adjacency matrix 기반)
    adj_matrix = cg.G.graph
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if adj_matrix[i, j] == 1:
                G.add_edge(node_names[i], node_names[j])
    
    return G, cg

# DoWhy 프레임워크 활용한 인과효과 추정
def estimate_causal_effect_dowhy(data, treatment, outcome, common_causes):
    """
    DoWhy를 이용한 체계적 인과효과 추정
    """
    # 1. 인과 모델 구성
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
        effect_modifiers=None
    )
    
    # 2. 식별 (Identification)
    identified_estimand = model.identify_effect()
    
    # 3. 추정 (Estimation) - 여러 방법 시도
    estimates = {}
    
    # Backdoor criterion
    try:
        backdoor_estimate = model.estimate_effect(identified_estimand,
                                                 method_name="backdoor.linear_regression")
        estimates['backdoor'] = backdoor_estimate.value
    except:
        estimates['backdoor'] = None
    
    # Instrumental variables
    try:
        iv_estimate = model.estimate_effect(identified_estimand,
                                           method_name="iv.instrumental_variable")
        estimates['iv'] = iv_estimate.value
    except:
        estimates['iv'] = None
    
    # 4. 반박검사 (Refutation)
    refutation_results = {}
    
    # Random common cause
    try:
        refute_random = model.refute_estimate(identified_estimand, backdoor_estimate,
                                            method_name="random_common_cause")
        refutation_results['random_common_cause'] = refute_random.new_effect
    except:
        pass
    
    # Placebo treatment
    try:
        refute_placebo = model.refute_estimate(identified_estimand, backdoor_estimate,
                                             method_name="placebo_treatment_refuter")
        refutation_results['placebo_treatment'] = refute_placebo.new_effect
    except:
        pass
    
    return estimates, refutation_results

# 베이지안 네트워크 구성
def build_bayesian_network(data, structure=None):
    """
    pgmpy 활용 베이지안 네트워크 구축 및 모수 학습
    """
    if structure is None:
        # Hill Climbing으로 구조 학습
        hc = HillClimbSearch(data)
        best_model = hc.estimate(scoring_method=BicScore(data))
        edges = best_model.edges()
    else:
        edges = structure.edges()
    
    # 베이지안 네트워크 생성
    model = BayesianNetwork(edges)
    
    # 모수 학습 (가우시안 분포 가정)
    from pgmpy.estimators import MaximumLikelihoodEstimator
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    
    # 추론 엔진 설정
    from pgmpy.inference import VariableElimination
    infer = VariableElimination(model)
    
    return model, infer



# PCMCI 시계열 인과발견
try:
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests import ParCorr
    
    def pcmci_time_series_analysis(time_series_data, tau_max=5):
        """
        PCMCI 알고리즘으로 시간지연 인과관계 분석
        """
        # 데이터 프레임 구성
        dataframe = pp.DataFrame(time_series_data.values, 
                               datatime=time_series_data.index,
                               var_names=time_series_data.columns.tolist())
        
        # PCMCI 설정
        parcorr = ParCorr(significance='analytic')
        pcmci = PCMCI(dataframe=dataframe, 
                     cond_ind_test=parcorr, 
                     verbosity=1)
        
        # PCMCI 실행
        results = pcmci.run_pcmci(tau_max=tau_max, 
                                 pc_alpha=0.05, 
                                 alpha_level=0.01)
        
        # 결과 해석
        causal_links = {}
        for j in range(len(time_series_data.columns)):
            causal_links[time_series_data.columns[j]] = []
            for i in range(len(time_series_data.columns)):
                for tau in range(tau_max + 1):
                    if results['graph'][i, j, tau] == "-->":
                        causal_links[time_series_data.columns[j]].append(
                            (time_series_data.columns[i], tau)
                        )
        
        return results, causal_links
    
except ImportError:
    print("TIGRAMITE not available. Using Granger causality as alternative.")

# Granger 인과성 분석 (PCMCI 대안)
from statsmodels.tsa.stattools import grangercausalitytests

def granger_causality_analysis(data, maxlag=5):
    """
    Granger 인과성 기반 시간지연 관계 분석
    """
    variables = data.columns.tolist()
    granger_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))),
                                 columns=variables, index=variables)
    
    for target in variables:
        for cause in variables:
            if target != cause:
                try:
                    test_data = data[[target, cause]].dropna()
                    test_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                    
                    # 최소 p-value 선택
                    p_values = []
                    for lag in range(1, maxlag + 1):
                        if lag in test_result:
                            p_val = test_result[lag][0]['ssr_ftest'][1]
                            p_values.append(p_val)
                    
                    if p_values:
                        granger_matrix.loc[cause, target] = min(p_values)
                except:
                    granger_matrix.loc[cause, target] = 1.0
    
    return granger_matrix




# SHAP 기반 특성 중요도 분석
def shap_analysis_acn_process(model, X_train, X_test, model_type='tree'):
    """
    SHAP을 이용한 ACN 공정 변수 중요도 분석
    """
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_train)
    else:
        explainer = shap.Explainer(model, X_train)
    
    # SHAP values 계산
    shap_values = explainer.shap_values(X_test)
    
    # 중요도 순위
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': np.abs(shap_values).mean(0)
    }).sort_values('mean_shap', ascending=False)
    
    return explainer, shap_values, feature_importance

# 시각화 함수
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_causal_results(dag, shap_importance, granger_matrix):
    """
    인과분석 결과 종합 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. DAG 시각화
    pos = nx.spring_layout(dag, k=3, iterations=50)
    nx.draw_networkx(dag, pos, ax=axes[0,0], 
                    node_color='lightblue', 
                    node_size=3000,
                    font_size=10, 
                    arrows=True,
                    arrowsize=20)
    axes[0,0].set_title('Causal DAG Structure')
    axes[0,0].axis('off')
    
    # 2. SHAP 중요도
    axes[0,1].barh(shap_importance['feature'], shap_importance['mean_shap'])
    axes[0,1].set_title('SHAP Feature Importance')
    axes[0,1].set_xlabel('Mean |SHAP value|')
    
    # 3. Granger 인과성 히트맵
    sns.heatmap(granger_matrix, annot=True, cmap='coolwarm', 
                center=0.05, ax=axes[1,0])
    axes[1,0].set_title('Granger Causality P-values')
    
    # 4. 결합된 중요도 분석
    # DAG에서의 연결 강도와 SHAP 중요도 비교
    dag_centrality = pd.Series(nx.degree_centrality(dag))
    combined_df = pd.DataFrame({
        'dag_centrality': dag_centrality,
        'shap_importance': shap_importance.set_index('feature')['mean_shap']
    }).fillna(0)
    
    axes[1,1].scatter(combined_df['dag_centrality'], 
                     combined_df['shap_importance'])
    axes[1,1].set_xlabel('DAG Centrality')
    axes[1,1].set_ylabel('SHAP Importance')
    axes[1,1].set_title('Causal Structure vs. Feature Importance')
    
    # 각 점에 변수명 라벨링
    for i, txt in enumerate(combined_df.index):
        axes[1,1].annotate(txt, 
                          (combined_df['dag_centrality'].iloc[i],
                           combined_df['shap_importance'].iloc[i]),
                          fontsize=8)
    
    plt.tight_layout()
    return fig




class ACNCausalAnalyzer:
    """
    ACN 공정 인과분석을 위한 통합 클래스
    """
    
    def __init__(self, data):
        self.raw_data = data
        self.processed_data = None
        self.causal_dag = None
        self.bayesian_model = None
        self.shap_results = None
        self.time_series_results = None
        
    def preprocess_data(self, vif_threshold=5):
        """데이터 전처리 파이프라인"""
        print("Step 1: Data preprocessing...")
        self.processed_data, self.scaler = preprocess_acn_data(self.raw_data)
        
        # VIF 기반 특성 선택
        features_to_keep = handle_multicollinearity(
            self.processed_data.select_dtypes(include=[np.number]), 
            threshold=vif_threshold
        )
        self.processed_data = self.processed_data[features_to_keep + ['acn_yield']]
        print(f"Kept {len(features_to_keep)} features after multicollinearity check")
        
    def discover_causal_structure(self, method='pc', alpha=0.05):
        """인과구조 발견"""
        print(f"Step 2: Causal structure discovery using {method}...")
        
        if method == 'pc':
            self.causal_dag, _ = discover_causal_structure_pc(
                self.processed_data, alpha=alpha
            )
        elif method == 'ges':
            # GES 구현
            from causallearn.search.ScoreBased.GES import ges
            Record = ges(self.processed_data.values)
            # 결과를 NetworkX 그래프로 변환하는 코드 추가 필요
        
        print(f"Discovered DAG with {len(self.causal_dag.edges())} edges")
        
    def analyze_time_series_causality(self, time_col=None, tau_max=5):
        """시계열 인과관계 분석"""
        print("Step 3: Time series causality analysis...")
        
        if time_col and time_col in self.processed_data.columns:
            ts_data = self.processed_data.set_index(time_col)
        else:
            ts_data = self.processed_data
        
        # Granger causality analysis
        self.time_series_results = granger_causality_analysis(ts_data, maxlag=tau_max)
        print("Granger causality analysis completed")
        
    def fit_bayesian_network(self):
        """베이지안 네트워크 학습"""
        print("Step 4: Bayesian network fitting...")
        
        self.bayesian_model, self.inference_engine = build_bayesian_network(
            self.processed_data, structure=self.causal_dag
        )
        print("Bayesian network fitted successfully")
        
    def explain_with_shap(self, target_var='acn_yield', model_type='rf'):
        """SHAP 기반 설명"""
        print("Step 5: SHAP analysis...")
        
        # 간단한 예측 모델 학습 (SHAP 분석용)
        X = self.processed_data.drop(columns=[target_var])
        y = self.processed_data[target_var]
        
        if model_type == 'rf':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # SHAP 분석
        explainer, shap_values, importance = shap_analysis_acn_process(
            model, X_train, X_test, model_type='tree'
        )
        
        self.shap_results = {
            'explainer': explainer,
            'shap_values': shap_values,
            'importance': importance,
            'model': model
        }
        
        print("SHAP analysis completed")
        
    def generate_optimization_recommendations(self):
        """최적화 권고사항 생성"""
        print("Step 6: Generating optimization recommendations...")
        
        recommendations = {}
        
        # SHAP 중요도 기반 권고
        if self.shap_results:
            top_features = self.shap_results['importance'].head(3)
            recommendations['top_influential_vars'] = top_features['feature'].tolist()
        
        # 인과구조 기반 권고
        if self.causal_dag:
            centrality = nx.degree_centrality(self.causal_dag)
            most_central = max(centrality, key=centrality.get)
            recommendations['most_connected_var'] = most_central
        
        # 시계열 분석 기반 권고
        if self.time_series_results is not None:
            # ACN yield에 가장 강한 Granger cause 찾기
            if 'acn_yield' in self.time_series_results.columns:
                granger_to_yield = self.time_series_results['acn_yield'].sort_values()
                strongest_cause = granger_to_yield.index[0]  # 가장 작은 p-value
                recommendations['strongest_temporal_cause'] = strongest_cause
        
        return recommendations
        
    def run_full_analysis(self):
        """전체 분석 파이프라인 실행"""
        print("Starting comprehensive ACN causal analysis...")
        
        # 1. 전처리
        self.preprocess_data()
        
        # 2. 인과구조 발견
        self.discover_causal_structure(method='pc')
        
        # 3. 시계열 분석
        self.analyze_time_series_causality()
        
        # 4. 베이지안 네트워크
        self.fit_bayesian_network()
        
        # 5. SHAP 분석
        self.explain_with_shap()
        
        # 6. 권고사항 생성
        recommendations = self.generate_optimization_recommendations()
        
        print("\\n=== Analysis Complete ===")
        print("Key Recommendations:")
        for key, value in recommendations.items():
            print(f"- {key}: {value}")
        
        return recommendations

# 사용 예제
def main():
    """
    ACN 공정 데이터 분석 실행 예제
    """
    # 샘플 데이터 생성 (실제 사용시 실공정 데이터 로드)
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'temperature': np.random.normal(450, 10, n_samples),
        'pressure': np.random.normal(150, 20, n_samples), 
        'nh3_c3h6_ratio': np.random.normal(1.15, 0.05, n_samples),
        'o2_c3h6_ratio': np.random.normal(1.75, 0.1, n_samples),
        'ghsv': np.random.normal(1000, 100, n_samples),
        'catalyst_age': np.random.uniform(0, 365, n_samples),
    })
    
    # ACN yield 계산 (temperature가 주요 영향인자로 설정)
    sample_data['acn_yield'] = (
        0.85 + 
        0.02 * (sample_data['temperature'] - 450) / 10 +
        0.01 * (sample_data['nh3_c3h6_ratio'] - 1.15) / 0.05 +
        -0.005 * sample_data['catalyst_age'] / 365 +
        np.random.normal(0, 0.02, n_samples)
    )
    
    # 분석 실행
    analyzer = ACNCausalAnalyzer(sample_data)
    recommendations = analyzer.run_full_analysis()
    
    # 시각화
    if analyzer.causal_dag and analyzer.shap_results and analyzer.time_series_results is not None:
        fig = visualize_causal_results(
            analyzer.causal_dag,
            analyzer.shap_results['importance'],
            analyzer.time_series_results
        )
        plt.show()

if __name__ == "__main__":
    main()
