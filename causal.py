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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# ACN 공정 데이터 전처리 함수 (시각화 및 결론 포함)
def preprocess_acn_data(raw_data, visualize=True):
    """
    ACN 공정 데이터 전처리 - 이상치 제거, 정규화, 결측치 처리
    시각화와 결론 출력 포함
    """
    print("=" * 60)
    print("ACN 공정 데이터 전처리 시작")
    print("=" * 60)
    
    # 1. 이상치 제거 (화학공정 특화)
    def detect_process_outliers(data, columns, method='IQR'):
        outlier_mask = pd.Series([False] * len(data))
        outlier_info = {}
        
        for col in columns:
            if method == 'IQR':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                col_outliers = (data[col] < lower) | (data[col] > upper)
                outlier_mask |= col_outliers
                outlier_info[col] = {
                    'count': col_outliers.sum(),
                    'percentage': (col_outliers.sum() / len(data)) * 100,
                    'bounds': (lower, upper)
                }
        
        return ~outlier_mask, outlier_info
    
    # 2. 공정 변수별 정규화
    process_vars = ['temperature', 'pressure', 'nh3_c3h6_ratio', 
                   'o2_c3h6_ratio', 'ghsv', 'catalyst_age']
    
    # 이상치 탐지 및 제거
    clean_mask, outlier_info = detect_process_outliers(raw_data, process_vars)
    clean_data = raw_data[clean_mask].copy()
    
    # 이상치 분석 결과 출력
    print(f"\n📊 이상치 분석 결과:")
    print(f"   - 원본 데이터: {len(raw_data)}개 샘플")
    print(f"   - 이상치 제거 후: {len(clean_data)}개 샘플")
    print(f"   - 제거된 샘플: {len(raw_data) - len(clean_data)}개 ({((len(raw_data) - len(clean_data))/len(raw_data)*100):.1f}%)")
    
    print(f"\n📈 변수별 이상치 현황:")
    for var, info in outlier_info.items():
        print(f"   - {var}: {info['count']}개 ({info['percentage']:.1f}%)")
    
    # 시각화
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(process_vars):
            if i < len(axes):
                # 박스플롯으로 이상치 시각화
                axes[i].boxplot([raw_data[var], clean_data[var]], 
                               labels=['Original', 'Cleaned'])
                axes[i].set_title(f'{var} - Outlier Detection')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('ACN Process Data - Outlier Detection Results', fontsize=16, y=1.02)
        plt.show()
    
    # 3. Robust Scaling (공정 데이터 특성 고려)
    scaler = RobustScaler()
    clean_data[process_vars] = scaler.fit_transform(clean_data[process_vars])
    
    print(f"\n🔧 데이터 정규화 완료 (RobustScaler 적용)")
    
    # 4. KNN 결측치 보간
    missing_before = clean_data[process_vars].isnull().sum().sum()
    imputer = KNNImputer(n_neighbors=5)
    clean_data[process_vars] = imputer.fit_transform(clean_data[process_vars])
    missing_after = clean_data[process_vars].isnull().sum().sum()
    
    print(f"\n🔍 결측치 처리:")
    print(f"   - 처리 전 결측치: {missing_before}개")
    print(f"   - 처리 후 결측치: {missing_after}개")
    
    # 전처리 후 데이터 분포 시각화
    if visualize:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, var in enumerate(process_vars):
            if i < len(axes):
                axes[i].hist(clean_data[var], bins=30, alpha=0.7, edgecolor='black')
                axes[i].set_title(f'{var} - Normalized Distribution')
                axes[i].set_xlabel('Normalized Value')
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.suptitle('ACN Process Data - After Preprocessing', fontsize=16, y=1.02)
        plt.show()
    
    print(f"\n✅ 데이터 전처리 완료!")
    print(f"   - 최종 데이터 크기: {clean_data.shape}")
    print(f"   - 처리된 변수: {len(process_vars)}개")
    
    return clean_data, scaler, outlier_info

# VIF 기반 다중공선성 처리 (시각화 및 결론 포함)
def handle_multicollinearity(df, threshold=5, visualize=True):
    """
    VIF 임계값 기반 다중공선성 변수 제거
    시각화와 결론 출력 포함
    """
    print("\n" + "=" * 60)
    print("다중공선성 분석 시작 (VIF 기반)")
    print("=" * 60)
    
    def calculate_vif(df):
        vif_data = pd.DataFrame()
        vif_data["feature"] = df.columns
        vif_data["VIF"] = [variance_inflation_factor(df.values, i)
                          for i in range(len(df.columns))]
        return vif_data
    
    # 초기 VIF 계산
    initial_vif = calculate_vif(df)
    print(f"\n📊 초기 VIF 분석 결과:")
    print(f"   - 분석 대상 변수: {len(df.columns)}개")
    print(f"   - VIF 임계값: {threshold}")
    
    # VIF 시각화
    if visualize:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 초기 VIF 막대 그래프
        bars = ax1.bar(range(len(initial_vif)), initial_vif['VIF'], 
                      color=['red' if vif > threshold else 'green' for vif in initial_vif['VIF']])
        ax1.set_xlabel('Variables')
        ax1.set_ylabel('VIF Value')
        ax1.set_title('Initial VIF Analysis')
        ax1.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        ax1.set_xticks(range(len(initial_vif)))
        ax1.set_xticklabels(initial_vif['feature'], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # VIF 값 라벨 추가
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    
    features_to_keep = df.columns.tolist()
    removed_features = []
    iteration = 0
    
    print(f"\n🔄 다중공선성 제거 과정:")
    
    while True:
        iteration += 1
        vif_df = calculate_vif(df[features_to_keep])
        max_vif = vif_df['VIF'].max()
        
        if max_vif <= threshold:
            break
            
        feature_to_remove = vif_df.loc[vif_df['VIF'].idxmax(), 'feature']
        features_to_keep.remove(feature_to_remove)
        removed_features.append((feature_to_remove, max_vif, iteration))
        print(f"   Iteration {iteration}: {feature_to_remove} 제거 (VIF: {max_vif:.2f})")
    
    # 최종 결과
    final_vif = calculate_vif(df[features_to_keep])
    
    print(f"\n✅ 다중공선성 분석 완료!")
    print(f"   - 제거된 변수: {len(removed_features)}개")
    print(f"   - 남은 변수: {len(features_to_keep)}개")
    print(f"   - 최대 VIF: {final_vif['VIF'].max():.2f}")
    
    if removed_features:
        print(f"\n📋 제거된 변수 목록:")
        for feature, vif, iter_num in removed_features:
            print(f"   - {feature} (VIF: {vif:.2f}, Iteration: {iter_num})")
    
    # 최종 VIF 시각화
    if visualize:
        # 최종 VIF 막대 그래프
        bars2 = ax2.bar(range(len(final_vif)), final_vif['VIF'], 
                       color=['green' if vif <= threshold else 'orange' for vif in final_vif['VIF']])
        ax2.set_xlabel('Variables')
        ax2.set_ylabel('VIF Value')
        ax2.set_title('Final VIF Analysis (After Removal)')
        ax2.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
        ax2.set_xticks(range(len(final_vif)))
        ax2.set_xticklabels(final_vif['feature'], rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # VIF 값 라벨 추가
        for i, bar in enumerate(bars2):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.suptitle('Multicollinearity Analysis (VIF)', fontsize=16, y=1.02)
        plt.show()
        
        # 제거 과정 히트맵
        if removed_features:
            fig, ax = plt.subplots(figsize=(10, 6))
            removal_data = pd.DataFrame(removed_features, 
                                      columns=['Feature', 'VIF', 'Iteration'])
            
            # 히트맵 데이터 준비
            heatmap_data = np.zeros((len(removed_features), 3))
            for i, (_, vif, iter_num) in enumerate(removed_features):
                heatmap_data[i, 0] = vif
                heatmap_data[i, 1] = iter_num
                heatmap_data[i, 2] = 1  # 제거됨을 표시
            
            im = ax.imshow(heatmap_data.T, cmap='Reds', aspect='auto')
            ax.set_xticks(range(len(removed_features)))
            ax.set_xticklabels(removal_data['Feature'], rotation=45, ha='right')
            ax.set_yticks(range(3))
            ax.set_yticklabels(['VIF Value', 'Iteration', 'Removed'])
            ax.set_title('Feature Removal Process')
            
            # 값 라벨 추가
            for i in range(len(removed_features)):
                ax.text(i, 0, f'{removal_data.iloc[i]["VIF"]:.1f}', 
                       ha='center', va='center', fontweight='bold')
                ax.text(i, 1, f'{removal_data.iloc[i]["Iteration"]}', 
                       ha='center', va='center', fontweight='bold')
            
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.show()
    
    return features_to_keep, removed_features, final_vif



# PC 알고리즘 기반 인과구조 발견 (시각화 및 결론 포함)
def discover_causal_structure_pc(data, alpha=0.05, visualize=True):
    """
    PC 알고리즘으로 ACN 공정의 인과구조 발견
    시각화와 결론 출력 포함
    """
    print("\n" + "=" * 60)
    print("PC 알고리즘 기반 인과구조 발견")
    print("=" * 60)
    
    print(f"\n🔍 PC 알고리즘 설정:")
    print(f"   - 유의수준 (alpha): {alpha}")
    print(f"   - 독립성 검정: Fisher's Z")
    print(f"   - 분석 변수: {len(data.columns)}개")
    print(f"   - 샘플 수: {len(data)}개")
    
    # PC 알고리즘 실행
    print(f"\n⚙️ PC 알고리즘 실행 중...")
    cg = pc(data.values, alpha=alpha, indep_test='fisherz', 
            stable=True, uc_rule=0)
    
    # NetworkX 그래프로 변환
    G = nx.DiGraph()
    node_names = data.columns.tolist()
    
    # 노드 추가
    G.add_nodes_from(node_names)
    
    # 엣지 추가 (adjacency matrix 기반)
    adj_matrix = cg.G.graph
    edges_found = []
    
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if adj_matrix[i, j] == 1:
                G.add_edge(node_names[i], node_names[j])
                edges_found.append((node_names[i], node_names[j]))
    
    # 결과 분석
    print(f"\n📊 인과구조 발견 결과:")
    print(f"   - 발견된 인과관계: {len(edges_found)}개")
    print(f"   - 노드 수: {len(G.nodes())}개")
    print(f"   - 연결 밀도: {nx.density(G):.3f}")
    
    if edges_found:
        print(f"\n🔗 발견된 인과관계:")
        for i, (source, target) in enumerate(edges_found, 1):
            print(f"   {i}. {source} → {target}")
    
    # 네트워크 중심성 분석
    centrality_measures = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G, max_iter=1000)
    }
    
    print(f"\n📈 네트워크 중심성 분석:")
    for measure_name, centrality in centrality_measures.items():
        if centrality:
            most_central = max(centrality, key=centrality.get)
            print(f"   - {measure_name} 중심성 최고: {most_central} ({centrality[most_central]:.3f})")
    
    # 시각화
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 인과 그래프 시각화
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        nx.draw_networkx(G, pos, ax=axes[0,0], 
                        node_color='lightblue', 
                        node_size=2000,
                        font_size=10, 
                        arrows=True,
                        arrowsize=20,
                        edge_color='gray',
                        with_labels=True)
        axes[0,0].set_title('Causal DAG Structure (PC Algorithm)', fontsize=14, fontweight='bold')
        axes[0,0].axis('off')
        
        # 2. 중심성 비교
        centrality_df = pd.DataFrame(centrality_measures)
        if not centrality_df.empty:
            centrality_df.plot(kind='bar', ax=axes[0,1], width=0.8)
            axes[0,1].set_title('Network Centrality Measures', fontsize=14, fontweight='bold')
            axes[0,1].set_xlabel('Variables')
            axes[0,1].set_ylabel('Centrality Score')
            axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[0,1].tick_params(axis='x', rotation=45)
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. 인접 행렬 히트맵
        adj_df = pd.DataFrame(adj_matrix, 
                             index=node_names, 
                             columns=node_names)
        sns.heatmap(adj_df, annot=True, cmap='Blues', 
                   ax=axes[1,0], cbar_kws={'label': 'Connection'})
        axes[1,0].set_title('Adjacency Matrix', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Target Variables')
        axes[1,0].set_ylabel('Source Variables')
        
        # 4. 네트워크 통계
        network_stats = {
            'Nodes': len(G.nodes()),
            'Edges': len(G.edges()),
            'Density': nx.density(G),
            'Avg Clustering': nx.average_clustering(G.to_undirected()) if len(G.edges()) > 0 else 0,
            'Connected Components': nx.number_connected_components(G.to_undirected())
        }
        
        stats_df = pd.DataFrame(list(network_stats.items()), 
                               columns=['Metric', 'Value'])
        axes[1,1].axis('off')
        
        # 테이블 형태로 통계 표시
        table_data = []
        for metric, value in network_stats.items():
            table_data.append([metric, f'{value:.3f}' if isinstance(value, float) else str(value)])
        
        table = axes[1,1].table(cellText=table_data,
                               colLabels=['Network Metric', 'Value'],
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        axes[1,1].set_title('Network Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.suptitle('PC Algorithm - Causal Structure Discovery', fontsize=16, y=1.02)
        plt.show()
        
        # 추가: 인과관계 강도 분석 (상관계수 기반)
        if edges_found:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            edge_strengths = []
            edge_labels = []
            
            for source, target in edges_found:
                if source in data.columns and target in data.columns:
                    corr = data[source].corr(data[target])
                    edge_strengths.append(abs(corr))
                    edge_labels.append(f"{source}→{target}\n(r={corr:.3f})")
            
            if edge_strengths:
                bars = ax.bar(range(len(edge_strengths)), edge_strengths, 
                             color='skyblue', edgecolor='navy', alpha=0.7)
                ax.set_xlabel('Causal Relationships')
                ax.set_ylabel('Correlation Strength (|r|)')
                ax.set_title('Causal Relationship Strengths', fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(edge_labels)))
                ax.set_xticklabels(edge_labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # 값 라벨 추가
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.show()
    
    print(f"\n✅ PC 알고리즘 분석 완료!")
    print(f"   - 인과구조 발견: {len(edges_found)}개 관계")
    print(f"   - 네트워크 밀도: {nx.density(G):.3f}")
    
    return G, cg, centrality_measures, edges_found

# DoWhy 프레임워크 활용한 인과효과 추정 (시각화 및 결론 포함)
def estimate_causal_effect_dowhy(data, treatment, outcome, common_causes, visualize=True):
    """
    DoWhy를 이용한 체계적 인과효과 추정
    시각화와 결론 출력 포함
    """
    print("\n" + "=" * 60)
    print("DoWhy 프레임워크 기반 인과효과 추정")
    print("=" * 60)
    
    print(f"\n🎯 인과효과 분석 설정:")
    print(f"   - 처리 변수 (Treatment): {treatment}")
    print(f"   - 결과 변수 (Outcome): {outcome}")
    print(f"   - 공통 원인 (Common Causes): {common_causes}")
    print(f"   - 데이터 크기: {data.shape}")
    
    # 1. 인과 모델 구성
    print(f"\n🔧 인과 모델 구성 중...")
    model = CausalModel(
        data=data,
        treatment=treatment,
        outcome=outcome,
        common_causes=common_causes,
        effect_modifiers=None
    )
    
    # 2. 식별 (Identification)
    print(f"\n🔍 인과효과 식별 중...")
    identified_estimand = model.identify_effect()
    print(f"   - 식별된 추정량: {identified_estimand}")
    
    # 3. 추정 (Estimation) - 여러 방법 시도
    print(f"\n📊 인과효과 추정 중...")
    estimates = {}
    estimate_details = {}
    
    # Backdoor criterion
    try:
        backdoor_estimate = model.estimate_effect(identified_estimand,
                                                 method_name="backdoor.linear_regression")
        estimates['backdoor'] = backdoor_estimate.value
        estimate_details['backdoor'] = {
            'value': backdoor_estimate.value,
            'confidence_interval': backdoor_estimate.confidence_interval,
            'p_value': getattr(backdoor_estimate, 'p_value', None)
        }
        print(f"   ✅ Backdoor 추정: {backdoor_estimate.value:.4f}")
    except Exception as e:
        estimates['backdoor'] = None
        print(f"   ❌ Backdoor 추정 실패: {str(e)}")
    
    # Instrumental variables
    try:
        iv_estimate = model.estimate_effect(identified_estimand,
                                           method_name="iv.instrumental_variable")
        estimates['iv'] = iv_estimate.value
        estimate_details['iv'] = {
            'value': iv_estimate.value,
            'confidence_interval': iv_estimate.confidence_interval,
            'p_value': getattr(iv_estimate, 'p_value', None)
        }
        print(f"   ✅ IV 추정: {iv_estimate.value:.4f}")
    except Exception as e:
        estimates['iv'] = None
        print(f"   ❌ IV 추정 실패: {str(e)}")
    
    # 4. 반박검사 (Refutation)
    print(f"\n🧪 반박검사 실행 중...")
    refutation_results = {}
    
    # Random common cause
    try:
        if 'backdoor' in estimates and estimates['backdoor'] is not None:
            refute_random = model.refute_estimate(identified_estimand, backdoor_estimate,
                                                method_name="random_common_cause")
            refutation_results['random_common_cause'] = refute_random.new_effect
            print(f"   ✅ Random Common Cause: {refute_random.new_effect:.4f}")
    except Exception as e:
        print(f"   ❌ Random Common Cause 실패: {str(e)}")
    
    # Placebo treatment
    try:
        if 'backdoor' in estimates and estimates['backdoor'] is not None:
            refute_placebo = model.refute_estimate(identified_estimand, backdoor_estimate,
                                                 method_name="placebo_treatment_refuter")
            refutation_results['placebo_treatment'] = refute_placebo.new_effect
            print(f"   ✅ Placebo Treatment: {refute_placebo.new_effect:.4f}")
    except Exception as e:
        print(f"   ❌ Placebo Treatment 실패: {str(e)}")
    
    # 결과 요약
    print(f"\n📋 인과효과 추정 결과 요약:")
    for method, value in estimates.items():
        if value is not None:
            print(f"   - {method}: {value:.4f}")
        else:
            print(f"   - {method}: 추정 실패")
    
    # 시각화
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 인과효과 추정값 비교
        valid_estimates = {k: v for k, v in estimates.items() if v is not None}
        if valid_estimates:
            methods = list(valid_estimates.keys())
            values = list(valid_estimates.values())
            
            bars = axes[0,0].bar(methods, values, 
                               color=['skyblue', 'lightcoral', 'lightgreen'][:len(methods)],
                               edgecolor='navy', alpha=0.7)
            axes[0,0].set_title('Causal Effect Estimates', fontsize=14, fontweight='bold')
            axes[0,0].set_ylabel('Effect Size')
            axes[0,0].grid(True, alpha=0.3)
            
            # 값 라벨 추가
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[0,0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                              f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[0,0].text(0.5, 0.5, 'No valid estimates', ha='center', va='center',
                          transform=axes[0,0].transAxes, fontsize=12)
            axes[0,0].set_title('Causal Effect Estimates', fontsize=14, fontweight='bold')
        
        # 2. 신뢰구간 시각화 (backdoor 추정이 있는 경우)
        if 'backdoor' in estimate_details and estimate_details['backdoor']:
            ci = estimate_details['backdoor']['confidence_interval']
            if ci:
                axes[0,1].errorbar(0, estimate_details['backdoor']['value'], 
                                 yerr=[[estimate_details['backdoor']['value'] - ci[0]], 
                                       [ci[1] - estimate_details['backdoor']['value']]], 
                                 fmt='o', capsize=10, capthick=2, markersize=8,
                                 color='blue', label='Backdoor Estimate')
                axes[0,1].set_xlim(-0.5, 0.5)
                axes[0,1].set_title('Confidence Interval (Backdoor)', fontsize=14, fontweight='bold')
                axes[0,1].set_ylabel('Effect Size')
                axes[0,1].legend()
                axes[0,1].grid(True, alpha=0.3)
                axes[0,1].set_xticks([])
        
        # 3. 반박검사 결과
        if refutation_results:
            ref_methods = list(refutation_results.keys())
            ref_values = list(refutation_results.values())
            
            bars = axes[1,0].bar(ref_methods, ref_values,
                               color=['orange', 'purple'][:len(ref_methods)],
                               edgecolor='black', alpha=0.7)
            axes[1,0].set_title('Refutation Test Results', fontsize=14, fontweight='bold')
            axes[1,0].set_ylabel('New Effect Size')
            axes[1,0].tick_params(axis='x', rotation=45)
            axes[1,0].grid(True, alpha=0.3)
            
            # 값 라벨 추가
            for bar, value in zip(bars, ref_values):
                height = bar.get_height()
                axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                              f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[1,0].text(0.5, 0.5, 'No refutation results', ha='center', va='center',
                          transform=axes[1,0].transAxes, fontsize=12)
            axes[1,0].set_title('Refutation Test Results', fontsize=14, fontweight='bold')
        
        # 4. 인과 그래프 시각화
        # 간단한 인과 그래프 생성
        G = nx.DiGraph()
        G.add_node(treatment, node_color='red')
        G.add_node(outcome, node_color='blue')
        for cause in common_causes:
            G.add_node(cause, node_color='gray')
            G.add_edge(cause, treatment)
            G.add_edge(cause, outcome)
        G.add_edge(treatment, outcome)
        
        pos = nx.spring_layout(G, seed=42)
        node_colors = ['red' if node == treatment else 'blue' if node == outcome else 'gray' 
                      for node in G.nodes()]
        
        nx.draw_networkx(G, pos, ax=axes[1,1], 
                        node_color=node_colors,
                        node_size=2000,
                        font_size=10,
                        arrows=True,
                        arrowsize=20,
                        with_labels=True)
        axes[1,1].set_title('Causal Graph Structure', fontsize=14, fontweight='bold')
        axes[1,1].axis('off')
        
        plt.tight_layout()
        plt.suptitle('DoWhy Framework - Causal Effect Estimation', fontsize=16, y=1.02)
        plt.show()
    
    # 결론 및 해석
    print(f"\n💡 인과효과 해석:")
    if estimates.get('backdoor'):
        effect = estimates['backdoor']
        if effect > 0:
            print(f"   - {treatment}이 1단위 증가하면 {outcome}이 {effect:.4f}만큼 증가")
        else:
            print(f"   - {treatment}이 1단위 증가하면 {outcome}이 {abs(effect):.4f}만큼 감소")
    
    if refutation_results:
        print(f"\n🔍 반박검사 결과:")
        for method, value in refutation_results.items():
            if abs(value) < 0.01:  # 임계값 설정
                print(f"   - {method}: 강건함 (효과 크기: {value:.4f})")
            else:
                print(f"   - {method}: 주의 필요 (효과 크기: {value:.4f})")
    
    print(f"\n✅ DoWhy 인과효과 추정 완료!")
    
    return estimates, refutation_results, estimate_details

# 베이지안 네트워크 구성 (시각화 및 결론 포함)
def build_bayesian_network(data, structure=None, visualize=True):
    """
    pgmpy 활용 베이지안 네트워크 구축 및 모수 학습
    시각화와 결론 출력 포함
    """
    print("\n" + "=" * 60)
    print("베이지안 네트워크 구축 및 학습")
    print("=" * 60)
    
    print(f"\n🔧 베이지안 네트워크 설정:")
    print(f"   - 데이터 크기: {data.shape}")
    print(f"   - 변수 수: {len(data.columns)}개")
    
    # 구조 학습 또는 기존 구조 사용
    if structure is None:
        print(f"\n🏗️ 네트워크 구조 학습 중 (Hill Climbing)...")
        hc = HillClimbSearch(data)
        best_model = hc.estimate(scoring_method=BicScore(data))
        edges = best_model.edges()
        print(f"   - 학습된 엣지 수: {len(edges)}개")
    else:
        edges = structure.edges()
        print(f"\n📋 기존 구조 사용:")
        print(f"   - 엣지 수: {len(edges)}개")
    
    if edges:
        print(f"\n🔗 네트워크 구조:")
        for i, (source, target) in enumerate(edges, 1):
            print(f"   {i}. {source} → {target}")
    
    # 베이지안 네트워크 생성
    print(f"\n⚙️ 베이지안 네트워크 생성 중...")
    model = BayesianNetwork(edges)
    
    # 모수 학습 (가우시안 분포 가정)
    print(f"\n📊 모수 학습 중...")
    from pgmpy.estimators import MaximumLikelihoodEstimator
    model.fit(data, estimator=MaximumLikelihoodEstimator)
    
    # 추론 엔진 설정
    from pgmpy.inference import VariableElimination
    infer = VariableElimination(model)
    
    # 네트워크 성능 평가
    print(f"\n📈 네트워크 성능 평가:")
    try:
        # BIC 점수 계산
        bic_score = BicScore(data).score(model)
        print(f"   - BIC 점수: {bic_score:.2f}")
        
        # AIC 점수 계산
        from pgmpy.estimators import AicScore
        aic_score = AicScore(data).score(model)
        print(f"   - AIC 점수: {aic_score:.2f}")
        
        # 네트워크 복잡도
        complexity = len(edges) + len(data.columns)  # 엣지 수 + 노드 수
        print(f"   - 네트워크 복잡도: {complexity}")
        
    except Exception as e:
        print(f"   - 성능 평가 중 오류: {str(e)}")
    
    # 시각화
    if visualize and edges:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 베이지안 네트워크 구조 시각화
        G = nx.DiGraph(edges)
        pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
        
        # 노드 크기를 연결도에 비례하게 설정
        node_sizes = [G.degree(node) * 500 + 1000 for node in G.nodes()]
        
        nx.draw_networkx(G, pos, ax=axes[0,0],
                        node_color='lightgreen',
                        node_size=node_sizes,
                        font_size=10,
                        arrows=True,
                        arrowsize=20,
                        edge_color='gray',
                        with_labels=True)
        axes[0,0].set_title('Bayesian Network Structure', fontsize=14, fontweight='bold')
        axes[0,0].axis('off')
        
        # 2. 노드 연결도 분포
        degrees = [G.degree(node) for node in G.nodes()]
        axes[0,1].hist(degrees, bins=max(1, len(set(degrees))), 
                      color='skyblue', edgecolor='black', alpha=0.7)
        axes[0,1].set_title('Node Degree Distribution', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Degree')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. 네트워크 통계
        network_stats = {
            'Nodes': len(G.nodes()),
            'Edges': len(G.edges()),
            'Density': nx.density(G),
            'Avg Degree': np.mean(degrees) if degrees else 0,
            'Max Degree': max(degrees) if degrees else 0,
            'Min Degree': min(degrees) if degrees else 0
        }
        
        stats_df = pd.DataFrame(list(network_stats.items()), 
                               columns=['Metric', 'Value'])
        axes[1,0].axis('off')
        
        # 테이블 형태로 통계 표시
        table_data = []
        for metric, value in network_stats.items():
            table_data.append([metric, f'{value:.3f}' if isinstance(value, float) else str(value)])
        
        table = axes[1,0].table(cellText=table_data,
                               colLabels=['Network Metric', 'Value'],
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        axes[1,0].set_title('Network Statistics', fontsize=14, fontweight='bold', pad=20)
        
        # 4. 조건부 확률 추정 (샘플)
        if len(data.columns) > 1:
            # 첫 번째 변수의 분포 시각화
            first_var = data.columns[0]
            axes[1,1].hist(data[first_var], bins=30, alpha=0.7, 
                          color='lightcoral', edgecolor='black')
            axes[1,1].set_title(f'Distribution of {first_var}', fontsize=14, fontweight='bold')
            axes[1,1].set_xlabel(first_var)
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5, 0.5, 'Insufficient variables\nfor distribution plot', 
                          ha='center', va='center', transform=axes[1,1].transAxes, fontsize=12)
            axes[1,1].set_title('Variable Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle('Bayesian Network Analysis', fontsize=16, y=1.02)
        plt.show()
        
        # 추가: 인과관계 강도 분석
        if len(edges) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            edge_strengths = []
            edge_labels = []
            
            for source, target in edges:
                if source in data.columns and target in data.columns:
                    # 상관계수로 인과관계 강도 추정
                    corr = data[source].corr(data[target])
                    edge_strengths.append(abs(corr))
                    edge_labels.append(f"{source}→{target}\n(r={corr:.3f})")
            
            if edge_strengths:
                bars = ax.bar(range(len(edge_strengths)), edge_strengths, 
                             color='lightgreen', edgecolor='darkgreen', alpha=0.7)
                ax.set_xlabel('Causal Relationships')
                ax.set_ylabel('Correlation Strength (|r|)')
                ax.set_title('Bayesian Network - Causal Relationship Strengths', 
                           fontsize=14, fontweight='bold')
                ax.set_xticks(range(len(edge_labels)))
                ax.set_xticklabels(edge_labels, rotation=45, ha='right')
                ax.grid(True, alpha=0.3)
                
                # 값 라벨 추가
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                plt.show()
    
    # 결론 및 해석
    print(f"\n💡 베이지안 네트워크 해석:")
    if edges:
        print(f"   - 총 {len(edges)}개의 인과관계가 발견됨")
        print(f"   - 네트워크 밀도: {nx.density(G):.3f}")
        
        # 가장 연결이 많은 노드 찾기
        if G.nodes():
            max_degree_node = max(G.nodes(), key=lambda x: G.degree(x))
            print(f"   - 가장 연결이 많은 변수: {max_degree_node} (연결도: {G.degree(max_degree_node)})")
    else:
        print(f"   - 인과관계가 발견되지 않음")
    
    print(f"\n✅ 베이지안 네트워크 구축 완료!")
    
    return model, infer, network_stats if 'network_stats' in locals() else {}



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

# Granger 인과성 분석 (PCMCI 대안) - 시각화 및 결론 포함
from statsmodels.tsa.stattools import grangercausalitytests

def granger_causality_analysis(data, maxlag=5, visualize=True):
    """
    Granger 인과성 기반 시간지연 관계 분석
    시각화와 결론 출력 포함
    """
    print("\n" + "=" * 60)
    print("Granger 인과성 분석")
    print("=" * 60)
    
    print(f"\n🔍 Granger 인과성 분석 설정:")
    print(f"   - 최대 지연 (maxlag): {maxlag}")
    print(f"   - 분석 변수: {len(data.columns)}개")
    print(f"   - 데이터 크기: {data.shape}")
    
    variables = data.columns.tolist()
    granger_matrix = pd.DataFrame(np.zeros((len(variables), len(variables))),
                                 columns=variables, index=variables)
    
    # 상세 분석 결과 저장
    detailed_results = {}
    significant_relationships = []
    
    print(f"\n⚙️ Granger 인과성 검정 실행 중...")
    
    for target in variables:
        detailed_results[target] = {}
        for cause in variables:
            if target != cause:
                try:
                    test_data = data[[target, cause]].dropna()
                    if len(test_data) < maxlag + 10:  # 최소 데이터 요구량
                        granger_matrix.loc[cause, target] = 1.0
                        detailed_results[target][cause] = {'p_value': 1.0, 'lag': 0, 'status': 'insufficient_data'}
                        continue
                    
                    test_result = grangercausalitytests(test_data, maxlag=maxlag, verbose=False)
                    
                    # 최소 p-value와 해당 lag 찾기
                    p_values = []
                    lag_results = {}
                    for lag in range(1, maxlag + 1):
                        if lag in test_result:
                            p_val = test_result[lag][0]['ssr_ftest'][1]
                            p_values.append(p_val)
                            lag_results[lag] = p_val
                    
                    if p_values:
                        min_p_value = min(p_values)
                        best_lag = min(lag_results, key=lag_results.get)
                        granger_matrix.loc[cause, target] = min_p_value
                        
                        detailed_results[target][cause] = {
                            'p_value': min_p_value,
                            'lag': best_lag,
                            'status': 'significant' if min_p_value < 0.05 else 'not_significant'
                        }
                        
                        if min_p_value < 0.05:
                            significant_relationships.append({
                                'cause': cause,
                                'target': target,
                                'p_value': min_p_value,
                                'lag': best_lag
                            })
                    else:
                        granger_matrix.loc[cause, target] = 1.0
                        detailed_results[target][cause] = {'p_value': 1.0, 'lag': 0, 'status': 'no_result'}
                        
                except Exception as e:
                    granger_matrix.loc[cause, target] = 1.0
                    detailed_results[target][cause] = {'p_value': 1.0, 'lag': 0, 'status': f'error: {str(e)}'}
    
    # 결과 분석
    print(f"\n📊 Granger 인과성 분석 결과:")
    print(f"   - 총 검정 쌍: {len(variables) * (len(variables) - 1)}개")
    print(f"   - 유의한 인과관계: {len(significant_relationships)}개")
    print(f"   - 유의수준: 0.05")
    
    if significant_relationships:
        print(f"\n🔗 발견된 유의한 인과관계:")
        for i, rel in enumerate(significant_relationships, 1):
            print(f"   {i}. {rel['cause']} → {rel['target']} (lag={rel['lag']}, p={rel['p_value']:.4f})")
    
    # 시각화
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Granger 인과성 히트맵
        mask = np.triu(np.ones_like(granger_matrix, dtype=bool))  # 상삼각 행렬 마스크
        sns.heatmap(granger_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0.05, ax=axes[0,0], cbar_kws={'label': 'P-value'})
        axes[0,0].set_title('Granger Causality P-values', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Target Variables')
        axes[0,0].set_ylabel('Cause Variables')
        
        # 2. 유의한 인과관계만 표시
        significant_matrix = granger_matrix.copy()
        significant_matrix[significant_matrix >= 0.05] = np.nan  # 유의하지 않은 값은 NaN으로
        
        if not significant_matrix.isna().all().all():
            sns.heatmap(significant_matrix, mask=mask, annot=True, cmap='Greens', 
                       ax=axes[0,1], cbar_kws={'label': 'P-value (Significant Only)'})
            axes[0,1].set_title('Significant Granger Causality (p < 0.05)', fontsize=14, fontweight='bold')
        else:
            axes[0,1].text(0.5, 0.5, 'No significant\nGranger causality\nfound', 
                          ha='center', va='center', transform=axes[0,1].transAxes, fontsize=12)
            axes[0,1].set_title('Significant Granger Causality', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('Target Variables')
        axes[0,1].set_ylabel('Cause Variables')
        
        # 3. 지연 시간 분포
        if significant_relationships:
            lags = [rel['lag'] for rel in significant_relationships]
            axes[1,0].hist(lags, bins=range(1, maxlag + 2), alpha=0.7, 
                          color='skyblue', edgecolor='black')
            axes[1,0].set_title('Distribution of Significant Lags', fontsize=14, fontweight='bold')
            axes[1,0].set_xlabel('Lag')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_xticks(range(1, maxlag + 1))
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5, 0.5, 'No significant\nrelationships\nfor lag analysis', 
                          ha='center', va='center', transform=axes[1,0].transAxes, fontsize=12)
            axes[1,0].set_title('Distribution of Significant Lags', fontsize=14, fontweight='bold')
        
        # 4. 변수별 인과성 통계
        causality_stats = {
            'Total Tests': len(variables) * (len(variables) - 1),
            'Significant': len(significant_relationships),
            'Significance Rate': len(significant_relationships) / (len(variables) * (len(variables) - 1)) * 100,
            'Avg P-value': granger_matrix.values[granger_matrix.values < 1.0].mean() if (granger_matrix.values < 1.0).any() else 1.0
        }
        
        axes[1,1].axis('off')
        
        # 테이블 형태로 통계 표시
        table_data = []
        for metric, value in causality_stats.items():
            if isinstance(value, float):
                table_data.append([metric, f'{value:.3f}'])
            else:
                table_data.append([metric, str(value)])
        
        table = axes[1,1].table(cellText=table_data,
                               colLabels=['Metric', 'Value'],
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        axes[1,1].set_title('Granger Causality Statistics', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.suptitle('Granger Causality Analysis', fontsize=16, y=1.02)
        plt.show()
        
        # 추가: 시간지연별 인과관계 강도 분석
        if significant_relationships:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 지연 시간별로 그룹화
            lag_groups = {}
            for rel in significant_relationships:
                lag = rel['lag']
                if lag not in lag_groups:
                    lag_groups[lag] = []
                lag_groups[lag].append(rel)
            
            # 지연 시간별 평균 p-value 계산
            lag_means = []
            lag_labels = []
            for lag in sorted(lag_groups.keys()):
                p_values = [rel['p_value'] for rel in lag_groups[lag]]
                lag_means.append(np.mean(p_values))
                lag_labels.append(f'Lag {lag}\n(n={len(p_values)})')
            
            bars = ax.bar(range(len(lag_means)), lag_means, 
                         color='lightcoral', edgecolor='darkred', alpha=0.7)
            ax.set_xlabel('Lag')
            ax.set_ylabel('Average P-value')
            ax.set_title('Average P-value by Lag (Significant Relationships Only)', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(range(len(lag_labels)))
            ax.set_xticklabels(lag_labels)
            ax.grid(True, alpha=0.3)
            
            # 값 라벨 추가
            for i, bar in enumerate(bars):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.show()
    
    # 결론 및 해석
    print(f"\n💡 Granger 인과성 해석:")
    if significant_relationships:
        print(f"   - {len(significant_relationships)}개의 유의한 시간지연 인과관계 발견")
        
        # 가장 강한 인과관계 찾기
        strongest = min(significant_relationships, key=lambda x: x['p_value'])
        print(f"   - 가장 강한 인과관계: {strongest['cause']} → {strongest['target']} (p={strongest['p_value']:.4f}, lag={strongest['lag']})")
        
        # 가장 일반적인 지연 시간
        if significant_relationships:
            most_common_lag = max(set(rel['lag'] for rel in significant_relationships), 
                                key=lambda x: sum(1 for rel in significant_relationships if rel['lag'] == x))
            print(f"   - 가장 일반적인 지연 시간: {most_common_lag}")
    else:
        print(f"   - 유의한 시간지연 인과관계가 발견되지 않음")
    
    print(f"\n✅ Granger 인과성 분석 완료!")
    
    return granger_matrix, detailed_results, significant_relationships




# SHAP 기반 특성 중요도 분석 (시각화 및 결론 포함)
def shap_analysis_acn_process(model, X_train, X_test, y_test=None, model_type='tree', visualize=True):
    """
    SHAP을 이용한 ACN 공정 변수 중요도 분석
    시각화와 결론 출력 포함
    """
    print("\n" + "=" * 60)
    print("SHAP 기반 특성 중요도 분석")
    print("=" * 60)
    
    print(f"\n🔍 SHAP 분석 설정:")
    print(f"   - 모델 타입: {model_type}")
    print(f"   - 훈련 데이터 크기: {X_train.shape}")
    print(f"   - 테스트 데이터 크기: {X_test.shape}")
    print(f"   - 특성 수: {len(X_test.columns)}개")
    
    # SHAP Explainer 설정
    print(f"\n⚙️ SHAP Explainer 설정 중...")
    try:
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, X_train)
        else:
            explainer = shap.Explainer(model, X_train)
        print(f"   ✅ {model_type} Explainer 설정 완료")
    except Exception as e:
        print(f"   ❌ Explainer 설정 실패: {str(e)}")
        return None, None, None
    
    # SHAP values 계산
    print(f"\n📊 SHAP values 계산 중...")
    try:
        shap_values = explainer.shap_values(X_test)
        print(f"   ✅ SHAP values 계산 완료")
    except Exception as e:
        print(f"   ❌ SHAP values 계산 실패: {str(e)}")
        return explainer, None, None
    
    # 중요도 순위 계산
    if len(shap_values.shape) > 1:
        mean_shap = np.abs(shap_values).mean(0)
    else:
        mean_shap = np.abs(shap_values).mean()
    
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'mean_shap': mean_shap,
        'std_shap': np.abs(shap_values).std(0) if len(shap_values.shape) > 1 else np.abs(shap_values).std()
    }).sort_values('mean_shap', ascending=False)
    
    # 모델 성능 평가
    if y_test is not None:
        try:
            y_pred = model.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            print(f"\n📈 모델 성능:")
            print(f"   - R² Score: {r2:.4f}")
            print(f"   - MSE: {mse:.4f}")
        except Exception as e:
            print(f"   - 모델 성능 평가 실패: {str(e)}")
    
    # 결과 분석
    print(f"\n📋 SHAP 중요도 분석 결과:")
    print(f"   - 가장 중요한 특성: {feature_importance.iloc[0]['feature']} ({feature_importance.iloc[0]['mean_shap']:.4f})")
    print(f"   - 가장 덜 중요한 특성: {feature_importance.iloc[-1]['feature']} ({feature_importance.iloc[-1]['mean_shap']:.4f})")
    print(f"   - 중요도 범위: {feature_importance['mean_shap'].min():.4f} ~ {feature_importance['mean_shap'].max():.4f}")
    
    # 상위 3개 특성 출력
    print(f"\n🏆 상위 3개 중요 특성:")
    for i in range(min(3, len(feature_importance))):
        feature = feature_importance.iloc[i]
        print(f"   {i+1}. {feature['feature']}: {feature['mean_shap']:.4f} ± {feature['std_shap']:.4f}")
    
    # 시각화
    if visualize:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 특성 중요도 막대 그래프
        top_features = feature_importance.head(10)  # 상위 10개만 표시
        bars = axes[0,0].barh(range(len(top_features)), top_features['mean_shap'], 
                             color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0,0].set_yticks(range(len(top_features)))
        axes[0,0].set_yticklabels(top_features['feature'])
        axes[0,0].set_xlabel('Mean |SHAP value|')
        axes[0,0].set_title('Top 10 Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # 값 라벨 추가
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0,0].text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                          f'{width:.3f}', ha='left', va='center', fontweight='bold')
        
        # 2. SHAP Summary Plot (상위 10개 특성)
        if len(shap_values.shape) > 1:
            # 샘플 수를 제한하여 시각화 성능 향상
            sample_size = min(100, len(X_test))
            sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
            
            shap.summary_plot(shap_values[sample_indices], X_test.iloc[sample_indices], 
                             max_display=10, show=False, ax=axes[0,1])
            axes[0,1].set_title('SHAP Summary Plot (Top 10 Features)', fontsize=14, fontweight='bold')
        else:
            axes[0,1].text(0.5, 0.5, 'SHAP Summary Plot\nnot available for this model type', 
                          ha='center', va='center', transform=axes[0,1].transAxes, fontsize=12)
            axes[0,1].set_title('SHAP Summary Plot', fontsize=14, fontweight='bold')
        
        # 3. 특성 중요도 분포
        axes[1,0].hist(feature_importance['mean_shap'], bins=20, alpha=0.7, 
                      color='lightcoral', edgecolor='black')
        axes[1,0].set_xlabel('Mean |SHAP value|')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Distribution of Feature Importance', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. 특성별 SHAP 값 분산
        axes[1,1].scatter(feature_importance['mean_shap'], feature_importance['std_shap'], 
                         alpha=0.7, s=100, color='green', edgecolor='darkgreen')
        axes[1,1].set_xlabel('Mean |SHAP value|')
        axes[1,1].set_ylabel('Std |SHAP value|')
        axes[1,1].set_title('Feature Importance vs Variability', fontsize=14, fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # 특성명 라벨 추가
        for i, feature in enumerate(feature_importance['feature']):
            axes[1,1].annotate(feature, 
                              (feature_importance.iloc[i]['mean_shap'], 
                               feature_importance.iloc[i]['std_shap']),
                              fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        plt.suptitle('SHAP Feature Importance Analysis', fontsize=16, y=1.02)
        plt.show()
        
        # 추가: Waterfall Plot (첫 번째 샘플)
        if len(shap_values.shape) > 1 and len(shap_values) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # 첫 번째 샘플의 SHAP 값으로 waterfall plot 생성
            sample_shap = shap_values[0]
            base_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
            
            # 특성별 기여도 계산
            contributions = []
            feature_names = []
            for i, feature in enumerate(X_test.columns):
                contributions.append(sample_shap[i])
                feature_names.append(feature)
            
            # 기여도 순으로 정렬
            sorted_indices = np.argsort(np.abs(contributions))[::-1]
            sorted_contributions = [contributions[i] for i in sorted_indices]
            sorted_features = [feature_names[i] for i in sorted_indices]
            
            # Waterfall plot 생성
            cumulative = base_value
            positions = range(len(sorted_contributions))
            
            for i, (contrib, feature) in enumerate(zip(sorted_contributions, sorted_features)):
                color = 'green' if contrib > 0 else 'red'
                ax.bar(i, contrib, bottom=cumulative, color=color, alpha=0.7, 
                      edgecolor='black', linewidth=0.5)
                cumulative += contrib
            
            ax.axhline(y=base_value, color='blue', linestyle='--', alpha=0.7, label=f'Base Value: {base_value:.3f}')
            ax.axhline(y=cumulative, color='purple', linestyle='--', alpha=0.7, label=f'Final Prediction: {cumulative:.3f}')
            
            ax.set_xticks(positions)
            ax.set_xticklabels(sorted_features, rotation=45, ha='right')
            ax.set_ylabel('SHAP Value')
            ax.set_title('SHAP Waterfall Plot (First Sample)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    # 결론 및 해석
    print(f"\n💡 SHAP 분석 해석:")
    print(f"   - 총 {len(feature_importance)}개 특성 분석")
    print(f"   - 가장 영향력 있는 특성: {feature_importance.iloc[0]['feature']}")
    print(f"   - 중요도 차이: {feature_importance.iloc[0]['mean_shap'] / feature_importance.iloc[-1]['mean_shap']:.1f}배")
    
    # 특성 그룹별 분석
    high_importance = feature_importance[feature_importance['mean_shap'] > feature_importance['mean_shap'].quantile(0.75)]
    medium_importance = feature_importance[(feature_importance['mean_shap'] > feature_importance['mean_shap'].quantile(0.25)) & 
                                         (feature_importance['mean_shap'] <= feature_importance['mean_shap'].quantile(0.75))]
    low_importance = feature_importance[feature_importance['mean_shap'] <= feature_importance['mean_shap'].quantile(0.25)]
    
    print(f"\n📊 특성 중요도 그룹별 분석:")
    print(f"   - 고중요도 특성: {len(high_importance)}개")
    print(f"   - 중중요도 특성: {len(medium_importance)}개")
    print(f"   - 저중요도 특성: {len(low_importance)}개")
    
    print(f"\n✅ SHAP 분석 완료!")
    
    return explainer, shap_values, feature_importance

# 통합 시각화 함수 (중복 제거 및 개선)
def create_comprehensive_visualization(dag, shap_importance, granger_matrix, 
                                     bayesian_stats=None, dowhy_results=None):
    """
    모든 인과분석 결과를 종합한 시각화
    """
    print("\n" + "=" * 60)
    print("종합 인과분석 결과 시각화")
    print("=" * 60)
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    
    # 1. DAG 시각화
    if dag and len(dag.edges()) > 0:
        pos = nx.spring_layout(dag, k=3, iterations=50, seed=42)
        node_sizes = [dag.degree(node) * 500 + 1000 for node in dag.nodes()]
        nx.draw_networkx(dag, pos, ax=axes[0,0], 
                        node_color='lightblue', 
                        node_size=node_sizes,
                        font_size=10, 
                        arrows=True,
                        arrowsize=20,
                        edge_color='gray')
        axes[0,0].set_title('Causal DAG Structure (PC Algorithm)', fontsize=14, fontweight='bold')
    else:
        axes[0,0].text(0.5, 0.5, 'No causal structure\nfound', ha='center', va='center',
                      transform=axes[0,0].transAxes, fontsize=12)
        axes[0,0].set_title('Causal DAG Structure', fontsize=14, fontweight='bold')
    axes[0,0].axis('off')
    
    # 2. SHAP 중요도
    if shap_importance is not None and len(shap_importance) > 0:
        top_features = shap_importance.head(8)  # 상위 8개만 표시
        bars = axes[0,1].barh(range(len(top_features)), top_features['mean_shap'],
                             color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0,1].set_yticks(range(len(top_features)))
        axes[0,1].set_yticklabels(top_features['feature'])
        axes[0,1].set_xlabel('Mean |SHAP value|')
        axes[0,1].set_title('SHAP Feature Importance', fontsize=14, fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # 값 라벨 추가
        for i, bar in enumerate(bars):
            width = bar.get_width()
            axes[0,1].text(width + 0.001, bar.get_y() + bar.get_height()/2.,
                          f'{width:.3f}', ha='left', va='center', fontweight='bold')
    else:
        axes[0,1].text(0.5, 0.5, 'SHAP analysis\nnot available', ha='center', va='center',
                      transform=axes[0,1].transAxes, fontsize=12)
        axes[0,1].set_title('SHAP Feature Importance', fontsize=14, fontweight='bold')
    
    # 3. Granger 인과성 히트맵
    if granger_matrix is not None:
        mask = np.triu(np.ones_like(granger_matrix, dtype=bool))
        sns.heatmap(granger_matrix, mask=mask, annot=True, cmap='RdYlBu_r', 
                   center=0.05, ax=axes[1,0], cbar_kws={'label': 'P-value'})
        axes[1,0].set_title('Granger Causality P-values', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('Target Variables')
        axes[1,0].set_ylabel('Cause Variables')
    else:
        axes[1,0].text(0.5, 0.5, 'Granger causality\nanalysis not available', ha='center', va='center',
                      transform=axes[1,0].transAxes, fontsize=12)
        axes[1,0].set_title('Granger Causality', fontsize=14, fontweight='bold')
    
    # 4. 결합된 중요도 분석
    if dag and shap_importance is not None:
        try:
            dag_centrality = pd.Series(nx.degree_centrality(dag))
            combined_df = pd.DataFrame({
                'dag_centrality': dag_centrality,
                'shap_importance': shap_importance.set_index('feature')['mean_shap']
            }).fillna(0)
            
            if len(combined_df) > 0:
                scatter = axes[1,1].scatter(combined_df['dag_centrality'], 
                                          combined_df['shap_importance'],
                                          s=100, alpha=0.7, c='green', edgecolor='darkgreen')
                axes[1,1].set_xlabel('DAG Centrality')
                axes[1,1].set_ylabel('SHAP Importance')
                axes[1,1].set_title('Causal Structure vs. Feature Importance', fontsize=14, fontweight='bold')
                axes[1,1].grid(True, alpha=0.3)
                
                # 각 점에 변수명 라벨링
                for i, txt in enumerate(combined_df.index):
                    axes[1,1].annotate(txt, 
                                      (combined_df['dag_centrality'].iloc[i],
                                       combined_df['shap_importance'].iloc[i]),
                                      fontsize=8, alpha=0.7)
            else:
                axes[1,1].text(0.5, 0.5, 'No data for\ncombined analysis', ha='center', va='center',
                              transform=axes[1,1].transAxes, fontsize=12)
                axes[1,1].set_title('Combined Analysis', fontsize=14, fontweight='bold')
        except Exception as e:
            axes[1,1].text(0.5, 0.5, f'Combined analysis\nfailed: {str(e)[:30]}...', ha='center', va='center',
                          transform=axes[1,1].transAxes, fontsize=12)
            axes[1,1].set_title('Combined Analysis', fontsize=14, fontweight='bold')
    else:
        axes[1,1].text(0.5, 0.5, 'Insufficient data for\ncombined analysis', ha='center', va='center',
                      transform=axes[1,1].transAxes, fontsize=12)
        axes[1,1].set_title('Combined Analysis', fontsize=14, fontweight='bold')
    
    # 5. 베이지안 네트워크 통계
    if bayesian_stats:
        stats_data = []
        for metric, value in bayesian_stats.items():
            stats_data.append([metric, f'{value:.3f}' if isinstance(value, float) else str(value)])
        
        axes[2,0].axis('off')
        table = axes[2,0].table(cellText=stats_data,
                               colLabels=['Bayesian Network Metric', 'Value'],
                               cellLoc='center',
                               loc='center',
                               bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        axes[2,0].set_title('Bayesian Network Statistics', fontsize=14, fontweight='bold', pad=20)
    else:
        axes[2,0].text(0.5, 0.5, 'Bayesian network\nstatistics not available', ha='center', va='center',
                      transform=axes[2,0].transAxes, fontsize=12)
        axes[2,0].set_title('Bayesian Network Statistics', fontsize=14, fontweight='bold')
    
    # 6. DoWhy 결과 요약
    if dowhy_results:
        estimates = dowhy_results.get('estimates', {})
        valid_estimates = {k: v for k, v in estimates.items() if v is not None}
        
        if valid_estimates:
            methods = list(valid_estimates.keys())
            values = list(valid_estimates.values())
            
            bars = axes[2,1].bar(methods, values, 
                               color=['skyblue', 'lightcoral', 'lightgreen'][:len(methods)],
                               edgecolor='navy', alpha=0.7)
            axes[2,1].set_title('DoWhy Causal Effect Estimates', fontsize=14, fontweight='bold')
            axes[2,1].set_ylabel('Effect Size')
            axes[2,1].grid(True, alpha=0.3)
            
            # 값 라벨 추가
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[2,1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                              f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        else:
            axes[2,1].text(0.5, 0.5, 'No valid DoWhy\nestimates available', ha='center', va='center',
                          transform=axes[2,1].transAxes, fontsize=12)
            axes[2,1].set_title('DoWhy Results', fontsize=14, fontweight='bold')
    else:
        axes[2,1].text(0.5, 0.5, 'DoWhy analysis\nnot available', ha='center', va='center',
                      transform=axes[2,1].transAxes, fontsize=12)
        axes[2,1].set_title('DoWhy Results', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('Comprehensive Causal Analysis Results', fontsize=18, y=1.02)
    plt.show()
    
    print("✅ 종합 시각화 완료!")
    
    return fig




class ACNCausalAnalyzer:
    """
    ACN 공정 인과분석을 위한 통합 클래스 (개선된 버전)
    모든 분석 과정에 시각화와 결론 출력 포함
    """
    
    def __init__(self, data, visualize=True):
        self.raw_data = data
        self.processed_data = None
        self.causal_dag = None
        self.bayesian_model = None
        self.shap_results = None
        self.time_series_results = None
        self.dowhy_results = None
        self.bayesian_stats = None
        self.visualize = visualize
        
    def preprocess_data(self, vif_threshold=5):
        """데이터 전처리 파이프라인 (시각화 및 결론 포함)"""
        self.processed_data, self.scaler, self.outlier_info = preprocess_acn_data(
            self.raw_data, visualize=self.visualize
        )
        
        # VIF 기반 특성 선택
        features_to_keep, self.removed_features, self.final_vif = handle_multicollinearity(
            self.processed_data.select_dtypes(include=[np.number]), 
            threshold=vif_threshold, visualize=self.visualize
        )
        
        # acn_yield가 있다면 포함
        if 'acn_yield' in self.processed_data.columns:
            features_to_keep = features_to_keep + ['acn_yield'] if 'acn_yield' not in features_to_keep else features_to_keep
        
        self.processed_data = self.processed_data[features_to_keep]
        
    def discover_causal_structure(self, method='pc', alpha=0.05):
        """인과구조 발견 (시각화 및 결론 포함)"""
        if method == 'pc':
            self.causal_dag, self.pc_result, self.centrality_measures, self.edges_found = discover_causal_structure_pc(
                self.processed_data, alpha=alpha, visualize=self.visualize
            )
        elif method == 'ges':
            # GES 구현 (향후 확장 가능)
            print("GES method not yet implemented. Using PC algorithm instead.")
            self.causal_dag, self.pc_result, self.centrality_measures, self.edges_found = discover_causal_structure_pc(
                self.processed_data, alpha=alpha, visualize=self.visualize
            )
        
    def analyze_time_series_causality(self, time_col=None, tau_max=5):
        """시계열 인과관계 분석 (시각화 및 결론 포함)"""
        if time_col and time_col in self.processed_data.columns:
            ts_data = self.processed_data.set_index(time_col)
        else:
            ts_data = self.processed_data
        
        # Granger causality analysis
        self.time_series_results, self.granger_details, self.significant_relationships = granger_causality_analysis(
            ts_data, maxlag=tau_max, visualize=self.visualize
        )
        
    def fit_bayesian_network(self):
        """베이지안 네트워크 학습 (시각화 및 결론 포함)"""
        self.bayesian_model, self.inference_engine, self.bayesian_stats = build_bayesian_network(
            self.processed_data, structure=self.causal_dag, visualize=self.visualize
        )
        
    def explain_with_shap(self, target_var='acn_yield', model_type='rf'):
        """SHAP 기반 설명 (시각화 및 결론 포함)"""
        # 간단한 예측 모델 학습 (SHAP 분석용)
        X = self.processed_data.drop(columns=[target_var])
        y = self.processed_data[target_var]
        
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # SHAP 분석
        explainer, shap_values, importance = shap_analysis_acn_process(
            model, X_train, X_test, y_test, model_type='tree', visualize=self.visualize
        )
        
        self.shap_results = {
            'explainer': explainer,
            'shap_values': shap_values,
            'importance': importance,
            'model': model,
            'X_test': X_test,
            'y_test': y_test
        }
        
    def estimate_causal_effects(self, treatment='temperature', outcome='acn_yield'):
        """DoWhy 기반 인과효과 추정 (시각화 및 결론 포함)"""
        # 공통 원인 설정 (treatment와 outcome을 제외한 변수들)
        common_causes = [col for col in self.processed_data.columns 
                        if col not in [treatment, outcome]]
        
        if len(common_causes) == 0:
            print("⚠️ 공통 원인이 없어 DoWhy 분석을 건너뜁니다.")
            return
        
        estimates, refutation_results, estimate_details = estimate_causal_effect_dowhy(
            self.processed_data, treatment, outcome, common_causes, visualize=self.visualize
        )
        
        self.dowhy_results = {
            'estimates': estimates,
            'refutation_results': refutation_results,
            'estimate_details': estimate_details,
            'treatment': treatment,
            'outcome': outcome,
            'common_causes': common_causes
        }
        
    def generate_optimization_recommendations(self):
        """최적화 권고사항 생성 (개선된 버전)"""
        print("\n" + "=" * 60)
        print("최적화 권고사항 생성")
        print("=" * 60)
        
        recommendations = {}
        
        # SHAP 중요도 기반 권고
        if self.shap_results and self.shap_results['importance'] is not None:
            top_features = self.shap_results['importance'].head(3)
            recommendations['top_influential_vars'] = top_features['feature'].tolist()
            print(f"\n🎯 SHAP 기반 권고:")
            print(f"   - 가장 영향력 있는 변수: {top_features.iloc[0]['feature']}")
            print(f"   - 상위 3개 변수: {', '.join(top_features['feature'].tolist())}")
        
        # 인과구조 기반 권고
        if self.causal_dag and len(self.causal_dag.edges()) > 0:
            centrality = nx.degree_centrality(self.causal_dag)
            most_central = max(centrality, key=centrality.get)
            recommendations['most_connected_var'] = most_central
            print(f"\n🔗 인과구조 기반 권고:")
            print(f"   - 가장 연결이 많은 변수: {most_central} (중심성: {centrality[most_central]:.3f})")
        
        # 시계열 분석 기반 권고
        if hasattr(self, 'significant_relationships') and self.significant_relationships:
            # ACN yield에 가장 강한 Granger cause 찾기
            yield_causes = [rel for rel in self.significant_relationships 
                          if rel['target'] == 'acn_yield']
            if yield_causes:
                strongest_cause = min(yield_causes, key=lambda x: x['p_value'])
                recommendations['strongest_temporal_cause'] = strongest_cause['cause']
                print(f"\n⏰ 시계열 분석 기반 권고:")
                print(f"   - ACN yield의 가장 강한 시간지연 원인: {strongest_cause['cause']}")
                print(f"   - 지연 시간: {strongest_cause['lag']}, p-value: {strongest_cause['p_value']:.4f}")
        
        # DoWhy 결과 기반 권고
        if self.dowhy_results and self.dowhy_results['estimates']:
            backdoor_effect = self.dowhy_results['estimates'].get('backdoor')
            if backdoor_effect is not None:
                treatment = self.dowhy_results['treatment']
                outcome = self.dowhy_results['outcome']
                recommendations['causal_effect'] = {
                    'treatment': treatment,
                    'outcome': outcome,
                    'effect_size': backdoor_effect
                }
                print(f"\n📊 인과효과 기반 권고:")
                print(f"   - {treatment} → {outcome} 인과효과: {backdoor_effect:.4f}")
        
        # 종합 권고사항
        print(f"\n💡 종합 최적화 권고사항:")
        if 'top_influential_vars' in recommendations:
            print(f"   1. 주요 제어 변수: {recommendations['top_influential_vars'][0]} 집중 관리")
        if 'most_connected_var' in recommendations:
            print(f"   2. 네트워크 중심 변수: {recommendations['most_connected_var']} 모니터링 강화")
        if 'strongest_temporal_cause' in recommendations:
            print(f"   3. 시간지연 고려: {recommendations['strongest_temporal_cause']} 조기 대응")
        
        return recommendations
        
    def run_full_analysis(self, treatment='temperature', outcome='acn_yield'):
        """전체 분석 파이프라인 실행 (개선된 버전)"""
        print("🚀 ACN 공정 종합 인과분석 시작")
        print("=" * 80)
        
        try:
            # 1. 전처리
            print("\n📋 Step 1: 데이터 전처리")
            self.preprocess_data()
            
            # 2. 인과구조 발견
            print("\n🔍 Step 2: 인과구조 발견")
            self.discover_causal_structure(method='pc')
            
            # 3. 시계열 분석
            print("\n⏰ Step 3: 시계열 인과관계 분석")
            self.analyze_time_series_causality()
            
            # 4. 베이지안 네트워크
            print("\n🧠 Step 4: 베이지안 네트워크 구축")
            self.fit_bayesian_network()
            
            # 5. SHAP 분석
            print("\n📊 Step 5: SHAP 특성 중요도 분석")
            self.explain_with_shap(target_var=outcome)
            
            # 6. DoWhy 인과효과 추정
            print("\n🎯 Step 6: DoWhy 인과효과 추정")
            self.estimate_causal_effects(treatment=treatment, outcome=outcome)
            
            # 7. 권고사항 생성
            print("\n💡 Step 7: 최적화 권고사항 생성")
            recommendations = self.generate_optimization_recommendations()
            
            # 8. 종합 시각화
            if self.visualize:
                print("\n📈 Step 8: 종합 결과 시각화")
                create_comprehensive_visualization(
                    self.causal_dag,
                    self.shap_results['importance'] if self.shap_results else None,
                    self.time_series_results,
                    self.bayesian_stats,
                    self.dowhy_results
                )
            
            print("\n" + "=" * 80)
            print("🎉 ACN 공정 종합 인과분석 완료!")
            print("=" * 80)
            
            return recommendations
            
        except Exception as e:
            print(f"\n❌ 분석 중 오류 발생: {str(e)}")
            print("부분적인 결과라도 확인해보세요.")
            return {}

# 사용 예제 (개선된 버전)
def main():
    """
    ACN 공정 데이터 분석 실행 예제 (시각화 및 결론 포함)
    """
    print("🔬 ACN 공정 인과추론 분석 시스템")
    print("=" * 50)
    
    # 샘플 데이터 생성 (실제 사용시 실공정 데이터 로드)
    np.random.seed(42)
    n_samples = 1000
    
    print(f"\n📊 샘플 데이터 생성 중... (n={n_samples})")
    
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
    
    print(f"✅ 샘플 데이터 생성 완료!")
    print(f"   - 데이터 크기: {sample_data.shape}")
    print(f"   - 변수: {list(sample_data.columns)}")
    
    # 분석 실행
    print(f"\n🚀 인과분석 시작...")
    analyzer = ACNCausalAnalyzer(sample_data, visualize=True)
    recommendations = analyzer.run_full_analysis(
        treatment='temperature', 
        outcome='acn_yield'
    )
    
    # 최종 결과 요약
    print(f"\n" + "=" * 80)
    print("📋 최종 분석 결과 요약")
    print("=" * 80)
    
    if recommendations:
        print(f"\n🎯 주요 발견사항:")
        for key, value in recommendations.items():
            if isinstance(value, list):
                print(f"   - {key}: {', '.join(map(str, value))}")
            elif isinstance(value, dict):
                print(f"   - {key}: {value}")
            else:
                print(f"   - {key}: {value}")
    
    print(f"\n💡 활용 방안:")
    print(f"   1. 공정 최적화: 주요 영향 변수 집중 관리")
    print(f"   2. 품질 향상: 인과관계 기반 제어 전략 수립")
    print(f"   3. 예측 모델: 발견된 인과구조를 활용한 예측 시스템 구축")
    print(f"   4. 모니터링: 시간지연 관계를 고려한 조기 경보 시스템")
    
    print(f"\n✅ 분석 완료! 모든 결과가 시각화와 함께 출력되었습니다.")

if __name__ == "__main__":
    main()
