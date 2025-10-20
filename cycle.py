import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from dtaidistance import dtw
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (14, 8)
sns.set_style("whitegrid")

# ============================================================================
# 1. 샘플 데이터 생성 (실제 데이터로 교체 필요)
# ============================================================================
def generate_sample_data(n_batches=50):
    """
    샘플 ACN 정제 배치 데이터 생성
    각 배치: phase1, phase2, phase3의 시계열 데이터
    """
    data = []
    
    for batch_id in range(n_batches):
        # 각 phase의 길이 (가변)
        len_p1 = np.random.randint(80, 150)
        len_p2 = np.random.randint(100, 200)
        len_p3 = np.random.randint(60, 120)
        
        # 기본 궤적에 노이즈 추가
        noise_scale = np.random.uniform(0.3, 1.5)
        golden_factor = np.random.uniform(0.6, 1.4)  # 고수율 근접도
        
        # Phase 1: 상승 추세
        p1 = 20 + 30 * np.linspace(0, 1, len_p1) + np.random.normal(0, noise_scale, len_p1)
        
        # Phase 2: 오실레이션 포함 (golden cycle은 작은 오실레이션)
        t2 = np.linspace(0, 4*np.pi, len_p2)
        p2 = 50 + 15*np.sin(t2) * golden_factor + np.random.normal(0, noise_scale, len_p2)
        
        # Phase 3: 하강 추세
        p3 = 65 - 20 * np.linspace(0, 1, len_p3) + np.random.normal(0, noise_scale, len_p3)
        
        # 수율 계산 (오실레이션 안정성, 기울기 등에 따라)
        osc_quality = 1.0 / (1.0 + np.std(np.diff(p2)))  # 작은 오실레이션이 좋음
        trend_quality = np.abs(np.mean(np.diff(p1))) / 10.0  # 적절한 상승 기울기
        yield_val = 70 + 20 * osc_quality + 10 * trend_quality + np.random.normal(0, 3)
        yield_val = np.clip(yield_val, 50, 95)
        
        batch_data = {
            'batch_id': f'B{batch_id:03d}',
            'phase1': p1.tolist(),
            'phase2': p2.tolist(),
            'phase3': p3.tolist(),
            'yield': yield_val
        }
        data.append(batch_data)
    
    return data

# ============================================================================
# 2. 시간 정규화 및 보간
# ============================================================================
def normalize_timeseries(phase_data, target_length=200):
    """
    가변 길이의 phase를 고정 길이로 정규화
    """
    original_length = len(phase_data)
    x_old = np.linspace(0, 1, original_length)
    x_new = np.linspace(0, 1, target_length)
    
    f = interp1d(x_old, phase_data, kind='cubic', fill_value='extrapolate')
    normalized = f(x_new)
    
    return normalized

def prepare_normalized_data(raw_data):
    """
    모든 배치의 phase를 정규화된 형태로 변환
    """
    normalized_data = []
    
    for batch in raw_data:
        normalized_batch = {
            'batch_id': batch['batch_id'],
            'phase1': normalize_timeseries(batch['phase1']),
            'phase2': normalize_timeseries(batch['phase2']),
            'phase3': normalize_timeseries(batch['phase3']),
            'yield': batch['yield']
        }
        normalized_data.append(normalized_batch)
    
    return normalized_data

# ============================================================================
# 3. 특성 추출 (Feature Extraction)
# ============================================================================
def extract_features(phase_data, phase_name):
    """
    단일 phase에서 다양한 특성 추출
    """
    features = {}
    
    # 기본 통계량
    features[f'{phase_name}_mean'] = np.mean(phase_data)
    features[f'{phase_name}_std'] = np.std(phase_data)
    features[f'{phase_name}_min'] = np.min(phase_data)
    features[f'{phase_name}_max'] = np.max(phase_data)
    features[f'{phase_name}_range'] = np.max(phase_data) - np.min(phase_data)
    
    # 트렌드 (1차 및 2차 미분)
    diff1 = np.diff(phase_data)
    diff2 = np.diff(diff1)
    
    features[f'{phase_name}_trend'] = np.mean(diff1)
    features[f'{phase_name}_trend_std'] = np.std(diff1)
    features[f'{phase_name}_acceleration'] = np.mean(diff2)
    
    # 오실레이션 분석
    features[f'{phase_name}_oscillation_amp'] = np.std(diff1)
    
    # FFT를 통한 주파수 분석
    fft_vals = np.abs(fft(phase_data - np.mean(phase_data)))
    freqs = fftfreq(len(phase_data))
    dominant_freq_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
    features[f'{phase_name}_dominant_freq'] = freqs[dominant_freq_idx]
    features[f'{phase_name}_spectral_power'] = fft_vals[dominant_freq_idx]
    
    # 극값 분석
    peaks = np.where((phase_data[1:-1] > phase_data[:-2]) & 
                     (phase_data[1:-1] > phase_data[2:]))[0] + 1
    valleys = np.where((phase_data[1:-1] < phase_data[:-2]) & 
                       (phase_data[1:-1] < phase_data[2:]))[0] + 1
    
    features[f'{phase_name}_n_peaks'] = len(peaks)
    features[f'{phase_name}_n_valleys'] = len(valleys)
    
    if len(peaks) > 0:
        features[f'{phase_name}_peak_amp'] = np.mean(phase_data[peaks])
        features[f'{phase_name}_peak_variance'] = np.var(phase_data[peaks])
    else:
        features[f'{phase_name}_peak_amp'] = 0
        features[f'{phase_name}_peak_variance'] = 0
    
    # 곡률 (Curvature)
    if len(diff2) > 0:
        curvature = np.abs(diff2) / (1 + np.abs(diff1[:-1])**2)**1.5
        features[f'{phase_name}_curvature_mean'] = np.mean(curvature)
    else:
        features[f'{phase_name}_curvature_mean'] = 0
    
    # 에너지
    features[f'{phase_name}_energy'] = np.sum(phase_data**2)
    
    return features

def extract_all_features(normalized_data):
    """
    모든 배치에서 특성 추출
    """
    feature_list = []
    batch_ids = []
    yields = []
    
    for batch in normalized_data:
        batch_features = {}
        
        for phase_name in ['phase1', 'phase2', 'phase3']:
            phase_features = extract_features(batch[phase_name], phase_name)
            batch_features.update(phase_features)
        
        feature_list.append(batch_features)
        batch_ids.append(batch['batch_id'])
        yields.append(batch['yield'])
    
    features_df = pd.DataFrame(feature_list)
    features_df['batch_id'] = batch_ids
    features_df['yield'] = yields
    
    return features_df

# ============================================================================
# 4. DTW 거리 계산
# ============================================================================
def calculate_dtw_distance(phase1, phase2):
    """
    두 시계열 간의 DTW 거리 계산
    """
    try:
        return dtw.distance(phase1, phase2)
    except:
        return np.inf

def calculate_batch_similarity_matrix(normalized_data):
    """
    모든 배치 쌍 간의 DTW 기반 유사도 행렬 계산
    """
    n_batches = len(normalized_data)
    similarity_matrix = np.zeros((n_batches, n_batches))
    
    for i in range(n_batches):
        for j in range(i, n_batches):
            # Phase별 거리의 가중 평균
            d1 = calculate_dtw_distance(normalized_data[i]['phase1'], 
                                        normalized_data[j]['phase1'])
            d2 = calculate_dtw_distance(normalized_data[i]['phase2'], 
                                        normalized_data[j]['phase2'])
            d3 = calculate_dtw_distance(normalized_data[i]['phase3'], 
                                        normalized_data[j]['phase3'])
            
            distance = (d1 + d2 + d3) / 3.0
            similarity_matrix[i, j] = distance
            similarity_matrix[j, i] = distance
    
    return similarity_matrix

# ============================================================================
# 4-1. 최적 클러스터 수 결정
# ============================================================================
def determine_optimal_clusters(features_df, top_features, max_clusters=10):
    """
    Elbow method, Silhouette score, Gap statistic을 이용한 최적 클러스터 수 결정
    """
    X = features_df[top_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)
    
    from sklearn.metrics import silhouette_score
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))
    
    # Elbow point 찾기 (이차 미분)
    elbow_point = np.argmax(np.diff(inertias, 2)) + 2
    
    # Silhouette score 최고점
    best_silhouette_k = list(K_range)[np.argmax(silhouette_scores)]
    
    print("\n=== 최적 클러스터 수 결정 ===")
    print(f"Elbow Method: {elbow_point}개 클러스터 추천")
    print(f"Silhouette Score: {best_silhouette_k}개 클러스터 추천 (점수: {max(silhouette_scores):.4f})")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow curve
    ax = axes[0]
    ax.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax.axvline(x=elbow_point, color='red', linestyle='--', linewidth=2, label=f'Elbow: K={elbow_point}')
    ax.set_xlabel('클러스터 수 (K)')
    ax.set_ylabel('관성 (Inertia)')
    ax.set_title('Elbow Method - 최적 K 결정')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Silhouette score
    ax = axes[1]
    ax.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax.axvline(x=best_silhouette_k, color='red', linestyle='--', linewidth=2, 
               label=f'Best: K={best_silhouette_k}')
    ax.set_xlabel('클러스터 수 (K)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score - 최적 K 결정')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 추천 클러스터 수 (두 방법의 평균)
    optimal_k = int(np.mean([elbow_point, best_silhouette_k]))
    
    return optimal_k, fig

# ============================================================================
# 4-2. Golden Batch 식별 (클러스터에 의존하지 않음)
# ============================================================================
def identify_golden_batches_statistical(features_df, yield_threshold_percentile=75):
    """
    수율 기반 통계적 임계값으로 Golden batch 식별 (클러스터 비의존적)
    
    Parameters:
    - yield_threshold_percentile: 상위 몇 퍼센타일을 golden으로 정의할 것인가
    """
    yield_threshold = np.percentile(features_df['yield'], yield_threshold_percentile)
    golden_mask = features_df['yield'] >= yield_threshold
    
    print(f"\n=== 통계적 Golden Batch 식별 (상위 {100-yield_threshold_percentile}%) ===")
    print(f"수율 임계값: {yield_threshold:.2f}%")
    print(f"Golden batch 수: {golden_mask.sum()}개 ({golden_mask.sum()/len(features_df)*100:.1f}%)")
    print(f"일반 batch 수: {(~golden_mask).sum()}개")
    
    return golden_mask, yield_threshold

def identify_golden_batches_multimodal(features_df, top_features):
    """
    다중 기준을 이용한 Golden batch 식별 (수율 + 안정성 + 특성)
    """
    from scipy.stats import zscore
    
    # 정규화
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df[top_features].values)
    
    # 1. 수율 기준 (Z-score > 1)
    yield_zscore = zscore(features_df['yield'].values)
    yield_criterion = yield_zscore > 0.5
    
    # 2. 안정성 기준 (변동성이 낮은 배치)
    # phase별 표준편차가 낮은 것이 좋음
    stability_features = [f for f in top_features if 'std' in f or 'oscillation' in f]
    if stability_features:
        stability_scores = np.mean(features_scaled[:, [list(top_features).index(f) 
                                                        for f in stability_features if f in top_features]], axis=1)
        stability_criterion = stability_scores < np.percentile(stability_scores, 50)
    else:
        stability_criterion = np.ones(len(features_df), dtype=bool)
    
    # 3. 트렌드 기준 (예상된 추이를 따르는 배치)
    trend_features = [f for f in top_features if 'trend' in f]
    if trend_features:
        trend_indices = [list(top_features).index(f) for f in trend_features if f in top_features]
        trend_scores = np.mean(np.abs(features_scaled[:, trend_indices]), axis=1)
        trend_criterion = trend_scores > np.percentile(trend_scores, 50)
    else:
        trend_criterion = np.ones(len(features_df), dtype=bool)
    
    # 통합 기준
    golden_mask = yield_criterion & stability_criterion & trend_criterion
    
    print(f"\n=== 다중 기준 Golden Batch 식별 ===")
    print(f"수율 기준: {yield_criterion.sum()}개")
    print(f"안정성 기준: {stability_criterion.sum()}개")
    print(f"트렌드 기준: {trend_criterion.sum()}개")
    print(f"통합 (모두 만족): {golden_mask.sum()}개 ({golden_mask.sum()/len(features_df)*100:.1f}%)")
    
    return golden_mask

def identify_golden_batches_dbscan(features_df, top_features, eps=0.5, min_samples=3):
    """
    DBSCAN을 이용한 밀도 기반 Golden batch 식별
    - 고수율 배치들의 밀집된 영역 찾기
    """
    from sklearn.cluster import DBSCAN
    
    X = features_df[top_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 수율 고려 추가 차원 (정규화)
    yield_scaled = (features_df['yield'].values - features_df['yield'].mean()) / features_df['yield'].std()
    X_with_yield = np.column_stack([X_scaled, yield_scaled])
    
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_with_yield)
    
    # 각 클러스터의 평균 수율
    cluster_yields = {}
    for cluster_id in set(clusters):
        if cluster_id == -1:  # 노이즈 포인트
            continue
        cluster_mask = clusters == cluster_id
        cluster_yields[cluster_id] = features_df[cluster_mask]['yield'].mean()
    
    if cluster_yields:
        golden_cluster_id = max(cluster_yields, key=cluster_yields.get)
        golden_mask = clusters == golden_cluster_id
    else:
        golden_mask = np.zeros(len(features_df), dtype=bool)
    
    print(f"\n=== DBSCAN 기반 Golden Batch 식별 ===")
    print(f"발견된 클러스터 수: {len(set(clusters)) - (1 if -1 in clusters else 0)}")
    print(f"노이즈 포인트: {(clusters == -1).sum()}개")
    print(f"Golden cluster ID: {golden_cluster_id if cluster_yields else 'None'}")
    print(f"Golden batch 수: {golden_mask.sum()}개 ({golden_mask.sum()/len(features_df)*100:.1f}%)")
    
    return golden_mask, clusters

def plot_identification_methods(features_df, top_features, mask_statistical, mask_multimodal):
    """
    다양한 Golden batch 식별 방법 비교 시각화
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PCA로 2차원 축소
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(StandardScaler().fit_transform(features_df[top_features].values))
    
    # 1. 수율 분포
    ax = axes[0, 0]
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=features_df['yield'].values, 
                        cmap='RdYlGn', s=100, alpha=0.6, edgecolors='black')
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('수율 (%)')
    ax.set_title('PCA 공간의 배치 분포 (수율 기반 색상)', fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.grid(True, alpha=0.3)
    
    # 2. 통계적 방법
    ax = axes[0, 1]
    colors = ['red' if m else 'lightblue' for m in mask_statistical]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=100, alpha=0.6, edgecolors='black')
    ax.set_title('통계적 방법 (상위 25%)', fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.grid(True, alpha=0.3)
    
    # 3. 다중 기준 방법
    ax = axes[1, 0]
    colors = ['red' if m else 'lightblue' for m in mask_multimodal]
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, s=100, alpha=0.6, edgecolors='black')
    ax.set_title('다중 기준 방법 (수율+안정성+트렌드)', fontweight='bold')
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
    ax.grid(True, alpha=0.3)
    
    # 4. 방법 간 일치도
    ax = axes[1, 1]
    agreement = (mask_statistical & mask_multimodal).sum()
    stat_only = (mask_statistical & ~mask_multimodal).sum()
    multi_only = (~mask_statistical & mask_multimodal).sum()
    neither = (~mask_statistical & ~mask_multimodal).sum()
    
    labels = ['Both\nMethods', 'Statistical\nOnly', 'Multi-Criteria\nOnly', 'Neither']
    sizes = [agreement, stat_only, multi_only, neither]
    colors_pie = ['green', 'orange', 'lightcoral', 'lightgray']
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors_pie, startangle=90)
    ax.set_title('식별 방법 간 일치도 비교', fontweight='bold')
    
    plt.tight_layout()
    return fig

# ============================================================================
# 5. 상관성 분석 및 특성 선택
# ============================================================================
def analyze_feature_correlation(features_df):
    """
    특성과 수율 간의 상관성 분석
    """
    correlation_results = []
    
    for col in features_df.columns:
        if col not in ['batch_id', 'yield']:
            pearson_corr, p_val = pearsonr(features_df[col], features_df['yield'])
            spearman_corr, sp_val = spearmanr(features_df[col], features_df['yield'])
            
            correlation_results.append({
                'feature': col,
                'pearson_corr': pearson_corr,
                'spearman_corr': spearman_corr,
                'importance': np.abs(pearson_corr)
            })
    
    corr_df = pd.DataFrame(correlation_results)
    corr_df = corr_df.sort_values('importance', ascending=False)
    
    return corr_df

def select_top_features(features_df, corr_df, top_n=15):
    """
    상관성 기반 상위 특성 선택
    """
    top_features = corr_df.head(top_n)['feature'].tolist()
    return features_df[top_features + ['batch_id', 'yield']]

# ============================================================================
# 6. 클러스터링 및 Golden Cycle 정의
# ============================================================================
def identify_golden_clusters(features_df, top_features, n_clusters=None):
    """
    특성 기반 클러스터링으로 고수율 그룹 식별
    n_clusters가 None이면 자동으로 최적값 결정
    """
    if n_clusters is None:
        n_clusters, _ = determine_optimal_clusters(features_df, top_features)
    
    X = features_df[top_features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    features_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 클러스터별 수율 통계
    cluster_stats = features_df.groupby('cluster')['yield'].agg([
        'mean', 'std', 'count'
    ]).sort_values('mean', ascending=False)
    
    print("\n=== 클러스터별 수율 통계 ===")
    print(cluster_stats)
    
    # 최고 수율 클러스터 (Golden Cluster)
    golden_cluster = cluster_stats.index[0]
    
    return features_df, golden_cluster, cluster_stats

def calculate_golden_trajectory(normalized_data, golden_cluster_batches):
    """
    고수율 배치들의 평균 궤적 계산
    """
    golden_traj = {
        'phase1': np.zeros(200),
        'phase2': np.zeros(200),
        'phase3': np.zeros(200)
    }
    
    for phase_name in golden_traj.keys():
        phase_data = np.array([b[phase_name] for b in golden_cluster_batches])
        golden_traj[phase_name] = np.mean(phase_data, axis=0)
    
    return golden_traj

# ============================================================================
# 7. 예측 모델 구축
# ============================================================================
def build_yield_prediction_model(features_df, top_features):
    """
    Random Forest를 이용한 수율 예측 모델 구축
    """
    X = features_df[top_features].values
    y = features_df['yield'].values
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, 
                                   max_depth=10, min_samples_split=5)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    
    print(f"\n=== 예측 모델 성능 ===")
    print(f"R² Score: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # 특성 중요도
    feature_importance = pd.DataFrame({
        'feature': top_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return model, feature_importance

# ============================================================================
# 8. 시각화
# ============================================================================
def plot_normalized_batches(normalized_data, golden_cluster_mask, title="Batch Trajectories"):
    """
    모든 배치의 정규화된 궤적 시각화
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    phases = ['phase1', 'phase2', 'phase3']
    
    for idx, phase_name in enumerate(phases):
        ax = axes[idx]
        
        for i, batch in enumerate(normalized_data):
            color = 'red' if golden_cluster_mask[i] else 'lightgray'
            alpha = 0.8 if golden_cluster_mask[i] else 0.3
            ax.plot(batch[phase_name], color=color, alpha=alpha, linewidth=1.5)
        
        ax.set_title(f'{phase_name.upper()} - 정규화된 궤적')
        ax.set_xlabel('정규화된 시간')
        ax.set_ylabel('측정값')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_golden_vs_others(normalized_data, golden_cluster_mask, normalized_data_full):
    """
    Golden cluster vs Others 비교
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    phases = ['phase1', 'phase2', 'phase3']
    
    for col, phase_name in enumerate(phases):
        golden_data = np.array([normalized_data[i][phase_name] 
                               for i in range(len(normalized_data)) 
                               if golden_cluster_mask[i]])
        other_data = np.array([normalized_data[i][phase_name] 
                              for i in range(len(normalized_data)) 
                              if not golden_cluster_mask[i]])
        
        # 평균 및 신뢰도 구간
        golden_mean = np.mean(golden_data, axis=0)
        golden_std = np.std(golden_data, axis=0)
        
        other_mean = np.mean(other_data, axis=0) if len(other_data) > 0 else np.zeros(200)
        other_std = np.std(other_data, axis=0) if len(other_data) > 0 else np.zeros(200)
        
        time_axis = np.linspace(0, 1, 200)
        
        # 상단: 평균 궤적
        ax = axes[0, col]
        ax.fill_between(time_axis, golden_mean - golden_std, golden_mean + golden_std,
                        alpha=0.3, color='red', label='Golden (±1 std)')
        ax.plot(time_axis, golden_mean, 'r-', linewidth=2, label='Golden Mean')
        
        if len(other_data) > 0:
            ax.fill_between(time_axis, other_mean - other_std, other_mean + other_std,
                           alpha=0.2, color='blue', label='Others (±1 std)')
            ax.plot(time_axis, other_mean, 'b--', linewidth=2, label='Others Mean')
        
        ax.set_title(f'{phase_name.upper()} - 평균 궤적 비교')
        ax.set_xlabel('정규화된 시간')
        ax.set_ylabel('측정값')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 하단: 표준편차 비교
        ax = axes[1, col]
        ax.plot(time_axis, golden_std, 'r-', linewidth=2, label='Golden Std')
        if len(other_data) > 0:
            ax.plot(time_axis, other_std, 'b--', linewidth=2, label='Others Std')
        ax.set_title(f'{phase_name.upper()} - 안정성 (표준편차)')
        ax.set_xlabel('정규화된 시간')
        ax.set_ylabel('표준편차')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correlation_heatmap(features_df, top_features):
    """
    상위 특성들 간의 상관성 히트맵
    """
    corr_matrix = features_df[top_features].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={'label': '상관계수'})
    ax.set_title('상위 특성들 간의 상관관계')
    plt.tight_layout()
    
    return fig

def plot_feature_importance(feature_importance, top_n=15):
    """
    특성 중요도 시각화
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = feature_importance.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['importance'].values, color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'].values)
    ax.set_xlabel('중요도 (Feature Importance)')
    ax.set_title('수율 예측에 미치는 특성 중요도')
    plt.tight_layout()
    
    return fig

def plot_golden_trajectory(golden_trajectory, normalized_data, golden_mask):
    """
    Golden Cycle의 최적 추이를 상세하게 시각화
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    phases = ['phase1', 'phase2', 'phase3']
    phase_names = ['Phase 1\n(상승 추세)', 'Phase 2\n(오실레이션)', 'Phase 3\n(하강 추세)']
    
    time_axis = np.linspace(0, 1, 200)
    
    # ====== 상단: 각 phase별 상세 분석 ======
    for col, (phase_name, phase_label) in enumerate(zip(phases, phase_names)):
        ax = fig.add_subplot(gs[0, col])
        
        # Golden cluster 데이터
        golden_data = np.array([normalized_data[i][phase_name] 
                               for i in range(len(normalized_data)) 
                               if golden_mask.iloc[i]])
        
        # 모든 배치
        all_data = np.array([normalized_data[i][phase_name] 
                            for i in range(len(normalized_data))])
        
        # 개별 배치 (투명도)
        for trajectory in all_data:
            ax.plot(time_axis, trajectory, 'gray', alpha=0.15, linewidth=0.8)
        
        # Golden 배치들
        for trajectory in golden_data:
            ax.plot(time_axis, trajectory, 'lightcoral', alpha=0.4, linewidth=1.2)
        
        # Golden 평균 (굵은 선)
        ax.plot(time_axis, golden_trajectory[phase_name], 'r-', linewidth=3.5, 
                label='Golden Mean', zorder=10)
        
        # 신뢰도 구간
        golden_std = np.std(golden_data, axis=0)
        ax.fill_between(time_axis, 
                        golden_trajectory[phase_name] - golden_std,
                        golden_trajectory[phase_name] + golden_std,
                        alpha=0.25, color='red', label='±1 Std Dev', zorder=5)
        
        ax.set_title(phase_label, fontsize=12, fontweight='bold')
        ax.set_xlabel('정규화된 시간 진행도')
        ax.set_ylabel('측정값')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # ====== 중단: 미분 분석 (트렌드/속도) ======
    for col, phase_name in enumerate(phases):
        ax = fig.add_subplot(gs[1, col])
        
        golden_data = np.array([normalized_data[i][phase_name] 
                               for i in range(len(normalized_data)) 
                               if golden_mask.iloc[i]])
        
        # 1차 미분 (변화 속도)
        golden_diff = np.diff(golden_trajectory[phase_name])
        golden_diff_std = np.std([np.diff(traj) for traj in golden_data], axis=0)
        
        time_axis_diff = np.linspace(0, 1, len(golden_diff))
        
        # 배경 (모든 배치)
        for trajectory in golden_data:
            ax.plot(time_axis_diff, np.diff(trajectory), 'lightgray', 
                   alpha=0.3, linewidth=0.8)
        
        # Golden 평균 변화율
        ax.plot(time_axis_diff, golden_diff, 'darkred', linewidth=2.5, label='Mean Trend')
        
        # 신뢰도 구간
        ax.fill_between(time_axis_diff, 
                        golden_diff - golden_diff_std,
                        golden_diff + golden_diff_std,
                        alpha=0.3, color='red', label='Trend ±1 Std')
        
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title(f'{phases[col].upper()}\n변화 속도 (1st Derivative)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('정규화된 시간 진행도')
        ax.set_ylabel('변화율 (dX/dt)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # ====== 하단: 2차 미분 (가속도/곡률) ======
    for col, phase_name in enumerate(phases):
        ax = fig.add_subplot(gs[2, col])
        
        # 2차 미분 (가속도)
        golden_diff2 = np.diff(np.diff(golden_trajectory[phase_name]))
        
        time_axis_diff2 = np.linspace(0, 1, len(golden_diff2))
        
        # Golden 가속도
        ax.plot(time_axis_diff2, golden_diff2, 'darkred', linewidth=2.5, 
               marker='o', markersize=4, label='Acceleration')
        
        # 0 기준선
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # 양수/음수 영역 강조
        ax.fill_between(time_axis_diff2, 0, golden_diff2, 
                        where=(golden_diff2 >= 0), alpha=0.3, color='green', 
                        label='Positive (Speeding up)')
        ax.fill_between(time_axis_diff2, 0, golden_diff2, 
                        where=(golden_diff2 < 0), alpha=0.3, color='red', 
                        label='Negative (Slowing down)')
        
        ax.set_title(f'{phases[col].upper()}\n가속도 (2nd Derivative)', 
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('정규화된 시간 진행도')
        ax.set_ylabel('가속도 (d²X/dt²)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    fig.suptitle('Golden Cycle 최적 추이 분석\n개별 궤적 | 트렌드 | 가속도', 
                fontsize=14, fontweight='bold', y=0.995)
    
    return fig

def plot_golden_trajectory_combined(golden_trajectory, normalized_data, golden_mask, 
                                    features_df_clustered, golden_cluster):
    """
    Golden Cycle의 전체 프로파일을 한 번에 표시
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    phases = ['phase1', 'phase2', 'phase3']
    phase_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    # ====== 좌상: 전체 프로세스 궤적 ======
    ax = axes[0, 0]
    
    time_axis = np.linspace(0, 1, 200)
    total_time = np.concatenate([
        time_axis,
        1 + time_axis,
        2 + time_axis
    ])
    
    golden_data = np.array([normalized_data[i]['phase1'] 
                           for i in range(len(normalized_data)) 
                           if golden_mask.iloc[i]])
    
    for trajectory in golden_data:
        full_traj = np.concatenate([
            normalized_data[np.where(golden_mask)[0][0]]['phase1'],
            normalized_data[np.where(golden_mask)[0][0]]['phase2'],
            normalized_data[np.where(golden_mask)[0][0]]['phase3']
        ])
        ax.plot(total_time, full_traj, 'lightgray', alpha=0.2, linewidth=0.8)
    
    # Golden 평균 전체 궤적
    full_golden = np.concatenate([
        golden_trajectory['phase1'],
        golden_trajectory['phase2'],
        golden_trajectory['phase3']
    ])
    
    ax.plot(total_time, full_golden, 'r-', linewidth=3, label='Golden Trajectory', zorder=10)
    
    # Phase 구분선
    for phase_sep in [1, 2]:
        ax.axvline(x=phase_sep, color='black', linestyle='--', alpha=0.5, linewidth=2)
    
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(['Phase 1\n상승', 'Phase 2\n오실레이션', 'Phase 3\n하강'])
    ax.set_title('전체 프로세스 Golden Trajectory', fontsize=12, fontweight='bold')
    ax.set_ylabel('측정값')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # ====== 우상: Phase별 특성 비교 ======
    ax = axes[0, 1]
    
    phase_stats = []
    for i, phase_name in enumerate(phases):
        golden_phase = np.array([normalized_data[j][phase_name] 
                                for j in range(len(normalized_data)) 
                                if golden_mask.iloc[j]])
        
        phase_stats.append({
            'phase': f'Phase {i+1}',
            'mean': np.mean(golden_trajectory[phase_name]),
            'std': np.std(golden_phase),
            'range': np.max(golden_trajectory[phase_name]) - np.min(golden_trajectory[phase_name]),
            'trend': np.mean(np.diff(golden_trajectory[phase_name]))
        })
    
    stat_names = ['mean', 'std', 'range', 'trend']
    x = np.arange(len(phases))
    width = 0.2
    
    for idx, stat_name in enumerate(stat_names):
        values = [stat['std' if stat_name == 'std' else stat_name] for stat in phase_stats]
        ax.bar(x + idx*width, values, width, label=stat_name.upper())
    
    ax.set_xlabel('Phase')
    ax.set_ylabel('값')
    ax.set_title('Phase별 특성 비교', fontsize=12, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([s['phase'] for s in phase_stats])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # ====== 좌하: 수율 분포 ======
    ax = axes[1, 0]
    
    golden_yields = features_df_clustered[golden_mask]['yield'].values
    other_yields = features_df_clustered[~golden_mask]['yield'].values
    
    ax.hist(golden_yields, bins=8, alpha=0.6, color='red', label=f'Golden (μ={golden_yields.mean():.1f}%)', edgecolor='darkred')
    ax.hist(other_yields, bins=8, alpha=0.6, color='blue', label=f'Others (μ={other_yields.mean():.1f}%)', edgecolor='darkblue')
    
    ax.axvline(golden_yields.mean(), color='red', linestyle='--', linewidth=2)
    ax.axvline(other_yields.mean(), color='blue', linestyle='--', linewidth=2)
    
    ax.set_xlabel('수율 (%)')
    ax.set_ylabel('배치 수')
    ax.set_title('Golden vs Others 수율 분포', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # ====== 우하: 통계 정보 ======
    ax = axes[1, 1]
    ax.axis('off')
    
    golden_count = golden_mask.sum()
    golden_yield_mean = golden_yields.mean()
    golden_yield_std = golden_yields.std()
    
    stats_text = f"""
    ╔════════════════════════════════════════╗
    ║       GOLDEN CYCLE 최적 조건           ║
    ╚════════════════════════════════════════╝
    
    📊 배치 통계
    ├─ Golden Batch 수: {golden_count}개
    ├─ 총 Batch 수: {len(normalized_data)}개
    └─ Golden 비율: {golden_count/len(normalized_data)*100:.1f}%
    
    🎯 수율 성능
    ├─ 평균 수율: {golden_yield_mean:.2f}%
    ├─ 표준편차: {golden_yield_std:.2f}%
    ├─ 최소값: {golden_yields.min():.2f}%
    └─ 최대값: {golden_yields.max():.2f}%
    
    📈 Phase별 특성
    ├─ Phase 1 범위: {np.max(golden_trajectory['phase1']) - np.min(golden_trajectory['phase1']):.2f}
    ├─ Phase 2 평균: {np.mean(golden_trajectory['phase2']):.2f}
    └─ Phase 3 기울기: {np.mean(np.diff(golden_trajectory['phase3'])):.4f}
    
    ✓ 권장 운전 조건
    ├─ Phase 1: 안정적 상승 추이 유지
    ├─ Phase 2: 제어된 오실레이션 (표준편차 최소화)
    └─ Phase 3: 일정한 속도로 감소
    """
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Golden Cycle 최적 조건 종합 분석', 
                fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    return fig

# ============================================================================
# 9. 메인 분석 파이프라인
# ============================================================================
def main():
    print("=" * 70)
    print("ACN 정제 배치 - Golden Cycle 분석")
    print("=" * 70)
    
    # 1. 데이터 생성 (실제 데이터로 교체)
    print("\n[1단계] 데이터 준비 중...")
    raw_data = generate_sample_data(n_batches=50)
    normalized_data = prepare_normalized_data(raw_data)
    
    # 2. 특성 추출
    print("[2단계] 특성 추출 중...")
    features_df = extract_all_features(normalized_data)
    print(f"추출된 특성 수: {len(features_df.columns) - 2}")
    
    # 3. 상관성 분석
    print("[3단계] 상관성 분석 중...")
    corr_df = analyze_feature_correlation(features_df)
    top_features = select_top_features(features_df, corr_df, top_n=15)
    
    print("\n=== 상위 상관 특성 (수율과의 관계) ===")
    print(corr_df.head(15)[['feature', 'pearson_corr']])
    
    # 4-1. 최적 클러스터 수 결정
    print("\n[4-1단계] 최적 클러스터 수 결정 중...")
    optimal_k, fig_optimal_k = determine_optimal_clusters(features_df, 
                                                          top_features['feature'].tolist())
    
    # 4-2. 클러스터링 (결정된 최적 K 사용)
    print("\n[4-2단계] 클러스터링 및 Golden Cycle 식별 중...")
    features_df_clustered, golden_cluster, cluster_stats = identify_golden_clusters(
        features_df, top_features['feature'].tolist(), n_clusters=optimal_k)
    
    # 4-3. 다양한 Golden batch 식별 방법 적용
    print("\n[4-3단계] 여러 방법으로 Golden batch 식별 중...")
    
    # 방법 1: 통계적 방법 (상위 25%)
    golden_mask_statistical, yield_threshold = identify_golden_batches_statistical(
        features_df_clustered, yield_threshold_percentile=75)
    
    # 방법 2: 다중 기준 (수율+안정성+트렌드)
    golden_mask_multimodal = identify_golden_batches_multimodal(
        features_df_clustered, top_features['feature'].tolist())
    
    # 방법 3: DBSCAN 기반
    golden_mask_dbscan, dbscan_clusters = identify_golden_batches_dbscan(
        features_df_clustered, top_features['feature'].tolist(), eps=1.0, min_samples=3)
    
    # 시각화
    fig_comparison = plot_identification_methods(features_df_clustered, 
                                                  top_features['feature'].tolist(),
                                                  golden_mask_statistical, 
                                                  golden_mask_multimodal)
    
    # 최종 결정: 다중 기준 방법 사용 (가장 엄밀함)
    golden_mask = golden_mask_multimodal
    
    print(f"\n✓ 최종 선택: 다중 기준 방법 ({golden_mask.sum()}개 배치)")
    golden_batch_ids = features_df_clustered[golden_mask]['batch_id'].tolist()
    golden_cluster_batches = [b for b in normalized_data 
                              if b['batch_id'].split('_')[0] + '...' not in golden_batch_ids]
    
    # 실제 매칭을 위해 인덱스 기반으로 재구성
    golden_cluster_batches = [normalized_data[i] for i in range(len(normalized_data)) 
                              if golden_mask.iloc[i]]
    
    golden_trajectory = calculate_golden_trajectory(normalized_data, golden_cluster_batches)
    
    print(f"\nGolden Cluster ID: {golden_cluster}")
    print(f"Golden Batch 수: {len(golden_cluster_batches)}")
    print(f"평균 수율: {features_df_clustered[golden_mask]['yield'].mean():.2f}%")
    
    # 5. 예측 모델
    print("\n[5단계] 수율 예측 모델 구축 중...")
    model, feature_importance = build_yield_prediction_model(
        features_df_clustered, top_features['feature'].tolist())
    
    print("\n=== 상위 중요 특성 ===")
    print(feature_importance.head(10))
    
    # 6. 시각화
    print("\n[6단계] 시각화 생성 중...")
    
    fig1 = plot_normalized_batches(normalized_data, golden_mask.values, 
                                   title="Batch Trajectories - Golden Batches Highlighted")
    fig1.suptitle('정규화된 배치 궤적 (빨강: Golden Cycle)', y=1.02)
    
    fig2 = plot_golden_vs_others(normalized_data, golden_mask.values, normalized_data)
    
    fig3 = plot_correlation_heatmap(features_df_clustered, top_features['feature'].tolist())
    
    fig4 = plot_feature_importance(feature_importance, top_n=15)
    
    fig5 = plot_golden_trajectory(golden_trajectory, normalized_data, golden_mask)
    
    fig6 = plot_golden_trajectory_combined(golden_trajectory, normalized_data, golden_mask, 
                                           features_df_clustered, golden_cluster)
    
    # 7. 결과 요약
    print("\n" + "=" * 70)
    print("분석 결과 요약")
    print("=" * 70)
    
    print("\n[Golden Cycle 특성]")
    golden_features = features_df_clustered[golden_mask][top_features['feature'].tolist()].mean()
    other_features = features_df_clustered[~golden_mask][top_features['feature'].tolist()].mean()
    
    print("\nGolden vs Others 비교 (상위 5개 중요 특성):")
    for feat in feature_importance.head(5)['feature'].values:
        g_val = golden_features[feat]
        o_val = other_features[feat]
        diff_pct = ((g_val - o_val) / (np.abs(o_val) + 1e-6)) * 100
        print(f"  {feat}: Golden={g_val:.4f}, Others={o_val:.4f} (차이: {diff_pct:+.1f}%)")
    
    plt.show()
    
    return {
        'normalized_data': normalized_data,
        'features_df': features_df_clustered,
        'model': model,
        'golden_trajectory': golden_trajectory,
        'golden_cluster': golden_cluster,
        'feature_importance': feature_importance
    }

if __name__ == "__main__":
    results = main()
