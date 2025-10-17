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
def identify_golden_clusters(features_df, top_features, n_clusters=4):
    """
    특성 기반 클러스터링으로 고수율 그룹 식별
    """
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
    
    # 4. 클러스터링
    print("\n[4단계] 클러스터링 및 Golden Cycle 식별 중...")
    features_df_clustered, golden_cluster, cluster_stats = identify_golden_clusters(
        features_df, top_features['feature'].tolist())
    
    # Golden cluster 배치 추출
    golden_mask = features_df_clustered['cluster'] == golden_cluster
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
    
    fig1 = plot_normalized_batches(normalized_data, golden_mask.values)
    fig1.suptitle('정규화된 배치 궤적 (빨강: Golden Cycle)', y=1.02)
    
    fig2 = plot_golden_vs_others(normalized_data, golden_mask.values, normalized_data)
    
    fig3 = plot_correlation_heatmap(features_df_clustered, top_features['feature'].tolist())
    
    fig4 = plot_feature_importance(feature_importance, top_n=15)
    
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
