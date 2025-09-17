import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(data_path=None, df=None):
    """
    ACN 정제 공정 데이터 로드 및 전처리
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    
    Returns:
    preprocessed_df: 전처리된 DataFrame
    """
    if df is not None:
        data = df.copy()
    elif data_path:
        data = pd.read_csv(data_path)
    else:
        raise ValueError("데이터 경로 또는 DataFrame을 제공해야 합니다.")
    
    print("=" * 80)
    print("ACN 정제 공정 데이터 전처리")
    print("=" * 80)
    print(f"원본 데이터 크기: {data.shape}")
    
    # 1. 최종 F/R Level에서 분석한 데이터만 필터링
    if 'Final_FR' in data.columns:
        # Final_FR이 최대값인 데이터만 선택 (최종 분석 데이터)
        max_fr_level = data['Final_FR'].max()
        data = data[data['Final_FR'] == max_fr_level].copy()
        print(f"최종 F/R Level 필터링 후 데이터 크기: {data.shape}")
        print(f"최종 F/R Level: {max_fr_level}")
    
    # 2. 품질값 정규화 (spec 기준)
    quality_columns = ['AN-10_200nm', 'AN-10_225nm', 'AN-10_250nm', 
                      'AN-50_200nm', 'AN-50_225nm', 'AN-50_250nm']
    
    # 품질값 정규화 (음수: spec out, 양수: spec in)
    for col in quality_columns:
        if col in data.columns:
            # 0을 기준으로 정규화 (실제 spec 값에 따라 조정 필요)
            data[f'{col}_normalized'] = data[col] - 0  # 실제 spec 값으로 변경 필요
    
    # 3. 데이터 타입 설정
    # 날짜 컬럼
    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    
    # 범주형 변수
    categorical_columns = ['Source', 'IsBubbled', 'IsBothChillerOn']
    for col in categorical_columns:
        if col in data.columns:
            data[col] = data[col].astype('category')
    
    # 수치형 변수
    numeric_columns = [col for col in data.columns if col not in categorical_columns + ['Date']]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    print(f"전처리 완료 후 데이터 크기: {data.shape}")
    return data

def feature_selection_analysis(df):
    """
    Feature Selection 분석
    
    Parameters:
    df: 전처리된 DataFrame
    
    Returns:
    feature_importance: 특성 중요도 결과
    """
    print("\n" + "=" * 80)
    print("Feature Selection 분석")
    print("=" * 80)
    
    if 'Yield' not in df.columns:
        print("Yield 컬럼이 없습니다.")
        return None
    
    # 1. 수치형 변수만 선택 (Yield 제외)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Yield' in numeric_cols:
        numeric_cols.remove('Yield')
    
    # 결측치가 있는 컬럼 제외
    numeric_cols = [col for col in numeric_cols if not df[col].isnull().all()]
    
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    y = df['Yield'].fillna(df['Yield'].median())
    
    print(f"분석 대상 특성 수: {len(numeric_cols)}")
    print(f"분석 대상 샘플 수: {len(X)}")
    
    # 2. 상관관계 기반 Feature Selection
    print("\n1. 상관관계 기반 Feature Selection")
    print("-" * 50)
    
    correlations = {}
    for col in numeric_cols:
        if col in X.columns:
            corr_pearson, p_value_pearson = pearsonr(X[col], y)
            corr_spearman, p_value_spearman = spearmanr(X[col], y)
            correlations[col] = {
                'pearson_corr': corr_pearson,
                'pearson_pvalue': p_value_pearson,
                'spearman_corr': corr_spearman,
                'spearman_pvalue': p_value_spearman,
                'abs_pearson': abs(corr_pearson),
                'abs_spearman': abs(corr_spearman)
            }
    
    # 상관관계 결과 정렬
    corr_df = pd.DataFrame(correlations).T
    corr_df = corr_df.sort_values('abs_pearson', ascending=False)
    
    print("Yield와의 상관관계 (상위 15개):")
    print(corr_df[['pearson_corr', 'pearson_pvalue', 'spearman_corr', 'spearman_pvalue']].head(15).round(4))
    
    # 3. 통계적 유의성 기반 Feature Selection
    print("\n2. 통계적 유의성 기반 Feature Selection (F-test)")
    print("-" * 50)
    
    selector_f = SelectKBest(score_func=f_regression, k='all')
    selector_f.fit(X, y)
    
    f_scores = pd.DataFrame({
        'feature': numeric_cols,
        'f_score': selector_f.scores_,
        'p_value': selector_f.pvalues_
    }).sort_values('f_score', ascending=False)
    
    print("F-test 결과 (상위 15개):")
    print(f_scores.head(15).round(4))
    
    # 4. 상호정보량 기반 Feature Selection
    print("\n3. 상호정보량 기반 Feature Selection")
    print("-" * 50)
    
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_df = pd.DataFrame({
        'feature': numeric_cols,
        'mutual_info': mi_scores
    }).sort_values('mutual_info', ascending=False)
    
    print("상호정보량 결과 (상위 15개):")
    print(mi_df.head(15).round(4))
    
    # 5. Random Forest 기반 Feature Importance
    print("\n4. Random Forest 기반 Feature Importance")
    print("-" * 50)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    rf_importance = pd.DataFrame({
        'feature': numeric_cols,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Random Forest 중요도 (상위 15개):")
    print(rf_importance.head(15).round(4))
    
    # 6. 종합 Feature Selection 점수 계산
    print("\n5. 종합 Feature Selection 점수")
    print("-" * 50)
    
    # 각 방법의 점수를 정규화하여 종합 점수 계산
    corr_df['corr_score'] = (corr_df['abs_pearson'] - corr_df['abs_pearson'].min()) / (corr_df['abs_pearson'].max() - corr_df['abs_pearson'].min())
    f_scores['f_score_norm'] = (f_scores['f_score'] - f_scores['f_score'].min()) / (f_scores['f_score'].max() - f_scores['f_score'].min())
    mi_df['mi_score_norm'] = (mi_df['mutual_info'] - mi_df['mutual_info'].min()) / (mi_df['mutual_info'].max() - mi_df['mutual_info'].min())
    rf_importance['rf_score_norm'] = (rf_importance['importance'] - rf_importance['importance'].min()) / (rf_importance['importance'].max() - rf_importance['importance'].min())
    
    # 종합 점수 계산
    combined_scores = pd.DataFrame({
        'feature': numeric_cols,
        'correlation_score': corr_df['corr_score'].values,
        'f_test_score': f_scores['f_score_norm'].values,
        'mutual_info_score': mi_df['mi_score_norm'].values,
        'rf_importance_score': rf_importance['rf_score_norm'].values
    })
    
    # 가중 평균으로 최종 점수 계산
    combined_scores['final_score'] = (
        combined_scores['correlation_score'] * 0.3 +
        combined_scores['f_test_score'] * 0.2 +
        combined_scores['mutual_info_score'] * 0.2 +
        combined_scores['rf_importance_score'] * 0.3
    )
    
    combined_scores = combined_scores.sort_values('final_score', ascending=False)
    
    print("종합 Feature Selection 점수 (상위 20개):")
    print(combined_scores.head(20).round(4))
    
    return {
        'correlations': corr_df,
        'f_test': f_scores,
        'mutual_info': mi_df,
        'rf_importance': rf_importance,
        'combined_scores': combined_scores
    }

def multivariate_pattern_analysis(df, top_features=None):
    """
    다변량 패턴 분석
    
    Parameters:
    df: 전처리된 DataFrame
    top_features: 상위 특성 리스트 (None이면 자동 선택)
    
    Returns:
    pattern_results: 패턴 분석 결과
    """
    print("\n" + "=" * 80)
    print("다변량 패턴 분석")
    print("=" * 80)
    
    if 'Yield' not in df.columns:
        print("Yield 컬럼이 없습니다.")
        return None
    
    # 1. 특성 선택
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Yield' in numeric_cols:
        numeric_cols.remove('Yield')
    
    # 결측치 처리
    X = df[numeric_cols].fillna(df[numeric_cols].median())
    y = df['Yield'].fillna(df['Yield'].median())
    
    # 상위 특성 선택 (기본값: 상위 10개)
    if top_features is None:
        # Yield와의 상관관계 기준으로 상위 10개 선택
        correlations = []
        for col in numeric_cols:
            if col in X.columns:
                corr, _ = pearsonr(X[col], y)
                correlations.append((col, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        top_features = [col for col, _ in correlations[:10]]
    
    print(f"분석 대상 특성: {top_features}")
    
    X_selected = X[top_features]
    
    # 2. 데이터 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # 3. PCA 분석
    print("\n1. 주성분 분석 (PCA)")
    print("-" * 50)
    
    pca = PCA()
    pca_result = pca.fit_transform(X_scaled)
    
    # 설명 분산 비율
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("주성분별 설명 분산 비율:")
    for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"PC{i+1}: {var_ratio:.4f} (누적: {cum_var:.4f})")
    
    # 95% 분산을 설명하는 주성분 수
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\n95% 분산을 설명하는 주성분 수: {n_components_95}")
    
    # 주성분 로딩
    pca_components = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(top_features))],
        index=top_features
    )
    
    print("\n주성분 로딩 (상위 3개 PC):")
    print(pca_components.iloc[:, :3].round(4))
    
    # 4. K-means 클러스터링
    print("\n2. K-means 클러스터링")
    print("-" * 50)
    
    # 최적 클러스터 수 찾기 (엘보우 방법)
    inertias = []
    K_range = range(2, min(11, len(X_scaled)//2))
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
    
    # 최적 클러스터 수 선택 (간단한 방법: 3-5개 중 선택)
    optimal_k = 4  # 실제로는 엘보우 포인트를 찾아야 함
    
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    print(f"선택된 클러스터 수: {optimal_k}")
    print("클러스터별 샘플 수:")
    unique, counts = np.unique(cluster_labels, return_counts=True)
    for cluster, count in zip(unique, counts):
        print(f"클러스터 {cluster}: {count}개")
    
    # 클러스터별 Yield 통계
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels
    
    print("\n클러스터별 Yield 통계:")
    cluster_yield_stats = df_clustered.groupby('Cluster')['Yield'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(cluster_yield_stats.round(4))
    
    # 5. DBSCAN 클러스터링 (이상치 탐지)
    print("\n3. DBSCAN 클러스터링 (이상치 탐지)")
    print("-" * 50)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"DBSCAN 클러스터 수: {n_clusters_dbscan}")
    print(f"이상치 수: {n_noise}")
    
    # 6. 패턴 분석
    print("\n4. 패턴 분석")
    print("-" * 50)
    
    # 고수율/저수율 그룹 분석
    yield_median = df['Yield'].median()
    high_yield_mask = df['Yield'] > yield_median
    low_yield_mask = df['Yield'] <= yield_median
    
    print(f"수율 중간값: {yield_median:.4f}")
    print(f"고수율 그룹: {high_yield_mask.sum()}개")
    print(f"저수율 그룹: {low_yield_mask.sum()}개")
    
    # 고수율/저수율 그룹별 특성 평균 비교
    high_yield_features = X_selected[high_yield_mask].mean()
    low_yield_features = X_selected[low_yield_mask].mean()
    
    feature_comparison = pd.DataFrame({
        'High_Yield_Mean': high_yield_features,
        'Low_Yield_Mean': low_yield_features,
        'Difference': high_yield_features - low_yield_features,
        'Relative_Diff': (high_yield_features - low_yield_features) / low_yield_features * 100
    }).sort_values('Relative_Diff', key=abs, ascending=False)
    
    print("\n고수율 vs 저수율 그룹 특성 비교:")
    print(feature_comparison.round(4))
    
    return {
        'pca': {
            'explained_variance_ratio': explained_variance_ratio,
            'cumulative_variance': cumulative_variance,
            'n_components_95': n_components_95,
            'components': pca_components
        },
        'kmeans': {
            'optimal_k': optimal_k,
            'cluster_labels': cluster_labels,
            'cluster_stats': cluster_yield_stats
        },
        'dbscan': {
            'n_clusters': n_clusters_dbscan,
            'n_noise': n_noise,
            'labels': dbscan_labels
        },
        'pattern_analysis': {
            'feature_comparison': feature_comparison,
            'yield_median': yield_median
        }
    }

def create_visualizations(df, feature_results, pattern_results):
    """
    시각화 생성
    
    Parameters:
    df: 전처리된 DataFrame
    feature_results: feature selection 결과
    pattern_results: 패턴 분석 결과
    """
    print("\n" + "=" * 80)
    print("시각화 생성")
    print("=" * 80)
    
    # 1. Feature Importance 시각화
    if feature_results and 'combined_scores' in feature_results:
        plt.figure(figsize=(15, 10))
        
        # 상위 15개 특성의 종합 점수
        plt.subplot(2, 2, 1)
        top_features = feature_results['combined_scores'].head(15)
        plt.barh(range(len(top_features)), top_features['final_score'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('종합 Feature Score')
        plt.title('상위 15개 특성의 종합 중요도')
        plt.gca().invert_yaxis()
        
        # 상관관계 히트맵
        plt.subplot(2, 2, 2)
        if 'Yield' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                # 상위 10개 특성만 선택
                top_corr_features = feature_results['combined_scores'].head(10)['feature'].tolist()
                if 'Yield' not in top_corr_features:
                    top_corr_features.append('Yield')
                
                corr_subset = corr_matrix.loc[top_corr_features, top_corr_features]
                sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, fmt='.3f')
                plt.title('상위 특성 간 상관관계')
        
        # PCA 설명 분산 비율
        plt.subplot(2, 2, 3)
        if pattern_results and 'pca' in pattern_results:
            pca_data = pattern_results['pca']
            plt.plot(range(1, len(pca_data['explained_variance_ratio']) + 1), 
                    pca_data['explained_variance_ratio'], 'bo-')
            plt.plot(range(1, len(pca_data['cumulative_variance']) + 1), 
                    pca_data['cumulative_variance'], 'ro-')
            plt.xlabel('주성분')
            plt.ylabel('설명 분산 비율')
            plt.title('PCA 설명 분산 비율')
            plt.legend(['개별', '누적'])
            plt.grid(True)
        
        # 클러스터별 Yield 분포
        plt.subplot(2, 2, 4)
        if pattern_results and 'kmeans' in pattern_results:
            df_clustered = df.copy()
            df_clustered['Cluster'] = pattern_results['kmeans']['cluster_labels']
            
            if 'Yield' in df_clustered.columns:
                sns.boxplot(data=df_clustered, x='Cluster', y='Yield')
                plt.title('클러스터별 Yield 분포')
        
        plt.tight_layout()
        plt.show()
    
    # 2. 패턴 분석 시각화
    if pattern_results and 'pattern_analysis' in pattern_results:
        plt.figure(figsize=(15, 8))
        
        # 고수율/저수율 그룹 특성 비교
        plt.subplot(1, 2, 1)
        feature_comp = pattern_results['pattern_analysis']['feature_comparison']
        top_diff_features = feature_comp.head(10)
        
        x = np.arange(len(top_diff_features))
        width = 0.35
        
        plt.bar(x - width/2, top_diff_features['High_Yield_Mean'], width, 
                label='고수율 그룹', alpha=0.8)
        plt.bar(x + width/2, top_diff_features['Low_Yield_Mean'], width, 
                label='저수율 그룹', alpha=0.8)
        
        plt.xlabel('특성')
        plt.ylabel('평균값')
        plt.title('고수율 vs 저수율 그룹 특성 비교')
        plt.xticks(x, top_diff_features.index, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Yield 분포
        plt.subplot(1, 2, 2)
        if 'Yield' in df.columns:
            plt.hist(df['Yield'], bins=30, alpha=0.7, edgecolor='black')
            plt.axvline(df['Yield'].median(), color='red', linestyle='--', 
                       label=f'중간값: {df["Yield"].median():.3f}')
            plt.xlabel('Yield')
            plt.ylabel('빈도')
            plt.title('Yield 분포')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def generate_analysis_report(df, feature_results, pattern_results):
    """
    종합 분석 리포트 생성
    
    Parameters:
    df: 전처리된 DataFrame
    feature_results: feature selection 결과
    pattern_results: 패턴 분석 결과
    
    Returns:
    report: 분석 리포트
    """
    print("\n" + "=" * 80)
    print("ACN 정제 공정 Yield 분석 종합 리포트")
    print("=" * 80)
    
    report = {}
    
    # 1. 데이터 개요
    print("\n1. 데이터 개요")
    print("-" * 50)
    print(f"총 샘플 수: {len(df)}")
    print(f"총 특성 수: {len(df.columns)}")
    
    if 'Yield' in df.columns:
        yield_stats = df['Yield'].describe()
        print(f"\nYield 통계:")
        print(f"평균: {yield_stats['mean']:.4f}")
        print(f"표준편차: {yield_stats['std']:.4f}")
        print(f"최솟값: {yield_stats['min']:.4f}")
        print(f"최댓값: {yield_stats['max']:.4f}")
        print(f"변동계수: {(yield_stats['std']/yield_stats['mean']*100):.2f}%")
        
        report['yield_stats'] = yield_stats
    
    # 2. 주요 영향 인자
    print("\n2. Yield에 주요한 영향 인자 (상위 10개)")
    print("-" * 50)
    
    if feature_results and 'combined_scores' in feature_results:
        top_features = feature_results['combined_scores'].head(10)
        print("특성명\t\t종합점수\t상관계수\tF-test\t상호정보량\tRF중요도")
        print("-" * 80)
        
        for _, row in top_features.iterrows():
            feature_name = row['feature'][:15]  # 이름 길이 제한
            print(f"{feature_name:<15}\t{row['final_score']:.4f}\t\t{row['correlation_score']:.4f}\t\t{row['f_test_score']:.4f}\t{row['mutual_info_score']:.4f}\t\t{row['rf_importance_score']:.4f}")
        
        report['top_features'] = top_features
    
    # 3. 패턴 분석 결과
    print("\n3. 패턴 분석 결과")
    print("-" * 50)
    
    if pattern_results:
        # PCA 결과
        if 'pca' in pattern_results:
            pca_data = pattern_results['pca']
            print(f"95% 분산을 설명하는 주성분 수: {pca_data['n_components_95']}")
            print(f"첫 번째 주성분 설명 분산: {pca_data['explained_variance_ratio'][0]:.4f}")
        
        # 클러스터링 결과
        if 'kmeans' in pattern_results:
            kmeans_data = pattern_results['kmeans']
            print(f"최적 클러스터 수: {kmeans_data['optimal_k']}")
            print("클러스터별 평균 Yield:")
            for cluster, stats in kmeans_data['cluster_stats'].iterrows():
                print(f"  클러스터 {cluster}: {stats['mean']:.4f} (n={stats['count']})")
        
        # 고수율/저수율 패턴
        if 'pattern_analysis' in pattern_results:
            pattern_data = pattern_results['pattern_analysis']
            print(f"\n고수율 그룹 기준 (Yield > {pattern_data['yield_median']:.4f})")
            print("고수율 그룹에서 주목할 만한 특성 (상위 5개):")
            top_pattern_features = pattern_data['feature_comparison'].head(5)
            for feature, row in top_pattern_features.iterrows():
                print(f"  {feature}: {row['Relative_Diff']:.2f}% 차이")
        
        report['pattern_analysis'] = pattern_results
    
    # 4. 권장사항
    print("\n4. Yield 최적화 권장사항")
    print("-" * 50)
    
    recommendations = []
    
    if feature_results and 'combined_scores' in feature_results:
        top_3_features = feature_results['combined_scores'].head(3)
        print("1. 주요 영향 인자 모니터링 강화:")
        for _, row in top_3_features.iterrows():
            print(f"   - {row['feature']}: 종합 중요도 {row['final_score']:.4f}")
            recommendations.append(f"{row['feature']} 모니터링 강화")
    
    if pattern_results and 'pattern_analysis' in pattern_results:
        pattern_data = pattern_results['pattern_analysis']
        top_positive_features = pattern_data['feature_comparison'][
            pattern_data['feature_comparison']['Relative_Diff'] > 0
        ].head(3)
        
        if len(top_positive_features) > 0:
            print("\n2. 고수율 달성을 위한 특성 조정:")
            for feature, row in top_positive_features.iterrows():
                print(f"   - {feature}: 고수율 그룹에서 {row['Relative_Diff']:.2f}% 높음")
                recommendations.append(f"{feature} 최적화")
    
    if pattern_results and 'kmeans' in pattern_results:
        kmeans_data = pattern_results['kmeans']
        best_cluster = kmeans_data['cluster_stats']['mean'].idxmax()
        print(f"\n3. 최적 운영 조건 (클러스터 {best_cluster}):")
        print(f"   - 평균 Yield: {kmeans_data['cluster_stats'].loc[best_cluster, 'mean']:.4f}")
        recommendations.append(f"클러스터 {best_cluster} 조건으로 운영")
    
    report['recommendations'] = recommendations
    
    return report

# 메인 실행 함수
def main_analysis(data_path=None, df=None):
    """
    ACN 정제 공정 Yield 분석 메인 함수
    
    Parameters:
    data_path: 데이터 파일 경로
    df: 이미 로드된 DataFrame
    
    Returns:
    results: 분석 결과 딕셔너리
    """
    # 1. 데이터 로드 및 전처리
    processed_df = load_and_preprocess_data(data_path, df)
    
    # 2. Feature Selection 분석
    feature_results = feature_selection_analysis(processed_df)
    
    # 3. 다변량 패턴 분석
    top_features = None
    if feature_results and 'combined_scores' in feature_results:
        top_features = feature_results['combined_scores'].head(10)['feature'].tolist()
    
    pattern_results = multivariate_pattern_analysis(processed_df, top_features)
    
    # 4. 시각화
    create_visualizations(processed_df, feature_results, pattern_results)
    
    # 5. 종합 리포트
    report = generate_analysis_report(processed_df, feature_results, pattern_results)
    
    return {
        'processed_data': processed_df,
        'feature_results': feature_results,
        'pattern_results': pattern_results,
        'report': report
    }

# 사용 예시
if __name__ == "__main__":
    print("ACN 정제 공정 Yield 분석을 시작합니다.")
    print("\n사용법:")
    print("1. CSV 파일에서 분석:")
    print("   results = main_analysis(data_path='your_data.csv')")
    print("\n2. DataFrame에서 분석:")
    print("   results = main_analysis(df=your_dataframe)")
    print("\n3. 결과 확인:")
    print("   print(results['report']['recommendations'])")
