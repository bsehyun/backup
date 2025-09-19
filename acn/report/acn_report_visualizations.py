# ACN 리포트 시각화 함수들
# 이 파일은 acn_report.py의 시각화 함수들을 별도로 분리한 것입니다.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def create_comprehensive_visualizations(analyzer):
    """
    종합 시각화 생성
    """
    print("\n" + "=" * 80)
    print("종합 시각화 생성")
    print("=" * 80)
    
    # 1. 기본 통계 시각화
    _create_basic_visualizations(analyzer)
    
    # 2. Feature Selection 시각화
    _create_feature_selection_visualizations(analyzer)
    
    # 3. 다변량 분석 시각화
    _create_multivariate_visualizations(analyzer)
    
    # 4. 구간별 분석 시각화
    _create_interval_visualizations(analyzer)
    
    # 5. Yield vs Input/Output 시각화
    _create_yield_analysis_visualizations(analyzer)

def _create_basic_visualizations(analyzer):
    """기본 통계 시각화"""
    plt.figure(figsize=(20, 12))
    
    # 1. Yield 분포
    plt.subplot(3, 4, 1)
    plt.hist(analyzer.df['Yield'], bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(analyzer.df['Yield'].mean(), color='red', linestyle='--', 
               label=f'평균: {analyzer.df["Yield"].mean():.3f}')
    plt.xlabel('Yield')
    plt.ylabel('빈도')
    plt.title('Yield 분포')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Final_FR vs Process 데이터 분포
    plt.subplot(3, 4, 2)
    if 'Final_FR' in analyzer.df.columns:
        final_counts = analyzer.df['Final_FR'].value_counts()
        plt.pie(final_counts.values, labels=['Process', 'Final'], autopct='%1.1f%%')
        plt.title('Final vs Process 데이터 비율')
    
    # 3. F/R Level 분포
    plt.subplot(3, 4, 3)
    if 'F/R Level' in analyzer.df.columns:
        plt.hist(analyzer.df['F/R Level'], bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('F/R Level (%)')
        plt.ylabel('빈도')
        plt.title('F/R Level 분포')
        plt.grid(True, alpha=0.3)
    
    # 4. Input vs Output 산점도
    plt.subplot(3, 4, 4)
    if 'Input_source' in analyzer.df.columns:
        plt.scatter(analyzer.df['Input_source'], analyzer.df['Calculated_Output'], alpha=0.6)
        plt.xlabel('Input Source')
        plt.ylabel('Calculated Output')
        plt.title('Input vs Output')
        plt.grid(True, alpha=0.3)
    
    # 5. 품질값 분포
    plt.subplot(3, 4, 5)
    quality_data = []
    quality_labels = []
    for col in analyzer.quality_columns:
        if col in analyzer.df.columns:
            quality_data.append(analyzer.df[col].dropna())
            quality_labels.append(col)
    
    if quality_data:
        plt.boxplot(quality_data, labels=quality_labels)
        plt.xticks(rotation=45)
        plt.title('품질값 분포')
        plt.grid(True, alpha=0.3)
    
    # 6. Batch별 분석 횟수
    plt.subplot(3, 4, 6)
    if 'No' in analyzer.df.columns:
        batch_counts = analyzer.df.groupby('No').size()
        plt.hist(batch_counts, bins=20, alpha=0.7, edgecolor='black')
        plt.xlabel('분석 횟수')
        plt.ylabel('Batch 수')
        plt.title('Batch별 분석 횟수 분포')
        plt.grid(True, alpha=0.3)
    
    # 7. 시간별 Yield 추이
    plt.subplot(3, 4, 7)
    if 'Date' in analyzer.df.columns:
        daily_yield = analyzer.df.groupby(analyzer.df['Date'].dt.date)['Yield'].mean()
        plt.plot(daily_yield.index, daily_yield.values, marker='o')
        plt.xlabel('날짜')
        plt.ylabel('평균 Yield')
        plt.title('시간별 Yield 추이')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
    
    # 8. Source별 Yield 비교
    plt.subplot(3, 4, 8)
    if 'Source' in analyzer.df.columns:
        source_yield = analyzer.df.groupby('Source')['Yield'].mean()
        plt.bar(source_yield.index.astype(str), source_yield.values)
        plt.xlabel('Source')
        plt.ylabel('평균 Yield')
        plt.title('Source별 Yield 비교')
        plt.grid(True, alpha=0.3)
    
    # 9. 상관관계 히트맵
    plt.subplot(3, 4, 9)
    if 'feature_selection' in analyzer.analysis_results:
        corr_data = analyzer.analysis_results['feature_selection']['correlations']
        top_features = corr_data.head(10).index.tolist()
        top_features.append('Yield')
        
        corr_matrix = analyzer.df[top_features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
        plt.title('상위 특성 간 상관관계')
    
    # 10. 고수율/저수율 그룹 비교
    plt.subplot(3, 4, 10)
    if 'Yield' in analyzer.df.columns:
        yield_median = analyzer.df['Yield'].median()
        high_yield = analyzer.df[analyzer.df['Yield'] > yield_median]['Yield']
        low_yield = analyzer.df[analyzer.df['Yield'] <= yield_median]['Yield']
        
        plt.hist([high_yield, low_yield], bins=20, alpha=0.7, 
                label=['고수율', '저수율'])
        plt.xlabel('Yield')
        plt.ylabel('빈도')
        plt.title('고수율 vs 저수율 그룹')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 11. Input 구간별 Yield
    plt.subplot(3, 4, 11)
    if 'Input_source' in analyzer.df.columns:
        input_bins = pd.cut(analyzer.df['Input_source'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        input_yield = analyzer.df.groupby(input_bins)['Yield'].mean()
        plt.bar(range(len(input_yield)), input_yield.values)
        plt.xticks(range(len(input_yield)), input_yield.index, rotation=45)
        plt.xlabel('Input 구간')
        plt.ylabel('평균 Yield')
        plt.title('Input 구간별 Yield')
        plt.grid(True, alpha=0.3)
    
    # 12. Output 일정성
    plt.subplot(3, 4, 12)
    if 'Calculated_Output' in analyzer.df.columns:
        plt.hist(analyzer.df['Calculated_Output'], bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(analyzer.df['Calculated_Output'].mean(), color='red', linestyle='--', 
                   label=f'평균: {analyzer.df["Calculated_Output"].mean():.3f}')
        plt.xlabel('Calculated Output')
        plt.ylabel('빈도')
        plt.title('Output 일정성 분석')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def _create_feature_selection_visualizations(analyzer):
    """Feature Selection 시각화"""
    if 'feature_selection' not in analyzer.analysis_results:
        return
    
    plt.figure(figsize=(15, 10))
    
    # 1. 상관관계 상위 15개
    plt.subplot(2, 2, 1)
    corr_data = analyzer.analysis_results['feature_selection']['correlations']
    top_corr = corr_data.head(15)
    
    plt.barh(range(len(top_corr)), top_corr['abs_pearson'])
    plt.yticks(range(len(top_corr)), top_corr.index)
    plt.xlabel('절댓값 상관계수')
    plt.title('Yield와의 상관관계 (상위 15개)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # 2. F-test 상위 15개
    plt.subplot(2, 2, 2)
    f_data = analyzer.analysis_results['feature_selection']['f_test']
    top_f = f_data.head(15)
    
    plt.barh(range(len(top_f)), top_f['f_score'])
    plt.yticks(range(len(top_f)), top_f['feature'])
    plt.xlabel('F-score')
    plt.title('F-test 결과 (상위 15개)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # 3. 상호정보량 상위 15개
    plt.subplot(2, 2, 3)
    mi_data = analyzer.analysis_results['feature_selection']['mutual_info']
    top_mi = mi_data.head(15)
    
    plt.barh(range(len(top_mi)), top_mi['mutual_info'])
    plt.yticks(range(len(top_mi)), top_mi['feature'])
    plt.xlabel('상호정보량')
    plt.title('상호정보량 결과 (상위 15개)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    # 4. 종합 중요도
    plt.subplot(2, 2, 4)
    # 각 방법의 점수를 정규화하여 종합
    corr_norm = (corr_data['abs_pearson'] - corr_data['abs_pearson'].min()) / (corr_data['abs_pearson'].max() - corr_data['abs_pearson'].min())
    f_norm = (f_data['f_score'] - f_data['f_score'].min()) / (f_data['f_score'].max() - f_data['f_score'].min())
    mi_norm = (mi_data['mutual_info'] - mi_data['mutual_info'].min()) / (mi_data['mutual_info'].max() - mi_data['mutual_info'].min())
    
    combined_score = (corr_norm * 0.4 + f_norm * 0.3 + mi_norm * 0.3)
    combined_df = pd.DataFrame({
        'feature': corr_data.index,
        'combined_score': combined_score
    }).sort_values('combined_score', ascending=False)
    
    top_combined = combined_df.head(15)
    plt.barh(range(len(top_combined)), top_combined['combined_score'])
    plt.yticks(range(len(top_combined)), top_combined['feature'])
    plt.xlabel('종합 점수')
    plt.title('종합 Feature Importance (상위 15개)')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def _create_multivariate_visualizations(analyzer):
    """다변량 분석 시각화"""
    if 'multivariate_analysis' not in analyzer.analysis_results:
        return
    
    plt.figure(figsize=(15, 10))
    
    # 1. PCA 설명 분산 비율
    plt.subplot(2, 2, 1)
    pca_data = analyzer.analysis_results['multivariate_analysis']['pca']
    explained_var = pca_data['explained_variance_ratio']
    cumulative_var = pca_data['cumulative_variance']
    
    plt.plot(range(1, len(explained_var) + 1), explained_var, 'bo-', label='개별')
    plt.plot(range(1, len(cumulative_var) + 1), cumulative_var, 'ro-', label='누적')
    plt.xlabel('주성분')
    plt.ylabel('설명 분산 비율')
    plt.title('PCA 설명 분산 비율')
    plt.legend()
    plt.grid(True)
    
    # 2. 클러스터별 Yield 분포
    plt.subplot(2, 2, 2)
    cluster_data = analyzer.analysis_results['multivariate_analysis']['clustering']
    cluster_stats = cluster_data['cluster_stats']
    
    plt.bar(cluster_stats.index, cluster_stats['mean'])
    plt.xlabel('클러스터')
    plt.ylabel('평균 Yield')
    plt.title('클러스터별 평균 Yield')
    plt.grid(True, alpha=0.3)
    
    # 3. 클러스터별 샘플 수
    plt.subplot(2, 2, 3)
    plt.bar(cluster_stats.index, cluster_stats['count'])
    plt.xlabel('클러스터')
    plt.ylabel('샘플 수')
    plt.title('클러스터별 샘플 수')
    plt.grid(True, alpha=0.3)
    
    # 4. 클러스터별 Yield 표준편차
    plt.subplot(2, 2, 4)
    plt.bar(cluster_stats.index, cluster_stats['std'])
    plt.xlabel('클러스터')
    plt.ylabel('Yield 표준편차')
    plt.title('클러스터별 Yield 변동성')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def _create_interval_visualizations(analyzer):
    """구간별 분석 시각화"""
    if 'interval_analysis' not in analyzer.analysis_results:
        return
    
    plt.figure(figsize=(15, 10))
    
    # 1. F/R Level 구간별 Yield
    plt.subplot(2, 2, 1)
    interval_data = analyzer.analysis_results['interval_analysis']['interval_yield_stats']
    
    plt.bar(range(len(interval_data)), interval_data['mean'])
    plt.xticks(range(len(interval_data)), interval_data.index, rotation=45)
    plt.xlabel('F/R Level 구간')
    plt.ylabel('평균 Yield')
    plt.title('F/R Level 구간별 평균 Yield')
    plt.grid(True, alpha=0.3)
    
    # 2. 구간별 품질값 변화
    plt.subplot(2, 2, 2)
    quality_data = analyzer.analysis_results['interval_analysis']['quality_interval_analysis']
    
    for col in analyzer.quality_columns:
        if col in quality_data:
            stats = quality_data[col]
            plt.plot(range(len(stats)), stats['mean'], marker='o', label=col)
    
    plt.xlabel('F/R Level 구간')
    plt.ylabel('평균 품질값')
    plt.title('구간별 품질값 변화')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 구간별 상관관계 히트맵
    plt.subplot(2, 2, 3)
    interval_corr = analyzer.analysis_results['interval_analysis']['interval_correlations']
    
    if interval_corr:
        # 상위 5개 특성의 구간별 상관관계
        top_features = list(interval_corr['75-100%'].keys())[:5] if '75-100%' in interval_corr else []
        
        if top_features:
            corr_matrix = []
            intervals = ['0-25%', '25-50%', '50-75%', '75-100%']
            
            for interval in intervals:
                if interval in interval_corr:
                    row = [interval_corr[interval].get(feat, {}).get('correlation', 0) for feat in top_features]
                    corr_matrix.append(row)
            
            if corr_matrix:
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                           xticklabels=top_features, yticklabels=intervals, fmt='.3f')
                plt.title('구간별 상관관계')
    
    # 4. 구간별 Yield 분포
    plt.subplot(2, 2, 4)
    if 'F/R Level' in analyzer.df.columns:
        level_ranges = pd.cut(analyzer.df['F/R Level'], 
                            bins=[0, 25, 50, 75, 100], 
                            labels=['0-25%', '25-50%', '50-75%', '75-100%'])
        
        yield_by_interval = [analyzer.df[level_ranges == interval]['Yield'].values 
                           for interval in ['0-25%', '25-50%', '50-75%', '75-100%']]
        
        plt.boxplot(yield_by_interval, labels=['0-25%', '25-50%', '50-75%', '75-100%'])
        plt.xlabel('F/R Level 구간')
        plt.ylabel('Yield')
        plt.title('구간별 Yield 분포')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def _create_yield_analysis_visualizations(analyzer):
    """Yield vs Input/Output 분석 시각화"""
    if 'yield_analysis' not in analyzer.analysis_results:
        return
    
    plt.figure(figsize=(15, 10))
    
    # 1. Input vs Output 산점도
    plt.subplot(2, 2, 1)
    plt.scatter(analyzer.df['Input_source'], analyzer.df['Calculated_Output'], alpha=0.6)
    
    # 회귀선 추가
    z = np.polyfit(analyzer.df['Input_source'], analyzer.df['Calculated_Output'], 1)
    p = np.poly1d(z)
    plt.plot(analyzer.df['Input_source'], p(analyzer.df['Input_source']), "r--", alpha=0.8)
    
    plt.xlabel('Input Source')
    plt.ylabel('Calculated Output')
    plt.title('Input vs Output (회귀선 포함)')
    plt.grid(True, alpha=0.3)
    
    # 2. Input 구간별 Output 변화
    plt.subplot(2, 2, 2)
    input_analysis = analyzer.analysis_results['yield_analysis']['input_output_analysis']
    
    plt.plot(range(len(input_analysis)), input_analysis['Calculated_Output'], marker='o')
    plt.xticks(range(len(input_analysis)), input_analysis.index, rotation=45)
    plt.xlabel('Input 구간')
    plt.ylabel('평균 Output')
    plt.title('Input 증가에 따른 Output 변화')
    plt.grid(True, alpha=0.3)
    
    # 3. Input 구간별 Yield 변화
    plt.subplot(2, 2, 3)
    plt.plot(range(len(input_analysis)), input_analysis['Yield'], marker='o', color='green')
    plt.xticks(range(len(input_analysis)), input_analysis.index, rotation=45)
    plt.xlabel('Input 구간')
    plt.ylabel('평균 Yield')
    plt.title('Input 증가에 따른 Yield 변화')
    plt.grid(True, alpha=0.3)
    
    # 4. Output 일정성 분석
    plt.subplot(2, 2, 4)
    plt.hist(analyzer.df['Calculated_Output'], bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(analyzer.df['Calculated_Output'].mean(), color='red', linestyle='--', 
               label=f'평균: {analyzer.df["Calculated_Output"].mean():.3f}')
    plt.axvline(analyzer.df['Calculated_Output'].mean() + analyzer.df['Calculated_Output'].std(), 
               color='orange', linestyle='--', label=f'+1σ: {analyzer.df["Calculated_Output"].mean() + analyzer.df["Calculated_Output"].std():.3f}')
    plt.axvline(analyzer.df['Calculated_Output'].mean() - analyzer.df['Calculated_Output'].std(), 
               color='orange', linestyle='--', label=f'-1σ: {analyzer.df["Calculated_Output"].mean() - analyzer.df["Calculated_Output"].std():.3f}')
    plt.xlabel('Calculated Output')
    plt.ylabel('빈도')
    plt.title('Output 일정성 분석')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
