import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def save_plots_to_pdf(plots_data, filename="eda_report.pdf"):
    """
    EDA 플롯들을 PDF로 저장
    
    Parameters:
    plots_data: dict - 플롯 데이터 딕셔너리
    filename: str - PDF 파일명
    """
    with PdfPages(filename) as pdf:
        # 커버 페이지
        create_cover_page(pdf, plots_data)
        
        # 요약 페이지
        create_summary_page(pdf, plots_data)
        
        # 각 분석 페이지들
        if 'correlation' in plots_data:
            create_correlation_page(pdf, plots_data['correlation'])
        
        if 'importance' in plots_data:
            create_importance_page(pdf, plots_data['importance'])
        
        if 'performance' in plots_data:
            create_performance_page(pdf, plots_data['performance'])
        
        if 'ensemble' in plots_data:
            create_ensemble_page(pdf, plots_data['ensemble'])
        
        # 결론 페이지
        create_conclusion_page(pdf, plots_data)
    
    print(f"PDF 리포트가 생성되었습니다: {filename}")
    return filename

def create_cover_page(pdf, data):
    """커버 페이지 생성"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 제목
    title = "제조 산업 센서 데이터\nEnsemble 모델 EDA 리포트"
    ax.text(0.5, 0.8, title, fontsize=24, fontweight='bold', 
            ha='center', va='center', transform=ax.transAxes)
    
    # 생성 정보
    info_text = f"""
    생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    분석 내용:
    • Basic Tag 변환 방법 분석
    • 상관관계 분석
    • Feature Importance 분석
    • 모델 성능 비교
    • Time Lag 분석
    • Ensemble 결정 분석
    """
    
    ax.text(0.5, 0.4, info_text, fontsize=12, ha='center', va='center',
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightblue", alpha=0.3))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_summary_page(pdf, data):
    """요약 페이지 생성"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 성능 요약
    if 'performance' in data:
        metrics = data['performance']['metrics']
        summary_text = f"""
        📊 모델 성능 요약
        
        Model A:
        - R² Score: {metrics['Model A']['R2']:.4f}
        - MSE: {metrics['Model A']['MSE']:.4f}
        - MAE: {metrics['Model A']['MAE']:.4f}
        
        Model B:
        - R² Score: {metrics['Model B']['R2']:.4f}
        - MSE: {metrics['Model B']['MSE']:.4f}
        - MAE: {metrics['Model B']['MAE']:.4f}
        
        Ensemble:
        - R² Score: {metrics['Ensemble']['R2']:.4f}
        - MSE: {metrics['Ensemble']['MSE']:.4f}
        - MAE: {metrics['Ensemble']['MAE']:.4f}
        
        🏆 최고 성능 모델: {max(metrics.keys(), key=lambda x: metrics[x]['R2'])}
        """
    else:
        summary_text = "성능 데이터가 없습니다."
    
    ax.text(0.05, 0.95, summary_text, fontsize=12, ha='left', va='top',
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightgreen", alpha=0.3))
    
    # 주요 발견사항
    findings_text = """
    🔍 주요 발견사항
    
    • Model A와 B는 서로 다른 feature engineering 전략 사용
    • Ensemble 모델이 개별 모델보다 성능 향상
    • Threshold 조정을 통한 성능 최적화 가능
    
    📈 개선 포인트
    • Feature engineering 최적화
    • Threshold 조정 실험
    • 새로운 변환 방법 도입 검토
    """
    
    ax.text(0.05, 0.4, findings_text, fontsize=12, ha='left', va='top',
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightyellow", alpha=0.3))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_correlation_page(pdf, data):
    """상관관계 분석 페이지"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('상관관계 분석', fontsize=16, fontweight='bold')
    
    # Target과의 상관관계 비교
    if 'target_corr_a' in data and 'target_corr_b' in data:
        target_corr_a = data['target_corr_a']
        target_corr_b = data['target_corr_b']
        
        # Model A Top 10 상관관계
        ax1 = axes[0, 0]
        top_target_corr_a = target_corr_a.head(10)
        bars = ax1.barh(range(len(top_target_corr_a)), top_target_corr_a.values,
                       color=['red' if x < 0 else 'blue' for x in top_target_corr_a.values])
        ax1.set_yticks(range(len(top_target_corr_a)))
        ax1.set_yticklabels(top_target_corr_a.index, fontsize=8)
        ax1.set_xlabel('Correlation with Target')
        ax1.set_title('Model A - Top 10 Correlations')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Model B Top 10 상관관계
        ax2 = axes[0, 1]
        top_target_corr_b = target_corr_b.head(10)
        bars = ax2.barh(range(len(top_target_corr_b)), top_target_corr_b.values,
                       color=['red' if x < 0 else 'blue' for x in top_target_corr_b.values])
        ax2.set_yticks(range(len(top_target_corr_b)))
        ax2.set_yticklabels(top_target_corr_b.index, fontsize=8)
        ax2.set_xlabel('Correlation with Target')
        ax2.set_title('Model B - Top 10 Correlations')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # 상관관계 분포 비교
        ax3 = axes[1, 0]
        ax3.hist(target_corr_a.values, bins=30, alpha=0.7, color='skyblue', 
                edgecolor='black', label='Model A')
        ax3.hist(target_corr_b.values, bins=30, alpha=0.7, color='lightcoral', 
                edgecolor='black', label='Model B')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Correlation with Target')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Correlation Distribution')
        ax3.legend()
        
        # 상관관계 히트맵 (Model A)
        ax4 = axes[1, 1]
        if 'correlation_matrix_a' in data:
            correlation_matrix_a = data['correlation_matrix_a']
            sns.heatmap(correlation_matrix_a.iloc[:10, :10], cmap='coolwarm', center=0, 
                       cbar_kws={'shrink': 0.8}, ax=ax4)
            ax4.set_title('Model A - Correlation Heatmap (Top 10)')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_importance_page(pdf, data):
    """Feature Importance 분석 페이지"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Importance 분석', fontsize=16, fontweight='bold')
    
    if 'importance_df_a' in data and 'importance_df_b' in data:
        importance_df_a = data['importance_df_a']
        importance_df_b = data['importance_df_b']
        
        # Model A Top 15 Importance
        ax1 = axes[0, 0]
        top_a = importance_df_a.nlargest(15, 'Importance')
        ax1.barh(range(len(top_a)), top_a['Importance'], color='skyblue')
        ax1.set_yticks(range(len(top_a)))
        ax1.set_yticklabels(top_a['Tag'], fontsize=8)
        ax1.set_xlabel('Importance')
        ax1.set_title('Model A - Top 15 Feature Importance')
        
        # Model B Top 15 Importance
        ax2 = axes[0, 1]
        top_b = importance_df_b.nlargest(15, 'Importance')
        ax2.barh(range(len(top_b)), top_b['Importance'], color='lightcoral')
        ax2.set_yticks(range(len(top_b)))
        ax2.set_yticklabels(top_b['Tag'], fontsize=8)
        ax2.set_xlabel('Importance')
        ax2.set_title('Model B - Top 15 Feature Importance')
        
        # Basic tag별 중요도 분석 (Model A)
        if 'basic_importance_a' in data:
            ax3 = axes[1, 0]
            basic_importance_a = data['basic_importance_a']
            if basic_importance_a:
                tags = list(basic_importance_a.keys())
                importance = list(basic_importance_a.values())
                ax3.bar(tags, importance, color='skyblue', alpha=0.8)
                ax3.set_xlabel('Basic Tags')
                ax3.set_ylabel('Average Importance')
                ax3.set_title('Model A - Basic Tag Importance')
                ax3.tick_params(axis='x', rotation=45)
        
        # Basic tag별 중요도 분석 (Model B)
        if 'basic_importance_b' in data:
            ax4 = axes[1, 1]
            basic_importance_b = data['basic_importance_b']
            if basic_importance_b:
                tags = list(basic_importance_b.keys())
                importance = list(basic_importance_b.values())
                ax4.bar(tags, importance, color='lightcoral', alpha=0.8)
                ax4.set_xlabel('Basic Tags')
                ax4.set_ylabel('Average Importance')
                ax4.set_title('Model B - Basic Tag Importance')
                ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_performance_page(pdf, data):
    """모델 성능 비교 페이지"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('모델 성능 비교', fontsize=16, fontweight='bold')
    
    if 'metrics' in data:
        metrics = data['metrics']
        model_names = list(metrics.keys())
        mse_values = [metrics[name]['MSE'] for name in model_names]
        mae_values = [metrics[name]['MAE'] for name in model_names]
        r2_values = [metrics[name]['R2'] for name in model_names]
        
        # MSE 비교
        ax1 = axes[0, 0]
        bars = ax1.bar(model_names, mse_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax1.set_ylabel('Mean Squared Error')
        ax1.set_title('MSE Comparison')
        ax1.tick_params(axis='x', rotation=45)
        
        # MAE 비교
        ax2 = axes[0, 1]
        bars = ax2.bar(model_names, mae_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('MAE Comparison')
        ax2.tick_params(axis='x', rotation=45)
        
        # R² 비교
        ax3 = axes[0, 2]
        bars = ax3.bar(model_names, r2_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax3.set_ylabel('R² Score')
        ax3.set_title('R² Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # 예측값 vs 실제값 비교
        if 'pred_a' in data and 'pred_b' in data and 'ensemble_pred' in data:
            ax4 = axes[1, 0]
            y = data.get('y', np.arange(len(data['pred_a'])))
            ax4.scatter(y, data['pred_a'], alpha=0.6, label='Model A', s=20)
            ax4.scatter(y, data['pred_b'], alpha=0.6, label='Model B', s=20)
            ax4.scatter(y, data['ensemble_pred'], alpha=0.6, label='Ensemble', s=20)
            ax4.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
            ax4.set_xlabel('Actual Values')
            ax4.set_ylabel('Predicted Values')
            ax4.set_title('Predicted vs Actual')
            ax4.legend()
        
        # 잔차 분석
        if 'residuals_a' in data and 'residuals_b' in data and 'residuals_ensemble' in data:
            ax5 = axes[1, 1]
            ax5.scatter(data['pred_a'], data['residuals_a'], alpha=0.6, label='Model A', s=20)
            ax5.scatter(data['pred_b'], data['residuals_b'], alpha=0.6, label='Model B', s=20)
            ax5.scatter(data['ensemble_pred'], data['residuals_ensemble'], alpha=0.6, label='Ensemble', s=20)
            ax5.axhline(y=0, color='r', linestyle='--')
            ax5.set_xlabel('Predicted Values')
            ax5.set_ylabel('Residuals')
            ax5.set_title('Residual Plot')
            ax5.legend()
        
        # 잔차 분포
        if 'residuals_a' in data and 'residuals_b' in data and 'residuals_ensemble' in data:
            ax6 = axes[1, 2]
            ax6.hist(data['residuals_a'], alpha=0.7, label='Model A', bins=30)
            ax6.hist(data['residuals_b'], alpha=0.7, label='Model B', bins=30)
            ax6.hist(data['residuals_ensemble'], alpha=0.7, label='Ensemble', bins=30)
            ax6.set_xlabel('Residuals')
            ax6.set_ylabel('Frequency')
            ax6.set_title('Residual Distribution')
            ax6.legend()
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_ensemble_page(pdf, data):
    """Ensemble 결정 분석 페이지"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Ensemble 결정 분석', fontsize=16, fontweight='bold')
    
    # 결정 분포
    if 'use_model_a' in data:
        ax1 = axes[0, 0]
        use_model_a = data['use_model_a']
        decision_counts = [np.sum(use_model_a), np.sum(~use_model_a)]
        ax1.pie(decision_counts, labels=['Model A', 'Model B'], autopct='%1.1f%%',
                colors=['skyblue', 'lightcoral'])
        ax1.set_title('Ensemble Decision Distribution')
    
    # 신뢰도 분포
    if 'confidence_a' in data:
        ax2 = axes[0, 1]
        confidence_a = data['confidence_a']
        threshold = data.get('threshold', 0.5)
        ax2.hist(confidence_a, bins=50, alpha=0.7, color='lightblue', edgecolor='black')
        ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {threshold}')
        ax2.set_xlabel('Model A Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Model A Confidence Distribution')
        ax2.legend()
    
    # 모델별 예측값 분포
    if 'pred_a' in data and 'pred_b' in data:
        ax3 = axes[0, 2]
        ax3.hist(data['pred_a'], alpha=0.7, label='Model A', bins=30, color='skyblue')
        ax3.hist(data['pred_b'], alpha=0.7, label='Model B', bins=30, color='lightcoral')
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Prediction Distribution')
        ax3.legend()
    
    # 예측값 차이 분석
    if 'pred_a' in data and 'pred_b' in data:
        ax4 = axes[1, 0]
        pred_diff = data['pred_a'] - data['pred_b']
        ax4.hist(pred_diff, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_xlabel('Prediction Difference (A - B)')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Prediction Difference Distribution')
    
    # 시간에 따른 결정 변화
    if 'use_model_a' in data and len(data['use_model_a']) > 100:
        ax5 = axes[1, 1]
        use_model_a = data['use_model_a']
        window_size = len(use_model_a) // 20
        decision_rate = []
        time_points = []
        
        for i in range(0, len(use_model_a) - window_size, window_size):
            window_decisions = use_model_a[i:i+window_size]
            decision_rate.append(np.mean(window_decisions))
            time_points.append(i + window_size//2)
        
        ax5.plot(time_points, decision_rate, marker='o', linewidth=2)
        ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        ax5.set_xlabel('Time Point')
        ax5.set_ylabel('Model A Usage Rate')
        ax5.set_title('Model A Usage Over Time')
        ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_conclusion_page(pdf, data):
    """결론 및 권장사항 페이지"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')
    
    # 성능 요약
    if 'performance' in data and 'metrics' in data['performance']:
        metrics = data['performance']['metrics']
        conclusion_text = f"""
        📊 분석 결과 요약
        
        🏆 성능 비교:
        • Model A R²: {metrics['Model A']['R2']:.4f}
        • Model B R²: {metrics['Model B']['R2']:.4f}
        • Ensemble R²: {metrics['Ensemble']['R2']:.4f}
        
        🎯 Ensemble 사용률:
        • Model A 사용률: {np.mean(data['ensemble']['use_model_a']):.2%}
        • Model B 사용률: {np.mean(~data['ensemble']['use_model_a']):.2%}
        
        📈 주요 발견사항:
        • {max(metrics.keys(), key=lambda x: metrics[x]['R2'])}가 최고 성능
        • 평균 신뢰도: {np.mean(data['ensemble']['confidence_a']):.3f}
        
        🔧 권장사항:
        • Feature engineering 최적화
        • Threshold 조정 실험
        • 새로운 변환 방법 도입 검토
        • 실시간 성능 모니터링 시스템 구축
        • 정기적인 모델 재훈련 계획 수립
        
        📋 다음 단계:
        1. 운영 환경에서의 성능 검증
        2. 새로운 센서 데이터에 대한 모델 적응성 테스트
        3. Ensemble 전략의 실시간 효과성 모니터링
        4. 추가 feature engineering 실험
        """
    else:
        conclusion_text = """
        📊 분석 결과 요약
        
        🔧 권장사항:
        • Feature engineering 최적화
        • Threshold 조정 실험
        • 새로운 변환 방법 도입 검토
        • 실시간 성능 모니터링 시스템 구축
        • 정기적인 모델 재훈련 계획 수립
        """
    
    ax.text(0.05, 0.95, conclusion_text, fontsize=12, ha='left', va='top',
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.5", 
            facecolor="lightblue", alpha=0.3))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

def create_html_report(plots_data, filename="eda_report.html"):
    """HTML 리포트 생성"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>제조 산업 센서 데이터 Ensemble 모델 EDA 리포트</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 10px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 5px; }}
            .highlight {{ background-color: #ffffcc; padding: 5px; border-radius: 3px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>제조 산업 센서 데이터 Ensemble 모델 EDA 리포트</h1>
            <p>생성일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>📊 모델 성능 요약</h2>
    """
    
    if 'performance' in plots_data and 'metrics' in plots_data['performance']:
        metrics = plots_data['performance']['metrics']
        html_content += f"""
            <div class="metric">
                <h3>Model A</h3>
                <p>R² Score: <span class="highlight">{metrics['Model A']['R2']:.4f}</span></p>
                <p>MSE: {metrics['Model A']['MSE']:.4f}</p>
                <p>MAE: {metrics['Model A']['MAE']:.4f}</p>
            </div>
            <div class="metric">
                <h3>Model B</h3>
                <p>R² Score: <span class="highlight">{metrics['Model B']['R2']:.4f}</span></p>
                <p>MSE: {metrics['Model B']['MSE']:.4f}</p>
                <p>MAE: {metrics['Model B']['MAE']:.4f}</p>
            </div>
            <div class="metric">
                <h3>Ensemble</h3>
                <p>R² Score: <span class="highlight">{metrics['Ensemble']['R2']:.4f}</span></p>
                <p>MSE: {metrics['Ensemble']['MSE']:.4f}</p>
                <p>MAE: {metrics['Ensemble']['MAE']:.4f}</p>
            </div>
        """
    
    html_content += """
        </div>
        
        <div class="section">
            <h2>🔍 주요 발견사항</h2>
            <ul>
                <li>Model A와 B는 서로 다른 feature engineering 전략 사용</li>
                <li>Ensemble 모델이 개별 모델보다 성능 향상</li>
                <li>Threshold 조정을 통한 성능 최적화 가능</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>🔧 권장사항</h2>
            <ul>
                <li>Feature engineering 최적화</li>
                <li>Threshold 조정 실험</li>
                <li>새로운 변환 방법 도입 검토</li>
                <li>실시간 성능 모니터링 시스템 구축</li>
                <li>정기적인 모델 재훈련 계획 수립</li>
            </ul>
        </div>
        
        <div class="section">
            <h2>📋 다음 단계</h2>
            <ol>
                <li>운영 환경에서의 성능 검증</li>
                <li>새로운 센서 데이터에 대한 모델 적응성 테스트</li>
                <li>Ensemble 전략의 실시간 효과성 모니터링</li>
                <li>추가 feature engineering 실험</li>
            </ol>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML 리포트가 생성되었습니다: {filename}")
    return filename

# 사용 예시
def generate_eda_reports(plots_data, base_filename="eda_report"):
    """
    EDA 결과를 PDF와 HTML로 저장
    
    Parameters:
    plots_data: dict - 플롯 데이터 딕셔너리
    base_filename: str - 기본 파일명
    
    Returns:
    tuple: (PDF 파일명, HTML 파일명)
    """
    pdf_filename = f"{base_filename}.pdf"
    html_filename = f"{base_filename}.html"
    
    # PDF 리포트 생성
    save_plots_to_pdf(plots_data, pdf_filename)
    
    # HTML 리포트 생성
    create_html_report(plots_data, html_filename)
    
    return pdf_filename, html_filename
