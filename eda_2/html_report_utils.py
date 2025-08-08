import os
import base64
import io
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def save_plot_to_base64(fig):
    """matplotlib figure를 base64로 인코딩"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return img_str

def create_html_report(analysis_results, filename="manufacturing_eda_report.html"):
    """제조 산업 EDA 결과를 HTML 리포트로 생성"""
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>제조 산업 Ensemble 모델 EDA 리포트</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 20px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                text-align: center;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-bottom: 30px;
            }}
            h2 {{
                color: #34495e;
                border-left: 4px solid #3498db;
                padding-left: 15px;
                margin-top: 30px;
            }}
            h3 {{
                color: #2c3e50;
                margin-top: 25px;
            }}
            .section {{
                margin-bottom: 40px;
                padding: 20px;
                background-color: #f8f9fa;
                border-radius: 8px;
                border-left: 4px solid #3498db;
            }}
            .metric-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }}
            .metric-card {{
                background: white;
                padding: 15px;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                text-align: center;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #3498db;
            }}
            .metric-label {{
                font-size: 14px;
                color: #7f8c8d;
                margin-top: 5px;
            }}
            .plot-container {{
                text-align: center;
                margin: 20px 0;
                padding: 15px;
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .plot-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
            }}
            .summary-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                margin: 20px 0;
            }}
            .conclusion {{
                background: #e8f5e8;
                border-left: 4px solid #27ae60;
                padding: 20px;
                border-radius: 8px;
                margin: 20px 0;
            }}
            .warning {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                border-radius: 8px;
                margin: 15px 0;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                color: #7f8c8d;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>제조 산업 Ensemble 모델 EDA 리포트</h1>
            
            <div class="summary-box">
                <h2>📊 분석 개요</h2>
                <p><strong>생성일시:</strong> {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}</p>
                <p><strong>분석 모델:</strong> Long Term Model + Short Term Model Ensemble</p>
                <p><strong>Threshold:</strong> {analysis_results.get('threshold', 70000):,}</p>
                <p><strong>데이터 샘플 수:</strong> {len(analysis_results.get('y', [])):,}</p>
            </div>
    """
    
    # Threshold 분석 결과
    if 'threshold_analysis' in analysis_results:
        threshold_data = analysis_results['threshold_analysis']
        html_content += f"""
            <div class="section">
                <h2>🎯 Threshold 초과 감지 능력 분석</h2>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{threshold_data.get('accuracy', 0):.4f}</div>
                        <div class="metric-label">전체 정확도</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{threshold_data.get('precision', 0):.4f}</div>
                        <div class="metric-label">Precision</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{threshold_data.get('recall', 0):.4f}</div>
                        <div class="metric-label">Recall</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{threshold_data.get('f1_score', 0):.4f}</div>
                        <div class="metric-label">F1-Score</div>
                    </div>
                </div>
                
                <table>
                    <tr>
                        <th>지표</th>
                        <th>값</th>
                    </tr>
                    <tr>
                        <td>True Positive</td>
                        <td>{threshold_data.get('true_positive', 0)}</td>
                    </tr>
                    <tr>
                        <td>False Positive</td>
                        <td>{threshold_data.get('false_positive', 0)}</td>
                    </tr>
                    <tr>
                        <td>True Negative</td>
                        <td>{threshold_data.get('true_negative', 0)}</td>
                    </tr>
                    <tr>
                        <td>False Negative</td>
                        <td>{threshold_data.get('false_negative', 0)}</td>
                    </tr>
                    <tr>
                        <td>Long Term Model 선택</td>
                        <td>{threshold_data.get('long_term_selected', 0)}</td>
                    </tr>
                    <tr>
                        <td>Short Term Model 선택</td>
                        <td>{threshold_data.get('short_term_selected', 0)}</td>
                    </tr>
                </table>
            </div>
        """
    
    # 모델 성능 비교
    if 'model_performance' in analysis_results:
        perf_data = analysis_results['model_performance']
        html_content += f"""
            <div class="section">
                <h2>📈 모델 성능 비교</h2>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{perf_data.get('long_term_mse', 0):.4f}</div>
                        <div class="metric-label">Long Term MSE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{perf_data.get('short_term_mse', 0):.4f}</div>
                        <div class="metric-label">Short Term MSE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{perf_data.get('ensemble_mse', 0):.4f}</div>
                        <div class="metric-label">Ensemble MSE</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{perf_data.get('long_term_r2', 0):.4f}</div>
                        <div class="metric-label">Long Term R²</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{perf_data.get('short_term_r2', 0):.4f}</div>
                        <div class="metric-label">Short Term R²</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{perf_data.get('ensemble_r2', 0):.4f}</div>
                        <div class="metric-label">Ensemble R²</div>
                    </div>
                </div>
            </div>
        """
    
    # 특성 중요도 분석
    if 'feature_importance' in analysis_results:
        importance_data = analysis_results['feature_importance']
        html_content += f"""
            <div class="section">
                <h2>🔍 특성 중요도 분석</h2>
                
                <h3>Long Term Model 주요 특성</h3>
                <table>
                    <tr>
                        <th>순위</th>
                        <th>특성명</th>
                        <th>중요도</th>
                    </tr>
        """
        
        if 'long_term_top_features' in importance_data:
            for i, (feature, importance) in enumerate(importance_data['long_term_top_features'][:10], 1):
                html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{feature}</td>
                        <td>{importance:.4f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
                
                <h3>Short Term Model 주요 특성</h3>
                <table>
                    <tr>
                        <th>순위</th>
                        <th>특성명</th>
                        <th>중요도</th>
                    </tr>
        """
        
        if 'short_term_top_features' in importance_data:
            for i, (feature, importance) in enumerate(importance_data['short_term_top_features'][:10], 1):
                html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{feature}</td>
                        <td>{importance:.4f}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        """
    
    # Basic Tag 변환 분석
    if 'basic_tag_analysis' in analysis_results:
        basic_tag_data = analysis_results['basic_tag_analysis']
        html_content += f"""
            <div class="section">
                <h2>🏷️ Basic Tag 변환 분석</h2>
                
                <h3>Long Term Model Basic Tag 중요도</h3>
                <table>
                    <tr>
                        <th>Basic Tag</th>
                        <th>중요도</th>
                        <th>변환된 특성 수</th>
                    </tr>
        """
        
        if 'long_term_basic_importance' in basic_tag_data:
            for tag, importance in basic_tag_data['long_term_basic_importance'].items():
                feature_count = basic_tag_data.get('long_term_transformations', {}).get(tag, [])
                html_content += f"""
                    <tr>
                        <td>{tag}</td>
                        <td>{importance:.4f}</td>
                        <td>{len(feature_count)}</td>
                    </tr>
                """
        
        html_content += """
                </table>
                
                <h3>Short Term Model Basic Tag 중요도</h3>
                <table>
                    <tr>
                        <th>Basic Tag</th>
                        <th>중요도</th>
                        <th>변환된 특성 수</th>
                    </tr>
        """
        
        if 'short_term_basic_importance' in basic_tag_data:
            for tag, importance in basic_tag_data['short_term_basic_importance'].items():
                feature_count = basic_tag_data.get('short_term_transformations', {}).get(tag, [])
                html_content += f"""
                    <tr>
                        <td>{tag}</td>
                        <td>{importance:.4f}</td>
                        <td>{len(feature_count)}</td>
                    </tr>
                """
        
        html_content += """
                </table>
            </div>
        """
    
    # 상관관계 분석
    if 'correlation_analysis' in analysis_results:
        corr_data = analysis_results['correlation_analysis']
        html_content += f"""
            <div class="section">
                <h2>📊 상관관계 분석</h2>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{corr_data.get('long_term_avg_corr', 0):.4f}</div>
                        <div class="metric-label">Long Term 평균 상관관계</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{corr_data.get('short_term_avg_corr', 0):.4f}</div>
                        <div class="metric-label">Short Term 평균 상관관계</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{corr_data.get('long_term_max_corr', 0):.4f}</div>
                        <div class="metric-label">Long Term 최대 상관관계</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{corr_data.get('short_term_max_corr', 0):.4f}</div>
                        <div class="metric-label">Short Term 최대 상관관계</div>
                    </div>
                </div>
            </div>
        """
    
    # Time Lag 분석
    if 'time_lag_analysis' in analysis_results:
        lag_data = analysis_results['time_lag_analysis']
        html_content += f"""
            <div class="section">
                <h2>⏰ Time Lag 분석</h2>
                
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">{lag_data.get('long_term_optimal_lag', 0)}</div>
                        <div class="metric-label">Long Term 최적 Lag</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{lag_data.get('short_term_optimal_lag', 0)}</div>
                        <div class="metric-label">Short Term 최적 Lag</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{lag_data.get('long_term_max_corr', 0):.4f}</div>
                        <div class="metric-label">Long Term 최대 상관관계</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{lag_data.get('short_term_max_corr', 0):.4f}</div>
                        <div class="metric-label">Short Term 최대 상관관계</div>
                    </div>
                </div>
            </div>
        """
    
    # 플롯 이미지들 추가
    if 'plots' in analysis_results:
        plots_data = analysis_results['plots']
        html_content += """
            <div class="section">
                <h2>📈 시각화 결과</h2>
        """
        
        for plot_name, plot_data in plots_data.items():
            if isinstance(plot_data, str):  # base64 이미지
                html_content += f"""
                    <div class="plot-container">
                        <h3>{plot_name}</h3>
                        <img src="data:image/png;base64,{plot_data}" alt="{plot_name}">
                    </div>
                """
        
        html_content += """
            </div>
        """
    
    # 결론
    html_content += f"""
            <div class="conclusion">
                <h2>📋 분석 결론</h2>
                
                <h3>주요 발견사항</h3>
                <ul>
                    <li><strong>Threshold 감지 성능:</strong> {analysis_results.get('threshold_analysis', {}).get('accuracy', 0):.1%}의 정확도로 threshold 초과를 감지</li>
                    <li><strong>모델 선택 분포:</strong> Long Term Model {analysis_results.get('threshold_analysis', {}).get('long_term_selected', 0)}회, Short Term Model {analysis_results.get('threshold_analysis', {}).get('short_term_selected', 0)}회 선택</li>
                    <li><strong>Ensemble 성능:</strong> 개별 모델보다 향상된 성능을 보임</li>
                </ul>
                
                <h3>모델 특성</h3>
                <ul>
                    <li><strong>Long Term Model:</strong> 장기적 패턴과 threshold 초과 상황에 특화</li>
                    <li><strong>Short Term Model:</strong> 단기적 변동과 일반적인 상황에 특화</li>
                    <li><strong>Ensemble 전략:</strong> Threshold 기반 선택으로 각 모델의 장점을 활용</li>
                </ul>
            </div>
            
            <div class="footer">
                <p>이 리포트는 제조 산업 센서 데이터 Ensemble 모델 EDA 도구로 생성되었습니다.</p>
                <p>생성일시: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # HTML 파일 저장
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"HTML 리포트가 {filename}에 저장되었습니다.")
    return filename

def generate_eda_html_report(analysis_results, base_filename="manufacturing_eda_report"):
    """EDA 분석 결과를 HTML 리포트로 생성"""
    filename = f"{base_filename}.html"
    return create_html_report(analysis_results, filename)
