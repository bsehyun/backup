import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정 (Windows)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ACNIntegratedReportGenerator:
    """
    ACN 정제 공정 종합 분석 결과 통합 리포트 생성기
    - main_advanced_experiments
    - main_ml_analysis  
    - main_analysis
    - main_comprehensive_analysis
    모든 결과를 통합하여 하나의 HTML 리포트 생성
    """
    
    def __init__(self):
        self.report_data = {}
        self.analysis_summary = {}
        
    def integrate_all_results(self, 
                            advanced_exp_results=None,
                            ml_analysis_results=None, 
                            basic_analysis_results=None,
                            comprehensive_results=None):
        """
        모든 분석 결과를 통합
        
        Parameters:
        advanced_exp_results: main_advanced_experiments 결과
        ml_analysis_results: main_ml_analysis 결과
        basic_analysis_results: main_analysis 결과  
        comprehensive_results: main_comprehensive_analysis 결과
        """
        print("=" * 80)
        print("ACN 정제 공정 종합 분석 결과 통합")
        print("=" * 80)
        
        self.report_data = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'advanced_experiments': advanced_exp_results,
            'ml_analysis': ml_analysis_results,
            'basic_analysis': basic_analysis_results,
            'comprehensive_analysis': comprehensive_results
        }
        
        # 분석 결과 요약 생성
        self._generate_analysis_summary()
        
        print("분석 결과 통합 완료")
        return self.report_data
    
    def _generate_analysis_summary(self):
        """분석 결과 요약 생성"""
        summary = {
            'data_overview': {},
            'key_findings': [],
            'recommendations': [],
            'model_performance': {},
            'feature_importance': {},
            'optimization_insights': {}
        }
        
        # 1. 데이터 개요
        if self.report_data['comprehensive_analysis']:
            comp_data = self.report_data['comprehensive_analysis']
            if 'analyzer' in comp_data:
                analyzer = comp_data['analyzer']
                summary['data_overview'] = {
                    'total_samples': len(analyzer.df),
                    'final_samples': len(analyzer.final_data) if analyzer.final_data is not None else 0,
                    'process_samples': len(analyzer.process_data) if analyzer.process_data is not None else 0,
                    'features': len(analyzer.df.columns)
                }
        
        # 2. 주요 발견사항 수집
        self._collect_key_findings(summary)
        
        # 3. 권장사항 수집
        self._collect_recommendations(summary)
        
        # 4. 모델 성능 수집
        self._collect_model_performance(summary)
        
        # 5. 특성 중요도 수집
        self._collect_feature_importance(summary)
        
        # 6. 최적화 인사이트 수집
        self._collect_optimization_insights(summary)
        
        self.analysis_summary = summary
    
    def _collect_key_findings(self, summary):
        """주요 발견사항 수집"""
        findings = []
        
        # 고급 실험 결과
        if self.report_data['advanced_experiments']:
            exp_data = self.report_data['advanced_experiments']
            if 'report' in exp_data and 'experiments' in exp_data['report']:
                experiments = exp_data['report']['experiments']
                
                if 'partial_regression' in experiments:
                    top_feature = experiments['partial_regression']['top_features'][0]
                    findings.append(f"Partial Regression: {top_feature['feature']}가 가장 높은 순수 상관관계 ({top_feature['correlation']:.4f})")
                
                if 'sobol' in experiments:
                    top_sensitive = experiments['sobol']['top_sensitive_features'][0]
                    findings.append(f"Sobol 민감도: {top_sensitive['feature']}가 가장 민감한 특성 (S1={top_sensitive['s1']:.4f})")
                
                if 'rsm' in experiments:
                    optimal_yield = experiments['rsm']['optimal_yield']
                    findings.append(f"RSM 최적화: 예측 가능한 최대 Yield {optimal_yield:.4f}")
        
        # ML 분석 결과
        if self.report_data['ml_analysis']:
            ml_data = self.report_data['ml_analysis']
            if 'report' in ml_data:
                report = ml_data['report']
                if 'best_model' in report:
                    best_model = report['best_model']
                    r2_score = report['model_performance'].iloc[0]['R²']
                    findings.append(f"ML 모델: {best_model}이 최고 성능 (R²={r2_score:.4f})")
        
        # 종합 분석 결과
        if self.report_data['comprehensive_analysis']:
            comp_data = self.report_data['comprehensive_analysis']
            if 'analysis_results' in comp_data:
                analysis_results = comp_data['analysis_results']
                if 'yield_analysis' in analysis_results:
                    yield_analysis = analysis_results['yield_analysis']
                    if 'input_output_correlation' in yield_analysis:
                        corr = yield_analysis['input_output_correlation']['correlation']
                        findings.append(f"Input-Output 관계: 상관계수 {corr:.4f}")
        
        summary['key_findings'] = findings
    
    def _collect_recommendations(self, summary):
        """권장사항 수집"""
        recommendations = []
        
        # 고급 실험 결과
        if self.report_data['advanced_experiments']:
            exp_data = self.report_data['advanced_experiments']
            if 'report' in exp_data and 'experiments' in exp_data['report']:
                experiments = exp_data['report']['experiments']
                
                if 'anova' in experiments:
                    significant_count = experiments['anova']['significant_features']
                    recommendations.append(f"ANOVA: {significant_count}개 유의한 특성에 집중하여 모니터링 강화")
                
                if 'sobol' in experiments:
                    top_sensitive = experiments['sobol']['top_sensitive_features'][0]
                    recommendations.append(f"민감도 분석: {top_sensitive['feature']}의 정밀한 제어 필요")
        
        # ML 분석 결과
        if self.report_data['ml_analysis']:
            ml_data = self.report_data['ml_analysis']
            if 'report' in ml_data and 'recommendations' in ml_data['report']:
                ml_recommendations = ml_data['report']['recommendations']
                for rec in ml_recommendations:
                    recommendations.append(f"ML 분석: {rec}")
        
        # 종합 분석 결과
        if self.report_data['comprehensive_analysis']:
            comp_data = self.report_data['comprehensive_analysis']
            if 'analysis_results' in comp_data:
                analysis_results = comp_data['analysis_results']
                if 'yield_analysis' in analysis_results:
                    yield_analysis = analysis_results['yield_analysis']
                    if 'input_output_correlation' in yield_analysis:
                        corr = yield_analysis['input_output_correlation']['correlation']
                        if corr < 0.5:
                            recommendations.append("Input 증가에 따른 Output 향상 방안 모색 필요")
                        else:
                            recommendations.append("Input과 Output의 높은 상관관계를 활용한 최적화")
        
        summary['recommendations'] = recommendations
    
    def _collect_model_performance(self, summary):
        """모델 성능 수집"""
        performance = {}
        
        # ML 분석 결과
        if self.report_data['ml_analysis']:
            ml_data = self.report_data['ml_analysis']
            if 'report' in ml_data and 'model_performance' in ml_data['report']:
                model_perf = ml_data['report']['model_performance']
                best_model = model_perf.iloc[0]
                performance['ml_best_model'] = {
                    'name': best_model['Model'],
                    'r2': best_model['R²'],
                    'rmse': best_model['RMSE'],
                    'cv_score': best_model['CV_Score']
                }
        
        # 고급 실험 결과
        if self.report_data['advanced_experiments']:
            exp_data = self.report_data['advanced_experiments']
            if 'report' in exp_data and 'experiments' in exp_data['report']:
                experiments = exp_data['report']['experiments']
                
                if 'anova' in experiments:
                    performance['anova'] = {
                        'r_squared': experiments['anova']['r_squared'],
                        'significant_features': experiments['anova']['significant_features']
                    }
                
                if 'rsm' in experiments:
                    performance['rsm'] = {
                        'model_type': experiments['rsm']['model_type'],
                        'r_squared': experiments['rsm']['r_squared'],
                        'cv_score': experiments['rsm']['cv_score']
                    }
        
        summary['model_performance'] = performance
    
    def _collect_feature_importance(self, summary):
        """특성 중요도 수집"""
        importance = {}
        
        # ML 분석 결과
        if self.report_data['ml_analysis']:
            ml_data = self.report_data['ml_analysis']
            if 'report' in ml_data and 'feature_importance' in ml_data['report']:
                feature_imp = ml_data['report']['feature_importance']
                top_features = feature_imp.head(5)
                importance['ml_top_features'] = [
                    {'feature': row['feature'], 'score': row['final_importance_score']}
                    for _, row in top_features.iterrows()
                ]
        
        # 고급 실험 결과
        if self.report_data['advanced_experiments']:
            exp_data = self.report_data['advanced_experiments']
            if 'report' in exp_data and 'experiments' in exp_data['report']:
                experiments = exp_data['report']['experiments']
                
                if 'partial_regression' in experiments:
                    top_partial = experiments['partial_regression']['top_features'][:3]
                    importance['partial_regression_top'] = top_partial
                
                if 'sobol' in experiments:
                    top_sobol = experiments['sobol']['top_sensitive_features'][:3]
                    importance['sobol_top'] = top_sobol
        
        # 종합 분석 결과
        if self.report_data['comprehensive_analysis']:
            comp_data = self.report_data['comprehensive_analysis']
            if 'analysis_results' in comp_data:
                analysis_results = comp_data['analysis_results']
                if 'feature_selection' in analysis_results:
                    feature_selection = analysis_results['feature_selection']
                    if 'correlations' in feature_selection:
                        corr_data = feature_selection['correlations']
                        top_corr = corr_data.head(3)
                        importance['correlation_top'] = [
                            {'feature': idx, 'correlation': row['pearson_corr']}
                            for idx, row in top_corr.iterrows()
                        ]
        
        summary['feature_importance'] = importance
    
    def _collect_optimization_insights(self, summary):
        """최적화 인사이트 수집"""
        insights = {}
        
        # 고급 실험 결과
        if self.report_data['advanced_experiments']:
            exp_data = self.report_data['advanced_experiments']
            if 'rsm' in exp_data:
                rsm_data = exp_data['rsm']
                if 'optimal_point' in rsm_data:
                    optimal = rsm_data['optimal_point']
                    insights['rsm_optimal'] = {
                        'optimal_point': optimal['optimal_point'],
                        'predicted_yield': optimal['predicted_yield']
                    }
        
        # 종합 분석 결과
        if self.report_data['comprehensive_analysis']:
            comp_data = self.report_data['comprehensive_analysis']
            if 'analysis_results' in comp_data:
                analysis_results = comp_data['analysis_results']
                if 'yield_analysis' in analysis_results:
                    yield_analysis = analysis_results['yield_analysis']
                    if 'output_consistency' in yield_analysis:
                        consistency = yield_analysis['output_consistency']
                        insights['output_consistency'] = {
                            'mean': consistency['mean'],
                            'std': consistency['std'],
                            'cv': consistency['cv']
                        }
        
        summary['optimization_insights'] = insights
    
    def generate_integrated_html_report(self, output_file='acn_integrated_report.html'):
        """
        통합 HTML 리포트 생성
        """
        print("\n" + "=" * 80)
        print("통합 HTML 리포트 생성")
        print("=" * 80)
        
        html_content = self._create_html_template()
        
        # 파일 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"통합 HTML 리포트가 '{output_file}'로 저장되었습니다.")
        return html_content
    
    def _create_html_template(self):
        """HTML 템플릿 생성"""
        html_template = f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>ACN 정제 공정 종합 분석 통합 리포트</title>
            <style>
                body {{ 
                    font-family: 'Malgun Gothic', Arial, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    line-height: 1.6; 
                    background-color: #f8f9fa;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background-color: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{ 
                    text-align: center; 
                    margin-bottom: 40px; 
                    border-bottom: 3px solid #3498db; 
                    padding-bottom: 20px;
                }}
                .header h1 {{ 
                    color: #2c3e50; 
                    margin: 0; 
                    font-size: 2.5em;
                }}
                .header p {{ 
                    color: #7f8c8d; 
                    margin: 10px 0 0 0; 
                    font-size: 1.1em;
                }}
                .nav {{ 
                    background-color: #34495e; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin-bottom: 30px;
                }}
                .nav ul {{ 
                    list-style: none; 
                    margin: 0; 
                    padding: 0; 
                    display: flex; 
                    justify-content: center; 
                    flex-wrap: wrap;
                }}
                .nav li {{ 
                    margin: 0 20px;
                }}
                .nav a {{ 
                    color: white; 
                    text-decoration: none; 
                    font-weight: bold; 
                    padding: 10px 15px; 
                    border-radius: 3px; 
                    transition: background-color 0.3s;
                }}
                .nav a:hover {{ 
                    background-color: #3498db;
                }}
                .section {{ 
                    margin: 40px 0; 
                    padding: 25px; 
                    border-radius: 8px; 
                    background-color: #f8f9fa;
                }}
                .section h2 {{ 
                    color: #2c3e50; 
                    border-left: 5px solid #3498db; 
                    padding-left: 15px; 
                    margin-top: 0;
                }}
                .section h3 {{ 
                    color: #34495e; 
                    margin-top: 25px; 
                    border-bottom: 2px solid #ecf0f1; 
                    padding-bottom: 10px;
                }}
                .summary-box {{ 
                    background-color: #ecf0f1; 
                    padding: 20px; 
                    border-radius: 5px; 
                    margin: 20px 0; 
                    border-left: 4px solid #3498db;
                }}
                .finding-box {{ 
                    background-color: #e8f5e8; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 10px 0; 
                    border-left: 4px solid #27ae60;
                }}
                .recommendation-box {{ 
                    background-color: #fff3cd; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 10px 0; 
                    border-left: 4px solid #f39c12;
                }}
                .insight-box {{ 
                    background-color: #e3f2fd; 
                    padding: 15px; 
                    border-radius: 5px; 
                    margin: 10px 0; 
                    border-left: 4px solid #2196f3;
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0; 
                    background-color: white;
                }}
                th, td {{ 
                    border: 1px solid #bdc3c7; 
                    padding: 12px; 
                    text-align: left;
                }}
                th {{ 
                    background-color: #34495e; 
                    color: white; 
                    font-weight: bold;
                }}
                tr:nth-child(even) {{ 
                    background-color: #f8f9fa;
                }}
                .metric {{ 
                    font-weight: bold; 
                    color: #e74c3c;
                }}
                .conclusion {{ 
                    background-color: #d4edda; 
                    padding: 25px; 
                    border-radius: 8px; 
                    margin: 30px 0; 
                    border: 2px solid #c3e6cb;
                }}
                .conclusion h3 {{ 
                    color: #155724; 
                    margin-top: 0;
                }}
                .analysis-grid {{ 
                    display: grid; 
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                    gap: 20px; 
                    margin: 20px 0;
                }}
                .analysis-card {{ 
                    background-color: white; 
                    padding: 20px; 
                    border-radius: 8px; 
                    border: 1px solid #dee2e6; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .analysis-card h4 {{ 
                    color: #2c3e50; 
                    margin-top: 0; 
                    border-bottom: 2px solid #3498db; 
                    padding-bottom: 10px;
                }}
                .status-indicator {{ 
                    display: inline-block; 
                    width: 12px; 
                    height: 12px; 
                    border-radius: 50%; 
                    margin-right: 8px;
                }}
                .status-completed {{ background-color: #27ae60; }}
                .status-partial {{ background-color: #f39c12; }}
                .status-missing {{ background-color: #e74c3c; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🧪 ACN 정제 공정 종합 분석 통합 리포트</h1>
                    <p>분석 일시: {self.report_data['analysis_date']}</p>
                </div>
                
                <nav class="nav">
                    <ul>
                        <li><a href="#overview">📊 개요</a></li>
                        <li><a href="#findings">🔍 주요 발견사항</a></li>
                        <li><a href="#recommendations">💡 권장사항</a></li>
                        <li><a href="#performance">📈 모델 성능</a></li>
                        <li><a href="#features">🎯 특성 중요도</a></li>
                        <li><a href="#optimization">⚡ 최적화 인사이트</a></li>
                        <li><a href="#conclusion">🎯 결론</a></li>
                    </ul>
                </nav>
                
                <section id="overview" class="section">
                    <h2>📊 분석 개요</h2>
                    <div class="summary-box">
                        <h3>데이터 정보</h3>
                        <p><strong>전체 샘플 수:</strong> {self.analysis_summary['data_overview'].get('total_samples', 'N/A')}개</p>
                        <p><strong>최종 분석 데이터:</strong> {self.analysis_summary['data_overview'].get('final_samples', 'N/A')}개</p>
                        <p><strong>과정 중 분석 데이터:</strong> {self.analysis_summary['data_overview'].get('process_samples', 'N/A')}개</p>
                        <p><strong>분석 특성 수:</strong> {self.analysis_summary['data_overview'].get('features', 'N/A')}개</p>
                    </div>
                    
                    <div class="analysis-grid">
                        <div class="analysis-card">
                            <h4>고급 실험 분석</h4>
                            <p><span class="status-indicator {'status-completed' if self.report_data['advanced_experiments'] else 'status-missing'}"></span>
                            {'완료' if self.report_data['advanced_experiments'] else '미실행'}</p>
                            <p>Partial Regression, ANOVA, Sobol 민감도, RSM</p>
                        </div>
                        <div class="analysis-card">
                            <h4>ML 분석</h4>
                            <p><span class="status-indicator {'status-completed' if self.report_data['ml_analysis'] else 'status-missing'}"></span>
                            {'완료' if self.report_data['ml_analysis'] else '미실행'}</p>
                            <p>다양한 ML 모델, Feature Importance, SHAP</p>
                        </div>
                        <div class="analysis-card">
                            <h4>기본 분석</h4>
                            <p><span class="status-indicator {'status-completed' if self.report_data['basic_analysis'] else 'status-missing'}"></span>
                            {'완료' if self.report_data['basic_analysis'] else '미실행'}</p>
                            <p>기초 통계, 상관관계, EDA</p>
                        </div>
                        <div class="analysis-card">
                            <h4>종합 분석</h4>
                            <p><span class="status-indicator {'status-completed' if self.report_data['comprehensive_analysis'] else 'status-missing'}"></span>
                            {'완료' if self.report_data['comprehensive_analysis'] else '미실행'}</p>
                            <p>구간별 분석, Yield-Input-Output 관계</p>
                        </div>
                    </div>
                </section>
                
                <section id="findings" class="section">
                    <h2>🔍 주요 발견사항</h2>
                    {self._generate_findings_html()}
                </section>
                
                <section id="recommendations" class="section">
                    <h2>💡 권장사항</h2>
                    {self._generate_recommendations_html()}
                </section>
                
                <section id="performance" class="section">
                    <h2>📈 모델 성능</h2>
                    {self._generate_performance_html()}
                </section>
                
                <section id="features" class="section">
                    <h2>🎯 특성 중요도</h2>
                    {self._generate_feature_importance_html()}
                </section>
                
                <section id="optimization" class="section">
                    <h2>⚡ 최적화 인사이트</h2>
                    {self._generate_optimization_html()}
                </section>
                
                <section id="conclusion" class="section">
                    <div class="conclusion">
                        <h3>🎯 종합 결론</h3>
                        <p>본 통합 분석을 통해 ACN 정제 공정의 Yield 최적화를 위한 종합적인 인사이트를 도출했습니다. 
                        다양한 분석 방법론을 통해 도출된 결과들을 종합하면, 다음과 같은 핵심 전략을 제안합니다:</p>
                        
                        <ul>
                            <li><strong>데이터 기반 의사결정:</strong> 통계적 유의성과 ML 모델 성능을 바탕으로 한 과학적 접근</li>
                            <li><strong>특성 우선순위 관리:</strong> 민감도 분석과 중요도 평가를 통한 효율적 자원 배분</li>
                            <li><strong>공정 최적화:</strong> RSM과 구간별 분석을 통한 최적 공정 조건 도출</li>
                            <li><strong>지속적 모니터링:</strong> 주요 영향 인자에 대한 실시간 모니터링 체계 구축</li>
                        </ul>
                        
                        <p>이러한 통합적 접근을 통해 ACN 정제 공정의 Yield를 체계적으로 개선할 수 있을 것으로 기대됩니다.</p>
                    </div>
                </section>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_findings_html(self):
        """주요 발견사항 HTML 생성"""
        if not self.analysis_summary['key_findings']:
            return "<p>주요 발견사항이 없습니다.</p>"
        
        findings_html = ""
        for finding in self.analysis_summary['key_findings']:
            findings_html += f'<div class="finding-box"><p>{finding}</p></div>'
        
        return findings_html
    
    def _generate_recommendations_html(self):
        """권장사항 HTML 생성"""
        if not self.analysis_summary['recommendations']:
            return "<p>권장사항이 없습니다.</p>"
        
        recommendations_html = ""
        for recommendation in self.analysis_summary['recommendations']:
            recommendations_html += f'<div class="recommendation-box"><p>{recommendation}</p></div>'
        
        return recommendations_html
    
    def _generate_performance_html(self):
        """모델 성능 HTML 생성"""
        performance = self.analysis_summary['model_performance']
        if not performance:
            return "<p>모델 성능 데이터가 없습니다.</p>"
        
        html = ""
        
        if 'ml_best_model' in performance:
            ml_model = performance['ml_best_model']
            html += f"""
            <div class="insight-box">
                <h4>ML 모델 최고 성능</h4>
                <p><strong>모델:</strong> {ml_model['name']}</p>
                <p><strong>R² Score:</strong> <span class="metric">{ml_model['r2']:.4f}</span></p>
                <p><strong>RMSE:</strong> {ml_model['rmse']:.4f}</p>
                <p><strong>CV Score:</strong> {ml_model['cv_score']:.4f}</p>
            </div>
            """
        
        if 'anova' in performance:
            anova = performance['anova']
            html += f"""
            <div class="insight-box">
                <h4>ANOVA 분석</h4>
                <p><strong>R² Score:</strong> <span class="metric">{anova['r_squared']:.4f}</span></p>
                <p><strong>유의한 특성 수:</strong> {anova['significant_features']}개</p>
            </div>
            """
        
        if 'rsm' in performance:
            rsm = performance['rsm']
            html += f"""
            <div class="insight-box">
                <h4>RSM 분석</h4>
                <p><strong>모델 타입:</strong> {rsm['model_type']}</p>
                <p><strong>R² Score:</strong> <span class="metric">{rsm['r_squared']:.4f}</span></p>
                <p><strong>CV Score:</strong> {rsm['cv_score']:.4f}</p>
            </div>
            """
        
        return html
    
    def _generate_feature_importance_html(self):
        """특성 중요도 HTML 생성"""
        importance = self.analysis_summary['feature_importance']
        if not importance:
            return "<p>특성 중요도 데이터가 없습니다.</p>"
        
        html = ""
        
        if 'ml_top_features' in importance:
            html += "<h4>ML 모델 특성 중요도 (상위 5개)</h4>"
            html += "<table><tr><th>특성</th><th>중요도 점수</th></tr>"
            for feature in importance['ml_top_features']:
                html += f"<tr><td>{feature['feature']}</td><td>{feature['score']:.4f}</td></tr>"
            html += "</table>"
        
        if 'partial_regression_top' in importance:
            html += "<h4>Partial Regression 상위 특성</h4>"
            html += "<table><tr><th>특성</th><th>상관계수</th><th>p-value</th></tr>"
            for feature in importance['partial_regression_top']:
                html += f"<tr><td>{feature['feature']}</td><td>{feature['correlation']:.4f}</td><td>{feature['p_value']:.4f}</td></tr>"
            html += "</table>"
        
        if 'sobol_top' in importance:
            html += "<h4>Sobol 민감도 상위 특성</h4>"
            html += "<table><tr><th>특성</th><th>1차 민감도 (S1)</th><th>총 민감도 (ST)</th></tr>"
            for feature in importance['sobol_top']:
                html += f"<tr><td>{feature['feature']}</td><td>{feature['s1']:.4f}</td><td>{feature['st']:.4f}</td></tr>"
            html += "</table>"
        
        return html
    
    def _generate_optimization_html(self):
        """최적화 인사이트 HTML 생성"""
        insights = self.analysis_summary['optimization_insights']
        if not insights:
            return "<p>최적화 인사이트 데이터가 없습니다.</p>"
        
        html = ""
        
        if 'rsm_optimal' in insights:
            optimal = insights['rsm_optimal']
            html += "<div class="insight-box">"
            html += "<h4>RSM 최적점</h4>"
            html += f"<p><strong>예측 최대 Yield:</strong> <span class="metric">{optimal['predicted_yield']:.4f}</span></p>"
            html += "<p><strong>최적 조건:</strong></p><ul>"
            for feature, value in optimal['optimal_point'].items():
                html += f"<li>{feature}: {value:.4f}</li>"
            html += "</ul></div>"
        
        if 'output_consistency' in insights:
            consistency = insights['output_consistency']
            html += f"""
            <div class="insight-box">
                <h4>Output 일정성 분석</h4>
                <p><strong>평균 Output:</strong> {consistency['mean']:.4f}</p>
                <p><strong>표준편차:</strong> {consistency['std']:.4f}</p>
                <p><strong>변동계수:</strong> <span class="metric">{consistency['cv']:.2f}%</span></p>
            </div>
            """
        
        return html

def create_integrated_report(advanced_exp_results=None,
                           ml_analysis_results=None, 
                           basic_analysis_results=None,
                           comprehensive_results=None,
                           output_file='acn_integrated_report.html'):
    """
    통합 리포트 생성 메인 함수
    
    Parameters:
    advanced_exp_results: main_advanced_experiments 결과
    ml_analysis_results: main_ml_analysis 결과
    basic_analysis_results: main_analysis 결과  
    comprehensive_results: main_comprehensive_analysis 결과
    output_file: 출력 파일명
    
    Returns:
    html_content: 생성된 HTML 내용
    """
    # 리포트 생성기 초기화
    generator = ACNIntegratedReportGenerator()
    
    # 모든 결과 통합
    generator.integrate_all_results(
        advanced_exp_results=advanced_exp_results,
        ml_analysis_results=ml_analysis_results,
        basic_analysis_results=basic_analysis_results,
        comprehensive_results=comprehensive_results
    )
    
    # HTML 리포트 생성
    html_content = generator.generate_integrated_html_report(output_file)
    
    return html_content

# 사용 예시
if __name__ == "__main__":
    print("ACN 정제 공정 통합 리포트 생성기를 시작합니다.")
    print("\n사용법:")
    print("1. 모든 분석 실행 후 결과 통합:")
    print("   html_content = create_integrated_report(")
    print("       advanced_exp_results=exp_results,")
    print("       ml_analysis_results=ml_results,")
    print("       basic_analysis_results=basic_results,")
    print("       comprehensive_results=comp_results")
    print("   )")
    print("\n2. 일부 분석만 있는 경우:")
    print("   html_content = create_integrated_report(")
    print("       advanced_exp_results=exp_results,")
    print("       ml_analysis_results=ml_results")
    print("   )")
