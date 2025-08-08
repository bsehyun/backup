"""
제조 산업 센서 데이터 Ensemble 모델 EDA 리포트 생성 예제

이 예제는 EDA 분석 결과를 PDF와 HTML 리포트로 생성하는 방법을 보여줍니다.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from manufacturing_ensemble_eda_notebook import (
    create_sample_data, 
    run_manufacturing_eda_with_report
)

def example_basic_usage():
    """기본 사용법 예제"""
    print("=== 기본 EDA 분석 및 리포트 생성 ===")
    
    # 1. 샘플 데이터 생성
    print("1. 데이터 생성 중...")
    X_a, X_b, y, basic_tags = create_sample_data(n_samples=1000, n_basic_tags=5)
    print(f"   - Model A features: {X_a.shape[1]}개")
    print(f"   - Model B features: {X_b.shape[1]}개")
    print(f"   - Basic tags: {basic_tags}")
    
    # 2. 모델 생성 및 훈련
    print("\n2. 모델 훈련 중...")
    model_a = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model_b = GradientBoostingRegressor(n_estimators=100, random_state=43)
    
    model_a.fit(X_a, y)
    model_b.fit(X_b, y)
    print("   - 모델 훈련 완료")
    
    # 3. EDA 분석 및 리포트 생성
    print("\n3. EDA 분석 및 리포트 생성 중...")
    results = run_manufacturing_eda_with_report(
        X_a=X_a,
        X_b=X_b,
        y=y,
        model_a=model_a,
        model_b=model_b,
        basic_tags=basic_tags,
        threshold=0.5,
        generate_report=True,
        report_filename="example_eda_report"
    )
    
    print("\n4. 분석 완료!")
    print(f"   - 최고 성능 모델: {max(results['metrics'].keys(), key=lambda x: results['metrics'][x]['R2'])}")
    print(f"   - Model A 사용률: {np.mean(results['use_model_a']):.2%}")
    print(f"   - Model B 사용률: {np.mean(~results['use_model_a']):.2%}")
    
    return results

def example_custom_report():
    """커스텀 리포트 생성 예제"""
    print("\n=== 커스텀 리포트 생성 예제 ===")
    
    # 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data(n_samples=500, n_basic_tags=3)
    
    # 모델 훈련
    model_a = GradientBoostingRegressor(n_estimators=50, random_state=42)
    model_b = GradientBoostingRegressor(n_estimators=50, random_state=43)
    
    model_a.fit(X_a, y)
    model_b.fit(X_b, y)
    
    # 커스텀 리포트 생성
    results = run_manufacturing_eda_with_report(
        X_a=X_a,
        X_b=X_b,
        y=y,
        model_a=model_a,
        model_b=model_b,
        basic_tags=basic_tags,
        threshold=0.3,  # 다른 threshold 사용
        generate_report=True,
        report_filename="custom_eda_report"  # 커스텀 파일명
    )
    
    print("커스텀 리포트 생성 완료!")
    return results

def example_without_report():
    """리포트 없이 분석만 수행하는 예제"""
    print("\n=== 리포트 없이 분석만 수행 ===")
    
    # 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data(n_samples=200, n_basic_tags=2)
    
    # 모델 훈련
    model_a = GradientBoostingRegressor(n_estimators=30, random_state=42)
    model_b = GradientBoostingRegressor(n_estimators=30, random_state=43)
    
    model_a.fit(X_a, y)
    model_b.fit(X_b, y)
    
    # 리포트 생성 없이 분석만 수행
    results = run_manufacturing_eda_with_report(
        X_a=X_a,
        X_b=X_b,
        y=y,
        model_a=model_a,
        model_b=model_b,
        basic_tags=basic_tags,
        threshold=0.5,
        generate_report=False  # 리포트 생성 안함
    )
    
    print("분석 완료 (리포트 생성 안함)")
    return results

def example_manual_report():
    """수동으로 리포트 데이터를 준비하여 생성하는 예제"""
    print("\n=== 수동 리포트 생성 예제 ===")
    
    from eda_report_utils import generate_eda_reports
    
    # 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data(n_samples=300, n_basic_tags=4)
    
    # 모델 훈련
    model_a = GradientBoostingRegressor(n_estimators=40, random_state=42)
    model_b = GradientBoostingRegressor(n_estimators=40, random_state=43)
    
    model_a.fit(X_a, y)
    model_b.fit(X_b, y)
    
    # 수동으로 리포트 데이터 준비
    from manufacturing_ensemble_eda_notebook import (
        correlation_analysis, feature_importance_analysis, 
        model_performance_comparison, ensemble_decision_analysis,
        analyze_basic_tag_importance
    )
    
    # 각 분석 수행
    corr_matrix_a, corr_matrix_b, target_corr_a, target_corr_b = correlation_analysis(X_a, X_b, y)
    importance_df_a, importance_df_b = feature_importance_analysis(X_a, X_b, y, model_a, model_b, basic_tags)
    metrics, ensemble_pred, pred_a, pred_b, confidence_a = model_performance_comparison(X_a, X_b, y, model_a, model_b, 0.5)
    use_model_a, confidence_a = ensemble_decision_analysis(X_a, X_b, y, model_a, model_b, 0.5)
    
    # 리포트 데이터 구성
    plots_data = {
        'correlation': {
            'target_corr_a': target_corr_a,
            'target_corr_b': target_corr_b,
            'correlation_matrix_a': corr_matrix_a,
            'correlation_matrix_b': corr_matrix_b
        },
        'importance': {
            'importance_df_a': importance_df_a,
            'importance_df_b': importance_df_b,
            'basic_importance_a': analyze_basic_tag_importance(importance_df_a, basic_tags, 'Model A'),
            'basic_importance_b': analyze_basic_tag_importance(importance_df_b, basic_tags, 'Model B')
        },
        'performance': {
            'metrics': metrics,
            'pred_a': pred_a,
            'pred_b': pred_b,
            'ensemble_pred': ensemble_pred,
            'y': y,
            'residuals_a': y - pred_a,
            'residuals_b': y - pred_b,
            'residuals_ensemble': y - ensemble_pred
        },
        'ensemble': {
            'use_model_a': use_model_a,
            'confidence_a': confidence_a,
            'threshold': 0.5,
            'pred_a': pred_a,
            'pred_b': pred_b
        }
    }
    
    # 리포트 생성
    pdf_filename, html_filename = generate_eda_reports(plots_data, "manual_eda_report")
    print(f"수동 리포트 생성 완료:")
    print(f"  - PDF: {pdf_filename}")
    print(f"  - HTML: {html_filename}")
    
    return plots_data

if __name__ == "__main__":
    print("제조 산업 센서 데이터 Ensemble 모델 EDA 리포트 생성 예제")
    print("=" * 60)
    
    # 1. 기본 사용법
    results1 = example_basic_usage()
    
    # 2. 커스텀 리포트
    results2 = example_custom_report()
    
    # 3. 리포트 없이 분석만
    results3 = example_without_report()
    
    # 4. 수동 리포트 생성
    results4 = example_manual_report()
    
    print("\n" + "=" * 60)
    print("모든 예제 실행 완료!")
    print("\n생성된 파일들:")
    print("- example_eda_report.pdf / .html")
    print("- custom_eda_report.pdf / .html")
    print("- manual_eda_report.pdf / .html")
    
    print("\n사용법:")
    print("1. 기본 사용: run_manufacturing_eda_with_report() 호출")
    print("2. 리포트 없이: generate_report=False 설정")
    print("3. 커스텀 파일명: report_filename='my_report' 설정")
    print("4. 수동 생성: generate_eda_reports() 직접 호출")
