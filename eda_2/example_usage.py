#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
제조 산업 Ensemble 모델 EDA 사용 예제

이 파일은 업데이트된 EDA 기능의 다양한 사용 방법을 보여줍니다.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from manufacturing_ensemble_eda import run_manufacturing_eda, create_sample_data

def example_basic_usage():
    """기본 사용법 예제"""
    print("=== 기본 사용법 예제 ===")
    
    # 1. 샘플 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data(n_samples=1000, n_basic_tags=5)
    
    # 2. 모델 생성 및 훈련
    long_term_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    short_term_model = GradientBoostingRegressor(n_estimators=100, random_state=43)
    
    # 3. 스케일러 생성 및 학습
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    
    # 모델 훈련
    long_term_model.fit(X_a, y)
    short_term_model.fit(X_b, y)
    
    # 스케일러 학습
    scaler_a.fit(X_a)
    scaler_b.fit(X_b)
    
    # 4. EDA 실행 (HTML 리포트 포함)
    results = run_manufacturing_eda(
        X_a=X_a,
        X_b=X_b,
        y=y,
        long_term_model=long_term_model,
        short_term_model=short_term_model,
        scaler_a=scaler_a,
        scaler_b=scaler_b,
        basic_tags=basic_tags,
        threshold=70000,
        generate_html_report=True
    )
    
    print("분석 완료!")
    return results

def example_custom_threshold():
    """커스텀 threshold 사용 예제"""
    print("=== 커스텀 Threshold 예제 ===")
    
    # 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data(n_samples=1000, n_basic_tags=5)
    
    # 모델 및 스케일러 설정
    long_term_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    short_term_model = GradientBoostingRegressor(n_estimators=100, random_state=43)
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    
    # 훈련
    long_term_model.fit(X_a, y)
    short_term_model.fit(X_b, y)
    scaler_a.fit(X_a)
    scaler_b.fit(X_b)
    
    # 다른 threshold 값으로 분석
    custom_threshold = 50000
    results = run_manufacturing_eda(
        X_a=X_a,
        X_b=X_b,
        y=y,
        long_term_model=long_term_model,
        short_term_model=short_term_model,
        scaler_a=scaler_a,
        scaler_b=scaler_b,
        basic_tags=basic_tags,
        threshold=custom_threshold,
        generate_html_report=True
    )
    
    print(f"커스텀 threshold ({custom_threshold})로 분석 완료!")
    return results

def example_without_html_report():
    """HTML 리포트 없이 분석하는 예제"""
    print("=== HTML 리포트 없이 분석 ===")
    
    # 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data(n_samples=1000, n_basic_tags=5)
    
    # 모델 및 스케일러 설정
    long_term_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    short_term_model = GradientBoostingRegressor(n_estimators=100, random_state=43)
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    
    # 훈련
    long_term_model.fit(X_a, y)
    short_term_model.fit(X_b, y)
    scaler_a.fit(X_a)
    scaler_b.fit(X_b)
    
    # HTML 리포트 없이 분석
    results = run_manufacturing_eda(
        X_a=X_a,
        X_b=X_b,
        y=y,
        long_term_model=long_term_model,
        short_term_model=short_term_model,
        scaler_a=scaler_a,
        scaler_b=scaler_b,
        basic_tags=basic_tags,
        threshold=70000,
        generate_html_report=False
    )
    
    print("HTML 리포트 없이 분석 완료!")
    return results

def example_external_models():
    """외부에서 훈련된 모델을 사용하는 예제"""
    print("=== 외부 모델 사용 예제 ===")
    
    # 실제 사용 시나리오: 외부에서 훈련된 모델과 스케일러를 받아서 사용
    # 이 예제에서는 샘플 데이터로 시뮬레이션
    
    # 1. 외부에서 제공되는 데이터 (실제로는 파일에서 로드)
    X_a, X_b, y, basic_tags = create_sample_data(n_samples=1000, n_basic_tags=5)
    
    # 2. 외부에서 제공되는 훈련된 모델들 (실제로는 pickle 등으로 로드)
    long_term_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    short_term_model = GradientBoostingRegressor(n_estimators=100, random_state=43)
    
    # 3. 외부에서 제공되는 훈련된 스케일러들
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    
    # 모델 훈련 (실제로는 이미 훈련된 모델)
    long_term_model.fit(X_a, y)
    short_term_model.fit(X_b, y)
    scaler_a.fit(X_a)
    scaler_b.fit(X_b)
    
    # 4. EDA 실행
    results = run_manufacturing_eda(
        X_a=X_a,
        X_b=X_b,
        y=y,
        long_term_model=long_term_model,
        short_term_model=short_term_model,
        scaler_a=scaler_a,
        scaler_b=scaler_b,
        basic_tags=basic_tags,
        threshold=70000,
        generate_html_report=True
    )
    
    print("외부 모델을 사용한 분석 완료!")
    return results

def example_threshold_analysis():
    """Threshold 분석에 집중하는 예제"""
    print("=== Threshold 분석 예제 ===")
    
    from manufacturing_ensemble_eda import ManufacturingEnsembleEDA
    
    # 데이터 생성
    X_a, X_b, y, basic_tags = create_sample_data(n_samples=1000, n_basic_tags=5)
    
    # 모델 및 스케일러 설정
    long_term_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    short_term_model = GradientBoostingRegressor(n_estimators=100, random_state=43)
    scaler_a = StandardScaler()
    scaler_b = StandardScaler()
    
    # 훈련
    long_term_model.fit(X_a, y)
    short_term_model.fit(X_b, y)
    scaler_a.fit(X_a)
    scaler_b.fit(X_b)
    
    # EDA 객체 생성
    eda = ManufacturingEnsembleEDA(
        X_a=X_a,
        X_b=X_b,
        y=y,
        long_term_model=long_term_model,
        short_term_model=short_term_model,
        scaler_a=scaler_a,
        scaler_b=scaler_b,
        basic_tags=basic_tags,
        threshold=70000
    )
    
    # Threshold 분석만 실행
    threshold_results = eda.threshold_analysis()
    
    print("Threshold 분석 완료!")
    print(f"정확도: {threshold_results['accuracy']:.4f}")
    print(f"Precision: {threshold_results['precision']:.4f}")
    print(f"Recall: {threshold_results['recall']:.4f}")
    print(f"F1-Score: {threshold_results['f1_score']:.4f}")
    
    return threshold_results

def example_manual_html_report():
    """수동으로 HTML 리포트 생성하는 예제"""
    print("=== 수동 HTML 리포트 생성 예제 ===")
    
    # 분석 실행
    results = example_basic_usage()
    
    # 수동으로 HTML 리포트 생성
    try:
        from html_report_utils import generate_eda_html_report
        html_filename = generate_eda_html_report(results, "manual_report.html")
        print(f"수동으로 HTML 리포트 생성 완료: {html_filename}")
    except ImportError:
        print("html_report_utils.py 파일이 필요합니다.")
    
    return results

if __name__ == "__main__":
    print("제조 산업 Ensemble 모델 EDA 사용 예제")
    print("=" * 50)
    
    # 다양한 예제 실행
    print("\n1. 기본 사용법")
    example_basic_usage()
    
    print("\n2. 커스텀 Threshold")
    example_custom_threshold()
    
    print("\n3. HTML 리포트 없이 분석")
    example_without_html_report()
    
    print("\n4. 외부 모델 사용")
    example_external_models()
    
    print("\n5. Threshold 분석만")
    example_threshold_analysis()
    
    print("\n6. 수동 HTML 리포트 생성")
    example_manual_html_report()
    
    print("\n모든 예제 실행 완료!")
